"""Raptor v5 with residual RadImageNet fusion inside CoAtNet.

Input is an already preprocessed [B, K, 3, 384, 384] triplet bag. Both
encoders receive exactly the same tensor; preprocessing belongs to the loader.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


COATNET_ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)


RAD_INPUTS = ("shared", "pm1_triplet", "pm1_middle")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FusionConfig:
    stage2_hidden: int = 1024
    stage4_hidden: int = 2048
    mlp_dropout: float = 0.0
    head_dropout: float = 0.2
    gradient_checkpointing: bool = False
    # What the RadImageNet trunk receives (CoAtNet always gets the ImageNet-normalized triplet):
    # 'shared' = the same triplet (original behaviour and the default for older checkpoints);
    # 'pm1_triplet' = the three slices rescaled to [-1, 1];
    # 'pm1_middle' = the middle slice in [-1, 1] copied to 3 channels (baseline Rad convention).
    rad_input: str = "shared"


def rad_input_transform(image: Tensor, mode: str) -> Tensor:
    """Map an ImageNet-normalized triplet [N,3,H,W] to the RadImageNet input convention."""
    if mode == "shared":
        return image
    if mode not in RAD_INPUTS:
        raise ValueError(f"Unknown rad_input {mode!r}; expected one of {RAD_INPUTS}")
    mean = torch.tensor(IMAGENET_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    pm1 = (image*std + mean)*2 - 1          # exact inverse of the ImageNet normalization, then [-1, 1]
    if mode == "pm1_middle":
        pm1 = pm1[:, 1:2].expand(-1, 3, -1, -1)
    return pm1.contiguous()


@contextmanager
def preserve_batchnorm_buffers(module: nn.Module):
    """Restore running statistics after a probe or checkpoint recomputation.

    BatchNorm, SyncBatchNorm and timm BatchNormAct subclasses keep their normal
    training behavior (including synchronization and trainable affine weights).
    Only running_mean, running_var and num_batches_tracked are restored, so a
    recomputed forward contributes gradients but does not count as a second
    observation. Enter this context for recomputation only, after the original
    forward has updated the statistics.
    """
    saved = [
        (buffer, buffer.detach().clone())
        for layer in module.modules()
        if isinstance(layer, nn.modules.batchnorm._BatchNorm)
        for name in ("running_mean", "running_var", "num_batches_tracked")
        if (buffer := getattr(layer, name, None)) is not None
    ]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in saved:
                buffer.copy_(value)


class ConcatenationMLP(nn.Module):
    """Three affine layers with GELU, applied to each spatial position.

    Concatenation precedes every learned operation. Only the output projection
    is zero-initialized, so the residual is exactly zero without a dead MLP.
    """

    def __init__(self, first_channels: int, second_channels: int,
                 out_channels: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        if min(first_channels, second_channels, out_channels, hidden) <= 0:
            raise ValueError("MLP channel dimensions must be positive")
        self.first_channels = first_channels
        self.second_channels = second_channels
        self.mlp = nn.Sequential(
            nn.LayerNorm(first_channels + second_channels),
            nn.Linear(first_channels + second_channels, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        if (first.ndim != 4 or second.ndim != 4
                or first.shape[0] != second.shape[0]
                or first.shape[2:] != second.shape[2:]
                or first.shape[1] != self.first_channels
                or second.shape[1] != self.second_channels):
            raise ValueError(f"Incompatible fusion features: {first.shape}, {second.shape}")
        joined = torch.cat((first, second), dim=1).permute(0, 2, 3, 1)
        return self.mlp(joined).permute(0, 3, 1, 2).contiguous()


class RadImageNetEncoder(nn.Module):
    """The same torchvision ResNet-50 trunk used by the baseline Rad encoder.

    Keep `backbone.0` ... `backbone.7` names for strict ResNet50.pt loading.
    conv3_x is backbone[5] (torchvision layer2); conv5_x is backbone[7]
    (layer4). No ImageNet weights are substituted for RadImageNet weights.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50

        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        x = image
        conv3 = None
        for index, layer in enumerate(self.backbone):
            x = layer(x)
            if index == 5:
                conv3 = x
        return conv3, x


class FusedCoAtNet(nn.Module):
    """Stage2 Z += F1(X,Z), Stage4 V += F2(V,Y), then normal CoAtNet pooling."""

    def __init__(self, coatnet: nn.Module, radimagenet: nn.Module,
                 config: FusionConfig = FusionConfig()):
        super().__init__()
        self.coatnet = coatnet
        self.radimagenet = radimagenet
        self.num_features = coatnet.num_features
        channels = [entry["num_chs"] for entry in coatnet.feature_info]
        if len(coatnet.stages) != 4 or channels[-4:] != [128, 256, 512, 1024]:
            raise ValueError("Expected the four-stage Raptor v5 CoAtNet RMLP-2")
        self.f1 = ConcatenationMLP(512, 256, 256, config.stage2_hidden, config.mlp_dropout)
        self.f2 = ConcatenationMLP(1024, 2048, 1024, config.stage4_hidden, config.mlp_dropout)
        self.gradient_checkpointing = config.gradient_checkpointing
        if config.rad_input not in RAD_INPUTS:
            raise ValueError(f"Unknown rad_input {config.rad_input!r}; expected one of {RAD_INPUTS}")
        self.rad_input = config.rad_input
        self.checkpoint_parts: frozenset[str] = frozenset()
        self.requires_grad_(True)
        self.coatnet.set_grad_checkpointing(config.gradient_checkpointing)

    CHECKPOINT_PARTS = ("rad", "s1", "s2", "s3", "s4", "fusion")

    def set_checkpoint_parts(self, parts) -> None:
        """Runtime-only activation recomputation for selected whole sections.

        Parts: rad (ResNet-50), s1 (stem+S1), s2, s3, s4 (CoAtNet stages) and
        fusion (both MLPs). Not stored in checkpoints; numerics are unchanged.
        These whole-section checkpoints preserve BatchNorm running buffers
        during recomputation. This does not cover timm's separate internal
        checkpointing; leave config.gradient_checkpointing=False when using
        trainable BatchNorm statistics with this path.
        """
        parts = frozenset(parts)
        unknown = parts - set(self.CHECKPOINT_PARTS)
        if unknown:
            raise ValueError(f"Unknown checkpoint parts: {sorted(unknown)}")
        self.checkpoint_parts = parts

    def _run(self, part: str, fn, *args):
        if part in self.checkpoint_parts and self.training and torch.is_grad_enabled():
            # s1 is a bound method spanning stem and stage 1; the other sections
            # are individual modules. Fusion MLPs have no BatchNorm buffers.
            scope = fn if isinstance(fn, nn.Module) else self.coatnet
            return checkpoint(
                fn, *args, use_reentrant=False,
                context_fn=lambda: (nullcontext(), preserve_batchnorm_buffers(scope)),
            )
        return fn(*args)

    def _stem_s1(self, image: Tensor) -> Tensor:
        return self.coatnet.stages[0](self.coatnet.stem(image))

    def _fuse(self, mlp: nn.Module, first: Tensor, second: Tensor) -> Tensor:
        if ((self.gradient_checkpointing or "fusion" in self.checkpoint_parts)
                and self.training and torch.is_grad_enabled()):
            return checkpoint(mlp, first, second, use_reentrant=False)
        return mlp(first, second)

    def forward_features(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 384, 384):
            raise ValueError("Both backbones require [N, 3, 384, 384] preprocessed triplets")
        # CoAtNet gets the triplet as saved; RadImageNet gets it in its configured convention.
        x, y = self._run("rad", self.radimagenet, rad_input_transform(image, self.rad_input))
        z = self._run("s1", self._stem_s1, image)
        z = self._run("s2", self.coatnet.stages[1], z)  # S2 convolution, [N,256,48,48]
        z = z + self._fuse(self.f1, x, z)  # F1 input order is (X,Z)
        v = self._run("s3", self.coatnet.stages[2], z)  # S3 receives the UPDATED Z
        v = self._run("s4", self.coatnet.stages[3], v)  # S4 transformer, [N,1024,12,12]
        v = v + self._fuse(self.f2, v, y)  # F2 input order is (V,Y)
        return self.coatnet.norm(v)

    def forward(self, image: Tensor) -> Tensor:
        return self.coatnet.forward_head(self.forward_features(image))


class RaptorClassifier(nn.Module):
    """Original Raptor attention-MIL head, retaining checkpoint parameter names."""

    def __init__(self, backbone: nn.Module, F_dim: int, n: int = 12, drop: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("Raptor expects a nonempty [B,K,3,384,384] triplet bag")
        b, k = x.shape[:2]
        return self.backbone(x.flatten(0, 1)).reshape(b, k, -1)

    def head(self, feats: Tensor) -> Tensor:
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encode(x))


def _checkpoint(source: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    return source if isinstance(source, Mapping) else torch.load(
        source, map_location="cpu", weights_only=True,
    )


def build_raptor_v5_radimagenet(
    raptor_checkpoint: str | Path | Mapping[str, Any],
    radimagenet_checkpoint: str | Path | Mapping[str, Any],
    config: FusionConfig = FusionConfig(),
) -> RaptorClassifier:
    """Strict-load both pretrained parents, add zero residuals, unfreeze all.

    Raptor checkpoint: {'model': state_dict, 'arch': ..., 'res': 384}.
    Rad checkpoint: the baseline ResNet50.pt state dict, or {'state_dict': ...}.
    Load and resume a *fused* checkpoint using load_fused_checkpoint instead.
    """
    import timm

    ck = _checkpoint(raptor_checkpoint)
    if ck.get("arch", COATNET_ARCH) != COATNET_ARCH or int(ck.get("res", 384)) != 384:
        raise ValueError("Parent checkpoint must be Raptor v5 CoAtNet RMLP-2 at res=384")
    coatnet = timm.create_model(COATNET_ARCH, pretrained=False, num_classes=0, in_chans=3, global_pool="avg")
    model = RaptorClassifier(coatnet, F_dim=coatnet.num_features, drop=config.head_dropout)
    model.load_state_dict(ck["model"], strict=True)
    rad = RadImageNetEncoder()
    rad_ck = _checkpoint(radimagenet_checkpoint)
    rad.load_state_dict(rad_ck.get("state_dict", rad_ck), strict=True)
    model.backbone = FusedCoAtNet(coatnet, rad, config)
    model.fusion_config = config
    model.requires_grad_(True)
    return model


def optimizer_groups(model: RaptorClassifier, backbone_lr: float = 1e-5,
                     fusion_lr: float = 1e-4, head_lr: float = 1e-4) -> list[dict[str, Any]]:
    """All parameters, exactly once, with both backbones unfrozen."""
    if not isinstance(model.backbone, FusedCoAtNet):
        raise TypeError("Expected the fused Raptor model")
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError("Both backbones and the fusion/head must remain unfrozen")
    groups = {"coatnet": [], "radimagenet": [], "fusion": [], "head": []}
    for name, parameter in model.named_parameters():
        if name.startswith("backbone.coatnet."):
            group = "coatnet"
        elif name.startswith("backbone.radimagenet."):
            group = "radimagenet"
        elif name.startswith(("backbone.f1.", "backbone.f2.")):
            group = "fusion"
        else:
            group = "head"
        groups[group].append(parameter)
    rates = dict(coatnet=backbone_lr, radimagenet=backbone_lr, fusion=fusion_lr, head=head_lr)
    return [{"name": name, "params": parameters, "lr": rates[name]}
            for name, parameters in groups.items()]


def fused_checkpoint(model: RaptorClassifier) -> dict[str, Any]:
    """Serialize model/config; trainers can add optimizer, scaler and epoch.

    Tensors are copied to CPU: state_dict() alone shares storage with the live parameters, so
    swapping weights back afterwards (e.g. EMA -> raw) would silently change the checkpoint.
    """
    state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    return {"format": "raptor_v5_radimagenet_v1", "arch": COATNET_ARCH, "res": 384,
            "labels": list(LABELS), "fusion_config": asdict(model.fusion_config),
            "model": state}


def load_fused_checkpoint(source: str | Path | Mapping[str, Any]) -> RaptorClassifier:
    import timm

    ck = _checkpoint(source)
    if (ck.get("format") != "raptor_v5_radimagenet_v1" or ck.get("arch") != COATNET_ARCH
            or ck.get("res") != 384 or ck.get("labels") != list(LABELS)):
        raise ValueError("Unsupported fused checkpoint architecture/input/label contract")
    config = FusionConfig(**ck["fusion_config"])
    coatnet = timm.create_model(COATNET_ARCH, pretrained=False, num_classes=0, in_chans=3, global_pool="avg")
    backbone = FusedCoAtNet(coatnet, RadImageNetEncoder(), config)
    model = RaptorClassifier(backbone, F_dim=backbone.num_features, drop=config.head_dropout)
    model.fusion_config = config
    model.load_state_dict(ck["model"], strict=True)
    model.requires_grad_(True)
    return model

"""CoAtNet-only Raptor v5 with the original per-finding attention-MIL head.

Consumes existing preprocessed [B, K, 3, 384, 384] triplet bags. Loading an old
fusion checkpoint keeps its trained CoAtNet/head tensors, without constructing
the ResNet encoder or either fusion block. All checkpoint loads are offline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


COATNET_ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
MODEL_FORMAT = "raptor_v5_coatnet_v1"
LEGACY_FUSED_FORMAT = "raptor_v5_resnet50_v3"
LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)


@dataclass(frozen=True)
class RaptorConfig:
    head_dropout: float = 0.2
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if not 0 <= self.head_dropout < 1:
            raise ValueError("head_dropout must be in [0, 1)")


class RaptorClassifier(nn.Module):
    """Original Raptor attention-MIL head, retaining its parameter names."""

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
        if (x.ndim != 5 or x.shape[0] == 0 or x.shape[1] == 0
                or tuple(x.shape[2:]) != (3, 384, 384)):
            raise ValueError("Raptor expects nonempty [B,K,3,384,384] preprocessed triplet bags")
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
    ck = source if isinstance(source, Mapping) else torch.load(
        source, map_location="cpu", weights_only=True,
    )
    if not isinstance(ck, Mapping):
        raise ValueError("Expected a checkpoint mapping")
    return ck


def _model_state(ck: Mapping[str, Any], weights: str) -> Mapping[str, Tensor]:
    if weights == "raw":
        state = ck.get("model")
    elif weights == "ema":
        ema = ck.get("ema")
        state = ema.get("shadow") if isinstance(ema, Mapping) else None
        if state is None:
            raise ValueError("Checkpoint has no embedded EMA shadow; use raw for a standalone EMA snapshot")
    else:
        raise ValueError("weights must be 'raw' or 'ema'")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint has no model state dictionary")
    return state


def _coatnet_state(state: Mapping[str, Tensor], legacy_fused: bool) -> dict[str, Tensor]:
    if not legacy_fused:
        return dict(state)
    kept = {}
    for name, value in state.items():
        if name.startswith("backbone.coatnet."):
            kept["backbone." + name.removeprefix("backbone.coatnet.")] = value
        elif name.startswith(("backbone.resnet50.", "backbone.f1.", "backbone.f2.")):
            continue
        elif name.startswith("backbone."):
            raise ValueError(f"Unexpected legacy backbone tensor: {name}")
        else:
            kept[name] = value
    return kept


def load_raptor_checkpoint(
    source: str | Path | Mapping[str, Any], *, weights: str = "raw",
    config: RaptorConfig | None = None,
) -> RaptorClassifier:
    """Load an original parent, a CoAtNet-only checkpoint, or strip v3 fusion.

    ``weights='ema'`` selects a trainer checkpoint's embedded ``ema.shadow``.
    A standalone EMA checkpoint stores its selected weights in ``model`` and
    uses the default ``weights='raw'``. Optimizer state is never migrated.
    """
    import timm

    ck = _checkpoint(source)
    fmt = ck.get("format")
    if fmt not in (None, MODEL_FORMAT, LEGACY_FUSED_FORMAT):
        raise ValueError(f"Unsupported Raptor checkpoint format: {fmt}")
    if ck.get("arch", COATNET_ARCH) != COATNET_ARCH or ck.get("res", 384) != 384:
        raise ValueError("Expected Raptor v5 CoAtNet RMLP-2 at res=384")
    if fmt is not None and any(key not in ck for key in ("arch", "res", "labels")):
        raise ValueError("Formatted checkpoint must declare architecture, resolution and label order")
    if "labels" in ck and list(ck["labels"]) != list(LABELS):
        raise ValueError("Checkpoint label order differs from Raptor v5")
    state = _coatnet_state(_model_state(ck, weights), fmt == LEGACY_FUSED_FORMAT)
    if config is None:
        if fmt == MODEL_FORMAT:
            config = RaptorConfig(**ck["raptor_config"])
        else:
            previous = ck.get("fusion_config", {})
            config = RaptorConfig(**{key: previous[key] for key in
                ("head_dropout", "gradient_checkpointing") if key in previous})
    backbone = timm.create_model(COATNET_ARCH, pretrained=False, num_classes=0,
                                 in_chans=3, global_pool="avg")
    backbone.set_grad_checkpointing(config.gradient_checkpointing)
    model = RaptorClassifier(backbone, F_dim=backbone.num_features, drop=config.head_dropout)
    model.load_state_dict(state, strict=True)
    model.raptor_config = config
    model.requires_grad_(True)
    return model


def optimizer_groups(model: RaptorClassifier, backbone_lr: float = 1e-5,
                     head_lr: float = 1e-4) -> list[dict[str, Any]]:
    """CoAtNet and the unchanged MIL head, each trainable parameter exactly once."""
    if not isinstance(model, RaptorClassifier) or not hasattr(model, "raptor_config"):
        raise TypeError("Expected a loaded CoAtNet-only Raptor model")
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError("CoAtNet and the head must remain unfrozen")
    groups = {"coatnet": [], "head": []}
    for name, parameter in model.named_parameters():
        groups["coatnet" if name.startswith("backbone.") else "head"].append(parameter)
    return [{"name": name, "params": parameters,
             "lr": backbone_lr if name == "coatnet" else head_lr}
            for name, parameters in groups.items()]


def raptor_checkpoint(model: RaptorClassifier) -> dict[str, Any]:
    """Serialize only CoAtNet and the MIL head, with their input/label contract."""
    if not isinstance(model, RaptorClassifier) or not hasattr(model, "raptor_config"):
        raise TypeError("Expected a loaded CoAtNet-only Raptor model")
    return {"format": MODEL_FORMAT, "arch": COATNET_ARCH, "res": 384,
            "labels": list(LABELS), "raptor_config": asdict(model.raptor_config),
            "model": model.state_dict()}

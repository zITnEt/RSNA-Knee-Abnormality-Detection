"""Codex points 1 and 6.

1. Checkpoints hold copies of the weights (fused_checkpoint clones to CPU), so an EMA "best"
   checkpoint really contains the EMA weights; the saved file is re-read and compared with the
   selected weights; a resumed run carries the previous fusion_best.pt into its outputs.
6. RadImageNet input convention: FusionConfig.rad_input selects what the ResNet-50 trunk sees.
   'shared'      - previous behaviour: the ImageNet-normalized triplet given to CoAtNet;
   'pm1_triplet' - the same three slices rescaled to [-1, 1] (u*2-1);
   'pm1_middle'  - the middle slice rescaled to [-1, 1] and copied to 3 channels
                   (the baseline Rad convention: uint8/127.5 - 1, grayscale replicated).
   Checkpoints without the field load as 'shared', so existing weights behave exactly as before.
   Training defaults now use 'pm1_middle'.

    python scripts/patch_codex_1_6.py
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path, replacements, marker):
    text = path.read_text(encoding='utf-8').replace('\r\n', '\n')
    if marker in text:
        print(path.name, 'already patched', hashlib.sha256(text.encode()).hexdigest())
        return
    for old, new in replacements:
        if text.count(old) != 1:
            raise SystemExit(f'{path.name}: expected one match for {old[:70]!r}, found {text.count(old)}')
        text = text.replace(old, new)
    path.write_bytes(text.encode('utf-8'))
    print(path.name, hashlib.sha256(text.encode()).hexdigest())


patch(ROOT/'models/raptor_v5_radimagenet.py', [
    ('''@dataclass(frozen=True)
class FusionConfig:
    stage2_hidden: int = 1024
    stage4_hidden: int = 2048
    mlp_dropout: float = 0.0
    head_dropout: float = 0.2
    gradient_checkpointing: bool = False
''', '''RAD_INPUTS = ("shared", "pm1_triplet", "pm1_middle")
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
'''),
    ('''        self.gradient_checkpointing = config.gradient_checkpointing
        self.checkpoint_parts: frozenset[str] = frozenset()''',
     '''        self.gradient_checkpointing = config.gradient_checkpointing
        if config.rad_input not in RAD_INPUTS:
            raise ValueError(f"Unknown rad_input {config.rad_input!r}; expected one of {RAD_INPUTS}")
        self.rad_input = config.rad_input
        self.checkpoint_parts: frozenset[str] = frozenset()'''),
    ('''        # Feed the identical image object to both trunks; no second normalization.
        x, y = self._run("rad", self.radimagenet, image)''',
     '''        # CoAtNet gets the triplet as saved; RadImageNet gets it in its configured convention.
        x, y = self._run("rad", self.radimagenet, rad_input_transform(image, self.rad_input))'''),
    ('''def fused_checkpoint(model: RaptorClassifier) -> dict[str, Any]:
    """Serialize model/config; trainers can add optimizer, scaler and epoch."""
    return {"format": "raptor_v5_radimagenet_v1", "arch": COATNET_ARCH, "res": 384,
            "labels": list(LABELS), "fusion_config": asdict(model.fusion_config),
            "model": model.state_dict()}''',
     '''def fused_checkpoint(model: RaptorClassifier) -> dict[str, Any]:
    """Serialize model/config; trainers can add optimizer, scaler and epoch.

    Tensors are copied to CPU: state_dict() alone shares storage with the live parameters, so
    swapping weights back afterwards (e.g. EMA -> raw) would silently change the checkpoint.
    """
    state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    return {"format": "raptor_v5_radimagenet_v1", "arch": COATNET_ARCH, "res": 384,
            "labels": list(LABELS), "fusion_config": asdict(model.fusion_config),
            "model": state}'''),
], marker='def rad_input_transform')


VERIFY = '''
                saved = torch.load('fusion_best.pt', map_location='cpu', weights_only=True)['model']
                names = [n for n, _ in model.named_parameters()]
                if len(names) != len(source) or not all(
                        torch.equal(saved[n].float(), s.detach().float().cpu()) for n, s in zip(names, source)):
                    raise RuntimeError('fusion_best.pt does not contain the selected weights')
                del saved'''

for script, indent_marker in (('train_fusion_ddp.py', 'ddp'), ('train_fusion_v2.py', 'v2')):
    patch(ROOT/'scripts'/script, [
        ("    smoke=False, smoke_updates=25, max_skipped_microbatches=20,\n)",
         "    smoke=False, smoke_updates=25, max_skipped_microbatches=20, rad_input='pm1_middle',\n)")
        if script == 'train_fusion_v2.py' else
        ("    max_hours=11.0, save_every_steps=100, keep_epoch_ema=False)}",
         "    max_hours=11.0, save_every_steps=100, keep_epoch_ema=False, rad_input='pm1_middle')}"),
        ('''                if kind == 'ema':
                    backup = ema.swap_in(model)
                    ck = fused_checkpoint(model)
                    ema.restore(model, backup)
                else:
                    ck = fused_checkpoint(model)
                ck.update(selection=best, training_config=cfg, recipe_sha256=RECIPE_SHA)
                atomic_save(ck, 'fusion_best.pt')''',
         '''                if kind == 'ema':
                    backup = ema.swap_in(model)
                    try:
                        ck = fused_checkpoint(model)   # copies taken while the EMA weights are installed
                    finally:
                        ema.restore(model, backup)
                    source = ema.params
                else:
                    ck = fused_checkpoint(model)
                    source = list(model.parameters())
                ck.update(selection=best, training_config=cfg, recipe_sha256=RECIPE_SHA)
                atomic_save(ck, 'fusion_best.pt')''' + VERIFY),
        ('''        model = build_raptor_v5_radimagenet(asset('raptor-knee-maxspan', 'raptor_ft_coatnet_v5_full_swa.pt'),
                                            rad, FusionConfig(gradient_checkpointing=False))
    model = model''',
         '''        model = build_raptor_v5_radimagenet(asset('raptor-knee-maxspan', 'raptor_ft_coatnet_v5_full_swa.pt'),
                                            rad, FusionConfig(gradient_checkpointing=False,
                                                              rad_input=cfg['rad_input']))
    model = model'''),
        ('''            if resume['training_config'][key] != cfg[key]:
                raise ValueError(f'Resume setting differs: {key}')
        model = load_fused_checkpoint(resume)''',
         '''            if resume['training_config'][key] != cfg[key]:
                raise ValueError(f'Resume setting differs: {key}')
        if resume['training_config'].get('rad_input', 'shared') != cfg['rad_input']:
            raise ValueError('Resume setting differs: rad_input (set it to the checkpoint value)')
        model = load_fused_checkpoint(resume)
        prior_best = Path(cfg['resume_path']).with_name('fusion_best.pt')
        if RANK == 0 and prior_best.is_file() and not Path('fusion_best.pt').exists():
            import shutil
            shutil.copyfile(prior_best, 'fusion_best.pt')   # keep the previous session's best in this output
            event('prior_best_carried', source=str(prior_best))''' if script == 'train_fusion_ddp.py' else
         '''            if resume['training_config'][key] != cfg[key]:
                raise ValueError(f'Resume setting differs: {key}')
        if resume['training_config'].get('rad_input', 'shared') != cfg['rad_input']:
            raise ValueError('Resume setting differs: rad_input (set it to the checkpoint value)')
        model = load_fused_checkpoint(resume)
        prior_best = Path(cfg['resume_path']).with_name('fusion_best.pt')
        if prior_best.is_file() and not Path('fusion_best.pt').exists():
            import shutil
            shutil.copyfile(prior_best, 'fusion_best.pt')   # keep the previous session's best in this output
            event('prior_best_carried', source=str(prior_best))'''),
    ], marker="rad_input='pm1_middle'")

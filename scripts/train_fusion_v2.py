"""Fusion fine-tune v2: fp16, frozen BN statistics, per-slot K=12 bags, GPU augmentation,
warmup+cosine, EMA, and gold74/validation AUC evaluation including a step-0 parent control.

Images are the saved, already-normalized float16 tensors. Augmentation is applied on the GPU
to training bags only; evaluation sees the saved tensors unchanged.
"""
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models.raptor_v5_radimagenet import (LABELS, FusionConfig, FusedCoAtNet, build_raptor_v5_radimagenet,
                                          optimizer_groups, fused_checkpoint, load_fused_checkpoint)
from raptor_data.ready_tensors import ReadyTripletDataset

RECIPE_SHA = '26ba1f6d2e41f37150239c8d46f738f2a0eca9e6b5a89a7542724c9f4c44aaca'
RAD_SHA = '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'
DEFAULTS = dict(
    epochs=4, k=12, sampling='per_slot', train_windows='pure', eval_windows='pure',
    seed=20261009, validation_fraction=0.1, accumulate=16, workers=3, gpus=1,
    backbone_lr=1e-5, fusion_lr=1e-4, head_lr=3e-5, weight_decay=0.01,
    warmup_steps=50, min_lr_ratio=0.01, ema_decay=0.998, clip_norm=3.0,
    amp='fp16', freeze_bn_stats=True, checkpoint_parts='auto', attention='auto', memory_limit_gib=13.5,
    augment=True, aug_rotate_deg=10.0, aug_scale=0.10, aug_shift=0.05,
    aug_contrast=0.10, aug_gamma=0.10, aug_brightness=0.05, aug_reverse_p=0.5,
    eval_chunk=16, eval_at_start=True, validation_eval='final',
    max_hours=10.5, save_every_steps=100, resume_path=None,
    smoke=False, smoke_updates=25, max_skipped_microbatches=20, rad_input='pm1_middle',
)
CHECKPOINT_LADDER = ((), ('s3', 's4'), ('s2', 's3', 's4', 'fusion'),
                     ('rad', 's1', 's2', 's3', 's4', 'fusion'))
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- utilities
def event(kind, **data):
    record = dict(event=kind, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **data)
    print(json.dumps(record), flush=True)
    with Path('training_events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record)+'\n')
    Path('progress.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


def atomic_save(data, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def freeze_bn_stats(model):
    """Put every BatchNorm in eval mode: running statistics are used and never updated.
    Affine weights stay trainable. Call after every model.train()."""
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def set_train_mode(model, cfg):
    model.train()
    return freeze_bn_stats(model) if cfg['freeze_bn_stats'] else 0


def lr_lambda_factory(warmup, total, min_ratio):
    warmup = max(0, int(warmup))
    total = max(total, warmup+1)

    def factor(step):
        if warmup and step < warmup:
            return min_ratio + (1-min_ratio)*(step+1)/warmup
        progress = min(1.0, (step-warmup)/max(1, total-warmup))
        return min_ratio + (1-min_ratio)*0.5*(1+math.cos(math.pi*progress))
    return factor


class EMA:
    """Float32 exponential moving average of all parameters (buffers are copied; BN is frozen)."""
    def __init__(self, model, decay):
        self.decay = decay
        self.params = [p.detach().clone().float() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1+step)/(10+step))
        current = [p.detach().float() if p.dtype != torch.float32 else p.detach() for p in model.parameters()]
        torch._foreach_mul_(self.params, d)
        torch._foreach_add_(self.params, current, alpha=1-d)

    @torch.no_grad()
    def swap_in(self, model):
        backup = [p.detach().to('cpu', copy=True) for p in model.parameters()]
        for p, e in zip(model.parameters(), self.params):
            p.copy_(e)
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for p, b in zip(model.parameters(), backup):
            p.copy_(b.to(p.device))

    def state_dict(self):
        return dict(decay=self.decay, params=[p.cpu() for p in self.params])

    def load_state_dict(self, state, device):
        self.decay = state['decay']
        self.params = [p.to(device) for p in state['params']]


class AutocastModule(torch.nn.Module):
    """Runs the wrapped module under autocast inside each DataParallel worker thread."""
    def __init__(self, module, dtype, enabled=True):
        super().__init__()
        self.module, self.dtype, self.enabled = module, dtype, enabled

    def forward(self, *args):
        with torch.autocast('cuda', dtype=self.dtype, enabled=self.enabled):
            return self.module(*args)


# ----------------------------------------------------------------------------- augmentation
def gpu_augment(bag, cfg, generator=None):
    """bag: [B,K,3,H,W] normalized tensor on GPU. Returns augmented tensor, same dtype/shape.

    Per window: random channel-order reversal (slice order), rotation/scale/shift, and
    contrast/brightness/gamma in [0,1] intensity space. No horizontal flip (medial/lateral)."""
    shape, dtype, device = bag.shape, bag.dtype, bag.device
    x = bag.reshape(-1, *shape[-3:]).float()
    n = x.shape[0]
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    u = x*std + mean

    def uniform(lo, hi):
        return lo + (hi-lo)*torch.rand(n, device=device, generator=generator)

    reverse = torch.rand(n, device=device, generator=generator) < cfg['aug_reverse_p']
    if reverse.any():
        u = torch.where(reverse.view(n, 1, 1, 1), u.flip(1), u)
    angle = uniform(-cfg['aug_rotate_deg'], cfg['aug_rotate_deg'])*math.pi/180
    scale = uniform(1-cfg['aug_scale'], 1+cfg['aug_scale'])
    tx = uniform(-cfg['aug_shift'], cfg['aug_shift'])*2   # grid coordinates span [-1, 1]
    ty = uniform(-cfg['aug_shift'], cfg['aug_shift'])*2
    cos, sin = torch.cos(angle)/scale, torch.sin(angle)/scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(u.shape), align_corners=False)
    u = F.grid_sample(u, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    gamma = torch.exp(uniform(-math.log1p(cfg['aug_gamma']), math.log1p(cfg['aug_gamma']))).view(n, 1, 1, 1)
    contrast = uniform(1-cfg['aug_contrast'], 1+cfg['aug_contrast']).view(n, 1, 1, 1)
    brightness = uniform(-cfg['aug_brightness'], cfg['aug_brightness']).view(n, 1, 1, 1)
    u = u.clamp(0, 1).pow(gamma)
    u = ((u-0.5)*contrast + 0.5 + brightness).clamp(0, 1)
    return ((u-mean)/std).to(dtype).reshape(shape)


# ----------------------------------------------------------------------------- data/labels
def discover_roots():
    roots = []
    for home in [Path('/kaggle/input'), Path('/kaggle/input/datasets/zafarnematov')]:
        roots.extend(home.glob('rsna-six-slot-raptor-v5-shard-*'))
    for home in [Path('/kaggle/input'), Path('/kaggle/input/notebooks/zafarnematov')]:
        for notebook in home.glob('rsna-six-slot-raptor-v5-preprocessing-*'):
            roots.extend(notebook.glob('six_slot_raptor_shard_*'))
    roots = sorted(set(p for p in roots if (p/'recipe.json').is_file()))
    if len(roots) != 32:
        raise ValueError(f'Expected all 32 completed shards; found {len(roots)}')
    for root in roots:
        recipe = json.loads((root/'recipe.json').read_text())
        manifest = json.loads((root/'manifest.json').read_text())
        if recipe['sha256'] != RECIPE_SHA or not recipe['normalized'] or not manifest['complete']:
            raise ValueError(f'Incorrect/incomplete preprocessing shard: {root}')
    return roots


def asset(slug, filename, base='/kaggle/input'):
    paths = sorted(Path(base).rglob(filename))
    paths = [p for p in paths if slug in str(p)]
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected one {slug}/{filename}; found {paths}')
    return paths[0]


def load_label_table(path, required_ids, allow_missing=False, name='labels'):
    table = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    if table.StudyInstanceUID.duplicated().any():
        raise ValueError(f'Duplicate study IDs in {name}')
    table = table.set_index('StudyInstanceUID')
    missing = set(required_ids) - set(table.index)
    if missing:
        raise ValueError(f'{name} lacks {len(missing)} required studies')
    values = table.loc[list(required_ids), list(LABELS)].to_numpy(dtype=np.float32)
    finite = np.isfinite(values)
    if not allow_missing and not finite.all():
        raise ValueError(f'Missing values in {name}')
    if not ((values[finite] >= 0) & (values[finite] <= 1)).all():
        raise ValueError(f'Out-of-range values in {name}')
    return dict(zip(required_ids, values))


def split_indices(n, seed, fraction):
    order = np.random.default_rng(seed).permutation(n)
    nval = round(n*fraction)
    if not 0 < nval < n:
        raise ValueError('Validation fraction must leave nonempty training and validation sets')
    return sorted(order[nval:].tolist()), sorted(order[:nval].tolist())


def macro_auc(y_true, y_score):
    """Per-finding ROC AUC on available (non-NaN) labels; findings with one class are skipped."""
    from sklearn.metrics import roc_auc_score
    per = {}
    for j, name in enumerate(LABELS):
        mask = np.isfinite(y_true[:, j])
        truth = y_true[mask, j]
        if mask.sum() and 0 < truth.sum() < len(truth):
            per[name] = float(roc_auc_score(truth, y_score[mask, j]))
    return (float(np.mean(list(per.values()))) if per else float('nan')), per


# ----------------------------------------------------------------------------- evaluation
@torch.inference_mode()
def predict(model, dataset, cfg, dtype, workers):
    """Full-bag logits for every study in dataset, splitting each bag over all GPUs."""
    model.eval()
    devices = list(range(min(torch.cuda.device_count(), cfg['gpus'])))
    replicas = [model.backbone] + [copy.deepcopy(model.backbone).to(f'cuda:{d}') for d in devices[1:]]
    wrapped = [AutocastModule(r, dtype) for r in replicas]
    event('predict_started', studies=len(dataset), devices=len(devices), chunk=cfg['eval_chunk'], workers=workers)
    loader = DataLoader(dataset, batch_size=1, num_workers=workers, pin_memory=True,
                        prefetch_factor=2 if workers else None)
    uids, logits = [], []
    size = cfg['eval_chunk']*len(devices)
    started, wait_s, gpu_s = time.monotonic(), 0.0, 0.0
    fetch = time.monotonic()
    for index, batch in enumerate(loader):
        got = time.monotonic()
        wait_s += got-fetch
        images = batch['images'][0]
        # Every chunk is padded to the same [eval_chunk*GPUs] shape (repeating the last window, whose
        # features are then discarded) so cudnn autotuning runs once instead of per bag length.
        n_windows = len(images)
        padded = -(-n_windows//size)*size
        if padded > n_windows:
            images = torch.cat([images, images[-1:].expand(padded-n_windows, *images.shape[1:])])
        features = []
        for chunk in images.split(size):
            inputs = [(p.to(f'cuda:{d}', non_blocking=True),) for p, d in zip(chunk.chunk(len(devices)), devices)]
            if not all(torch.isfinite(x).all() for (x,) in inputs):
                raise ValueError('Non-finite saved evaluation tensor')
            if len(devices) == 1:   # stay on the main thread: per-call threads reset cuDNN plan caches
                outputs = [wrapped[0](inputs[0][0])]
            else:
                outputs = torch.nn.parallel.parallel_apply(wrapped, inputs, devices=devices)
            features.extend(o.to('cuda:0').float() for o in outputs)
        with torch.autocast('cuda', dtype=dtype):
            out = model.head(torch.cat(features, 0)[:n_windows][None])
        logits.append(out.float().cpu()[0])
        uids.append(batch['study_uid'][0])
        gpu_s += time.monotonic()-got
        if index in (0, 1, 2) or (index+1) % 25 == 0:
            event('predict_progress', studies=index+1, total=len(loader), windows=n_windows,
                  seconds=round(time.monotonic()-started, 1), data_wait_s=round(wait_s, 1), gpu_s=round(gpu_s, 1))
        fetch = time.monotonic()
    del replicas, wrapped
    torch.cuda.empty_cache()
    return uids, torch.stack(logits).numpy()


def run_evaluation(tag, model, datasets, truth, cfg, dtype, step):
    """datasets: name -> dataset; truth: name -> dict(binary=..., soft=...)."""
    results = {}
    Path('predictions').mkdir(exist_ok=True)
    for name, dataset in datasets.items():
        started = time.monotonic()
        uids, logits = predict(model, dataset, cfg, dtype, cfg['workers'])
        probs = 1/(1+np.exp(-logits))
        frame = pd.DataFrame(probs, columns=list(LABELS))
        frame.insert(0, 'StudyInstanceUID', uids)
        frame.to_csv(f'predictions/{tag}_{name}.csv', index=False)
        y = np.stack([truth[name]['binary'][u] for u in uids])
        auc, per = macro_auc(y, probs)
        record = dict(macro_auc=auc, per_finding=per, studies=len(uids),
                      seconds=round(time.monotonic()-started, 1))
        if truth[name].get('soft') is not None:
            soft = np.stack([truth[name]['soft'][u] for u in uids])
            record['soft_bce'] = float(F.binary_cross_entropy_with_logits(
                torch.from_numpy(logits), torch.from_numpy(soft)).item())
        results[name] = record
        event('evaluation', tag=tag, split=name, step=step, **record)
    return results


# ----------------------------------------------------------------------------- memory/speed probe
def set_fused_attention(model, fused):
    """Toggle timm's fused SDPA attention (True) vs explicit matmul attention (False).
    Both compute the same function; speed differs by GPU/precision. Returns modules changed."""
    count = 0
    for module in model.modules():
        if hasattr(module, 'fused_attn'):
            module.fused_attn = bool(fused)
            count += 1
    return count


def time_training_step(model, network, images, y, cfg, scaler, passes=3):
    """Peak memory and per-pass seconds of forward+backward on one batch (no optimizer step)."""
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)
    timings = []
    x = logits = loss = None
    try:
        for _ in range(passes):   # first pass includes cudnn autotuning
            torch.cuda.synchronize()
            started = time.monotonic()
            x = gpu_augment(images, cfg) if cfg['augment'] else images
            logits = network(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            timings.append(round(time.monotonic()-started, 2))
            x = logits = loss = None
            model.zero_grad(set_to_none=True)
    finally:
        x = logits = loss = None
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    peak = [round(torch.cuda.max_memory_allocated(d)/2**30, 2) for d in range(torch.cuda.device_count())]
    return peak, timings


def probe_checkpointing(model, network, batch, cfg, scaler, dtype):
    """For each attention implementation, find the cheapest checkpoint setting that fits in memory;
    then keep the (attention, checkpoint) pair with the fastest steady-state training step."""
    ladder = CHECKPOINT_LADDER if cfg['checkpoint_parts'] == 'auto' else (tuple(cfg['checkpoint_parts']),)
    modes = {'auto': (True, False), 'fused': (True,), 'math': (False,)}[cfg['attention']]
    images = batch['images'].cuda(non_blocking=True)
    y = batch['targets'].cuda(non_blocking=True)
    report, candidates = [], []
    for fused in modes:
        set_fused_attention(model, fused)
        for parts in ladder:
            model.backbone.set_checkpoint_parts(parts)
            entry = dict(fused_attention=fused, parts=list(parts))
            try:
                peak, timings = time_training_step(model, network, images, y, cfg, scaler)
            except RuntimeError as error:
                if 'out of memory' not in str(error).lower():
                    raise
                torch.cuda.empty_cache()
                report.append({**entry, 'oom': True})
                continue
            fits = max(peak) <= cfg['memory_limit_gib']
            report.append({**entry, 'peak_gib': peak, 'seconds': timings, 'fits': fits})
            event('probe_result', **report[-1])
            if fits:
                candidates.append((min(timings[1:]), fused, parts))
                break
    if not candidates:
        event('checkpoint_probe', results=report, chosen=None)
        raise RuntimeError('No attention/activation-checkpoint setting fits in GPU memory')
    seconds, fused, parts = min(candidates, key=lambda c: c[0])
    set_fused_attention(model, fused)
    model.backbone.set_checkpoint_parts(parts)
    event('checkpoint_probe', results=report, chosen=dict(fused_attention=fused, parts=list(parts),
                                                          seconds_per_microbatch=seconds))
    return dict(fused_attention=fused, parts=list(parts))


# ----------------------------------------------------------------------------- main
def main(config=None):
    started = time.monotonic()
    cfg = {**DEFAULTS, **(config or {})}
    if not torch.cuda.is_available():
        raise RuntimeError('GPU training requested; refusing CPU fallback')
    if cfg['amp'] not in ('fp16', 'bf16'):
        raise ValueError('amp must be fp16 or bf16')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = True
    dtype = torch.float16 if cfg['amp'] == 'fp16' else torch.bfloat16
    capability = torch.cuda.get_device_capability(0)
    if dtype == torch.bfloat16 and capability < (8, 0):
        raise ValueError(f'bf16 has no tensor-core support on compute capability {capability}; use fp16')

    roots = discover_roots()
    all_train = ReadyTripletDataset(roots, split='train', k=None)
    gold_files = ReadyTripletDataset(roots, split='held_out_gold', k=None).files
    if len(all_train) != 4333 or len(gold_files) != 74:
        raise ValueError(f'Corpus split mismatch: {len(all_train)}, {len(gold_files)}')
    train_ids_all = [p.stem for p in all_train.files]
    gold_ids = [p.stem for p in gold_files]
    soft = load_label_table(asset('jev-knee-labels', 'HYBRID_labels_scores.csv'), train_ids_all, name='HYBRID soft')
    binary = load_label_table(asset('jev-knee-labels', 'HYBRID_labels_binary.csv'), train_ids_all, name='HYBRID binary')
    gold = load_label_table(asset('jev-knee-labels', 'gold74_eval_set.csv'), gold_ids, allow_missing=True, name='gold74')

    train_data = ReadyTripletDataset(roots, split='train', k=cfg['k'], labels=soft,
                                     sampling=cfg['sampling'], windows=cfg['train_windows'])
    val_full = ReadyTripletDataset(roots, split='train', k=None, windows=cfg['eval_windows'])
    gold_data = ReadyTripletDataset(roots, split='held_out_gold', k=None, windows=cfg['eval_windows'])
    gold_all_windows = ReadyTripletDataset(roots, split='held_out_gold', k=None, windows='all')
    assert train_data.files == all_train.files == val_full.files
    train_idx, val_idx = split_indices(len(train_data), cfg['seed'], cfg['validation_fraction'])
    split = dict(train=[train_ids_all[i] for i in train_idx], validation=[train_ids_all[i] for i in val_idx],
                 heldout_gold=gold_ids)
    assert not (set(split['train']) | set(split['validation'])) & set(gold_ids)
    Path('training_split.json').write_text(json.dumps(split, indent=2))
    Path('training_config.json').write_text(json.dumps({**cfg, 'recipe_sha256': RECIPE_SHA, 'labels': list(LABELS),
        'target_source': 'HYBRID_labels_scores.csv', 'validation_binary': 'HYBRID_labels_binary.csv',
        'gold': 'gold74_eval_set.csv (missing cells masked per finding)'}, indent=2))
    if cfg['validation_eval'] not in ('every', 'final', 'never'):
        raise ValueError('validation_eval must be every, final or never')
    gold_sets = {'gold74': gold_data}
    validation_sets = {'validation': Subset(val_full, val_idx)}
    truth = {'gold74': dict(binary=gold), 'gold74_allwindows': dict(binary=gold),
             'validation': dict(binary=binary, soft=soft)}
    event('inputs_verified', shards=len(roots), training=len(train_idx), validation=len(val_idx),
          heldout_gold=len(gold_ids), gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
          capability=list(capability), config=cfg)

    resume = None
    if cfg['resume_path']:
        resume = torch.load(cfg['resume_path'], map_location='cpu', weights_only=True)
        if resume['training_split'] != split or resume['training_config']['recipe_sha256'] != RECIPE_SHA:
            raise ValueError('Resume dataset/split differs')
        for key in ('epochs', 'k', 'sampling', 'seed', 'accumulate', 'backbone_lr', 'fusion_lr', 'head_lr'):
            if resume['training_config'][key] != cfg[key]:
                raise ValueError(f'Resume setting differs: {key}')
        if resume['training_config'].get('rad_input', 'shared') != cfg['rad_input']:
            raise ValueError('Resume setting differs: rad_input (set it to the checkpoint value)')
        model = load_fused_checkpoint(resume)
        prior_best = Path(cfg['resume_path']).with_name('fusion_best.pt')
        if prior_best.is_file() and not Path('fusion_best.pt').exists():
            import shutil
            shutil.copyfile(prior_best, 'fusion_best.pt')   # keep the previous session's best in this output
            event('prior_best_carried', source=str(prior_best))
    else:
        import hashlib
        rad = asset('resnet-50-radimagenet-marwan', 'ResNet50.pt')
        with rad.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != RAD_SHA:
                raise ValueError('RadImageNet asset hash differs from branch manifest')
        model = build_raptor_v5_radimagenet(asset('raptor-knee-maxspan', 'raptor_ft_coatnet_v5_full_swa.pt'),
                                            rad, FusionConfig(gradient_checkpointing=False,
                                                              rad_input=cfg['rad_input']))
    model = model.cuda()
    bn_frozen = set_train_mode(model, cfg)
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg['backbone_lr'], cfg['fusion_lr'], cfg['head_lr']),
                                  weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16, init_scale=1024)
    n_gpu = torch.cuda.device_count()
    # One GPU by default: DataParallel/parallel_apply start new threads on every call, which on
    # Kaggle T4s made each step ~8x slower than the single-thread benchmark (0.71 s per 12-window
    # forward+backward). Effective batch = batch_size * accumulate.
    batch_size = max(1, min(n_gpu, cfg['gpus'], 2))
    wrapped = AutocastModule(model, dtype)
    network = torch.nn.DataParallel(wrapped, device_ids=list(range(batch_size))) if batch_size > 1 else wrapped
    updates_per_epoch = math.ceil(math.ceil(len(train_idx)/batch_size)/cfg['accumulate'])
    total_updates = cfg['smoke_updates'] if cfg['smoke'] else cfg['epochs']*updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(cfg['warmup_steps'], total_updates, cfg['min_lr_ratio']))
    ema = EMA(model, cfg['ema_decay'])

    epoch_start, batch_start, global_step = 0, 0, 0
    best = dict(gold_auc=-1.0)
    history = []
    carried = (0.0, 0)
    if resume:
        optimizer.load_state_dict(resume['optimizer']); scaler.load_state_dict(resume['scaler'])
        scheduler.load_state_dict(resume['scheduler']); ema.load_state_dict(resume['ema'], 'cuda:0')
        epoch_start, batch_start = resume['epoch_index'], resume['next_batch']
        global_step, best, history = resume['global_step'], resume['best'], resume['history']
        carried = (resume['epoch_running'], resume['epoch_seen'])
        torch.set_rng_state(resume['torch_rng']); torch.cuda.set_rng_state_all(resume['cuda_rng'])
        random.setstate(resume['python_rng'])
        del resume

    # Memory probe on one real training batch (no optimizer step).
    probe_loader = DataLoader(Subset(train_data, train_idx[:batch_size]), batch_size=batch_size, num_workers=0)
    speed = probe_checkpointing(model, network, next(iter(probe_loader)), cfg, scaler, dtype)
    event('model_ready', trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
          frozen_batchnorm_layers=bn_frozen, **speed,
          optimizer_groups=[dict(name=g['name'], lr=g['lr']) for g in optimizer.param_groups],
          batch_size=batch_size, k=cfg['k'], effective_batch=batch_size*cfg['accumulate'],
          updates_per_epoch=updates_per_epoch, total_updates=total_updates, amp=str(dtype))

    def evaluate_all(tag, step, include_ema=True, validation=False):
        eval_sets = {**gold_sets, **(validation_sets if validation and not cfg['smoke'] else {})}
        results = {'raw': run_evaluation(f'{tag}_raw', model, eval_sets, truth, cfg, dtype, step)}
        if include_ema:
            backup = ema.swap_in(model)
            try:
                results['ema'] = run_evaluation(f'{tag}_ema', model, eval_sets, truth, cfg, dtype, step)
            finally:
                ema.restore(model, backup)
        set_train_mode(model, cfg)
        return results

    def consider_best(results, tag, step):
        nonlocal best
        for kind, by_split in results.items():
            auc = by_split['gold74']['macro_auc']
            if auc > best['gold_auc']:
                best = dict(gold_auc=auc, weights=kind, tag=tag, step=step)
                if kind == 'ema':
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
                atomic_save(ck, 'fusion_best.pt')
                saved = torch.load('fusion_best.pt', map_location='cpu', weights_only=True)['model']
                names = [n for n, _ in model.named_parameters()]
                if len(names) != len(source) or not all(
                        torch.equal(saved[n].float(), s.detach().float().cpu()) for n, s in zip(names, source)):
                    raise RuntimeError('fusion_best.pt does not contain the selected weights')
                del saved
                event('best_updated', **best)

    def save(epoch, next_batch, running, seen):
        ck = fused_checkpoint(model)
        ck.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), scheduler=scheduler.state_dict(),
                  ema=ema.state_dict(), epoch_index=epoch, next_batch=next_batch, global_step=global_step,
                  best=best, history=history, epoch_running=running, epoch_seen=seen,
                  training_split=split, training_config={**cfg, 'recipe_sha256': RECIPE_SHA},
                  torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                  python_rng=random.getstate())
        atomic_save(ck, 'fusion_last.pt')
        event('checkpoint_saved', epoch_index=epoch, next_batch=next_batch, optimizer_steps=global_step)

    if cfg['eval_at_start'] and global_step == 0:
        # Step 0: zero residuals => exactly the parent Raptor (EMA equals raw here).
        start = evaluate_all('step0', 0, include_ema=False, validation=cfg['validation_eval'] == 'every')
        start['raw'].update(run_evaluation('step0_raw', model, {'gold74_allwindows': gold_all_windows},
                                           truth, cfg, dtype, 0))
        set_train_mode(model, cfg)
        history.append(dict(step=0, epoch=0, evaluation=start))
        consider_best({'raw': start['raw']}, 'step0', 0)

    finished = False
    for epoch in range(epoch_start, cfg['epochs']):
        generator = torch.Generator().manual_seed(cfg['seed']+epoch)
        order = torch.randperm(len(train_idx), generator=generator).tolist()
        loader = DataLoader(Subset(train_data, [train_idx[i] for i in order]), batch_size=batch_size,
                            num_workers=cfg['workers'], pin_memory=True, drop_last=False,
                            prefetch_factor=2 if cfg['workers'] else None, persistent_workers=False)
        optimizer.zero_grad(set_to_none=True)
        running, seen = carried if epoch == epoch_start else (0.0, 0)
        skipped, amp_skips = 0, 0
        wait_s, compute_s, timed = 0.0, 0.0, 0
        event('epoch_started', epoch=epoch+1, batches=len(loader), resume_batch=batch_start)
        fetch_start = time.monotonic()
        for batch_index, batch in enumerate(loader):
            if batch_index < batch_start:
                fetch_start = time.monotonic()
                continue
            got = time.monotonic()
            images = batch['images'].cuda(non_blocking=True)
            y = batch['targets'].cuda(non_blocking=True)
            if not torch.isfinite(images).all():
                raise ValueError('Non-finite saved training tensor')
            if cfg['augment']:
                images = gpu_augment(images, cfg)
            group_start = (batch_index//cfg['accumulate'])*cfg['accumulate']
            group_size = min(cfg['accumulate'], len(loader)-group_start)
            logits = network(images)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            if torch.isfinite(loss):
                scaler.scale(loss/group_size).backward()
                running += float(loss.detach())*len(y); seen += len(y)
            else:
                skipped += 1
                event('microbatch_skipped_nonfinite', epoch=epoch+1, batch=batch_index+1, skipped=skipped)
                if skipped > cfg['max_skipped_microbatches']:
                    raise FloatingPointError('Too many non-finite training losses')
            del images, logits, loss
            torch.cuda.synchronize()
            done = time.monotonic()
            if batch_index > batch_start + 2:   # ignore warm-up iterations in timing
                wait_s += got-fetch_start; compute_s += done-got; timed += 1
            if batch_index < batch_start + 3:
                event('microbatch_complete', epoch=epoch+1, batch=batch_index+1, loss=running/max(seen, 1),
                      seconds=round(done-fetch_start, 2),
                      peak_gpu_gib=[round(torch.cuda.max_memory_allocated(i)/2**30, 2) for i in range(n_gpu)])
            boundary = (batch_index+1) % cfg['accumulate'] == 0 or batch_index+1 == len(loader)
            if boundary:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_norm'])
                before = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                advanced = (not scaler.is_enabled()) or scaler.get_scale() >= before
                optimizer.zero_grad(set_to_none=True)
                if advanced:
                    scheduler.step(); global_step += 1
                    ema.update(model, global_step)
                    amp_skips = 0
                else:
                    amp_skips += 1
                    event('amp_update_skipped', consecutive=amp_skips, scale=scaler.get_scale())
                    if amp_skips >= 8:
                        raise FloatingPointError('Eight consecutive non-finite AMP updates')
                if advanced and (global_step <= 3 or global_step % 10 == 0):
                    per_mb = (wait_s+compute_s)/max(timed, 1)
                    remaining_mb = (len(loader)-batch_index-1) + (cfg['epochs']-epoch-1)*len(loader)
                    if cfg['smoke']:
                        remaining_mb = max(0, (total_updates-global_step)*cfg['accumulate'])
                    event('training_progress', epoch=epoch+1, batch=batch_index+1, batches=len(loader),
                          optimizer_steps=global_step, loss=running/max(seen, 1), gradient_norm=float(grad_norm),
                          lr=[g['lr'] for g in optimizer.param_groups], scale=scaler.get_scale(),
                          sec_per_microbatch=round(per_mb, 3),
                          data_wait_fraction=round(wait_s/max(wait_s+compute_s, 1e-9), 3),
                          projected_epoch_hours=round(per_mb*len(loader)/3600, 2),
                          projected_remaining_training_hours=round(per_mb*remaining_mb/3600, 2),
                          hours=round((time.monotonic()-started)/3600, 3),
                          peak_gpu_gib=[round(torch.cuda.max_memory_allocated(i)/2**30, 2) for i in range(n_gpu)])
                if advanced and global_step % cfg['save_every_steps'] == 0:
                    save(epoch, batch_index+1, running, seen)
                if cfg['smoke'] and global_step >= total_updates:
                    finished = True
                    break
                if time.monotonic()-started > cfg['max_hours']*3600:
                    save(epoch, batch_index+1, running, seen)
                    event('paused_time_budget', epoch=epoch+1, next_batch=batch_index+1,
                          optimizer_steps=global_step, resume_checkpoint='fusion_last.pt')
                    return
            fetch_start = time.monotonic()
        batch_start = 0
        tag = 'smoke_end' if finished else f'epoch{epoch+1}'
        last_epoch = epoch+1 == cfg['epochs']
        results = evaluate_all(tag, global_step, validation=cfg['validation_eval'] == 'every'
                               or (cfg['validation_eval'] == 'final' and last_epoch))
        history.append(dict(step=global_step, epoch=epoch+1, training_loss=running/max(seen, 1),
                            evaluation=results))
        consider_best(results, tag, global_step)
        if not cfg['smoke']:
            backup = ema.swap_in(model)
            atomic_save(fused_checkpoint(model), f'fusion_ema_epoch{epoch+1}.pt')
            ema.restore(model, backup)
        Path('history.json').write_text(json.dumps(history, indent=2))
        save(epoch+1, 0, 0.0, 0)
        summary = {kind: {name: r['macro_auc'] for name, r in by.items()} for kind, by in results.items()}
        event('epoch_complete', epoch=epoch+1, optimizer_steps=global_step, training_loss=running/max(seen, 1),
              macro_auc=summary, best=best)
        if finished:
            break
        if time.monotonic()-started > cfg['max_hours']*3600:
            event('paused_time_budget', epoch=epoch+1, optimizer_steps=global_step, resume_checkpoint='fusion_last.pt')
            return
    event('training_complete', optimizer_steps=global_step, best=best,
          hours=round((time.monotonic()-started)/3600, 3))


if __name__ == '__main__':
    main(globals().get('RUN_CONFIG'))

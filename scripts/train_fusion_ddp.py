"""Two-GPU DistributedDataParallel version of the fusion v2 fine-tune.

Launch one process per GPU:
    python -m torch.distributed.run --standalone --nproc_per_node=2 train_fusion_ddp.py --config '{"epochs": 20}'

Each process owns one GPU and runs everything on its main thread (no per-step worker threads).
Recipe as train_fusion_v2: fp16 + GradScaler, configurable frozen/synchronized BN, per-slot K=12 bags without
series-boundary windows, GPU augmentation, warmup+cosine, EMA, gold74 AUC each epoch (raw and EMA,
step 0 = parent Raptor), validation AUC after the final epoch. Every rank sees a disjoint half of
each epoch's study order; gradients are averaged once per optimizer step (no_sync while
accumulating). Evaluation studies are split across ranks and gathered on rank 0, which alone
writes events, predictions and checkpoints.

Use studies_per_gpu=2, effective_batch=32, freeze_bn_stats=False and sync_batchnorm=True
for four-study SyncBatchNorm forwards and eight accumulation microsteps on two GPUs.
"""
import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models.raptor_v5_radimagenet import (LABELS, FusionConfig, build_raptor_v5_radimagenet, optimizer_groups,
                                          fused_checkpoint, load_fused_checkpoint, preserve_batchnorm_buffers)
from raptor_data.ready_tensors import ReadyTripletDataset
from raptor_data.cv_folds import METRIC_COLUMNS, combined_score, metric_rows, stratified_folds
from train_fusion_v2 import (DEFAULTS as V2_DEFAULTS, RECIPE_SHA, RAD_SHA, AutocastModule, asset,
                             atomic_save, discover_roots, gpu_augment, load_label_table, lr_lambda_factory,
                             macro_auc, set_fused_attention, set_train_mode, split_indices)

DEFAULTS = {**V2_DEFAULTS, **dict(
    epochs=20, effective_batch=16, workers=2, checkpoint_parts=(), attention='fused',
    max_hours=11.0, save_every_steps=100, keep_epoch_ema=False, rad_input='pm1_middle',
    studies_per_gpu=1, sync_batchnorm=False, probe_reserve_gib=2.0,
    fold=None, cv_folds=5, cv_seed=20261010, smoke_validation_studies=24)}
DEFAULTS.pop('accumulate', None)
DEFAULTS.pop('gpus', None)
RANK, WORLD, LOCAL = 0, 1, 0


def accumulation_steps(effective_batch, studies_per_gpu, world):
    """Optimizer batch = studies/GPU * GPUs * accumulation microsteps."""
    values = (effective_batch, studies_per_gpu, world)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in values):
        raise ValueError('Batch sizes and world size must be positive integers')
    global_microbatch = studies_per_gpu * world
    if effective_batch % global_microbatch:
        raise ValueError('effective_batch must be divisible by studies_per_gpu * world_size')
    return effective_batch // global_microbatch


def microbatch_loss_weight(batch_index, studies_per_rank, studies_per_gpu, accumulate):
    """Weight a mean loss by study count, including a short final batch/group."""
    group_start = (batch_index // accumulate) * accumulate * studies_per_gpu
    group_studies = min(accumulate * studies_per_gpu, studies_per_rank - group_start)
    batch_studies = min(studies_per_gpu, studies_per_rank - batch_index * studies_per_gpu)
    if batch_studies < 1 or group_studies < 1:
        raise ValueError('Microbatch index is outside the rank shard')
    return batch_studies / group_studies


def configure_batchnorm(model, cfg, world):
    """Use timm's converter so CoAtNet's fused BN+activation modules keep their activation."""
    if cfg['sync_batchnorm']:
        if cfg['freeze_bn_stats']:
            raise ValueError('SyncBatchNorm requires freeze_bn_stats=False for this experiment')
        from timm.layers.norm_act import convert_sync_batchnorm
        model = convert_sync_batchnorm(model)
    elif world > 1 and not cfg['freeze_bn_stats']:
        raise ValueError('Unfrozen multi-GPU BatchNorm requires sync_batchnorm=True')
    return model


def event(kind, **data):
    if RANK != 0:
        return
    record = dict(event=kind, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **data)
    print(json.dumps(record), flush=True)
    with Path('training_events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record)+'\n')
    Path('progress.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


def setup():
    global RANK, WORLD, LOCAL
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        LOCAL = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(LOCAL)
        dist.init_process_group('nccl')
        RANK, WORLD = dist.get_rank(), dist.get_world_size()
    else:
        torch.cuda.set_device(0)
    return torch.device('cuda', LOCAL)


def all_max(value, device):
    if WORLD == 1:
        return value
    t = torch.tensor([float(value)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


class HostEMA:
    """Float32 EMA of all parameters kept in host memory (frees ~0.45 GB per GPU).
    Every rank holds an identical copy, since parameters are identical after each step.
    EMA evaluation uses the current model's synchronized BN buffers, not EMA-averaged
    statistics. Those same buffers are included in saved EMA checkpoints.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.params = [p.detach().to('cpu', torch.float32, copy=True) for p in model.parameters()]

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1+step)/(10+step))
        current = [p.detach().float().cpu() for p in model.parameters()]
        torch._foreach_mul_(self.params, d)
        torch._foreach_add_(self.params, current, alpha=1-d)

    @torch.no_grad()
    def swap_in(self, model):
        backup = [p.detach().to('cpu', copy=True) for p in model.parameters()]
        for p, e in zip(model.parameters(), self.params):
            p.copy_(e.to(p.device, p.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for p, b in zip(model.parameters(), backup):
            p.copy_(b.to(p.device))

    def state_dict(self):
        return dict(decay=self.decay, params=self.params)

    def load_state_dict(self, state):
        self.decay = state['decay']
        self.params = [p.detach().to('cpu', torch.float32, copy=True) for p in state['params']]


def probe_step(model, network, images, y, cfg, scaler, device, passes=3):
    """Peak memory (this GPU only) and seconds per forward+backward; no optimizer step."""
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    # Probing must not train the running statistics or alter step-0 predictions.
    with preserve_batchnorm_buffers(model):
        for _ in range(passes):   # first pass includes cudnn autotuning
            torch.cuda.synchronize(device)
            t = time.monotonic()
            x = gpu_augment(images, cfg) if cfg['augment'] else images
            loss = F.binary_cross_entropy_with_logits(network(x).float(), y)
            scaler.scale(loss).backward()
            torch.cuda.synchronize(device)
            timings.append(round(time.monotonic()-t, 2))
            del x, loss
            model.zero_grad(set_to_none=True)
    peak = round(torch.cuda.max_memory_allocated(device)/2**30, 2)
    torch.cuda.empty_cache()
    return peak, timings


def rank_shard(indices):
    """Disjoint, equally long shard of `indices` for this rank (drops at most WORLD-1 items)."""
    per = len(indices)//WORLD
    return list(indices[RANK::WORLD][:per])


# ----------------------------------------------------------------------------- evaluation
@torch.inference_mode()
def predict_shard(model, dataset, cfg, dtype, device):
    """Logits for this rank's share of `dataset` (every WORLD-th study), main thread, one GPU."""
    model.eval()
    indices = list(range(RANK, len(dataset), WORLD))
    loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=cfg['workers'], pin_memory=True,
                        prefetch_factor=2 if cfg['workers'] else None)
    size = cfg['eval_chunk']
    uids, logits = [], []
    for batch in loader:
        images = batch['images'][0]
        n_windows = len(images)
        padded = -(-n_windows//size)*size   # fixed chunk shape: cudnn tunes once
        if padded > n_windows:
            images = torch.cat([images, images[-1:].expand(padded-n_windows, *images.shape[1:])])
        features = []
        for chunk in images.split(size):
            x = chunk.to(device, non_blocking=True)
            if not torch.isfinite(x).all():
                raise ValueError('Non-finite saved evaluation tensor')
            with torch.autocast('cuda', dtype=dtype):
                features.append(model.backbone(x).float())
        with torch.autocast('cuda', dtype=dtype):
            out = model.head(torch.cat(features, 0)[:n_windows][None])
        logits.append(out.float().cpu()[0].numpy())
        uids.append(batch['study_uid'][0])
    return uids, logits


def run_evaluation(tag, model, datasets, truth, cfg, dtype, step, device):
    """All ranks predict their shard; rank 0 gathers, scores and returns results (others None)."""
    results = {}
    for name, dataset in datasets.items():
        started = time.monotonic()
        uids, logits = predict_shard(model, dataset, cfg, dtype, device)
        if WORLD > 1:
            gathered = [None]*WORLD
            dist.all_gather_object(gathered, (uids, logits))
            uids = [u for part in gathered for u in part[0]]
            logits = [v for part in gathered for v in part[1]]
        if RANK != 0:
            continue
        logits = np.stack(logits)
        if len(set(uids)) != len(dataset):
            raise RuntimeError(f'Evaluation coverage mismatch on {name}: {len(set(uids))}/{len(dataset)}')
        probs = 1/(1+np.exp(-logits))
        Path('predictions').mkdir(exist_ok=True)
        frame = pd.DataFrame(probs, columns=list(LABELS))
        frame.insert(0, 'StudyInstanceUID', uids)
        frame.sort_values('StudyInstanceUID').to_csv(f'predictions/{tag}_{name}.csv', index=False)
        y = np.stack([truth[name]['binary'][u] for u in uids])
        auc, per = macro_auc(y, probs)
        record = dict(macro_auc=auc, per_finding=per, studies=len(uids), seconds=round(time.monotonic()-started, 1))
        if truth[name].get('soft') is not None:
            soft = np.stack([truth[name]['soft'][u] for u in uids])
            record['soft_bce'] = float(F.binary_cross_entropy_with_logits(
                torch.from_numpy(logits), torch.from_numpy(soft)).item())
        results[name] = record
        event('evaluation', tag=tag, split=name, step=step, **record)
    return results if RANK == 0 else None


# ----------------------------------------------------------------------------- main
def main(config=None):
    started = time.monotonic()
    device = setup()
    cfg = {**DEFAULTS, **(config or {})}
    cfg['accumulate'] = accumulation_steps(cfg['effective_batch'], cfg['studies_per_gpu'], WORLD)
    if cfg['k'] is None and cfg['studies_per_gpu'] > 1:
        raise ValueError('Multiple studies per GPU require fixed K bags')
    if cfg['sync_batchnorm'] and cfg['freeze_bn_stats']:
        raise ValueError('Set freeze_bn_stats=False when enabling SyncBatchNorm')
    if WORLD > 1 and not cfg['freeze_bn_stats'] and not cfg['sync_batchnorm']:
        raise ValueError('Unfrozen multi-GPU BatchNorm requires sync_batchnorm=True')
    cfg['world_size'] = WORLD
    if cfg['amp'] != 'fp16':
        raise ValueError('Use fp16 on T4 (no bf16 tensor cores)')
    dtype = torch.float16
    random.seed(cfg['seed']+RANK); np.random.seed(cfg['seed']+RANK)
    torch.manual_seed(cfg['seed']+RANK); torch.cuda.manual_seed(cfg['seed']+RANK)
    torch.backends.cudnn.benchmark = True

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
    fold_of = None
    if cfg['fold'] is None:
        train_idx, val_idx = split_indices(len(train_data), cfg['seed'], cfg['validation_fraction'])
    else:
        if not 0 <= cfg['fold'] < cfg['cv_folds']:
            raise ValueError(f"fold must be in [0, {cfg['cv_folds']})")
        if cfg['validation_eval'] != 'every':
            raise ValueError("Cross-validation scores every epoch: set validation_eval='every'")
        fold_of = stratified_folds(train_ids_all, np.stack([binary[u] for u in train_ids_all]),
                                   cfg['cv_folds'], cfg['cv_seed'])
        train_idx = [i for i, u in enumerate(train_ids_all) if fold_of[u] != cfg['fold']]
        val_idx = [i for i, u in enumerate(train_ids_all) if fold_of[u] == cfg['fold']]
    split = dict(train=[train_ids_all[i] for i in train_idx], validation=[train_ids_all[i] for i in val_idx],
                 heldout_gold=gold_ids)
    assert not (set(split['train']) | set(split['validation'])) & set(gold_ids)
    gold_sets = {'gold74': gold_data}
    validation_sets = {'validation': Subset(val_full, val_idx[:cfg['smoke_validation_studies']]
                                            if cfg['smoke'] else val_idx)}
    truth = {'gold74': dict(binary=gold), 'gold74_allwindows': dict(binary=gold),
             'validation': dict(binary=binary, soft=soft)}
    if RANK == 0:
        Path('training_split.json').write_text(json.dumps(split, indent=2))
        if fold_of is not None:
            pd.DataFrame(sorted(fold_of.items()), columns=['StudyInstanceUID', 'fold']).to_csv(
                'cv_folds.csv', index=False)
        Path('training_config.json').write_text(json.dumps({**cfg, 'recipe_sha256': RECIPE_SHA,
            'labels': list(LABELS), 'target_source': 'HYBRID_labels_scores.csv',
            'validation_binary': 'HYBRID_labels_binary.csv', 'gold': 'gold74_eval_set.csv (missing masked)'},
            indent=2, default=list))
    if RANK == 0:
        event('inputs_verified', shards=len(roots), training=len(train_idx), validation=len(val_idx),
              heldout_gold=len(gold_ids), world_size=WORLD, gpu=torch.cuda.get_device_name(device), config=cfg)

    resume = None
    if cfg['resume_path']:
        resume = torch.load(cfg['resume_path'], map_location='cpu', weights_only=True)
        if resume['training_split'] != split or resume['training_config']['recipe_sha256'] != RECIPE_SHA:
            raise ValueError('Resume dataset/split differs')
        for key in ('epochs', 'k', 'sampling', 'seed', 'effective_batch', 'world_size',
                    'backbone_lr', 'fusion_lr', 'head_lr'):
            if resume['training_config'][key] != cfg[key]:
                raise ValueError(f'Resume setting differs: {key}')
        for key, old_default in (('studies_per_gpu', 1), ('sync_batchnorm', False),
                                 ('freeze_bn_stats', True)):
            if resume['training_config'].get(key, old_default) != cfg[key]:
                raise ValueError(f'Resume setting differs: {key}; start a separate experiment')
        if resume['training_config'].get('rad_input', 'shared') != cfg['rad_input']:
            raise ValueError('Resume setting differs: rad_input (set it to the checkpoint value)')
        model = load_fused_checkpoint(resume)
        prior_best = Path(cfg['resume_path']).with_name('fusion_best.pt')
        if RANK == 0 and prior_best.is_file() and not Path('fusion_best.pt').exists():
            import shutil
            shutil.copyfile(prior_best, 'fusion_best.pt')   # keep the previous session's best in this output
            event('prior_best_carried', source=str(prior_best))
    else:
        import hashlib
        rad = asset('resnet-50-radimagenet-marwan', 'ResNet50.pt')
        with rad.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != RAD_SHA:
                raise ValueError('RadImageNet asset hash differs from branch manifest')
        # Same file on every rank; the fusion MLPs' random layers are then synchronised by DDP.
        model = build_raptor_v5_radimagenet(asset('raptor-knee-maxspan', 'raptor_ft_coatnet_v5_full_swa.pt'),
                                            rad, FusionConfig(gradient_checkpointing=False,
                                                              rad_input=cfg['rad_input']))
    model = model.to(device)
    bn_frozen = set_train_mode(model, cfg)
    set_fused_attention(model, cfg['attention'] != 'math')
    model.backbone.set_checkpoint_parts(cfg['checkpoint_parts'])
    wrapped = AutocastModule(model, dtype)

    # Per-rank timing/memory probe on the bare model, before DDP hooks and SyncBN exist.
    probe_indices = rank_shard(train_idx)[:cfg['studies_per_gpu']]
    probe = next(iter(DataLoader(Subset(train_data, probe_indices), batch_size=cfg['studies_per_gpu'],
                                num_workers=0)))
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    # Probe ordinary BN before conversion: an OOM on one rank cannot strand another
    # inside a SyncBatchNorm collective. Abort together if either GPU cannot fit.
    probe_error, timings = None, []
    try:
        peak, timings = probe_step(model, wrapped, probe['images'].to(device), probe['targets'].to(device),
                                   cfg, scaler, device)
    except torch.cuda.OutOfMemoryError as exc:
        peak, probe_error = float('inf'), str(exc)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    del probe
    peak = all_max(peak, device)
    if peak + cfg['probe_reserve_gib'] > cfg['memory_limit_gib']:
        raise RuntimeError(f'Probe peak {peak} GiB + optimizer/DDP reserve '
                           f"{cfg['probe_reserve_gib']} GiB exceeds {cfg['memory_limit_gib']} GiB; "
                           f'enable more checkpoint_parts. Local OOM: {probe_error}')

    model = configure_batchnorm(model, cfg, WORLD)
    bn_frozen = set_train_mode(model, cfg)
    wrapped = AutocastModule(model, dtype)

    if WORLD > 1:
        network = torch.nn.parallel.DistributedDataParallel(
            wrapped, device_ids=[LOCAL], output_device=LOCAL, broadcast_buffers=False,
            gradient_as_bucket_view=True)
    else:
        network = wrapped
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg['backbone_lr'], cfg['fusion_lr'], cfg['head_lr']),
                                  weight_decay=cfg['weight_decay'])
    microbatches = math.ceil(len(rank_shard(list(range(len(train_idx)))))/cfg['studies_per_gpu'])
    updates_per_epoch = math.ceil(microbatches/cfg['accumulate'])
    total_updates = cfg['smoke_updates'] if cfg['smoke'] else cfg['epochs']*updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(cfg['warmup_steps'], total_updates, cfg['min_lr_ratio']))
    ema = HostEMA(model, cfg['ema_decay'])   # created after DDP's initial broadcast: identical on all ranks

    epoch_start, batch_start, global_step = 0, 0, 0
    best = dict(gold_auc=-1.0, score=-1.0)
    history = []
    carried = (0.0, 0)
    if resume:
        optimizer.load_state_dict(resume['optimizer']); scaler.load_state_dict(resume['scaler'])
        scheduler.load_state_dict(resume['scheduler']); ema.load_state_dict(resume['ema'])
        epoch_start, batch_start = resume['epoch_index'], resume['next_batch']
        global_step, best, history = resume['global_step'], resume['best'], resume['history']
        carried = (resume['epoch_running'], resume['epoch_seen'])
        del resume
    event('model_ready', trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
          frozen_batchnorm_layers=bn_frozen, fused_attention=cfg['attention'] != 'math',
          checkpoint_parts=list(cfg['checkpoint_parts']), probe_peak_gib=peak, probe_seconds=timings,
          world_size=WORLD, microbatches_per_rank=microbatches, accumulate_per_rank=cfg['accumulate'],
          studies_per_gpu=cfg['studies_per_gpu'], sync_batchnorm=cfg['sync_batchnorm'],
          batchnorm_studies_per_forward=(cfg['studies_per_gpu'] * WORLD if cfg['sync_batchnorm'] else
                                        (0 if cfg['freeze_bn_stats'] else cfg['studies_per_gpu'])),
          effective_batch=cfg['effective_batch'], updates_per_epoch=updates_per_epoch,
          total_updates=total_updates, amp=str(dtype),
          optimizer_groups=[dict(name=g['name'], lr=g['lr']) for g in optimizer.param_groups])

    def evaluate_all(tag, step, include_ema=True, validation=False):
        sets = {**gold_sets, **(validation_sets if validation else {})}
        results = {'raw': run_evaluation(f'{tag}_raw', model, sets, truth, cfg, dtype, step, device)}
        if include_ema:
            backup = ema.swap_in(model)
            try:
                results['ema'] = run_evaluation(f'{tag}_ema', model, sets, truth, cfg, dtype, step, device)
            finally:
                ema.restore(model, backup)
        set_train_mode(model, cfg)
        return results if RANK == 0 else None

    def consider_best(results, tag, step):
        nonlocal best
        if RANK != 0:
            return
        for kind, by_split in results.items():
            score = combined_score(by_split)   # mean of gold74 and left-out-fold macro AUC
            if score > best.get('score', best['gold_auc']):
                best = dict(score=score, gold_auc=by_split['gold74']['macro_auc'],
                            fold_auc=by_split.get('validation', {}).get('macro_auc'),
                            weights=kind, tag=tag, step=step)
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

    def write_metrics():
        if RANK != 0:
            return
        rows = metric_rows(history, cfg['fold'])
        pd.DataFrame(rows, columns=list(METRIC_COLUMNS)).to_csv('cv_metrics.csv', index=False)
        Path('cv_metrics.json').write_text(json.dumps(dict(
            fold=cfg['fold'], definition='validation_accuracy = (gold74 macro AUC + left-out fold macro AUC) / 2',
            best=best, rows=rows), indent=2))

    def save(epoch, next_batch, running, seen):
        if RANK != 0:
            return
        ck = fused_checkpoint(model)
        ck.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), scheduler=scheduler.state_dict(),
                  ema=ema.state_dict(), epoch_index=epoch, next_batch=next_batch, global_step=global_step,
                  best=best, history=history, epoch_running=running, epoch_seen=seen,
                  training_split=split, training_config={**cfg, 'recipe_sha256': RECIPE_SHA})
        atomic_save(ck, 'fusion_last.pt')
        event('checkpoint_saved', epoch_index=epoch, next_batch=next_batch, optimizer_steps=global_step)

    if cfg['eval_at_start'] and global_step == 0:
        start = evaluate_all('step0', 0, include_ema=False, validation=cfg['validation_eval'] == 'every')
        extra = run_evaluation('step0_raw', model, {'gold74_allwindows': gold_all_windows}, truth, cfg, dtype, 0,
                               device)
        set_train_mode(model, cfg)
        if RANK == 0:
            start['raw'].update(extra)
            history.append(dict(step=0, epoch=0, evaluation=start))
            consider_best({'raw': start['raw']}, 'step0', 0)
            write_metrics()

    finished = stop = False
    for epoch in range(epoch_start, cfg['epochs']):
        generator = torch.Generator().manual_seed(cfg['seed']+epoch)   # same order on every rank
        order = torch.randperm(len(train_idx), generator=generator).tolist()
        mine = [train_idx[i] for i in rank_shard(order)]
        n_mb = math.ceil(len(mine)/cfg['studies_per_gpu'])
        loader = DataLoader(Subset(train_data, mine[batch_start*cfg['studies_per_gpu']:]),
                            batch_size=cfg['studies_per_gpu'], num_workers=cfg['workers'],
                            pin_memory=True, prefetch_factor=2 if cfg['workers'] else None)
        optimizer.zero_grad(set_to_none=True)
        running, seen = carried if epoch == epoch_start else (0.0, 0)
        nonfinite, amp_skips = 0, 0
        wait_s, compute_s, timed = 0.0, 0.0, 0
        event('epoch_started', epoch=epoch+1, microbatches_per_rank=n_mb, resume_batch=batch_start)
        fetch_start = time.monotonic()
        stop = False
        for batch_index, batch in enumerate(loader, start=batch_start):
            got = time.monotonic()
            images = batch['images'].to(device, non_blocking=True)
            y = batch['targets'].to(device, non_blocking=True)
            if not torch.isfinite(images).all():
                raise ValueError('Non-finite saved training tensor')
            if cfg['augment']:
                images = gpu_augment(images, cfg)
            loss_weight = microbatch_loss_weight(batch_index, len(mine), cfg['studies_per_gpu'],
                                                cfg['accumulate'])
            boundary = (batch_index+1) % cfg['accumulate'] == 0 or batch_index+1 == n_mb
            sync = network.no_sync() if (WORLD > 1 and not boundary) else nullcontext()
            with sync:
                logits = network(images)
                loss = F.binary_cross_entropy_with_logits(logits.float(), y)
                # Always backward (keeps ranks in lockstep); a non-finite loss gives non-finite
                # gradients, which GradScaler detects identically on all ranks and skips.
                scaler.scale(loss*loss_weight).backward()
            if torch.isfinite(loss):
                running += float(loss.detach()) * len(y); seen += len(y)
            else:
                nonfinite += 1
            del images, logits, loss
            torch.cuda.synchronize(device)
            done = time.monotonic()
            if batch_index > batch_start + 2:
                wait_s += got-fetch_start; compute_s += done-got; timed += 1
            if batch_index < batch_start + 3:
                event('microbatch_complete', epoch=epoch+1, batch=batch_index+1, loss=running/max(seen, 1),
                      seconds=round(done-fetch_start, 2),
                      peak_gpu_gib=round(torch.cuda.max_memory_allocated(device)/2**30, 2))
            if boundary:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_norm'])
                before = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                advanced = scaler.get_scale() >= before
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
                if advanced and (global_step <= 3 or global_step % 20 == 0):
                    per_mb = (wait_s+compute_s)/max(timed, 1)
                    remaining = (n_mb-batch_index-1) + (cfg['epochs']-epoch-1)*n_mb
                    if cfg['smoke']:
                        remaining = max(0, (total_updates-global_step)*cfg['accumulate'])
                    event('training_progress', epoch=epoch+1, batch=batch_index+1, batches=n_mb,
                          optimizer_steps=global_step, loss=running/max(seen, 1), gradient_norm=float(grad_norm),
                          lr=[g['lr'] for g in optimizer.param_groups], scale=scaler.get_scale(),
                          nonfinite_losses=nonfinite, sec_per_microbatch=round(per_mb, 3),
                          data_wait_fraction=round(wait_s/max(wait_s+compute_s, 1e-9), 3),
                          projected_epoch_minutes=round(per_mb*n_mb/60, 1),
                          projected_remaining_training_hours=round(per_mb*remaining/3600, 2),
                          hours=round((time.monotonic()-started)/3600, 3),
                          peak_gpu_gib=round(torch.cuda.max_memory_allocated(device)/2**30, 2))
                if advanced and global_step % cfg['save_every_steps'] == 0:
                    save(epoch, batch_index+1, running, seen)
                if cfg['smoke'] and global_step >= total_updates:
                    finished = True
                    break
                stop = all_max(time.monotonic()-started > cfg['max_hours']*3600, device) > 0
                if stop:
                    save(epoch, batch_index+1, running, seen)
                    event('paused_time_budget', epoch=epoch+1, next_batch=batch_index+1,
                          optimizer_steps=global_step, resume_checkpoint='fusion_last.pt')
                    break
            fetch_start = time.monotonic()
        if stop:
            break
        batch_start = 0
        last_epoch = finished or epoch+1 == cfg['epochs']
        tag = 'smoke_end' if finished else f'epoch{epoch+1}'
        results = evaluate_all(tag, global_step, validation=cfg['validation_eval'] == 'every'
                               or (cfg['validation_eval'] == 'final' and last_epoch))
        if RANK == 0:
            history.append(dict(step=global_step, epoch=epoch+1, training_loss=running/max(seen, 1),
                                evaluation=results))
            consider_best(results, tag, global_step)
            write_metrics()
            if (cfg['keep_epoch_ema'] or last_epoch) and not cfg['smoke']:
                backup = ema.swap_in(model)
                atomic_save(fused_checkpoint(model), f'fusion_ema_epoch{epoch+1}.pt')
                ema.restore(model, backup)
            Path('history.json').write_text(json.dumps(history, indent=2))
        save(epoch+1, 0, 0.0, 0)
        if RANK == 0:
            summary = {kind: {name: r['macro_auc'] for name, r in by.items()} for kind, by in results.items()}
            event('epoch_complete', epoch=epoch+1, optimizer_steps=global_step,
                  training_loss=running/max(seen, 1), macro_auc=summary, best=best,
                  validation_accuracy={kind: combined_score(by) for kind, by in results.items()},
                  hours=round((time.monotonic()-started)/3600, 3))
        if finished:
            break
        if all_max(time.monotonic()-started > cfg['max_hours']*3600, device) > 0:
            event('paused_time_budget', epoch=epoch+1, optimizer_steps=global_step, resume_checkpoint='fusion_last.pt')
            break
    else:
        event('training_complete', optimizer_steps=global_step, best=best,
              hours=round((time.monotonic()-started)/3600, 3))
    if finished:
        event('training_complete', optimizer_steps=global_step, best=best,
              hours=round((time.monotonic()-started)/3600, 3))
    if not stop and (cfg.get('extra_checkpoints') or cfg.get('validate_parent')):
        final = {}
        if cfg.get('validate_parent'):
            import hashlib
            rad = asset('resnet-50-radimagenet-marwan', 'ResNet50.pt')
            with rad.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != RAD_SHA:
                    raise ValueError('RadImageNet asset hash differs from branch manifest')
            parent = build_raptor_v5_radimagenet(asset('raptor-knee-maxspan', 'raptor_ft_coatnet_v5_full_swa.pt'),
                                                 rad, FusionConfig(gradient_checkpointing=False))
            weights = [('parent_step0', parent.state_dict())]
            del parent
        else:
            weights = []
        for name, path in (cfg.get('extra_checkpoints') or {}).items():
            weights.append((name, torch.load(path, map_location='cpu', weights_only=True)['model']))
        for name, state in weights:
            model.load_state_dict(state, strict=True)
            final[name] = run_evaluation(f'final_{name}', model, {**gold_sets, **validation_sets}, truth, cfg,
                                         dtype, global_step, device)
            event('final_validation', model=name,
                  macro_auc=None if final[name] is None else {k: v['macro_auc'] for k, v in final[name].items()})
        if RANK == 0:
            Path('final_validation.json').write_text(json.dumps(final, indent=2))
    if WORLD > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='{}')
    main(json.loads(parser.parse_args().config))

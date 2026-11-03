"""Single-split minimum-overlap V2 training; raw/EMA evaluation and resumable checkpoints."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, Subset

from models.raptor_v5_resnet50 import (LABELS, FusionConfig, build_raptor_v5_resnet50,
    fused_checkpoint, load_fused_checkpoint, optimizer_groups)
from raptor_data.series_triplets import balanced_slot_quotas

RECIPE_SHA = 'aad96c9225fa13edd3d2dff0a105f2e7de6b3efc91a0ad45a8229bd15acfd35c'


def atomic_json(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def event(kind, **kwargs):
    row = dict(event=kind, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs)
    print(json.dumps(row, allow_nan=False), flush=True)
    with open('events.jsonl', 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
    atomic_json(row, 'progress.json')


def label_table(path, ids, missing=False, binary=False):
    table = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    if table.index.duplicated().any() or not set(ids) <= set(table.index):
        raise ValueError(f'Duplicate or missing study IDs: {path}')
    values = table.loc[ids, list(LABELS)].to_numpy(dtype=np.float32)
    finite = np.isfinite(values)
    if not missing and not finite.all():
        raise ValueError(f'Missing labels: {path}')
    if not ((values[finite] >= 0) & (values[finite] <= 1)).all():
        raise ValueError(f'Labels out of range: {path}')
    if binary and not np.isin(values[finite], [0, 1]).all():
        raise ValueError(f'Expected binary evaluation labels: {path}')
    return dict(zip(ids, values))


def make_split(ids, gold, seed):
    ids, gold = sorted(ids), sorted(gold)
    if len(set(ids)) != len(ids) or len(set(gold)) != len(gold) or set(ids) & set(gold):
        raise ValueError('Duplicate or overlapping study IDs')
    order = np.random.default_rng(seed).permutation(len(ids))
    nval = round(len(ids) * 0.1)
    return dict(train=sorted(ids[i] for i in order[nval:]),
                validation=sorted(ids[i] for i in order[:nval]), heldout_gold=gold)


def discover(data):
    roots = sorted(p.parent for p in Path(data).rglob('recipe.json')
                   if json.loads(p.read_text()).get('format') == 'rsna_min_overlap_384_v1')
    if len(roots) != 32:
        raise ValueError(f'Expected exactly 32 minimum-overlap V2 shards; found {len(roots)}')
    files, rows = {}, []
    for root in roots:
        recipe = json.loads((root / 'recipe.json').read_text())
        manifest = json.loads((root / 'manifest.json').read_text())
        if (recipe['sha256'] != RECIPE_SHA or manifest['recipe_sha256'] != RECIPE_SHA
                or not manifest['complete'] or manifest['failures']):
            raise ValueError(f'Wrong or incomplete shard: {root}')
        for row in manifest['completed']:
            uid, file = row['study_uid'], root / row['file']
            if uid in files or not file.is_file() or file.stat().st_size != row['bytes']:
                raise ValueError(f'Missing, duplicated or wrong-size study: {uid}')
            files[uid] = file
            rows.append(row)
    train = sorted(r['study_uid'] for r in rows if r['split'] == 'train')
    gold = sorted(r['study_uid'] for r in rows if r['split'] == 'held_out_gold')
    if len(files) != 4407 or len(train) != 4333 or len(gold) != 74:
        raise ValueError('Corpus counts differ from the verified V2 export')
    return files, train, gold


class StudyDataset(Dataset):
    """Ready tensors; K=24 training sampling is balanced over populated slots."""
    def __init__(self, files, ids, labels, seed=20261013, train=False):
        self.files, self.ids, self.labels = files, list(ids), labels
        self.seed, self.train, self.epoch = seed, train, 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        uid = self.ids[i]
        with np.load(self.files[uid], allow_pickle=False) as saved:
            images = torch.from_numpy(saved['images'])
            indices = saved['triplet_indices']
            owners = saved['slice_series_uids'][indices]
            if (not np.all(owners == owners[:, :1]) or not np.all(saved['same_series'])
                    or not np.all(owners[:, 0] == saved['triplet_series_uids'])):
                raise ValueError(f'Cross-series triplet: {uid}')
            slots = saved['triplet_slot_ids']
            if slots.shape != (len(images), 3) or not np.all(slots == slots[:, :1]):
                raise ValueError(f'Inconsistent triplet slots: {uid}')
            slots = slots[:, 0].copy()
        if images.dtype != torch.float16 or images.ndim != 4 or tuple(images.shape[1:]) != (3, 384, 384):
            raise ValueError(f'Invalid saved images: {uid}')
        if not 0 < len(images) <= 60 or not torch.isfinite(images).all():
            raise ValueError(f'Invalid bag: {uid}')
        if self.train:
            # Deterministic per epoch/study, including worker and mid-epoch restarts.
            value = int.from_bytes(hashlib.sha256(f'{self.seed}:{self.epoch}:{uid}'.encode()).digest()[:8], 'little')
            generator = torch.Generator().manual_seed(value)
            pools = [torch.from_numpy(np.flatnonzero(slots == s)) for s in sorted(set(slots.tolist()))]
            quotas = balanced_slot_quotas([len(p) for p in pools], min(24, len(images)), generator)
            picks = [p[torch.randperm(len(p), generator=generator)[:q]] for p, q in zip(pools, quotas) if q]
            images = images[torch.cat(picks)]
            # Mild per-study intensity augmentation, using the existing ImageNet normalization.
            mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
            std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
            factor = .9 + .2 * torch.rand((), generator=generator)
            images = (((images.float() * std + mean) * factor).clamp(0, 1) - mean) / std
            images = images.half()
        return dict(images=images, targets=torch.tensor(self.labels[uid]), study_uid=uid)


class EMA:
    """Named float32 EMA state, including floating buffers; integers copied exactly."""
    def __init__(self, model, decay=.998):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0,1)')
        self.decay, self.updates = decay, 0
        self.shadow = {k: (v.detach().float().clone() if v.is_floating_point() else v.detach().clone())
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise ValueError('EMA state keys changed')
        for key, value in current.items():
            if value.is_floating_point():
                self.shadow[key].lerp_(value.detach().float(), 1 - decay)
            else:
                self.shadow[key].copy_(value)

    def state_dict(self):
        return dict(decay=self.decay, updates=self.updates,
                    shadow={k: v.detach().cpu().clone() for k, v in self.shadow.items()})

    def load_state_dict(self, state, model):
        if state['shadow'].keys() != model.state_dict().keys():
            raise ValueError('EMA checkpoint keys differ')
        self.decay, self.updates = state['decay'], state['updates']
        self.shadow = {k: v.to(model.state_dict()[k].device).clone() for k, v in state['shadow'].items()}

    @contextlib.contextmanager
    def installed(self, model):
        backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        try:
            model.load_state_dict(self.shadow, strict=True)
            yield
        finally:
            model.load_state_dict(backup, strict=True)


def metrics(truth, logits):
    from sklearn.metrics import roc_auc_score
    probability = torch.from_numpy(logits).sigmoid().numpy()
    mask = np.isfinite(truth)
    if not mask.any() or not np.isin(truth[mask], [0, 1]).all():
        raise ValueError('Evaluation requires available binary labels')
    predicted = probability >= .5
    per = {}
    for j, name in enumerate(LABELS):
        valid = mask[:, j]
        y = truth[valid, j]
        per[name] = dict(available=int(valid.sum()),
                        accuracy=float(np.mean(predicted[valid, j] == y)) if len(y) else None,
                        auc=float(roc_auc_score(y, probability[valid, j])) if len(np.unique(y)) == 2 else None)
    aucs = [v['auc'] for v in per.values() if v['auc'] is not None]
    loss = F.binary_cross_entropy_with_logits(torch.from_numpy(logits[mask]), torch.from_numpy(truth[mask])).item()
    return dict(accuracy=float(np.mean(predicted[mask] == truth[mask])),
                macro_auc=float(np.mean(aucs)) if aucs else None, loss=loss,
                labeled_entries=int(mask.sum()), threshold=.5, per_finding=per)


def train_mode(model):
    model.train()
    # Small study batches and checkpoint recomputation: keep pretrained BN statistics;
    # affine BN parameters and all other parameters remain trainable.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def bag_logits(model, images, chunk, amp, training=False):
    features = []
    with torch.autocast('cuda', dtype=amp):
        for x in images.split(chunk):
            if training:
                features.append(checkpoint(model.backbone, x, use_reentrant=False))
            else:
                features.append(model.backbone(x))
        return model.head(torch.cat(features)[None]).float()


@torch.inference_mode()
def evaluate(model, loader, amp, chunk, tag):
    model.eval()
    ids, labels, logits = [], [], []
    for i, item in enumerate(loader):
        out = bag_logits(model, item['images'][0].cuda(non_blocking=True), chunk, amp)
        if not torch.isfinite(out).all():
            raise FloatingPointError('Nonfinite evaluation logits')
        ids.append(item['study_uid'][0])
        labels.append(item['targets'][0].numpy())
        logits.append(out[0].cpu().numpy())
        if (i + 1) % 50 == 0:
            event('evaluation_progress', tag=tag, studies=i + 1, total=len(loader))
    logits, labels = np.asarray(logits), np.asarray(labels)
    np.savez_compressed(f'predictions/{tag}.npz', study_uid=np.asarray(ids), labels=labels, logits=logits)
    result = metrics(labels, logits)
    event('evaluation_complete', tag=tag, **result)
    return result


def rng_state():
    return dict(python=random.getstate(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def run(cfg):
    output = Path(cfg['output']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    for folder in ['checkpoints', 'predictions']:
        Path(folder).mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing CPU training')
    if cfg['epochs'] != 10:
        raise ValueError('This campaign is exactly 10 epochs')
    random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    files, ids, gold = discover(cfg['data'])
    split = make_split(ids, gold, cfg['seed'])
    if tuple(len(split[k]) for k in ('train', 'validation', 'heldout_gold')) != (3900, 433, 74):
        raise ValueError('Wrong 90/10 split')
    labels = Path(cfg['labels'])
    soft = label_table(labels / 'HYBRID_labels_scores.csv', ids)
    binary = label_table(labels / 'HYBRID_labels_binary.csv', ids, binary=True)
    gold_labels = label_table(labels / 'gold74_eval_set.csv', gold, missing=True, binary=True)
    cfg = {**cfg, 'recipe_sha256': RECIPE_SHA, 'labels_sha256': {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in labels.glob('*.csv')},
        'model_format': 'raptor_v5_resnet50_v3', 'train_k': 24, 'eval_k': 'all',
        'freeze_bn_stats': True, 'ema_policy': 'fp32 parameters and floating buffers; integer buffers copied',
        'accuracy_definition': 'fraction of available binary finding labels correct at sigmoid >= 0.5',
        'selection_metric': 'validation macro AUROC only; Gold74 never selects checkpoints'}
    for name, value in [('training_split.json', split), ('training_config.json', cfg)]:
        if Path(name).exists() and json.loads(Path(name).read_text()) != value:
            # resume_path is an operational argument, not a training setting.
            prior = json.loads(Path(name).read_text())
            if name != 'training_config.json' or {k:v for k,v in prior.items() if k!='resume'} != {k:v for k,v in value.items() if k!='resume'}:
                raise ValueError(f'Existing {name} differs; use an isolated output directory')
        atomic_json(value, name)
    data = StudyDataset(files, split['train'], soft, cfg['seed'], train=True)
    eval_loaders = {name: DataLoader(StudyDataset(files, split[key], targets), batch_size=1,
        num_workers=cfg['workers'], pin_memory=True) for name, key, targets in [
            ('validation', 'validation', binary), ('gold74', 'heldout_gold', gold_labels)]}
    resume = torch.load(cfg['resume'], map_location='cpu', weights_only=True) if cfg['resume'] else None
    if resume:
        if resume['training_split'] != split or {k:v for k,v in resume['training_config'].items() if k!='resume'} != {k:v for k,v in cfg.items() if k!='resume'}:
            raise ValueError('Resume split/config mismatch')
        model = load_fused_checkpoint(resume)
    else:
        model = build_raptor_v5_resnet50(cfg['raptor'], cfg['resnet'], FusionConfig())
    model.cuda()
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg['backbone_lr'], cfg['fusion_lr'], cfg['head_lr']),
                                  weight_decay=.01)
    steps_per_epoch = math.ceil(len(data) / cfg['accumulate'])
    total_steps = steps_per_epoch * cfg['epochs']
    warmup = min(100, steps_per_epoch)
    def lr_factor(step):
        if step < warmup:
            return .01 + .99 * (step + 1) / warmup
        return .01 + .99 * .5 * (1 + math.cos(math.pi * min(1, (step - warmup) / (total_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp == torch.float16)
    ema = EMA(model, cfg['ema_decay'])
    start_epoch, start_pos, global_step, history, best = 0, 0, 0, [], {'raw': -1., 'ema': -1.}
    running, seen = 0., 0
    if resume:
        optimizer.load_state_dict(resume['optimizer']); scheduler.load_state_dict(resume['scheduler'])
        scaler.load_state_dict(resume['scaler']); ema.load_state_dict(resume['ema'], model)
        start_epoch, start_pos, global_step = resume['epoch_index'], resume['next_study'], resume['global_step']
        history, best = resume['history'], resume['best']
        running, seen = resume['running_loss'], resume['seen']
        restore_rng(resume['rng']); del resume
    event('run_ready', gpu=torch.cuda.get_device_name(), train=3900, validation=433, gold74=74,
          epochs=10, amp=str(amp), parameters=sum(p.numel() for p in model.parameters()), config=cfg)

    def save(path, epoch_index, next_study, running_loss, count):
        ck = fused_checkpoint(model)
        ck.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                  ema=ema.state_dict(), global_step=global_step, epoch_index=epoch_index,
                  next_study=next_study, running_loss=running_loss, seen=count,
                  training_split=split, training_config=cfg, history=history, best=best, rng=rng_state())
        atomic_save(ck, path)

    if not cfg['resume']:
        save('checkpoints/initial.pt', 0, 0, 0., 0)
    if start_epoch == 0 and start_pos == 0 and global_step == 0:
        # Also run this when resuming an initial checkpoint from a failed preflight.
        # A first real backward pass verifies memory, gradients and both input paths.
        probe_rng = rng_state()
        item = data[0]
        train_mode(model)
        probe = F.binary_cross_entropy_with_logits(bag_logits(model, item['images'].cuda(), cfg['chunk'], amp, True), item['targets'].cuda()[None])
        probe.backward()
        if not torch.isfinite(probe) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Preflight loss/gradients are not finite')
        optimizer.zero_grad(set_to_none=True)
        del probe, item
        restore_rng(probe_rng)
        torch.cuda.empty_cache()
        event('training_preflight_passed', peak_gib=torch.cuda.max_memory_allocated()/2**30)
        if cfg['preflight_only']:
            return

    for epoch in range(start_epoch, 10):
        data.epoch = epoch
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(cfg['seed'] + epoch)).tolist()
        offset = start_pos if epoch == start_epoch else 0
        if epoch != start_epoch:
            running, seen = 0., 0
        # Dedicated loader generator does not consume model/dropout RNG on resume.
        loader = DataLoader(Subset(data, order[offset:]), batch_size=1, num_workers=cfg['workers'],
                            pin_memory=True, generator=torch.Generator().manual_seed(cfg['seed'] + epoch))
        train_mode(model)
        optimizer.zero_grad(set_to_none=True)
        event('epoch_started', epoch=epoch+1, next_study=offset, total=len(data))
        began = time.monotonic()
        for local, item in enumerate(loader):
            index = offset + local
            group_start = (index // cfg['accumulate']) * cfg['accumulate']
            group_size = min(cfg['accumulate'], len(data) - group_start)
            images, targets = item['images'][0].cuda(non_blocking=True), item['targets'].cuda(non_blocking=True)
            logits = bag_logits(model, images, cfg['chunk'], amp, True)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch+1}, study {index}')
            scaler.scale(loss / group_size).backward()
            running += float(loss.detach()); seen += 1
            boundary = (index + 1) % cfg['accumulate'] == 0 or index + 1 == len(data)
            if boundary:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
                if not torch.isfinite(norm):
                    # No silent omitted studies or EMA updates for failed optimizer steps.
                    raise FloatingPointError('Nonfinite gradients; last successful checkpoint preserved')
                previous_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() < previous_scale:
                    raise FloatingPointError('AMP skipped an optimizer update')
                global_step += 1; scheduler.step(); ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                if global_step % cfg['save_every'] == 0:
                    save(f'checkpoints/step_{global_step:06d}.pt', epoch, index+1, running, seen)
                    atomic_json({'path': f'checkpoints/step_{global_step:06d}.pt'}, 'latest_checkpoint.json')
                if global_step % 5 == 0 or index+1 == len(data):
                    event('training_progress', epoch=epoch+1, studies=index+1, total=len(data),
                          loss=running/seen, optimizer_step=global_step, ema_updates=ema.updates,
                          seconds=time.monotonic()-began, peak_gib=torch.cuda.max_memory_allocated()/2**30)
            del images, targets, logits, loss

        # Preserve the trained epoch immediately, before potentially lengthy evaluation.
        save(f'checkpoints/epoch_{epoch+1:02d}_before_eval.pt', epoch, len(data), running, seen)
        atomic_json({'path': f'checkpoints/epoch_{epoch+1:02d}_before_eval.pt'}, 'latest_checkpoint.json')
        result = {'epoch': epoch+1, 'train_loss': running/seen, 'raw': {}, 'ema': {}}
        for name, loader in eval_loaders.items():
            result['raw'][name] = evaluate(model, loader, amp, cfg['eval_chunk'], f'epoch_{epoch+1:02d}_raw_{name}')
        with ema.installed(model):
            for name, loader in eval_loaders.items():
                result['ema'][name] = evaluate(model, loader, amp, cfg['eval_chunk'], f'epoch_{epoch+1:02d}_ema_{name}')
            ck = fused_checkpoint(model)
            ck.update(epoch=epoch+1, weight_kind='ema', ema_updates=ema.updates, metrics=result['ema'],
                      training_split=split, training_config=cfg)
            atomic_save(ck, f'checkpoints/epoch_{epoch+1:02d}_ema.pt')
            score = result['ema']['validation']['macro_auc']
            if score is not None and score > best['ema']:
                best['ema'] = score
                atomic_save(ck, 'checkpoints/best_validation_ema.pt')
        score = result['raw']['validation']['macro_auc']
        if score is not None and score > best['raw']:
            best['raw'] = score
            atomic_save(fused_checkpoint(model), 'checkpoints/best_validation_raw.pt')
        history.append(result)
        atomic_json(history, 'history.json')
        save(f'checkpoints/epoch_{epoch+1:02d}_raw.pt', epoch+1, 0, 0., 0)
        atomic_json({'path': f'checkpoints/epoch_{epoch+1:02d}_raw.pt'}, 'latest_checkpoint.json')
        event('epoch_complete', **result)
    event('training_complete', epochs=10, optimizer_steps=global_step, ema_updates=ema.updates,
          best_validation_auc=best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'labels', 'raptor', 'resnet', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--resume')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--accumulate', type=int, default=16)
    parser.add_argument('--chunk', type=int, default=6)
    parser.add_argument('--eval-chunk', type=int, default=12)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--ema-decay', type=float, default=.998)
    parser.add_argument('--backbone-lr', type=float, default=1e-5)
    parser.add_argument('--fusion-lr', type=float, default=1e-4)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--preflight-only', action='store_true')
    cfg = vars(parser.parse_args())
    for key in ['data', 'labels', 'raptor', 'resnet', 'output', 'resume']:
        if cfg[key]:
            cfg[key] = str(Path(cfg[key]).resolve())
    run(cfg)


if __name__ == '__main__':
    main()

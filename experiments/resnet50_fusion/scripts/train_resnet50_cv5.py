"""Five-fold ResNet50 fusion fine-tuning on minimum-overlap V2 tensors.

Each invocation trains one study-level fold from the original pretrained parents.
Gold74 is evaluated, but only the left-out training fold selects checkpoints.
"""
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Subset

from models.raptor_v5_resnet50 import (
    FusionConfig, build_raptor_v5_resnet50, fused_checkpoint,
    load_fused_checkpoint, optimizer_groups,
)
from scripts.train_resnet50_full_gold4 import (
    EMA, RECIPE_SHA, StudyDataset, atomic_json, atomic_save, bag_logits,
    discover, evaluate, event, label_table, reference_lr_factor, restore_rng,
    rng_state, train_mode,
)


FOLDS_SHA256 = '5e899b9bd07c55759d4a8cd2425d2095bba94605196f88849f6f18c16dacc869'
LABEL_SHA256 = {
    'HYBRID_labels_scores.csv': '40f4d6c4669913a8aedf946ce4645ed16972d554536dcd1c7b1d1fde854dfed6',
    'HYBRID_labels_binary.csv': 'aa64723de96d45408588c6fb6c234980d610e996e3f1f4241111213aeefe0316',
    'gold74_eval_set.csv': 'bb34e7642532e49b1626078f56148375715cf9add3e3079260a4e9e981368782',
}
FOLD_SIZES = (866, 866, 867, 867, 867)
LATEST_CHECKPOINT = 'checkpoints/latest.pt'
BEST_CHECKPOINTS = {
    'raw': 'checkpoints/best_validation_raw.pt',
    'ema': 'checkpoints/best_validation_ema.pt',
}
PENDING_BEST_CHECKPOINTS = {
    kind: path.replace('.pt', '.pending.pt') for kind, path in BEST_CHECKPOINTS.items()
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_label_hashes(labels_dir):
    labels_dir = Path(labels_dir)
    actual = {name: sha256(labels_dir / name) for name in LABEL_SHA256}
    if actual != LABEL_SHA256:
        raise ValueError(f'Label CSV hash mismatch: {actual}')
    return actual


def load_cv_split(ids, gold, folds_csv, fold):
    """Validate the pinned study assignment and return one disjoint CV split."""
    if fold not in range(5):
        raise ValueError('fold must be 0..4')
    ids, gold = sorted(map(str, ids)), sorted(map(str, gold))
    if (len(ids) != 4333 or len(set(ids)) != 4333 or len(gold) != 74
            or len(set(gold)) != 74 or set(ids) & set(gold)):
        raise ValueError('Expected disjoint 4,333 training and 74 Gold studies')
    actual_sha = sha256(folds_csv)
    if actual_sha != FOLDS_SHA256:
        raise ValueError(f'CV assignment hash mismatch: {actual_sha}')
    table = pd.read_csv(folds_csv, dtype=str, keep_default_na=False)
    if list(table.columns) != ['StudyInstanceUID', 'fold'] or len(table) != 4333:
        raise ValueError('Wrong CV assignment schema or row count')
    if table['StudyInstanceUID'].duplicated().any() or set(table['StudyInstanceUID']) != set(ids):
        raise ValueError('CV assignment has missing, extra, or duplicate study IDs')
    if not table['fold'].isin([str(i) for i in range(5)]).all():
        raise ValueError('CV assignment has a fold outside 0..4')
    counts = tuple(int((table['fold'] == str(i)).sum()) for i in range(5))
    if counts != FOLD_SIZES:
        raise ValueError(f'Unexpected CV fold sizes: {counts}')
    fold_of = dict(zip(table['StudyInstanceUID'], table['fold']))
    validation = [uid for uid in ids if fold_of[uid] == str(fold)]
    train = [uid for uid in ids if fold_of[uid] != str(fold)]
    if len(train) + len(validation) != 4333 or set(train) & set(validation):
        raise ValueError('CV train/validation coverage or disjointness failed')
    return dict(train=train, validation=validation, heldout_gold=gold)


def cv_lr_factor(step, new_steps_per_epoch, reference_steps_per_epoch=244,
                 reference_total_steps=2440, warmup=100):
    """Match each prior epoch's first/last LR at fold-relative positions.

    A CV fold has 217 steps, fewer than the reference 244. Rounded endpoint
    interpolation skips some intermediate reference factors but exactly keeps
    the two endpoints; the old 100-step warmup and ten-epoch cosine are intact.
    """
    if step < 0 or new_steps_per_epoch < 2 or reference_steps_per_epoch < 2:
        raise ValueError('Invalid CV learning-rate schedule position or dimensions')
    epoch, local = divmod(step, new_steps_per_epoch)
    old_local = round(local * (reference_steps_per_epoch - 1) / (new_steps_per_epoch - 1))
    old_step = epoch * reference_steps_per_epoch + old_local
    return reference_lr_factor(old_step, reference_steps_per_epoch,
                               reference_steps_per_epoch, reference_total_steps, warmup)


def initialize_model(cfg, resume):
    """Never initialize a new fold from a fusion model trained on all studies."""
    if resume is not None:
        return load_fused_checkpoint(resume)
    return build_raptor_v5_resnet50(cfg['raptor'], cfg['resnet'], FusionConfig())


def save_rolling_checkpoint(state):
    """Atomically replace the one full-state restart point for this fold."""
    Path(LATEST_CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)
    atomic_save(state, LATEST_CHECKPOINT)
    atomic_json({'path': LATEST_CHECKPOINT}, 'latest_checkpoint.json')


def save_best_checkpoint(state, kind, *, staged=False):
    """Write best weights; a staged write becomes visible after resume commit."""
    if kind not in BEST_CHECKPOINTS:
        raise ValueError(f'Unknown weight kind: {kind}')
    path = PENDING_BEST_CHECKPOINTS[kind] if staged else BEST_CHECKPOINTS[kind]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_save(state, path)
    return path


def commit_best_checkpoints(best):
    """Finalize candidates recorded by latest.pt; discard interrupted candidates.

    Best weight files are never overwritten before the full-state checkpoint
    records the new best epoch and score. On a restart this reconciliation can
    safely finish a committed replacement or remove an uncommitted candidate.
    """
    for kind in ('raw', 'ema'):
        pending = Path(PENDING_BEST_CHECKPOINTS[kind])
        if not pending.exists():
            continue
        saved = torch.load(pending, map_location='cpu', weights_only=True)
        recorded = best[kind]
        matches = (saved.get('weight_kind') == kind
                   and saved.get('epoch') == recorded['epoch']
                   and saved.get('metrics', {}).get('validation', {}).get('macro_auc')
                   == recorded['score'])
        del saved
        if matches:
            pending.replace(BEST_CHECKPOINTS[kind])
        else:
            if recorded['epoch'] is not None and not Path(BEST_CHECKPOINTS[kind]).is_file():
                raise FileNotFoundError(f'Committed {kind} best checkpoint is missing')
            pending.unlink()


def select_best(best, result, epoch):
    """Update raw/EMA winners using only left-out-fold macro AUROC."""
    improved = []
    for kind in ('raw', 'ema'):
        score = result[kind]['validation']['macro_auc']
        if score is None or not math.isfinite(float(score)):
            raise ValueError(f'Undefined validation macro AUROC for {kind}')
        if float(score) > best[kind]['score']:
            best[kind] = dict(
                score=float(score), epoch=epoch,
                checkpoint=BEST_CHECKPOINTS[kind],
                predictions=f'predictions/epoch_{epoch:02d}_{kind}_validation.npz',
            )
            improved.append(kind)
    return improved


def selection_payload(best, fold):
    entries = {}
    for kind in ('raw', 'ema'):
        item = best[kind]
        if item['epoch'] is None:
            raise ValueError(f'No selected {kind} epoch')
        if not Path(item['checkpoint']).is_file() or not Path(item['predictions']).is_file():
            raise FileNotFoundError(f'Selected {kind} checkpoint/predictions missing: {item}')
        entries[kind] = dict(epoch=item['epoch'], macro_auc=item['score'],
                             checkpoint=item['checkpoint'], predictions=item['predictions'])
    kind = max(('raw', 'ema'), key=lambda k: (entries[k]['macro_auc'], k == 'ema'))
    return dict(fold=fold, selection_metric='validation_macro_auc', raw=entries['raw'],
                ema=entries['ema'], selected=dict(weight_kind=kind, **entries[kind]))


def stable_config(cfg):
    # These are launch mechanics, not the training recipe. The launcher does
    # a preflight-only invocation followed by a resume from latest.pt.
    return {k: v for k, v in cfg.items() if k not in ('resume', 'preflight_only')}


def run(cfg):
    output = Path(cfg['output']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    for folder in ('checkpoints', 'predictions'):
        Path(folder).mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing CPU training')
    if not 1 <= cfg['epochs'] <= 5:
        raise ValueError('CV fine-tuning is limited to at most five epochs per fold')
    if cfg['accumulate'] != 16:
        raise ValueError('Expected effective batch of 16 studies')
    if (cfg['workers'] < 0 or cfg['chunk'] < 1 or cfg['eval_chunk'] < 1
            or cfg['save_every'] < 1):
        raise ValueError('Invalid worker, chunk, or checkpoint cadence setting')
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True

    files, ids, gold = discover(cfg['data'])
    split = load_cv_split(ids, gold, cfg['folds_csv'], cfg['fold'])
    label_hashes = verify_label_hashes(cfg['labels'])
    labels = Path(cfg['labels'])
    soft = label_table(labels / 'HYBRID_labels_scores.csv', ids)
    binary = label_table(labels / 'HYBRID_labels_binary.csv', ids, binary=True)
    gold_labels = label_table(labels / 'gold74_eval_set.csv', gold, missing=True, binary=True)
    cfg = {
        **cfg, 'recipe_sha256': RECIPE_SHA, 'folds_sha256': FOLDS_SHA256,
        'labels_sha256': label_hashes, 'model_format': 'raptor_v5_resnet50_v3',
        'train_k': 24, 'eval_k': 'all', 'freeze_bn_stats': True,
        'ema_policy': 'fp32 parameters and floating buffers; integer buffers copied',
        'accuracy_definition': 'fraction of available binary finding labels correct at sigmoid >= 0.5',
        'selection_metric': 'left-out-fold validation macro AUROC only; Gold74 never selects checkpoints',
        'learning_rate_reference': dict(
            prior_steps_per_epoch=244, prior_total_steps=2440, warmup=100,
            mapping='rounded per-epoch endpoint interpolation; first and last prior LR factors match'),
    }
    training_config = stable_config(cfg)
    for name, value in (('training_split.json', split), ('training_config.json', training_config)):
        path = Path(name)
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'Existing {name} differs; use an isolated fold output directory')
        atomic_json(value, path)
    complete = Path('complete.json')
    if complete.exists():
        marker = json.loads(complete.read_text())
        if (marker.get('fold') != cfg['fold'] or marker.get('epochs') != cfg['epochs']
                or len(json.loads(Path('history.json').read_text())) != cfg['epochs']
                or not Path('latest_checkpoint.json').exists()):
            raise ValueError('Inconsistent completed fold marker')
        selection_payload(marker['best'], cfg['fold'])
        return
    if cfg['resume'] is None and Path('latest_checkpoint.json').exists():
        raise ValueError('Existing fold checkpoint found; resume it instead of restarting from parents')

    data = StudyDataset(files, split['train'], soft, cfg['seed'], train=True)
    eval_loaders = {
        'validation': DataLoader(StudyDataset(files, split['validation'], binary), batch_size=1,
                                 num_workers=cfg['workers'], pin_memory=True),
        'gold74': DataLoader(StudyDataset(files, split['heldout_gold'], gold_labels), batch_size=1,
                             num_workers=cfg['workers'], pin_memory=True),
    }
    resume = torch.load(cfg['resume'], map_location='cpu', weights_only=True) if cfg['resume'] else None
    if resume is not None and (resume['training_split'] != split
                               or stable_config(resume['training_config']) != training_config):
        raise ValueError('Resume split/config mismatch')
    model = initialize_model(cfg, resume).cuda()
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, cfg['backbone_lr'], cfg['fusion_lr'], cfg['head_lr']), weight_decay=.01)
    steps_per_epoch = math.ceil(len(data) / cfg['accumulate'])
    if steps_per_epoch != 217:
        raise ValueError(f'Expected 217 optimizer steps per CV epoch, found {steps_per_epoch}')
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cv_lr_factor(step, steps_per_epoch))
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp == torch.float16)
    ema = EMA(model, cfg['ema_decay'])
    start_epoch, start_pos, global_step, history = 0, 0, 0, []
    best = {'raw': dict(score=-1., epoch=None), 'ema': dict(score=-1., epoch=None)}
    running, seen = 0., 0
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        scaler.load_state_dict(resume['scaler'])
        ema.load_state_dict(resume['ema'], model)
        start_epoch, start_pos, global_step = (
            resume['epoch_index'], resume['next_study'], resume['global_step'])
        history, best = resume['history'], resume['best']
        running, seen = resume['running_loss'], resume['seen']
        if (ema.updates != global_step or global_step != start_epoch * steps_per_epoch
                + math.ceil(start_pos / cfg['accumulate'])
                or not 0 <= start_epoch <= cfg['epochs']
                or not 0 <= start_pos <= len(data) or len(history) != start_epoch):
            raise ValueError('Resume optimizer/EMA/epoch position is inconsistent')
        restore_rng(resume['rng'])
        del resume
    commit_best_checkpoints(best)
    if all(best[kind]['epoch'] is not None for kind in ('raw', 'ema')):
        atomic_json(selection_payload(best, cfg['fold']), 'best_checkpoint.json')
    event('run_ready', fold=cfg['fold'], gpu=torch.cuda.get_device_name(),
          train=len(data), validation=len(split['validation']), gold74=len(gold),
          epochs=cfg['epochs'], steps_per_epoch=steps_per_epoch, amp=str(amp),
          parameters=sum(p.numel() for p in model.parameters()), config=training_config)

    def save(epoch_index, next_study, running_loss, count):
        state = fused_checkpoint(model)
        state.update(
            fold=cfg['fold'], weight_kind='raw',
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(), ema=ema.state_dict(), global_step=global_step,
            epoch_index=epoch_index, next_study=next_study, running_loss=running_loss,
            seen=count, training_split=split, training_config=training_config,
            history=history, best=best, rng=rng_state())
        save_rolling_checkpoint(state)

    if cfg['resume'] is None:
        save(0, 0, 0., 0)
    if start_epoch == start_pos == global_step == 0:
        probe_rng = rng_state()
        item = data[0]
        train_mode(model)
        probe = F.binary_cross_entropy_with_logits(
            bag_logits(model, item['images'].cuda(), cfg['chunk'], amp, True),
            item['targets'].cuda()[None])
        probe.backward()
        if not torch.isfinite(probe) or not all(
                p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Preflight loss/gradients are not finite')
        optimizer.zero_grad(set_to_none=True)
        del probe, item
        restore_rng(probe_rng)
        torch.cuda.empty_cache()
        event('training_preflight_passed', fold=cfg['fold'],
              peak_gib=torch.cuda.max_memory_allocated() / 2**30)
    if cfg['preflight_only']:
        return

    for epoch in range(start_epoch, cfg['epochs']):
        data.epoch = epoch
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(
            cfg['seed'] + epoch)).tolist()
        offset = start_pos if epoch == start_epoch else 0
        if epoch != start_epoch:
            running, seen = 0., 0
        loader = DataLoader(
            Subset(data, order[offset:]), batch_size=1, num_workers=cfg['workers'],
            pin_memory=True, generator=torch.Generator().manual_seed(cfg['seed'] + epoch))
        train_mode(model)
        optimizer.zero_grad(set_to_none=True)
        event('epoch_started', fold=cfg['fold'], epoch=epoch + 1,
              next_study=offset, total=len(data))
        began = time.monotonic()
        for local, item in enumerate(loader):
            index = offset + local
            group_start = (index // cfg['accumulate']) * cfg['accumulate']
            group_size = min(cfg['accumulate'], len(data) - group_start)
            images = item['images'][0].cuda(non_blocking=True)
            targets = item['targets'].cuda(non_blocking=True)
            logits = bag_logits(model, images, cfg['chunk'], amp, True)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch + 1}, study {index}')
            scaler.scale(loss / group_size).backward()
            running += float(loss.detach())
            seen += 1
            boundary = (index + 1) % cfg['accumulate'] == 0 or index + 1 == len(data)
            if boundary:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite gradients; last successful checkpoint preserved')
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    raise FloatingPointError('AMP skipped an optimizer update')
                global_step += 1
                scheduler.step()
                ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                if global_step % cfg['save_every'] == 0:
                    save(epoch, index + 1, running, seen)
                if global_step % 5 == 0 or index + 1 == len(data):
                    event('training_progress', fold=cfg['fold'], epoch=epoch + 1,
                          studies=index + 1, total=len(data), loss=running / seen,
                          optimizer_step=global_step, ema_updates=ema.updates,
                          seconds=time.monotonic() - began,
                          peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            del images, targets, logits, loss

        save(epoch, len(data), running, seen)
        result = {'epoch': epoch + 1, 'train_loss': running / seen,
                  'raw': {}, 'ema': {}}
        for name, eval_loader in eval_loaders.items():
            result['raw'][name] = evaluate(
                model, eval_loader, amp, cfg['eval_chunk'],
                f'epoch_{epoch + 1:02d}_raw_{name}')
        with ema.installed(model):
            for name, eval_loader in eval_loaders.items():
                result['ema'][name] = evaluate(
                    model, eval_loader, amp, cfg['eval_chunk'],
                    f'epoch_{epoch + 1:02d}_ema_{name}')
            improved = select_best(best, result, epoch + 1)
            if 'ema' in improved:
                ema_checkpoint = fused_checkpoint(model)
                ema_checkpoint.update(
                    fold=cfg['fold'], epoch=epoch + 1, weight_kind='ema',
                    ema_updates=ema.updates, metrics=result['ema'],
                    training_split=split, training_config=training_config)
                save_best_checkpoint(ema_checkpoint, 'ema', staged=True)
        if 'raw' in improved:
            raw_checkpoint = fused_checkpoint(model)
            raw_checkpoint.update(
                fold=cfg['fold'], epoch=epoch + 1, weight_kind='raw',
                ema_updates=ema.updates, metrics=result['raw'],
                training_split=split, training_config=training_config)
            save_best_checkpoint(raw_checkpoint, 'raw', staged=True)
        history.append(result)
        atomic_json(history, 'history.json')
        save(epoch + 1, 0, 0., 0)
        commit_best_checkpoints(best)
        atomic_json(selection_payload(best, cfg['fold']), 'best_checkpoint.json')
        event('epoch_complete', fold=cfg['fold'], improved=improved, **result)

    if (len(history) != cfg['epochs'] or global_step != steps_per_epoch * cfg['epochs']
            or ema.updates != global_step):
        raise ValueError('CV training ended with incomplete epochs or EMA updates')
    selection = selection_payload(best, cfg['fold'])
    atomic_json(selection, 'best_checkpoint.json')
    atomic_json(dict(fold=cfg['fold'], epochs=cfg['epochs'], optimizer_steps=global_step,
                     ema_updates=ema.updates, best=best, selected=selection['selected'],
                     latest_checkpoint=json.loads(Path('latest_checkpoint.json').read_text())['path']),
                'complete.json')
    event('training_complete', fold=cfg['fold'], epochs=cfg['epochs'],
          optimizer_steps=global_step, ema_updates=ema.updates,
          selected=selection['selected'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'labels', 'raptor', 'resnet', 'output', 'folds-csv'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--resume')
    parser.add_argument('--epochs', type=int, default=5)
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
    for key in ('data', 'labels', 'raptor', 'resnet', 'output', 'folds_csv', 'resume'):
        if cfg[key]:
            cfg[key] = str(Path(cfg[key]).resolve())
    run(cfg)


if __name__ == '__main__':
    main()

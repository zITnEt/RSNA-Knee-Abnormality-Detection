"""Aggregate completed five-fold predictions without using Gold74 for selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.raptor_v5_resnet50 import LABELS
from scripts.train_resnet50_full_gold4 import atomic_json, metrics


def read_predictions(path, expected_ids):
    with np.load(path, allow_pickle=False) as saved:
        ids = saved['study_uid'].astype(str)
        labels = saved['labels'].astype(np.float32)
        logits = saved['logits'].astype(np.float32)
    expected_shape = (len(ids), len(LABELS))
    if (len(ids) != len(expected_ids) or len(set(ids)) != len(ids)
            or set(ids) != set(expected_ids)
            or labels.shape != expected_shape or logits.shape != expected_shape
            or not np.isfinite(logits).all()):
        raise ValueError(f'Incomplete or invalid predictions: {path}')
    return ids, labels, logits


def source_for(selection, variant):
    if variant == 'selected':
        item = selection['selected']
        return item['weight_kind'], item['epoch']
    if variant in ('raw_best', 'ema_best'):
        kind = variant.split('_')[0]
        return kind, selection[kind]['epoch']
    kind = variant.split('_')[0]
    return kind, 5


def summarize(root):
    root = Path(root).resolve()
    folds = []
    for fold in range(5):
        directory = root / f'fold_{fold}'
        complete = json.loads((directory / 'complete.json').read_text())
        split = json.loads((directory / 'training_split.json').read_text())
        selection = json.loads((directory / 'best_checkpoint.json').read_text())
        if (complete['fold'] != fold or complete['epochs'] != 5
                or selection['fold'] != fold
                or selection['selection_metric'] != 'validation_macro_auc'):
            raise ValueError(f'Wrong completion or checkpoint selection: fold {fold}')
        folds.append((directory, split, selection))
    all_ids = [uid for _, split, _ in folds for uid in split['validation']]
    gold_ids = folds[0][1]['heldout_gold']
    if (len(all_ids) != 4333 or len(set(all_ids)) != 4333
            or len(gold_ids) != 74 or len(set(gold_ids)) != 74
            or set(all_ids) & set(gold_ids)
            or any(split['heldout_gold'] != gold_ids for _, split, _ in folds)):
        raise ValueError('Fold validation or Gold74 coverage differs from pinned corpus')

    report = dict(folds=5, non_gold_studies=len(all_ids), gold74_studies=len(gold_ids),
                  selection_metric='per-fold validation macro AUROC', variants={})
    for variant in ('selected', 'raw_best', 'ema_best', 'raw_epoch05', 'ema_epoch05'):
        oof = {}
        gold_probabilities = []
        gold_truth = None
        chosen = []
        for fold, (directory, split, selection) in enumerate(folds):
            kind, epoch = source_for(selection, variant)
            if kind not in ('raw', 'ema') or not 1 <= epoch <= 5:
                raise ValueError(f'Invalid selected checkpoint: fold {fold}')
            if variant in ('selected', 'raw_best', 'ema_best'):
                item = selection['selected'] if variant == 'selected' else selection[kind]
                if not (directory / item['checkpoint']).is_file():
                    raise FileNotFoundError(f'Selected weight file missing: fold {fold}')
            prefix = f'epoch_{epoch:02d}_{kind}'
            val_ids, val_truth, val_logits = read_predictions(
                directory / 'predictions' / f'{prefix}_validation.npz',
                split['validation'])
            for uid, truth, logit in zip(val_ids, val_truth, val_logits):
                if uid in oof:
                    raise ValueError(f'Duplicate out-of-fold study {uid}')
                oof[uid] = (truth, logit)
            this_gold_ids, this_gold_truth, gold_logits = read_predictions(
                directory / 'predictions' / f'{prefix}_gold74.npz', gold_ids)
            positions = {uid: i for i, uid in enumerate(this_gold_ids)}
            order = [positions[uid] for uid in gold_ids]
            this_gold_truth = this_gold_truth[order]
            gold_logits = gold_logits[order]
            if gold_truth is None:
                gold_truth = this_gold_truth
            elif not np.array_equal(gold_truth, this_gold_truth, equal_nan=True):
                raise ValueError('Gold74 labels differ across folds')
            gold_probabilities.append(torch.from_numpy(gold_logits).sigmoid().numpy())
            chosen.append(dict(fold=fold, weight_kind=kind, epoch=epoch))

        if set(oof) != set(all_ids):
            raise ValueError(f'Missing out-of-fold studies: {variant}')
        ordered = sorted(oof)
        oof_truth = np.asarray([oof[uid][0] for uid in ordered])
        oof_logits = np.asarray([oof[uid][1] for uid in ordered])
        ensemble_prob = np.mean(gold_probabilities, axis=0)
        clipped = np.clip(ensemble_prob, 1e-6, 1 - 1e-6)
        gold_ensemble_logits = np.log(clipped / (1 - clipped))
        np.savez_compressed(root / f'{variant}_oof.npz',
                            study_uid=np.asarray(ordered), labels=oof_truth,
                            logits=oof_logits)
        np.savez_compressed(root / f'{variant}_gold74_ensemble.npz',
                            study_uid=np.asarray(gold_ids), labels=gold_truth,
                            probabilities=ensemble_prob)
        report['variants'][variant] = dict(
            fold_models=chosen, oof=metrics(oof_truth, oof_logits),
            gold74_ensemble=metrics(gold_truth, gold_ensemble_logits))
    atomic_json(report, root / 'cv5_summary.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args()
    report = summarize(args.run_root)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

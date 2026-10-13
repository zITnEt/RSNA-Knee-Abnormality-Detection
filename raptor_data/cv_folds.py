"""Deterministic K-fold split of the 4,333-study training corpus and the per-epoch CV metrics.

The held-out gold74 studies are a separate preprocessing split and never enter these folds.
No patient identifiers are available, so folds are study-level, stratified over the 12
findings with iterative multilabel stratification (Sechidis et al., 2011).

Validation accuracy (the selection score) = mean of the gold74 macro AUC and the left-out-fold
macro AUC, computed separately for raw and EMA weights.
"""
import numpy as np

METRIC_COLUMNS = ('fold', 'epoch', 'optimizer_step', 'weights', 'gold74_macro_auc', 'fold_macro_auc',
                  'validation_accuracy', 'training_loss', 'note')


def stratified_folds(study_ids, labels, n_folds=5, seed=20261010):
    """Map each study ID to a fold in [0, n_folds); independent of the input order."""
    ids = [str(u) for u in study_ids]
    y = np.asarray(labels)
    if y.ndim != 2 or len(ids) != len(y):
        raise ValueError('labels must be [studies, findings] and match study_ids')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate study IDs')
    if not np.isin(y, (0, 1)).all():
        raise ValueError('Stratification needs binary labels')
    order = np.argsort(np.array(ids))
    ids = [ids[i] for i in order]
    y = y[order].astype(np.int64)
    n = len(ids)
    rng = np.random.default_rng(seed)
    visit = rng.permutation(n)
    fold_of = np.full(n, -1, dtype=np.int64)
    remaining = np.ones(n, dtype=bool)
    capacity = np.full(n_folds, n/n_folds)                       # studies still wanted per fold
    wanted = np.tile(y.sum(0)/n_folds, (n_folds, 1)).astype(float)   # positives still wanted per fold/label

    def pick(candidates):
        candidates = candidates[capacity[candidates] == capacity[candidates].max()]
        return int(candidates[0] if len(candidates) == 1 else candidates[rng.integers(len(candidates))])

    while remaining.any():
        counts = y[remaining].sum(0)
        present = np.flatnonzero(counts > 0)
        if len(present) == 0:                                    # studies with no positive finding
            for i in visit[remaining[visit]]:
                fold = pick(np.arange(n_folds))
                fold_of[i] = fold; remaining[i] = False; capacity[fold] -= 1
            break
        label = present[np.argmin(counts[present])]              # rarest remaining finding first
        for i in visit[remaining[visit] & (y[visit, label] == 1)]:
            need = wanted[:, label]
            fold = pick(np.flatnonzero(need == need.max()))
            fold_of[i] = fold; remaining[i] = False
            capacity[fold] -= 1; wanted[fold] -= y[i]
    return dict(zip(ids, fold_of.tolist()))


def combined_score(by_split):
    """Validation accuracy: mean of gold74 and left-out-fold macro AUC (gold74 alone if no fold)."""
    gold = by_split['gold74']['macro_auc']
    if 'validation' not in by_split:
        return gold
    return (gold + by_split['validation']['macro_auc'])/2


def metric_rows(history, fold):
    """One row per (evaluation, weights); step 0 is the parent, where EMA weights equal raw weights."""
    rows = []
    for entry in history:
        evaluation = entry.get('evaluation') or {}
        kinds = dict(evaluation)
        note = ''
        if entry['epoch'] == 0 and 'ema' not in kinds and 'raw' in kinds:
            kinds['ema'] = kinds['raw']
            note = 'step 0 (parent): EMA is initialised from these weights, so EMA = raw'
        for kind in ('raw', 'ema'):
            if kind not in kinds:
                continue
            by_split = kinds[kind]
            rows.append(dict(
                fold=fold, epoch=entry['epoch'], optimizer_step=entry['step'], weights=kind,
                gold74_macro_auc=by_split['gold74']['macro_auc'],
                fold_macro_auc=by_split['validation']['macro_auc'] if 'validation' in by_split else None,
                validation_accuracy=combined_score(by_split) if 'validation' in by_split else None,
                training_loss=entry.get('training_loss'), note=note))
    return rows

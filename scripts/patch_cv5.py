"""5-fold cross-validation for the DDP fusion trainer.

* cfg['fold'] in [0, cv_folds): train on the other folds of the 4,333 training studies, validate on
  this fold (gold74 is a separate split and is never trained on); fold=None keeps the old 90/10 split.
* Every evaluation (step 0 and each epoch) scores raw and EMA weights on gold74 and the left-out fold;
  validation accuracy = mean of the two macro AUCs; fusion_best.pt is selected by it.
* cv_metrics.csv / cv_metrics.json hold the per-epoch table; cv_folds.csv holds the fold assignment.
* Smoke runs validate on the first smoke_validation_studies studies of the fold.

    python scripts/patch_cv5.py
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


patch(ROOT/'scripts/train_fusion_ddp.py', [
    ("""from raptor_data.ready_tensors import ReadyTripletDataset
""",
     """from raptor_data.ready_tensors import ReadyTripletDataset
from raptor_data.cv_folds import METRIC_COLUMNS, combined_score, metric_rows, stratified_folds
"""),
    ("""    studies_per_gpu=1, sync_batchnorm=False, probe_reserve_gib=2.0)}""",
     """    studies_per_gpu=1, sync_batchnorm=False, probe_reserve_gib=2.0,
    fold=None, cv_folds=5, cv_seed=20261010, smoke_validation_studies=24)}"""),
    ("""    train_idx, val_idx = split_indices(len(train_data), cfg['seed'], cfg['validation_fraction'])
    split = dict(""",
     """    fold_of = None
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
    split = dict("""),
    ("""    validation_sets = {'validation': Subset(val_full, val_idx)}""",
     """    validation_sets = {'validation': Subset(val_full, val_idx[:cfg['smoke_validation_studies']]
                                            if cfg['smoke'] else val_idx)}"""),
    ("""    if RANK == 0:
        Path('training_split.json').write_text(json.dumps(split, indent=2))""",
     """    if RANK == 0:
        Path('training_split.json').write_text(json.dumps(split, indent=2))
        if fold_of is not None:
            pd.DataFrame(sorted(fold_of.items()), columns=['StudyInstanceUID', 'fold']).to_csv(
                'cv_folds.csv', index=False)"""),
    ("""        sets = {**gold_sets, **(validation_sets if validation and not cfg['smoke'] else {})}""",
     """        sets = {**gold_sets, **(validation_sets if validation else {})}"""),
    ("""    best = dict(gold_auc=-1.0)
    history = []""",
     """    best = dict(gold_auc=-1.0, score=-1.0)
    history = []"""),
    ("""        for kind, by_split in results.items():
            auc = by_split['gold74']['macro_auc']
            if auc > best['gold_auc']:
                best = dict(gold_auc=auc, weights=kind, tag=tag, step=step)""",
     """        for kind, by_split in results.items():
            score = combined_score(by_split)   # mean of gold74 and left-out-fold macro AUC
            if score > best.get('score', best['gold_auc']):
                best = dict(score=score, gold_auc=by_split['gold74']['macro_auc'],
                            fold_auc=by_split.get('validation', {}).get('macro_auc'),
                            weights=kind, tag=tag, step=step)"""),
    ("""    def save(epoch, next_batch, running, seen):""",
     """    def write_metrics():
        if RANK != 0:
            return
        rows = metric_rows(history, cfg['fold'])
        pd.DataFrame(rows, columns=list(METRIC_COLUMNS)).to_csv('cv_metrics.csv', index=False)
        Path('cv_metrics.json').write_text(json.dumps(dict(
            fold=cfg['fold'], definition='validation_accuracy = (gold74 macro AUC + left-out fold macro AUC) / 2',
            best=best, rows=rows), indent=2))

    def save(epoch, next_batch, running, seen):"""),
    ("""            start['raw'].update(extra)
            history.append(dict(step=0, epoch=0, evaluation=start))
            consider_best({'raw': start['raw']}, 'step0', 0)""",
     """            start['raw'].update(extra)
            history.append(dict(step=0, epoch=0, evaluation=start))
            consider_best({'raw': start['raw']}, 'step0', 0)
            write_metrics()"""),
    ("""            consider_best(results, tag, global_step)
            if (cfg['keep_epoch_ema'] or last_epoch) and not cfg['smoke']:""",
     """            consider_best(results, tag, global_step)
            write_metrics()
            if (cfg['keep_epoch_ema'] or last_epoch) and not cfg['smoke']:"""),
    ("""            event('epoch_complete', epoch=epoch+1, optimizer_steps=global_step,
                  training_loss=running/max(seen, 1), macro_auc=summary, best=best,""",
     """            event('epoch_complete', epoch=epoch+1, optimizer_steps=global_step,
                  training_loss=running/max(seen, 1), macro_auc=summary, best=best,
                  validation_accuracy={kind: combined_score(by) for kind, by in results.items()},"""),
], marker='stratified_folds')

patch(ROOT/'scripts/build_fusion_v2_notebook.py', [
    ("""    'raptor_data/ready_tensors.py': ROOT/'raptor_data/ready_tensors.py',
""",
     """    'raptor_data/ready_tensors.py': ROOT/'raptor_data/ready_tensors.py',
    'raptor_data/cv_folds.py': ROOT/'raptor_data/cv_folds.py',
"""),
    ("""  and after every epoch, for raw and EMA weights. `fusion_best.pt` is selected by gold74 AUC.""",
     """  and after every epoch, for raw and EMA weights. With `fold` set, the validation set is that
  left-out fold and `fusion_best.pt` is selected by validation accuracy = mean of the gold74 and
  fold macro AUCs (`cv_metrics.csv`); otherwise by gold74 AUC."""),
], marker='cv_folds.py')

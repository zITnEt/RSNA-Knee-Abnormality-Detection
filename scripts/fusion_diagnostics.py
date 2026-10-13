"""Fusion diagnostics on Kaggle (2 GPUs, one process per GPU via DIAG_RANK / CUDA_VISIBLE_DEVICES).

For every trained checkpoint (5 CV folds, the 20-epoch run's saved best and final EMA):
  * weights: drift from the parent per component; fusion layer norms (last layers were zero at init).
  * activations, per window: ||F1 residual|| / ||S2 features||, cos(F1 residual, S2 features), the same
    for F2 at stage 4, and the relative change of the pooled 1024-d feature when the fusion is removed.
  * predictions in three modes on the same studies:
        full       - the model as trained;
        no_fusion  - both residuals removed (CoAtNet path of the fine-tuned model only);
        rad_mean   - RadImageNet features replaced by their average over the evaluated windows, so the
                     fusion MLPs still run but receive no study-specific RadImageNet information.
Evaluation sets: gold74 (all checkpoints) and, for each CV fold model, 150 studies of its own left-out fold.
The parent Raptor (step 0) is evaluated on the same sets for a fine-tuning vs fusion decomposition.
Writes /kaggle/working/diag/*. No training, no labels are used here (scoring happens offline).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.raptor_v5_radimagenet import (LABELS, FusionConfig, build_raptor_v5_radimagenet,
                                          load_fused_checkpoint, rad_input_transform)
from raptor_data.cv_folds import stratified_folds
from raptor_data.ready_tensors import ReadyTripletDataset
from train_fusion_v2 import asset, discover_roots, load_label_table

CHUNK = 16
PER_FOLD = 150
SEED = 20261012
OUT = Path('/kaggle/working/diag')
RANK = int(os.environ.get('DIAG_RANK', 0))
MODES = ('full', 'no_fusion', 'rad_mean')


def log(kind, **data):
    print(json.dumps(dict(event=kind, rank=RANK, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **data)),
          flush=True)


def kernel_dir(slug):
    for p in (Path('/kaggle/input/notebooks/zafarnematov')/slug, Path('/kaggle/input')/slug):
        if p.is_dir():
            return p
    raise FileNotFoundError(slug)


def checkpoints():
    ck = {}
    for k, slug in ((0, 'rsna-fusion-cv5-f01'), (1, 'rsna-fusion-cv5-f01'), (2, 'rsna-fusion-cv5-f23'),
                    (3, 'rsna-fusion-cv5-f23'), (4, 'rsna-fusion-cv5-f4')):
        path = kernel_dir(slug)/f'fold{k}'/'fusion_best.pt'
        assert path.is_file(), path
        ck[f'cv_fold{k}'] = path
    ck['run20_saved_best_raw_ep2'] = next(kernel_dir('rsna-fusion-ddp-train').rglob('fusion_best.pt'))
    ck['run20_ema_epoch20'] = next(kernel_dir('rsna-fusion-ddp-resume').rglob('fusion_ema_epoch20.pt'))
    return ck


def build_parent():
    return build_raptor_v5_radimagenet(asset('raptor-knee-maxspan', 'raptor_ft_coatnet_v5_full_swa.pt'),
                                       asset('resnet-50-radimagenet-marwan', 'ResNet50.pt'),
                                       FusionConfig(rad_input='pm1_middle'))


# ----------------------------------------------------------------------------- weights
def weight_stats(parent_sd, sd):
    groups = {'coatnet': ('backbone.coatnet.',), 'radimagenet': ('backbone.radimagenet.',),
              'head': ('norm.', 'att.', 'clsW', 'clsb')}
    is_param = lambda k: not any(s in k for s in ('running_mean', 'running_var', 'num_batches_tracked'))
    out = {}
    for name, prefixes in groups.items():
        keys = [k for k in sd if k.startswith(prefixes) and is_param(k)]
        a = torch.cat([sd[k].float().flatten() for k in keys]); b = torch.cat([parent_sd[k].float().flatten() for k in keys])
        out[name] = dict(parameters=int(a.numel()), norm=float(a.norm()), drift_norm=float((a - b).norm()),
                         relative_drift=float((a - b).norm()/b.norm()))
    for name in ('f1', 'f2'):
        layers = {}
        for idx, label in ((0, 'layernorm'), (1, 'linear1'), (4, 'linear2'), (7, 'linear3_output_zero_init')):
            w = sd[f'backbone.{name}.mlp.{idx}.weight'].float(); bias = sd[f'backbone.{name}.mlp.{idx}.bias'].float()
            layers[label] = dict(weight_norm=float(w.norm()), weight_rms=float(w.pow(2).mean().sqrt()),
                                 bias_norm=float(bias.norm()),
                                 **({'weight_mean_minus_1': float((w - 1).mean())} if idx == 0 else {}))
        out[name] = layers
    buffers = [k for k in sd if not is_param(k) and 'num_batches' not in k]
    out['batchnorm_buffers_max_abs_change'] = max(float((sd[k].float() - parent_sd[k].float()).abs().max())
                                                  for k in buffers)
    return out


# ----------------------------------------------------------------------------- activations
def padded_chunks(images):
    n = len(images)
    padded = -(-n//CHUNK)*CHUNK
    x = images if padded == n else torch.cat([images, images[-1:].expand(padded-n, *images.shape[1:])])
    for start in range(0, padded, CHUNK):
        yield x[start:start+CHUNK], max(0, min(CHUNK, n-start))


@torch.inference_mode()
def rad_means(model, dataset, device):
    bb = model.backbone
    sx = sy = None; count = 0
    for i in range(len(dataset)):
        for chunk, valid in padded_chunks(dataset[i]['images']):
            if not valid:
                continue
            with torch.autocast('cuda', dtype=torch.float16):
                x, y = bb.radimagenet(rad_input_transform(chunk[:valid].to(device).float(), bb.rad_input))
            x, y = x.float().sum(0), y.float().sum(0)
            sx = x if sx is None else sx + x; sy = y if sy is None else sy + y; count += valid
    return (sx/count)[None], (sy/count)[None]


def norm_rows(t):
    return t.flatten(1).float().norm(dim=1)


def cos_rows(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(1).float(), b.flatten(1).float(), dim=1)


@torch.inference_mode()
def run_study(model, images, device, means, modes):
    """Per-mode study probabilities + per-window statistics (full mode only)."""
    bb = model.backbone
    feats = {m: [] for m in modes}
    stats = []
    for chunk, valid in padded_chunks(images):
        c = chunk.to(device, non_blocking=True).float()
        with torch.autocast('cuda', dtype=torch.float16):
            x, y = bb.radimagenet(rad_input_transform(c, bb.rad_input))
            z = bb.coatnet.stages[1](bb._stem_s1(c))
            tail = lambda zz: bb.coatnet.stages[3](bb.coatnet.stages[2](zz))
            pool = lambda vv: bb.coatnet.forward_head(bb.coatnet.norm(vv))
            r1 = bb.f1(x, z)
            v = tail(z + r1)
            r2 = bb.f2(v, y)
            f_full = pool(v + r2)
            feats['full'].append(f_full.float())
            if 'no_fusion' in modes:
                f_nf = pool(tail(z))
                feats['no_fusion'].append(f_nf.float())
            if 'rad_mean' in modes:
                xm, ym = means
                vm = tail(z + bb.f1(xm.to(x.dtype).expand(len(c), -1, -1, -1), z))
                f_rm = pool(vm + bb.f2(vm, ym.to(y.dtype).expand(len(c), -1, -1, -1)))
                feats['rad_mean'].append(f_rm.float())
        if valid:
            s = slice(0, valid)
            row = dict(s2_residual_ratio=(norm_rows(r1[s])/norm_rows(z[s])), s2_cos=cos_rows(r1[s], z[s]),
                       s4_residual_ratio=(norm_rows(r2[s])/norm_rows(v[s])), s4_cos=cos_rows(r2[s], v[s]))
            if 'no_fusion' in modes:
                row['feat_change_no_fusion'] = norm_rows(f_full[s] - f_nf[s])/norm_rows(f_full[s])
            if 'rad_mean' in modes:
                row['feat_change_rad_mean'] = norm_rows(f_full[s] - f_rm[s])/norm_rows(f_full[s])
            stats.append({k: t.cpu().numpy() for k, t in row.items()})
    n = len(images)
    probs = {}
    with torch.autocast('cuda', dtype=torch.float16):
        for m in modes:
            probs[m] = torch.sigmoid(model.head(torch.cat(feats[m], 0)[:n][None]).float())[0].cpu().numpy()
    window_stats = {k: np.concatenate([s[k] for s in stats]) for k in stats[0]}
    return probs, window_stats


@torch.inference_mode()
def run_parent(model, images, device):
    feats = []
    for chunk, _ in padded_chunks(images):
        with torch.autocast('cuda', dtype=torch.float16):
            feats.append(model.backbone(chunk.to(device, non_blocking=True).float()).float())
    with torch.autocast('cuda', dtype=torch.float16):
        return torch.sigmoid(model.head(torch.cat(feats, 0)[:len(images)][None]).float())[0].cpu().numpy()


# ----------------------------------------------------------------------------- jobs
class Subset:
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]


def evaluate(name, model, datasets, device, modes):
    for set_name, dataset in datasets.items():
        started = time.time()
        means = rad_means(model, dataset, device) if 'rad_mean' in modes else None
        rows, window = [], {}
        for i in range(len(dataset)):
            item = dataset[i]
            probs, ws = run_study(model, item['images'], device, means, modes)
            for m, p in probs.items():
                rows.append([item['study_uid'], m, *p.tolist()])
            for k, v in ws.items():
                window.setdefault(k, []).append(v)
                window.setdefault(f'{k}__study_mean', []).append(float(np.mean(v)))
            window.setdefault('study_uid', []).append(item['study_uid'])
        pd.DataFrame(rows, columns=['StudyInstanceUID', 'mode', *LABELS]).to_csv(
            OUT/f'preds_{name}_{set_name}.csv', index=False)
        per_window = {k: np.concatenate(v) for k, v in window.items() if not k.endswith('__study_mean')
                      and k != 'study_uid'}
        np.savez_compressed(OUT/f'windows_{name}_{set_name}.npz', **per_window)
        pd.DataFrame({k: v for k, v in window.items() if k.endswith('__study_mean') or k == 'study_uid'}).to_csv(
            OUT/f'study_stats_{name}_{set_name}.csv', index=False)
        log('evaluated', checkpoint=name, set=set_name, studies=len(dataset),
            windows=int(len(per_window['s2_residual_ratio'])), minutes=round((time.time()-started)/60, 2),
            median={k: float(np.median(v)) for k, v in per_window.items()})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda', 0)
    torch.backends.cudnn.benchmark = True
    roots = discover_roots()
    gold = ReadyTripletDataset(roots, split='held_out_gold', k=None, windows='pure')
    train = ReadyTripletDataset(roots, split='train', k=None, windows='pure')
    ids = [p.stem for p in train.files]
    binary = load_label_table(asset('jev-knee-labels', 'HYBRID_labels_binary.csv'), ids, name='HYBRID binary')
    fold_of = stratified_folds(ids, np.stack([binary[u] for u in ids]), 5, 20261010)
    saved = pd.read_csv(kernel_dir('rsna-fusion-cv5-f01')/'fold0'/'cv_folds.csv', dtype={'StudyInstanceUID': str})
    assert dict(zip(saved.StudyInstanceUID, saved.fold)) == fold_of, 'fold assignment differs from training'
    rng = np.random.default_rng(SEED)
    fold_sets = {}
    for k in range(5):
        members = [i for i, u in enumerate(ids) if fold_of[u] == k]
        fold_sets[k] = Subset(train, sorted(rng.choice(members, PER_FOLD, replace=False).tolist()))
    if RANK == 0:
        pd.DataFrame([(train.files[i].stem, k) for k, s in fold_sets.items() for i in s.indices],
                     columns=['StudyInstanceUID', 'fold']).to_csv(OUT/'fold_subsets.csv', index=False)
    ck = checkpoints()
    jobs = {0: ['cv_fold0', 'cv_fold1', 'run20_saved_best_raw_ep2', 'run20_ema_epoch20'],
            1: ['cv_fold2', 'cv_fold3', 'cv_fold4']}[RANK]
    parent_folds = {0: [0, 1], 1: [2, 3, 4]}[RANK]
    log('start', jobs=jobs, parent_folds=parent_folds, gold=len(gold), per_fold=PER_FOLD)

    parent = build_parent()
    parent_sd = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    parent = parent.to(device).eval()
    sets = {f'fold{k}': fold_sets[k] for k in parent_folds}
    if RANK == 0:
        sets = {'gold74': gold, **sets}
    for set_name, dataset in sets.items():
        rows = []
        for i in range(len(dataset)):
            item = dataset[i]
            rows.append([item['study_uid'], 'parent', *run_parent(parent, item['images'], device).tolist()])
        pd.DataFrame(rows, columns=['StudyInstanceUID', 'mode', *LABELS]).to_csv(
            OUT/f'preds_parent_{set_name}.csv', index=False)
        log('parent_evaluated', set=set_name, studies=len(dataset))
    del parent
    torch.cuda.empty_cache()

    weights = {}
    for name in jobs:
        state = torch.load(ck[name], map_location='cpu', weights_only=True)
        meta = dict(checkpoint=str(ck[name]), selection=state.get('selection'),
                    fold=state.get('training_config', {}).get('fold'),
                    rad_input=state.get('fusion_config', {}).get('rad_input', 'shared (absent in checkpoint)'))
        model = load_fused_checkpoint(state)
        weights[name] = dict(**meta, **weight_stats(parent_sd, model.state_dict()))
        del state
        model = model.to(device).eval()
        sets = {'gold74': gold}
        if name.startswith('cv_fold'):
            sets[f'fold{name[-1]}'] = fold_sets[int(name[-1])]
        evaluate(name, model, sets, device, MODES)
        del model
        torch.cuda.empty_cache()
    (OUT/f'weights_rank{RANK}.json').write_text(json.dumps(weights, indent=2))
    log('done')


if __name__ == '__main__':
    main()

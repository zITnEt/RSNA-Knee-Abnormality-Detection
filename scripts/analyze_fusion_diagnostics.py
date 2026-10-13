"""Fusion-specific statistics from the rsna-fusion-diagnostics Kaggle outputs (results/fusion_diagnostics/diag).

Modes per checkpoint: full (as trained), no_fusion (both residuals removed), rad_mean (RadImageNet features
replaced by their average, so fusion runs but gets no study-specific RadImageNet information); parent = step 0.
Decomposition on identical studies:  full - parent = (no_fusion - parent) + (full - no_fusion)
  no_fusion - parent : change of the fine-tuned CoAtNet/head path with the fusion switched off
  full - no_fusion   : what the trained model gets from its fusion residuals
  full - rad_mean    : the part of that which depends on study-specific RadImageNet information
Sets: gold74 (gold labels, selection set) and each CV fold model's own 150 left-out studies (weak HYBRID labels).

    python scripts/analyze_fusion_diagnostics.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_fusion_posttrain import JEV, LABELS, ece, log_loss, per_finding_auc, read_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__import__('os').environ.get('DIAG_OUT', ROOT/'results/fusion_diagnostics'))
SRC = OUT/'diag'
B, SEED = 2000, 20261012
MODES = ['parent', 'no_fusion', 'rad_mean', 'full']
COLORS = {'full': '#2a78d6', 'no_fusion': '#eb6834', 'rad_mean': '#1baf7a', 'parent': '#8a8a83'}
CKPTS = [f'cv_fold{k}' for k in range(5)] + ['run20_saved_best_raw_ep2', 'run20_ema_epoch20']
INK, MUTED, GRID = '#1f1f1d', '#6b6b66', '#e4e3dc'


def load_preds(name, set_name, uids):
    frame = pd.read_csv(SRC/f'preds_{name}_{set_name}.csv', dtype={'StudyInstanceUID': str})
    out = {}
    for mode, g in frame.groupby('mode'):
        g = g.set_index('StudyInstanceUID')
        assert set(g.index) == set(uids), (name, set_name, mode)
        out[mode] = g.loc[uids, LABELS].to_numpy(float)
    return out


def macro(Y, P):
    aucs = per_finding_auc(Y, P)
    br, ec, ll = [], [], []
    for j in range(Y.shape[1]):
        m = np.isfinite(Y[:, j]); y, p = Y[m, j], P[m, j]
        br.append(np.mean((p - y)**2)); ec.append(ece(y, p)); ll.append(log_loss(y, p))
    return dict(macro_auc=float(np.nanmean(aucs)), brier=float(np.mean(br)), ece=float(np.mean(ec)),
                log_loss=float(np.mean(ll))), aucs


def paired_boot(Y, A, Bp, seed):
    """Paired study bootstrap of macro-AUC(A) - macro-AUC(B); replicates with a single-class finding dropped."""
    rng = np.random.default_rng(seed)
    n = len(Y); diffs, per = [], []
    for _ in range(B):
        ii = rng.integers(0, n, n)
        a, b = per_finding_auc(Y[ii], A[ii]), per_finding_auc(Y[ii], Bp[ii])
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        diffs.append(a.mean() - b.mean()); per.append(a - b)
    diffs, per = np.array(diffs), np.array(per)
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(np.mean(diffs <= 0)),
            np.percentile(per, 2.5, axis=0), np.percentile(per, 97.5, axis=0), len(diffs))


PAIRS = [('full', 'no_fusion'), ('full', 'rad_mean'), ('no_fusion', 'parent'), ('full', 'parent')]


def score_set(set_name, ck, Y, P, rows, pairs, per_finding, seed, S=None):
    for mode in MODES:
        m, aucs = macro(Y, P[mode])
        extra = {}
        if S is not None:
            p = np.clip(P[mode], 1e-6, 1 - 1e-6)
            extra['soft_bce_vs_hybrid_scores'] = float(-np.mean(S*np.log(p) + (1 - S)*np.log(1 - p)))
        rows.append(dict(set=set_name, checkpoint=ck, mode=mode, studies=len(Y), **m, **extra))
    for i, (a, b) in enumerate(PAIRS):
        lo, hi, p, plo, phi, nb = paired_boot(Y, P[a], P[b], seed + i)
        fa, fb = per_finding_auc(Y, P[a]), per_finding_auc(Y, P[b])
        pairs.append(dict(set=set_name, checkpoint=ck, comparison=f'{a} - {b}', delta_macro_auc=float(fa.mean() - fb.mean()),
                          ci_low=lo, ci_high=hi, p_boot_delta_le_0=p, valid_replicates=nb))
        for j, f in enumerate(LABELS):
            per_finding.append(dict(set=set_name, checkpoint=ck, comparison=f'{a} - {b}', finding=f,
                                    delta_auc=float(fa[j] - fb[j]), ci_low=float(plo[j]), ci_high=float(phi[j])))
    full, nf = P['full'], P['no_fusion']
    return dict(set=set_name, checkpoint=ck, mean_abs_prob_change_full_vs_no_fusion=float(np.abs(full - nf).mean()),
                p95_abs_prob_change=float(np.percentile(np.abs(full - nf), 95)),
                decision_flips_at_0_5=float(((full >= .5) != (nf >= .5)).mean()),
                mean_abs_prob_change_full_vs_rad_mean=float(np.abs(full - P['rad_mean']).mean()),
                mean_abs_prob_change_full_vs_parent=float(np.abs(full - P['parent']).mean()))


def main():
    gold = read_labels(JEV/'gold74_eval_set.csv')
    g_uids = sorted(gold.index)
    Yg = gold.loc[g_uids, LABELS].to_numpy(float)
    hyb, soft = read_labels(JEV/'HYBRID_labels_binary.csv'), read_labels(JEV/'HYBRID_labels_scores.csv')
    subsets = pd.read_csv(SRC/'fold_subsets.csv', dtype={'StudyInstanceUID': str})
    rows, pairs, per_finding, change = [], [], [], []

    parent_g = load_preds('parent', 'gold74', g_uids)['parent']
    gold_preds = {}
    for i, ck in enumerate(CKPTS):
        P = load_preds(ck, 'gold74', g_uids); P['parent'] = parent_g
        gold_preds[ck] = P
        change.append(score_set('gold74', ck, Yg, P, rows, pairs, per_finding, SEED + 10*i))
    ens = {m: np.mean([gold_preds[f'cv_fold{k}'][m] for k in range(5)], 0) for m in ('no_fusion', 'rad_mean', 'full')}
    ens['parent'] = parent_g
    change.append(score_set('gold74', '5-fold ensemble', Yg, ens, rows, pairs, per_finding, SEED + 100))
    # official-label subset: the 16 report-verified studies follow a different labelling convention
    m58 = gold.loc[g_uids, 'source'].to_numpy() == 'original_gold58'
    for i, (ck, P) in enumerate([('5-fold ensemble', ens)] + [(c, gold_preds[c]) for c in CKPTS]):
        change.append(score_set('gold58 official', ck, Yg[m58], {m: v[m58] for m, v in P.items()},
                                rows, pairs, per_finding, SEED + 150 + 10*i))

    pooled = {m: [] for m in MODES}; Yp, Sp = [], []
    for k in range(5):
        uids = sorted(subsets[subsets.fold == k].StudyInstanceUID)
        P = load_preds(f'cv_fold{k}', f'fold{k}', uids)
        P['parent'] = load_preds('parent', f'fold{k}', uids)['parent']
        Y, S = hyb.loc[uids, LABELS].to_numpy(float), soft.loc[uids, LABELS].to_numpy(float)
        change.append(score_set(f'fold{k} left-out 150 (weak labels)', f'cv_fold{k}', Y, P, rows, pairs, per_finding,
                                SEED + 200 + 10*k, S))
        for m in MODES:
            pooled[m].append(P[m])
        Yp.append(Y); Sp.append(S)
    pooled = {m: np.concatenate(v) for m, v in pooled.items()}
    change.append(score_set('pooled left-out 750 (weak labels, each fold by its own model)', 'cv folds pooled',
                            np.concatenate(Yp), pooled, rows, pairs, per_finding, SEED + 300, np.concatenate(Sp)))

    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows); metrics.to_csv(OUT/'fusion_ablation_metrics.csv', index=False)
    pairs = pd.DataFrame(pairs); pairs.to_csv(OUT/'fusion_paired_deltas.csv', index=False)
    per_finding = pd.DataFrame(per_finding); per_finding.to_csv(OUT/'fusion_per_finding_deltas.csv', index=False)
    change = pd.DataFrame(change); change.to_csv(OUT/'fusion_prediction_change.csv', index=False)

    # residual magnitudes (per window, gold74 and left-out fold sets)
    wrows, box = [], {}
    for ck in CKPTS:
        for f in sorted(SRC.glob(f'windows_{ck}_*.npz')):
            set_name = f.stem.split(f'{ck}_')[1]
            z = np.load(f)
            row = dict(checkpoint=ck, set=set_name, windows=len(z['s2_residual_ratio']))
            for key in z.files:
                v = z[key]
                row.update({f'{key}_median': float(np.median(v)), f'{key}_p10': float(np.percentile(v, 10)),
                            f'{key}_p90': float(np.percentile(v, 90))})
            wrows.append(row)
            if set_name == 'gold74':
                box[ck] = {k: z[k] for k in z.files}
    windows = pd.DataFrame(wrows); windows.to_csv(OUT/'fusion_residual_magnitudes.csv', index=False)

    weights = {}
    for f in sorted(SRC.glob('weights_rank*.json')):
        weights.update(json.loads(f.read_text()))
    wt = []
    for ck, w in weights.items():
        wt.append(dict(checkpoint=ck, rad_input=w['rad_input'],
                       coatnet_relative_drift=w['coatnet']['relative_drift'],
                       radimagenet_relative_drift=w['radimagenet']['relative_drift'],
                       head_relative_drift=w['head']['relative_drift'],
                       f1_output_weight_rms=w['f1']['linear3_output_zero_init']['weight_rms'],
                       f2_output_weight_rms=w['f2']['linear3_output_zero_init']['weight_rms'],
                       f1_linear1_rms=w['f1']['linear1']['weight_rms'], f2_linear1_rms=w['f2']['linear1']['weight_rms'],
                       batchnorm_buffers_max_abs_change=w['batchnorm_buffers_max_abs_change']))
    wt = pd.DataFrame(wt).set_index('checkpoint').loc[CKPTS].reset_index()
    wt.to_csv(OUT/'fusion_weight_stats.csv', index=False)
    plots(metrics, pairs, per_finding, box)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(wt.round(4).to_string(index=False))
    print(windows[['checkpoint', 'set', 'windows', 's2_residual_ratio_median', 's4_residual_ratio_median',
                   's2_cos_median', 's4_cos_median', 'feat_change_no_fusion_median',
                   'feat_change_rad_mean_median']].round(4).to_string(index=False))
    print(metrics.pivot_table(index=['set', 'checkpoint'], columns='mode', values='macro_auc').round(4).to_string())
    print(pairs.round(4).to_string(index=False))
    print(change.round(4).to_string(index=False))


def style(ax, title):
    ax.set_title(title, fontsize=10.5, color=INK, loc='left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def plots(metrics, pairs, per_finding, box):
    (OUT/'plots').mkdir(exist_ok=True)
    short = {c: c.replace('run20_saved_best_raw_ep2', '20ep best').replace('run20_ema_epoch20', '20ep EMA20')
             .replace('cv_fold', 'fold ') for c in CKPTS}

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, key, title in ((axes[0], 's2_residual_ratio', 'F1 (stage 2): ||residual|| / ||CoAtNet features||'),
                           (axes[1], 's4_residual_ratio', 'F2 (stage 4): ||residual|| / ||CoAtNet features||')):
        data = [box[c][key] for c in CKPTS if c in box]
        ax.boxplot(data, showfliers=False, widths=0.5, medianprops=dict(color=COLORS['full'], lw=2),
                   boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        ax.set_xticks(range(1, len(data) + 1), [short[c] for c in CKPTS if c in box], rotation=20)
        ax.grid(axis='y', color=GRID, lw=0.8)
        style(ax, title)
    fig.suptitle('Size of the fusion residual relative to the feature map it is added to (gold74 windows; '
                 'whiskers 5-95%)', x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT/'plots/residual_magnitudes.png', dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[8, 6]))
    for ax, sets, title in ((axes[0], ['gold74'], 'gold74 (gold labels): each checkpoint and the 5-fold ensemble'),
                            (axes[1], None, 'each fold model on its own 150 left-out studies (weak labels)')):
        m = metrics[metrics.set.isin(sets)] if sets else metrics[metrics.set.str.startswith('fold')]
        cks = list(dict.fromkeys(m.checkpoint))
        for i, ck in enumerate(cks):
            for j, mode in enumerate(MODES):
                v = m[(m.checkpoint == ck) & (m['mode'] == mode)].macro_auc.iloc[0]
                ax.plot(i + (j - 1.5)*0.12, v, 'o', ms=8, color=COLORS[mode], label=mode if i == 0 else None)
        ax.set_xticks(range(len(cks)), [short.get(c, c) for c in cks], rotation=20)
        ax.grid(axis='y', color=GRID, lw=0.8)
        style(ax, title)
        ax.set_ylabel('macro AUC', fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, ncol=4, loc='lower left')
    fig.suptitle('Ablations: parent / fusion switched off / RadImageNet replaced by its mean / full model',
                 x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT/'plots/fusion_ablation_auc.png', dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for ax, (set_prefix, ck, title) in zip(axes, (
            ('pooled', 'cv folds pooled', 'pooled left-out 750 studies (weak labels)'),
            ('gold74', '5-fold ensemble', 'gold74, 5-fold ensemble (gold labels)'))):
        d = per_finding[(per_finding.set.str.startswith(set_prefix)) & (per_finding.checkpoint == ck)
                        & (per_finding.comparison == 'full - no_fusion')].set_index('finding').loc[LABELS[::-1]]
        y = np.arange(len(d))
        ax.errorbar(d.delta_auc, y, xerr=[d.delta_auc - d.ci_low, d.ci_high - d.delta_auc], fmt='o',
                    color=COLORS['full'], ecolor='#b7d3f6', elinewidth=3, capsize=0, ms=7)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(y, d.index)
        ax.grid(axis='x', color=GRID, lw=0.8)
        style(ax, title)
        ax.set_xlabel('AUC(full) - AUC(fusion off), 95% paired bootstrap CI', fontsize=9, color=MUTED)
    fig.suptitle('Per-finding effect of the trained fusion residuals', x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT/'plots/per_finding_fusion_effect.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()

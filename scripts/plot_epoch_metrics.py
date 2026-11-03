"""Macro AUC, Brier, ECE and log loss per epoch (raw and EMA) for the 20-epoch fusion run, on gold74
and on the 433-study validation split (weak HYBRID binary labels). Uses saved predictions only.

Validation predictions exist only for epoch 0 (parent), raw epoch 2 (saved fusion_best.pt) and epoch 20
(raw and EMA): the run evaluated validation only at the end (validation_eval='final').

    python scripts/plot_epoch_metrics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_fusion_posttrain import (LABELS, RUN1, RES, JEV, ece, log_loss, per_finding_auc,  # noqa: E402
                                      read_labels, read_preds)

OUT = RUN1/'posttrain_analysis'
RAW, EMA, PARENT, INK, MUTED = '#2a78d6', '#eb6834', '#6b6b66', '#1f1f1d', '#8a8a83'


def macro_metrics(Y, P):
    """Macro over the 12 findings; each finding uses its available (non-missing) labels."""
    aucs, briers, eces, lls = [], [], [], []
    for j in range(Y.shape[1]):
        m = np.isfinite(Y[:, j]); y, p = Y[m, j], P[m, j]
        briers.append(float(np.mean((p - y)**2))); eces.append(ece(y, p)); lls.append(log_loss(y, p))
    aucs = per_finding_auc(Y, P)
    return dict(macro_auc=float(np.nanmean(aucs)), brier=float(np.mean(briers)), ece=float(np.mean(eces)),
                log_loss=float(np.mean(lls)))


def main():
    gold = read_labels(JEV/'gold74_eval_set.csv')
    g_uids = sorted(gold.index)
    Yg = gold.loc[g_uids, LABELS].to_numpy(float)
    import json
    va = sorted(json.loads((RUN1/'training_split.json').read_text())['validation'])
    Yv = read_labels(JEV/'HYBRID_labels_binary.csv').loc[va, LABELS].to_numpy(float)

    rows = []
    rows.append(dict(set='gold74', epoch=0, weights='parent', **macro_metrics(
        Yg, read_preds(RUN1/'predictions/step0_raw_gold74.csv', g_uids))))
    for e in range(1, 21):
        folder = RUN1 if e < 20 else RES
        for kind in ('raw', 'ema'):
            P = read_preds(folder/f'predictions/epoch{e}_{kind}_gold74.csv', g_uids)
            rows.append(dict(set='gold74', epoch=e, weights=kind, **macro_metrics(Yg, P)))
    for e, kind, name in ((0, 'parent', 'final_parent_step0'), (2, 'raw', 'final_epoch2_ema_best'),
                          (20, 'raw', 'epoch20_raw'), (20, 'ema', 'epoch20_ema')):
        P = read_preds(RES/f'predictions/{name}_validation.csv', va)
        rows.append(dict(set='validation (433, weak labels)', epoch=e, weights=kind, **macro_metrics(Yv, P)))
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'epoch_metrics_gold74_validation.csv', index=False)

    metrics = [('macro_auc', 'Macro AUC (higher is better)'), ('brier', 'Brier score (lower is better)'),
               ('ece', 'ECE, 10 bins (lower is better)'), ('log_loss', 'Log loss (lower is better)')]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.2))
    for r, setname in enumerate(('gold74', 'validation (433, weak labels)')):
        t = table[table.set == setname]
        parent = t[t.weights == 'parent'].iloc[0]
        for c, (col, title) in enumerate(metrics):
            ax = axes[r, c]
            ax.axhline(parent[col], color=PARENT, lw=1, ls='--', zorder=1)
            ax.text(20.4, parent[col], 'parent', color=MUTED, fontsize=8, va='center')
            for kind, color in (('raw', RAW), ('ema', EMA)):
                k = t[(t.weights == kind) | (t.weights == 'parent')].sort_values('epoch')
                style = dict(color=color, lw=2, marker='o', ms=4) if setname == 'gold74' else \
                    dict(color=color, lw=0, marker='o', ms=8)
                ax.plot(k.epoch, k[col], label=kind.upper() if kind == 'ema' else 'raw', zorder=3, **style)
            if setname == 'gold74':
                sel = t[(t.epoch == 2) & (t.weights == 'ema')].iloc[0]
                ax.plot([2], [sel[col]], marker='o', ms=12, mfc='none', mec=INK, mew=1.5, zorder=4)
            ax.set_xlim(-0.5, 21.8)
            ax.set_xticks([0, 2, 5, 10, 15, 20])
            ax.grid(axis='y', color='#e4e3dc', lw=0.8)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.set_title(title, fontsize=10.5, color=INK, loc='left')
            if c == 0:
                ax.set_ylabel(setname, fontsize=11, color=INK)
            if r == 1:
                ax.set_xlabel('epoch (0 = parent)', fontsize=9, color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=9, loc='lower left')
    fig.suptitle('20-epoch fusion run: metrics per epoch. Ring = selected checkpoint (epoch 2 EMA). '
                 'Validation predictions exist only at epochs 0, 2 (raw) and 20.', fontsize=11, color=INK, x=0.01,
                 ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT/'plots/epoch_metrics_gold74_validation.png', dpi=140)
    pd.set_option('display.width', 200)
    print(table.round(4).to_string(index=False))


if __name__ == '__main__':
    main()

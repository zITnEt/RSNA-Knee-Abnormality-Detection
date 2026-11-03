"""Per-finding AUC over the 20-epoch fusion run: gold74 at every epoch (raw and EMA) and the 433-study
validation split (weak HYBRID labels) where predictions exist (epoch 0 parent, raw epoch 2, epoch 20 raw/EMA).
Reads tables produced by scripts/analyze_fusion_posttrain.py.

    python scripts/plot_per_finding_epochs.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = ROOT/'results/fusion_ddp_run1/posttrain_analysis'
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
RAW, EMA, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1f1f1d', '#6b6b66', '#e4e3dc'

ep = pd.read_csv(A/'training_per_epoch_per_finding_auc.csv')
pf = pd.read_csv(A/'metrics_per_finding.csv')
val = pf[pf.set.str.startswith('validation')]
gold_counts = pf[(pf.set.str.startswith('gold74')) & (pf.model == 'parent_step0')].set_index('finding')
val_counts = val[val.model == 'parent_step0'].set_index('finding')
vpoints = {'parent_step0': (0, 'parent'), 'saved_fusion_best': (2, 'raw'), 'epoch20_raw': (20, 'raw'),
           'epoch20_ema': (20, 'ema')}

rows = []
for f in LABELS:
    for e in range(21):
        for kind in (('raw',) if e == 0 else ('raw', 'ema')):
            v = ep[(ep.epoch == e) & (ep.weights == kind) & (ep.finding == f)].gold74_auc.iloc[0]
            rows.append(dict(finding=f, set='gold74', epoch=e, weights='parent' if e == 0 else kind, auc=v))
    for model, (e, kind) in vpoints.items():
        v = val[(val.model == model) & (val.finding == f)].auroc.iloc[0]
        rows.append(dict(finding=f, set='validation433_weak', epoch=e, weights=kind, auc=v))
table = pd.DataFrame(rows)
table.to_csv(A/'per_finding_auc_by_epoch.csv', index=False)

fig, axes = plt.subplots(3, 4, figsize=(19, 12), sharex=True)
for ax, f in zip(axes.ravel(), LABELS):
    g = table[(table.finding == f) & (table.set == 'gold74')]
    parent = g[g.weights == 'parent'].auc.iloc[0]
    for kind, color, lw in (('raw', RAW, 1.3), ('ema', EMA, 2)):
        k = g[(g.weights == kind) | (g.weights == 'parent')].sort_values('epoch')
        ax.plot(k.epoch, k.auc, color=color, lw=lw, marker='o', ms=3, label=f'gold74 {kind.upper() if kind == "ema" else kind}')
    v = table[(table.finding == f) & (table.set == 'validation433_weak')]
    for _, r in v.iterrows():
        face = INK if r.weights in ('raw', 'parent') else 'white'
        ax.plot(r.epoch + (0.35 if r.weights == 'ema' else 0), r.auc, marker='D', ms=7, mfc=face, mec=INK, mew=1.4,
                ls='none', label='validation (raw / parent)' if r.weights == 'raw' and r.epoch == 2 else
                ('validation (EMA)' if r.weights == 'ema' else None))
    ax.axhline(parent, color=MUTED, lw=0.8, ls='--')
    gc, vc = gold_counts.loc[f], val_counts.loc[f]
    ax.set_title(f'{f}  (gold74 pos {int(gc.positives)}/{int(gc.n)}, val pos {int(vc.positives)}/{int(vc.n)})',
                 fontsize=10, color=INK, loc='left')
    ax.grid(axis='y', color=GRID, lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_xticks([0, 2, 5, 10, 15, 20])
for ax in axes[-1]:
    ax.set_xlabel('epoch (0 = parent)', fontsize=9, color=MUTED)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.005, 0.968), ncol=4, frameon=False, fontsize=9.5)
fig.suptitle('Per-finding AUC, 20-epoch fusion run. Lines: gold74 every epoch; diamonds: validation (433, weak labels) '
             'where predictions exist; dashed: parent on gold74.', x=0.01, ha='left', fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.935))
fig.savefig(A/'plots/per_finding_auc_by_epoch.png', dpi=130)

wide = table[table.set == 'gold74'].pivot_table(index='finding', columns=['weights', 'epoch'], values='auc')
summary = pd.DataFrame({
    'gold74 parent': wide[('parent', 0)],
    'gold74 EMA ep1': wide[('ema', 1)], 'gold74 EMA ep2': wide[('ema', 2)], 'gold74 EMA ep5': wide[('ema', 5)],
    'gold74 EMA ep10': wide[('ema', 10)], 'gold74 EMA ep20': wide[('ema', 20)],
    'gold74 best EMA epoch': table[(table.set == 'gold74') & (table.weights == 'ema')].loc[
        lambda d: d.groupby('finding').auc.idxmax()].set_index('finding').epoch,
    'val parent': val[val.model == 'parent_step0'].set_index('finding').auroc,
    'val raw ep2': val[val.model == 'saved_fusion_best'].set_index('finding').auroc,
    'val EMA ep20': val[val.model == 'epoch20_ema'].set_index('finding').auroc}).loc[LABELS]
summary.to_csv(A/'per_finding_auc_summary.csv')
pd.set_option('display.width', 250)
print(summary.round(3).to_string())

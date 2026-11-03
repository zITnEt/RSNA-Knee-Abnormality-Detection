"""Why some findings fall and others rise on gold74 over the 20-epoch fusion run.

For every finding, using saved per-study predictions (no new inference):
  1. gold74 AUC against gold labels AND against the weak HYBRID labels of the same 74 studies, every epoch
     (if the model is learning the weak-label definition, the weak-label AUC holds while the gold AUC falls);
  2. paired bootstrap CIs for EMA ep20 - EMA ep2 and EMA ep2 - parent (gold74), and the same on validation;
  3. how concentrated the ep2 -> ep20 change is: per-study rank contributions, how many studies make up 80%;
  4. whether the studies that move are ones where the gold and weak labels disagree;
  5. score separation (positives vs negatives, logit) and the training-target profile per finding.

    python scripts/explain_per_finding_trends.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RUN1 = ROOT/'results/fusion_ddp_run1'
RES = ROOT/'results/fusion_ddp_resume/output'
JEV = ROOT/'data/external/jev_knee/original'
OUT = RUN1/'posttrain_analysis/per_finding_trends'
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
B, SEED, EPS = 2000, 20261012, 1e-6


def labels(path):
    return pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')


def preds(path, uids):
    f = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    assert set(f.index) == set(uids), path
    return f.loc[uids, LABELS].to_numpy(float)


def auc(y, s):
    m = np.isfinite(y)
    y, s = y[m], s[m]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1*(n1 + 1)/2)/(n1*n0))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p/(1 - p))


def boot_delta(y, a, b, rng_seed):
    """Paired bootstrap of AUC(b) - AUC(a) over studies."""
    m = np.isfinite(y)
    y, a, b = y[m], a[m], b[m]
    rng = np.random.default_rng(rng_seed)
    out = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            out.append(auc(y[i], b[i]) - auc(y[i], a[i]))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())


def contributions(y, s):
    """Per-study share of the AUC: positives = fraction of negatives they outrank (ties 1/2), divided by n1;
    negatives = -(fraction of positives ranked at or below them) / n0... expressed so that the AUC equals the
    sum over positives and the 'loss' is attributable to individual negatives too."""
    m = np.isfinite(y)
    pos, neg = s[m & (y == 1)], s[m & (y == 0)]
    n1, n0 = len(pos), len(neg)
    cp = ((pos[:, None] > neg[None, :]) + 0.5*(pos[:, None] == neg[None, :])).mean(1)   # per positive
    cn = ((pos[:, None] > neg[None, :]) + 0.5*(pos[:, None] == neg[None, :])).mean(0)   # per negative
    return cp, cn, n1, n0


gold = labels(JEV/'gold74_eval_set.csv')
g = sorted(gold.index)
Yg = gold.loc[g, LABELS].to_numpy(float)
src = gold.loc[g, 'source'].to_numpy()
hb, hs = labels(JEV/'HYBRID_labels_binary.csv'), labels(JEV/'HYBRID_labels_scores.csv')
Wg = hb.loc[g, LABELS].to_numpy(float)          # weak binary labels of the gold74 studies
Sg = hs.loc[g, LABELS].to_numpy(float)          # weak soft scores of the gold74 studies
split = json.loads((RUN1/'training_split.json').read_text())
v = sorted(split['validation'])
Yv = hb.loc[v, LABELS].to_numpy(float)
Str = hs.loc[split['train'], LABELS].to_numpy(float)

P = {0: {'raw': preds(RUN1/'predictions/step0_raw_gold74.csv', g)}}
P[0]['ema'] = P[0]['raw']
for e in range(1, 20):
    P[e] = {k: preds(RUN1/f'predictions/epoch{e}_{k}_gold74.csv', g) for k in ('raw', 'ema')}
P[20] = {k: preds(RES/f'predictions/epoch20_{k}_gold74.csv', g) for k in ('raw', 'ema')}
V = {'parent': preds(RES/'predictions/final_parent_step0_validation.csv', v),
     'ep2': preds(RES/'predictions/final_epoch2_ema_best_validation.csv', v),
     'ep20_ema': preds(RES/'predictions/epoch20_ema_validation.csv', v)}

# 1. gold74 AUC vs gold labels and vs weak labels, every epoch (EMA; epoch 0 = parent)
curve = []
for e in range(21):
    for j, f in enumerate(LABELS):
        s = P[e]['ema'][:, j]
        curve.append(dict(epoch=e, finding=f, auc_gold_labels=auc(Yg[:, j], s), auc_weak_labels=auc(Wg[:, j], s),
                          auc_gold58=auc(np.where(src == 'original_gold58', Yg[:, j], np.nan), s),
                          auc_gold16=auc(np.where(src != 'original_gold58', Yg[:, j], np.nan), s)))
curve = pd.DataFrame(curve)
curve.to_csv(OUT/'gold74_auc_gold_vs_weak_labels_by_epoch.csv', index=False)

rows, movers = [], []
for j, f in enumerate(LABELS):
    y, w = Yg[:, j], Wg[:, j]
    a0, a2, a20 = P[0]['ema'][:, j], P[2]['ema'][:, j], P[20]['ema'][:, j]
    c = curve[curve.finding == f].set_index('epoch')
    d20 = boot_delta(y, a2, a20, SEED + j)
    d2 = boot_delta(y, a0, a2, SEED + 100 + j)
    dv = boot_delta(Yv[:, j], V['ep2'][:, j], V['ep20_ema'][:, j], SEED + 200 + j)

    # 3. concentration of the ep2 -> ep20 change across studies
    cp2, cn2, n1, n0 = contributions(y, a2)
    cp20, cn20, _, _ = contributions(y, a20)
    dpos = (cp20 - cp2)/n1            # change in AUC contributed by each positive (sums to the AUC change)
    total = dpos.sum()
    order = np.argsort(dpos if total < 0 else -dpos)
    cum = np.cumsum(dpos[order])
    k80 = int(np.searchsorted(cum/total, 0.8) + 1) if total != 0 else 0
    dneg = (cn20 - cn2)/n0            # per negative: the fraction of positives it is ranked below changes
    order_n = np.argsort(dneg if total < 0 else -dneg)
    cum_n = np.cumsum(dneg[order_n])
    k80n = int(np.searchsorted(cum_n/total, 0.8) + 1) if total != 0 else 0

    # 4. are the movers gold/weak disagreements?
    m = np.isfinite(y)
    yy, ww, ss, l2, l20 = y[m], w[m], Sg[m, j], logit(a2[m]), logit(a20[m])
    dl = l20 - l2
    pos_idx = np.flatnonzero(yy == 1); neg_idx = np.flatnonzero(yy == 0)
    disagree = yy != ww
    for name, idx, dc in (('positive', pos_idx, dpos), ('negative', neg_idx, dneg)):
        for rank, t in enumerate(np.argsort(dc if total < 0 else -dc)[:5]):
            movers.append(dict(finding=f, gold_label=name, rank_of_change=rank + 1, auc_change_contribution=float(dc[t]),
                               weak_label=int(ww[idx[t]]), weak_soft_score=float(ss[idx[t]]),
                               prob_ep2=float(a2[m][idx[t]]), prob_ep20=float(a20[m][idx[t]]),
                               source=src[m][idx[t]]))

    # 5. separation and training-target profile
    tr = Str[:, j]
    rows.append(dict(
        finding=f, gold_pos=int(np.nansum(y)), gold_n=int(m.sum()),
        gold_auc_parent=c.loc[0, 'auc_gold_labels'], gold_auc_ep2=c.loc[2, 'auc_gold_labels'],
        gold_auc_ep20=c.loc[20, 'auc_gold_labels'],
        delta_ep2_vs_parent=c.loc[2, 'auc_gold_labels'] - c.loc[0, 'auc_gold_labels'], ci_ep2_vs_parent=f'[{d2[0]:+.3f}, {d2[1]:+.3f}]',
        delta_ep20_vs_ep2=c.loc[20, 'auc_gold_labels'] - c.loc[2, 'auc_gold_labels'], ci_ep20_vs_ep2=f'[{d20[0]:+.3f}, {d20[1]:+.3f}]',
        weaklabel_auc_ep2=c.loc[2, 'auc_weak_labels'], weaklabel_auc_ep20=c.loc[20, 'auc_weak_labels'],
        gold58_delta_ep20_vs_ep2=c.loc[20, 'auc_gold58'] - c.loc[2, 'auc_gold58'],
        gold16_delta_ep20_vs_ep2=c.loc[20, 'auc_gold16'] - c.loc[2, 'auc_gold16'],
        val_auc_ep2=auc(Yv[:, j], V['ep2'][:, j]), val_auc_ep20=auc(Yv[:, j], V['ep20_ema'][:, j]),
        val_ci_ep20_vs_ep2=f'[{dv[0]:+.3f}, {dv[1]:+.3f}]',
        positives_making_80pct_of_change=k80, negatives_making_80pct_of_change=k80n,
        gold_weak_disagreements=int(disagree.sum()),
        gold_pos_weak_neg=int(((yy == 1) & (ww == 0)).sum()), gold_neg_weak_pos=int(((yy == 0) & (ww == 1)).sum()),
        mean_logit_change_gold_pos=float(dl[pos_idx].mean()), mean_logit_change_gold_neg=float(dl[neg_idx].mean()),
        mean_logit_change_disagreeing=float(dl[disagree].mean()) if disagree.any() else np.nan,
        separation_ep2=float(l2[pos_idx].mean() - l2[neg_idx].mean()),
        separation_ep20=float(l20[pos_idx].mean() - l20[neg_idx].mean()),
        train_mean_soft_target=float(np.nanmean(tr)),
        train_uncertain_0_2_to_0_8=float(((tr > .2) & (tr < .8)).mean()),
        train_missing=float(np.isnan(tr).mean())))

summary = pd.DataFrame(rows)
summary.to_csv(OUT/'per_finding_trend_explanation.csv', index=False)
pd.DataFrame(movers).to_csv(OUT/'top_movers_ep2_to_ep20.csv', index=False)
pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.float_format', '{:.3f}'.format)
print(summary.set_index('finding').T.to_string())
print()
mv = pd.DataFrame(movers)
focus = ['Fracture', 'Effusion', 'Medial Meniscus', 'Contusion', 'Lateral OA', 'Synovitis']
print(mv[mv.finding.isin(focus) & (mv.rank_of_change <= 4)].to_string(index=False))
print()
w = curve[curve.finding.isin(focus) & curve.epoch.isin([0, 1, 2, 5, 10, 15, 20])]
print(w.pivot(index='finding', columns='epoch', values=['auc_gold_labels', 'auc_weak_labels']).to_string())

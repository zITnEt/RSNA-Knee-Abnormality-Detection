"""Statistics for a fusion fine-tuning run: gold74 AUC per epoch (raw/EMA), per-finding tables,
bootstrap CIs and paired differences vs the step-0 parent, subset (official gold58 vs manual)
analysis, calibration, prediction drift, and training dynamics from training_events.jsonl.

    python scripts/analyze_fusion_run.py results/fusion_ddp_run1 data/external/jev_knee/original/gold74_eval_set.csv
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
B = int(__import__("os").environ.get("BOOT", 2000))
RNG = np.random.default_rng(20261010)


def auc(y, s):
    """Rank-based ROC AUC (ties averaged); nan if one class."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1*(n1+1)/2)/(n1*n0)


def per_finding(y, s, rows=None):
    out = {}
    for j, name in enumerate(LABELS):
        yy, ss = y[:, j], s[:, j]
        if rows is not None:
            yy, ss = yy[rows], ss[rows]
        m = np.isfinite(yy)
        out[name] = auc(yy[m].astype(int), ss[m])
    return out


def macro(d):
    v = [x for x in d.values() if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def main(run_dir, gold_csv):
    run = Path(run_dir)
    out = run/'analysis'
    out.mkdir(exist_ok=True)
    gold = pd.read_csv(gold_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    uids = sorted(gold.index)
    y = gold.loc[uids, LABELS].to_numpy(float)
    source = gold.loc[uids, 'source'].to_numpy()

    # ----------------------------------------------------------------- predictions
    preds = {}
    for f in sorted((run/'predictions').glob('*_gold74*.csv')):
        m = re.match(r'(step0|epoch\d+)_(raw|ema)_(gold74(?:_allwindows)?)$', f.stem)
        if not m:
            continue
        tag, kind, split = m.groups()
        p = pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[uids, LABELS]
        preds[(tag, kind, split)] = p.to_numpy(float)
    base = preds[('step0', 'raw', 'gold74')]

    def epoch_of(tag):
        return 0 if tag == 'step0' else int(tag[5:])

    rows = []
    for (tag, kind, split), s in preds.items():
        pf = per_finding(y, s)
        sub58 = per_finding(y, s, source == 'original_gold58')
        rho = [spearmanr(s[:, j], base[:, j]).correlation for j in range(12)]
        brier = np.nanmean((s - y)**2, axis=0)
        eps = 1e-6
        ll = -np.nanmean(y*np.log(np.clip(s, eps, 1)) + (1-y)*np.log(np.clip(1-s, eps, 1)), axis=0)
        logit = lambda p: np.log(np.clip(p, eps, 1-eps)/np.clip(1-p, eps, 1-eps))
        rows.append(dict(epoch=epoch_of(tag), weights=kind, split=split, macro_auc=macro(pf),
                         macro_auc_gold58=macro(sub58), **{f'auc_{k}': v for k, v in pf.items()},
                         mean_spearman_vs_step0=float(np.mean(rho)),
                         mean_abs_logit_change_vs_step0=float(np.mean(np.abs(logit(s)-logit(base)))),
                         macro_brier=float(np.mean(brier)), macro_logloss=float(np.mean(ll)),
                         mean_pred=float(np.mean(s))))
    table = pd.DataFrame(rows).sort_values(['split', 'epoch', 'weights']).reset_index(drop=True)
    table.to_csv(out/'per_epoch_gold74.csv', index=False)

    # ----------------------------------------------------------------- bootstrap (study resampling)
    n = len(uids)
    idx = RNG.integers(0, n, size=(B, n))
    gold_only = table[table.split == 'gold74']
    key = [('step0', 'raw')]
    best_row = gold_only.loc[gold_only.macro_auc.idxmax()]
    best_tag = 'step0' if best_row.epoch == 0 else f'epoch{int(best_row.epoch)}'
    last = int(gold_only.epoch.max())
    for e in sorted(set([1, 2, 3, 5, 10, last] + [int(best_row.epoch)]) - {0}):
        for kind in ('raw', 'ema'):
            if (f'epoch{e}', kind, 'gold74') in preds:
                key.append((f'epoch{e}', kind))

    def boot_macro(s):
        vals = np.empty(B)
        for b in range(B):
            ii = idx[b]
            vals[b] = macro(per_finding(y[ii], s[ii]))
        return vals

    base_boot = boot_macro(base)
    boot_rows = []
    for tag, kind in key:
        s = preds[(tag, kind, 'gold74')]
        bs = base_boot if (tag, kind) == ('step0', 'raw') else boot_macro(s)
        d = bs - base_boot
        point = macro(per_finding(y, s))
        boot_rows.append(dict(model=f'{tag}_{kind}', epoch=epoch_of(tag), weights=kind, macro_auc=point,
                              ci_low=float(np.nanpercentile(bs, 2.5)), ci_high=float(np.nanpercentile(bs, 97.5)),
                              diff_vs_step0=point - macro(per_finding(y, base)),
                              diff_ci_low=float(np.nanpercentile(d, 2.5)), diff_ci_high=float(np.nanpercentile(d, 97.5)),
                              p_not_better=float(np.mean(d <= 0)) if (tag, kind) != ('step0', 'raw') else np.nan))
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(out/'bootstrap_gold74.csv', index=False)

    # per-finding paired bootstrap: best model vs step0
    best_s = preds[(best_tag, best_row.weights, 'gold74')]
    last_ema = preds[(f'epoch{last}', 'ema', 'gold74')]
    pf_rows = []
    for j, name in enumerate(LABELS):
        m = np.isfinite(y[:, j])
        yy = y[m, j].astype(int)
        b0, b1, bl = base[m, j], best_s[m, j], last_ema[m, j]
        nn = m.sum()
        ii_all = RNG.integers(0, nn, size=(B, nn))
        diffs, diffs_l = [], []
        for ii in ii_all:
            a0 = auc(yy[ii], b0[ii])
            diffs.append(auc(yy[ii], b1[ii]) - a0)
            diffs_l.append(auc(yy[ii], bl[ii]) - a0)
        diffs, diffs_l = np.array(diffs), np.array(diffs_l)
        pf_rows.append(dict(finding=name, n=int(nn), positives=int(yy.sum()), prevalence=float(yy.mean()),
                            auc_step0=auc(yy, b0), auc_best=auc(yy, b1), auc_last_ema=auc(yy, bl),
                            best_minus_step0=auc(yy, b1)-auc(yy, b0),
                            best_diff_ci_low=float(np.nanpercentile(diffs, 2.5)),
                            best_diff_ci_high=float(np.nanpercentile(diffs, 97.5)),
                            last_minus_step0=auc(yy, bl)-auc(yy, b0),
                            last_diff_ci_low=float(np.nanpercentile(diffs_l, 2.5)),
                            last_diff_ci_high=float(np.nanpercentile(diffs_l, 97.5)),
                            mean_pred_step0=float(b0.mean()), mean_pred_best=float(b1.mean()),
                            mean_pred_last_ema=float(bl.mean())))
    pf = pd.DataFrame(pf_rows)
    pf.to_csv(out/'per_finding_best_vs_step0.csv', index=False)

    # exploratory ensembles (selected on gold74 -> optimistic)
    ens = {}
    ema = {e: preds[(f'epoch{e}', 'ema', 'gold74')] for e in range(1, last+1) if (f'epoch{e}', 'ema', 'gold74') in preds}
    def rank_avg(arrs):
        return np.mean([np.apply_along_axis(rankdata, 0, a) for a in arrs], axis=0)
    ens['step0+best (rank mean)'] = macro(per_finding(y, rank_avg([base, best_s])))
    ens['EMA epochs 1-3 (rank mean)'] = macro(per_finding(y, rank_avg([ema[e] for e in (1, 2, 3) if e in ema])))
    ens['EMA epochs 1-5 (rank mean)'] = macro(per_finding(y, rank_avg([ema[e] for e in range(1, 6) if e in ema])))
    ens['step0 + EMA epochs 1-3 (rank mean)'] = macro(per_finding(y, rank_avg([base] + [ema[e] for e in (1, 2, 3) if e in ema])))
    ens[f'all EMA epochs 1-{last} (rank mean)'] = macro(per_finding(y, rank_avg(list(ema.values()))))

    # ----------------------------------------------------------------- training dynamics
    ev = [json.loads(l) for l in (run/'training_events.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    prog = pd.DataFrame([e for e in ev if e['event'] == 'training_progress'])
    prog['lr_backbone'] = prog.lr.str[0]
    prog['lr_fusion'] = prog.lr.str[2]
    prog['lr_head'] = prog.lr.str[3]
    prog.drop(columns=['lr']).to_csv(out/'training_progress.csv', index=False)
    epochs = pd.DataFrame([e for e in ev if e['event'] == 'epoch_complete'])
    starts = {e['epoch']: pd.Timestamp(e['utc']) for e in ev if e['event'] == 'epoch_started'}
    evals = pd.DataFrame([e for e in ev if e['event'] == 'evaluation'])
    epochs['epoch_wall_min'] = [(pd.Timestamp(r.utc) - starts[r.epoch]).total_seconds()/60 for r in epochs.itertuples()]
    eval_secs = evals.assign(ep=evals.tag.str.extract(r'epoch(\d+)')[0].astype(float)).groupby('ep').seconds.sum()
    epochs['eval_min'] = epochs.epoch.map(lambda e: eval_secs.get(float(e), np.nan)/60)
    epochs['train_min'] = epochs.epoch_wall_min - epochs.eval_min
    epochs['raw_gold74'] = epochs.macro_auc.map(lambda d: d['raw']['gold74'])
    epochs['ema_gold74'] = epochs.macro_auc.map(lambda d: d['ema']['gold74'])
    epochs[['epoch', 'optimizer_steps', 'training_loss', 'raw_gold74', 'ema_gold74', 'epoch_wall_min', 'eval_min',
            'train_min', 'hours']].to_csv(out/'epochs.csv', index=False)
    model_ready = next(e for e in ev if e['event'] == 'model_ready')
    paused = [e for e in ev if e['event'] == 'paused_time_budget']
    amp_skips = sum(e['event'] == 'amp_update_skipped' for e in ev)
    best_updates = [e for e in ev if e['event'] == 'best_updated']
    nonfinite = int(prog.nonfinite_losses.max()) if 'nonfinite_losses' in prog else 0

    steady = prog[prog.optimizer_steps > 20]
    summary = dict(
        run=str(run), studies_gold=n, positives={k: int(np.nansum(y[:, j])) for j, k in enumerate(LABELS)},
        sources={k: int(v) for k, v in pd.Series(source).value_counts().items()},
        step0=dict(macro_auc=macro(per_finding(y, base)),
                   allwindows=macro(per_finding(y, preds[('step0', 'raw', 'gold74_allwindows')]))
                   if ('step0', 'raw', 'gold74_allwindows') in preds else None),
        best=dict(epoch=int(best_row.epoch), weights=best_row.weights, macro_auc=float(best_row.macro_auc)),
        last_epoch=last, best_updates=best_updates, paused=paused[0] if paused else None,
        amp_skipped_updates=amp_skips, nonfinite_losses_max_in_epoch=nonfinite,
        total_optimizer_steps=int(prog.optimizer_steps.max()),
        model_ready={k: model_ready[k] for k in ('trainable_parameters', 'frozen_batchnorm_layers', 'probe_peak_gib',
                                                  'probe_seconds', 'updates_per_epoch', 'effective_batch',
                                                  'microbatches_per_rank')},
        throughput=dict(sec_per_microbatch_median=float(steady.sec_per_microbatch.median()),
                        sec_per_microbatch_p95=float(steady.sec_per_microbatch.quantile(.95)),
                        data_wait_fraction_max=float(steady.data_wait_fraction.max()),
                        peak_gpu_gib_max=float(prog.peak_gpu_gib.max()),
                        epoch_wall_min_median=float(epochs.epoch_wall_min.median()),
                        train_min_median=float(epochs.train_min.median()),
                        eval_min_median=float(epochs.eval_min.median())),
        grad_norm=dict(median=float(steady.gradient_norm.median()), p95=float(steady.gradient_norm.quantile(.95)),
                       max=float(prog.gradient_norm.max()), clip=3.0,
                       fraction_above_clip=float((steady.gradient_norm > 3.0).mean())),
        loss_scale=dict(min=float(prog.scale.min()), max=float(prog.scale.max()), last=float(prog.scale.iloc[-1])),
        training_loss=dict(epoch1=float(epochs.training_loss.iloc[0]), last=float(epochs.training_loss.iloc[-1])),
        ensembles_exploratory=ens,
        raw_vs_ema=dict(ema_better_epochs=int((epochs.ema_gold74 > epochs.raw_gold74).sum()), epochs=len(epochs)),
        correlation_trainloss_gold=float(np.corrcoef(epochs.training_loss, epochs.ema_gold74)[0, 1]),
    )
    (out/'summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')

    # ----------------------------------------------------------------- console report
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print('=== per-epoch gold74 macro AUC ===')
    g = table[table.split == 'gold74'].pivot(index='epoch', columns='weights', values='macro_auc')
    g58 = table[table.split == 'gold74'].pivot(index='epoch', columns='weights', values='macro_auc_gold58')
    rho = table[table.split == 'gold74'].pivot(index='epoch', columns='weights', values='mean_spearman_vs_step0')
    cal = table[table.split == 'gold74'].pivot(index='epoch', columns='weights', values='macro_brier')
    print(pd.concat({'auc': g, 'auc_gold58': g58, 'rho_vs_step0': rho, 'brier': cal}, axis=1).round(4).to_string())
    print('\n=== bootstrap (B=%d, study resampling) ===' % B)
    print(boot.round(4).to_string(index=False))
    print('\n=== per finding: best (%s %s) and last EMA vs step0 ===' % (best_tag, best_row.weights))
    print(pf.round(3).to_string(index=False))
    print('\n=== epochs ===')
    print(epochs[['epoch', 'optimizer_steps', 'training_loss', 'raw_gold74', 'ema_gold74', 'epoch_wall_min',
                  'eval_min', 'train_min']].round(4).to_string(index=False))
    print('\n=== exploratory ensembles (selected on gold74, optimistic) ===')
    for k, v in ens.items():
        print(f'{k}: {v:.4f}')
    print('\n=== summary ===')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('best_updates',)}, indent=1, default=float))


if __name__ == '__main__':
    main(*sys.argv[1:3])

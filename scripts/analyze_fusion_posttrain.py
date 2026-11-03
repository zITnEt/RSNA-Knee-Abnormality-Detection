"""Post-training analysis of the completed 20-epoch Raptor V5 + RadImageNet fusion run
(Kaggle kernels rsna-fusion-ddp-train [epochs 1-19, epoch 20 to batch 1896] and
rsna-fusion-ddp-resume [rest of epoch 20 + final validation]).

Reads saved predictions, logs and label files only: no training, no inference, no Kaggle calls,
no checkpoint deserialisation, no MRI data. Outputs go to results/fusion_ddp_run1/posttrain_analysis.

    python scripts/analyze_fusion_posttrain.py              # B = 2000 bootstrap replicates
    $env:BOOT=200; python scripts/analyze_fusion_posttrain.py   # quick check
"""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN1 = ROOT/'results/fusion_ddp_run1'
RES = ROOT/'results/fusion_ddp_resume/output'
OUT = RUN1/'posttrain_analysis'
JEV = ROOT/'data/external/jev_knee/original'
OFFICIAL = ROOT/'audit/training/labels/train.csv'
SERIES = ROOT/'audit/preprocessing_metadata/train_series.csv'
CORPUS = ROOT/'results/six_slot_preprocessing/corpus_inventory.csv'
PRIOR = ROOT/'audit/training'
SUBMIT_LOG = ROOT/'results/fusion_submit/run1/rsna-fusion-submit.log'
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
SLOTS = ['SAG_FLUID_FS', 'COR_FLUID_FS', 'AX_FLUID_FS', 'SAG_FLUID_NOFS', 'COR_T1', 'SAG_T1']
B = int(os.environ.get('BOOT', 2000))
SEED = 20261010
EPS = 1e-6
INPUTS = {}
# Parameter counts measured by constructing the model code (timm 1.0.22, random init) in the analysis
# container; the total equals the trainable_parameters logged by the run's model_ready event.
PARAMETERS = {'coatnet_rmlp_2_rw_384 (Raptor V5 trunk)': 72856264, 'radimagenet_resnet50 trunk': 23508032,
              'fusion F1 (stage2: conv3_x 512 + CoAtNet 256 -> 256)': 2100992,
              'fusion F2 (stage4: CoAtNet 1024 + conv5_x 2048 -> 1024)': 12594176,
              'attention-MIL pooling (att)': 265484, 'feature LayerNorm (norm)': 2048,
              'linear classifier (1024 -> 12)': 12300}


def use(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = path
    return path


def ref(uid):
    """Non-identifying case reference (study UIDs are not exported)."""
    return 'case_' + hashlib.sha256(uid.encode()).hexdigest()[:8]


def read_preds(path, uids):
    frame = pd.read_csv(use(path), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    if frame.index.duplicated().any() or set(frame.index) != set(uids):
        raise ValueError(f'{path.name}: prediction IDs do not match the evaluation set')
    values = frame.loc[uids, LABELS].to_numpy(float)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError(f'{path.name}: non-finite or out-of-range probabilities')
    return values


# ----------------------------------------------------------------------------- metrics
def auc(y, s):
    y = np.asarray(y)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1*(n1 + 1)/2)/(n1*n0))


def per_finding_auc(Y, P):
    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        m = np.isfinite(Y[:, j])
        out[j] = auc(Y[m, j], P[m, j])
    return out


def evaluable(Y):
    """Findings with both classes present among available labels (others cannot have an AUC)."""
    return np.array([0 < np.nansum(Y[:, j]) < np.isfinite(Y[:, j]).sum() for j in range(Y.shape[1])])


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p/(1 - p))


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y*np.log(p) + (1 - y)*np.log(1 - p)))


def ece(y, p, bins=10):
    """Expected calibration error: 10 equal-width probability bins, weighted by bin count."""
    which = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            total += m.mean()*abs(y[m].mean() - p[m].mean())
    return float(total)


def calibration_fit(y, p):
    """Calibration slope/intercept: logistic regression of the label on logit(p) (unpenalised)."""
    if y.sum() < 2 or (1 - y).sum() < 2:
        return np.nan, np.nan
    fit = LogisticRegression(C=1e8, max_iter=5000).fit(logit(p)[:, None], y.astype(int))
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def bootstrap(Y, preds, n, seed):
    """Study-level bootstrap: same resampled indices for every model (paired). Returns [B,12] AUCs."""
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    out = {k: np.empty((B, Y.shape[1])) for k in preds}
    for b, ii in enumerate(idx):
        Yb = Y[ii]
        for k, P in preds.items():
            out[k][b] = per_finding_auc(Yb, P[ii])
    return out


def ci(values):
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def macro_replicates(boot_k, ev):
    """Macro AUC per replicate over the full-sample evaluable findings; replicates where any of
    those findings has a single class are dropped (counted), never re-averaged over fewer findings."""
    sub = boot_k[:, ev]
    valid = np.isfinite(sub).all(1)
    vals = np.full(len(sub), np.nan)
    vals[valid] = sub[valid].mean(1)
    return vals, int((~valid).sum())


def summarize(set_name, Y, preds, boot, note):
    ev = evaluable(Y)
    summary, findings = [], []
    complete = np.isfinite(Y).all(1)
    for k, P in preds.items():
        pf = per_finding_auc(Y, P)
        reps, dropped = macro_replicates(boot[k], ev)
        lo, hi = ci(reps)
        m_all = np.isfinite(Y)
        yy, pp = Y[m_all], P[m_all]
        aps, lls, brs, eces = [], [], [], []
        for j, name in enumerate(LABELS):
            m = np.isfinite(Y[:, j]); y = Y[m, j]; p = P[m, j]
            ap = float(average_precision_score(y, p)) if ev[j] else np.nan
            ll, br = log_loss(y, p), float(np.mean((p - y)**2))
            e = ece(y, p); slope, intercept = calibration_fit(y, p)
            flo, fhi = ci(boot[k][:, j])
            aps.append(ap); lls.append(ll); brs.append(br); eces.append(e)
            findings.append(dict(set=set_name, model=k, finding=name, n=int(m.sum()), positives=int(y.sum()),
                                 missing=int((~m).sum()), prevalence=float(y.mean()), auroc=pf[j],
                                 auroc_ci_low=flo, auroc_ci_high=fhi, average_precision=ap,
                                 ap_baseline_prevalence=float(y.mean()), log_loss=ll, brier=br,
                                 ece_10bin=e, calibration_slope=slope, calibration_intercept=intercept,
                                 mean_prediction=float(p.mean())))
        summary.append(dict(set=set_name, model=k, studies=len(Y), complete_case_studies=int(complete.sum()),
                            evaluable_findings=int(ev.sum()), macro_auc=float(np.nanmean(pf[ev])),
                            macro_auc_ci_low=lo, macro_auc_ci_high=hi, bootstrap_replicates=B,
                            bootstrap_dropped_single_class=dropped,
                            macro_auc_complete_case=float(np.nanmean(per_finding_auc(Y[complete], P[complete])[ev]))
                            if complete.sum() > 1 else np.nan,
                            micro_auc_pooled_cells=auc(yy, pp), macro_average_precision=float(np.nanmean(aps)),
                            macro_log_loss=float(np.mean(lls)), macro_brier=float(np.mean(brs)),
                            macro_ece_10bin=float(np.mean(eces)), note=note))
    return summary, findings


def paired(set_name, Y, preds, boot, pairs):
    ev = evaluable(Y)
    rows = []
    for a, b in pairs:
        if a not in preds or b not in preds:
            continue
        ra, _ = macro_replicates(boot[a], ev)
        rb, _ = macro_replicates(boot[b], ev)
        d = ra - rb
        pa, pb = per_finding_auc(Y, preds[a]), per_finding_auc(Y, preds[b])
        lo, hi = ci(d)
        rows.append(dict(set=set_name, model=a, reference=b, finding='MACRO',
                         difference=float(np.nanmean(pa[ev]) - np.nanmean(pb[ev])), ci_low=lo, ci_high=hi,
                         p_boot_not_better=float(np.mean(d[np.isfinite(d)] <= 0))))
        for j, name in enumerate(LABELS):
            dj = boot[a][:, j] - boot[b][:, j]
            lo, hi = ci(dj)
            rows.append(dict(set=set_name, model=a, reference=b, finding=name, difference=pa[j] - pb[j],
                             ci_low=lo, ci_high=hi, p_boot_not_better=float(np.mean(dj[np.isfinite(dj)] <= 0))))
    return rows


def confusion(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    div = lambda a, b: a/b if b else np.nan
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    ppv, npv = div(tp, tp + fp), div(tn, tn + fn)
    f1 = div(2*tp, 2*tp + fp + fn)
    return dict(threshold=float(t), tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=sens, specificity=spec, ppv=ppv,
                npv=npv, f1=f1, balanced_accuracy=(sens + spec)/2 if np.isfinite(sens) and np.isfinite(spec)
                else np.nan, predicted_positive_rate=float(pred.mean()))


def threshold_for_sensitivity(y, p, target):
    """Highest threshold whose sensitivity (p >= t) is at least `target` on these cases."""
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return np.nan
    return float(pos[int(np.ceil(target*len(pos))) - 1])


def read_labels(path):
    frame = pd.read_csv(use(path), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    if frame.index.duplicated().any():
        raise ValueError(f'Duplicate IDs in {path}')
    return frame


def save(frame, name):
    frame = pd.DataFrame(frame)
    frame.to_csv(OUT/name, index=False)
    return frame


# ----------------------------------------------------------------------------- 1. inputs and verification
def load_everything():
    checks = {}
    split = json.loads(use(RUN1/'training_split.json').read_text())
    split_resume = json.loads(use(RES/'training_split.json').read_text())
    tr, va, gl = split['train'], split['validation'], split['heldout_gold']
    checks['split'] = dict(
        train=len(tr), validation=len(va), heldout_gold=len(gl),
        all_ids_unique=len(set(tr) | set(va) | set(gl)) == len(tr) + len(va) + len(gl),
        train_validation_overlap=len(set(tr) & set(va)), train_gold_overlap=len(set(tr) & set(gl)),
        validation_gold_overlap=len(set(va) & set(gl)), resume_session_split_identical=split == split_resume)
    gold = read_labels(JEV/'gold74_eval_set.csv')
    g_uids = sorted(gold.index)
    Yg = gold.loc[g_uids, LABELS].to_numpy(float)
    source = gold.loc[g_uids, 'source'].to_numpy()
    official = read_labels(OFFICIAL)
    complete = official.dropna(subset=LABELS)
    g58 = source == 'original_gold58'
    ids58 = np.array(g_uids)[g58]
    checks['gold'] = dict(
        gold74_ids_equal_heldout_split=set(g_uids) == set(gl), official_train_studies=len(official),
        official_complete_label_studies=len(complete),
        gold58_equals_official_complete=set(ids58) == set(complete.index),
        gold58_labels_equal_official=bool(np.array_equal(Yg[g58], complete.loc[ids58, LABELS].to_numpy(float))),
        sources={k: int(v) for k, v in pd.Series(source).value_counts().items()},
        missing_label_cells=int(np.isnan(Yg).sum()))
    hyb_b, hyb_s = read_labels(JEV/'HYBRID_labels_binary.csv'), read_labels(JEV/'HYBRID_labels_scores.csv')
    v_uids = sorted(va)
    Yv = hyb_b.loc[v_uids, LABELS].to_numpy(float)
    Sv = hyb_s.loc[v_uids, LABELS].to_numpy(float)

    P = {'parent_step0': read_preds(RUN1/'predictions/step0_raw_gold74.csv', g_uids)}
    for e in range(1, 20):
        for kind in ('raw', 'ema'):
            P[f'epoch{e}_{kind}'] = read_preds(RUN1/f'predictions/epoch{e}_{kind}_gold74.csv', g_uids)
    for kind in ('raw', 'ema'):
        P[f'epoch20_{kind}'] = read_preds(RES/f'predictions/epoch20_{kind}_gold74.csv', g_uids)
    P['saved_fusion_best'] = read_preds(RES/'predictions/final_epoch2_ema_best_gold74.csv', g_uids)
    parent_rerun = read_preds(RES/'predictions/final_parent_step0_gold74.csv', g_uids)
    allwin = read_preds(RUN1/'predictions/step0_raw_gold74_allwindows.csv', g_uids)
    V = {'parent_step0': read_preds(RES/'predictions/final_parent_step0_validation.csv', v_uids),
         'saved_fusion_best': read_preds(RES/'predictions/final_epoch2_ema_best_validation.csv', v_uids),
         'epoch20_raw': read_preds(RES/'predictions/epoch20_raw_validation.csv', v_uids),
         'epoch20_ema': read_preds(RES/'predictions/epoch20_ema_validation.csv', v_uids)}
    mad = lambda a, b: float(np.abs(a - b).max())
    checks['predictions'] = dict(
        gold74_files=len(P) + 2, validation_files=len(V),
        parent_step0_session1_vs_session2_max_abs_diff=mad(P['parent_step0'], parent_rerun),
        saved_fusion_best_vs_epoch2_raw_max_abs_diff=mad(P['saved_fusion_best'], P['epoch2_raw']),
        saved_fusion_best_vs_epoch2_ema_max_abs_diff=mad(P['saved_fusion_best'], P['epoch2_ema']),
        parent_pure_vs_all_windows_max_abs_diff=mad(P['parent_step0'], allwin),
        parent_pure_vs_all_windows_macro_auc=[float(np.nanmean(per_finding_auc(Yg, P['parent_step0']))),
                                              float(np.nanmean(per_finding_auc(Yg, allwin)))])
    return dict(checks=checks, split=split, g_uids=g_uids, Yg=Yg, source=source, v_uids=v_uids, Yv=Yv, Sv=Sv,
                hyb_b=hyb_b, hyb_s=hyb_s, P=P, V=V)


# ----------------------------------------------------------------------------- 2. data audit
def data_audit(d):
    corpus = pd.read_csv(use(CORPUS), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    slots = np.array([json.loads(s) for s in corpus.slot_present], dtype=bool)
    corpus[SLOTS] = slots
    corpus['n_slots'] = slots.sum(1)
    series = pd.read_csv(use(SERIES), dtype=str)
    corpus['n_series'] = series.groupby('StudyInstanceUID').SeriesInstanceUID.nunique().reindex(corpus.index)
    split, src = d['split'], d['source']
    g = np.array(d['g_uids'])
    groups = {'train (gradient updates)': split['train'], 'validation (433, weak labels)': split['validation'],
              'gold74 (selection set)': list(g), 'gold58 official subset': list(g[src == 'original_gold58']),
              'gold16 report-verified subset': list(g[src != 'original_gold58'])}
    rows = []
    for name, ids in groups.items():
        c = corpus.loc[ids]
        rows.append(dict(split=name, studies=len(ids), series_total=int(c.n_series.sum()),
                         series_per_study_mean=float(c.n_series.mean()), triplet_windows_total=int(c.windows.sum()),
                         windows_mean=float(c.windows.mean()), windows_min=int(c.windows.min()),
                         windows_max=int(c.windows.max()), studies_with_cross_series_windows=int((c.cross_series_windows > 0).sum()),
                         all_six_slots=int((c.n_slots == 6).sum()),
                         **{f'slot_{s}_present': int(c[s].sum()) for s in SLOTS}))
    save(rows, 'data_split_counts.csv')
    d['corpus'] = corpus
    audit = dict(corpus_studies=len(corpus), duplicate_study_ids=int(corpus.index.duplicated().sum()),
                 duplicate_tensor_sha256=int(corpus.sha256.duplicated().sum()),
                 failed_preprocessing='0 (corpus_summary failure_count)', patient_ids='not available in any local file')

    lab = []
    tr = split['train']
    soft_tr = d['hyb_s'].loc[tr, LABELS]
    soft_va = d['hyb_s'].loc[d['v_uids'], LABELS]
    for j, f in enumerate(LABELS):
        s = soft_tr[f].to_numpy()
        lab.append(dict(set='train soft targets (HYBRID scores)', finding=f, n=int(np.isfinite(s).sum()),
                        missing=int(np.isnan(s).sum()), mean_soft_target=float(np.nanmean(s)),
                        soft_lt_0_1=int((s < .1).sum()), soft_0_1_to_0_4=int(((s >= .1) & (s < .4)).sum()),
                        uncertain_0_4_to_0_6=int(((s >= .4) & (s <= .6)).sum()),
                        soft_0_6_to_0_9=int(((s > .6) & (s <= .9)).sum()), soft_gt_0_9=int((s > .9).sum())))
        yv = d['Yv'][:, j]
        lab.append(dict(set='validation weak binary (HYBRID)', finding=f, n=len(yv), positives=int(yv.sum()),
                        negatives=int((yv == 0).sum()), missing=0, prevalence=float(yv.mean()),
                        mean_soft_target=float(soft_va[f].mean()),
                        uncertain_0_4_to_0_6=int(((soft_va[f] >= .4) & (soft_va[f] <= .6)).sum())))
        for name, mask in (('gold74', np.ones(len(src), bool)), ('gold58 official', src == 'original_gold58'),
                           ('gold16 report-verified', src != 'original_gold58')):
            y = d['Yg'][mask, j]
            lab.append(dict(set=name, finding=f, n=int(np.isfinite(y).sum()), positives=int(np.nansum(y)),
                            negatives=int((y == 0).sum()), missing=int(np.isnan(y).sum()),
                            prevalence=float(np.nanmean(y))))
    save(lab, 'label_counts.csv')

    # gold versus weak (HYBRID) labels on the gold studies; HYBRID was itself tuned against gold labels
    dis = []
    weak = d['hyb_b'].loc[d['g_uids'], LABELS].to_numpy(float)
    for name, mask in (('gold58 official', src == 'original_gold58'), ('gold16 report-verified', src != 'original_gold58')):
        for j, f in enumerate(LABELS):
            y, w = d['Yg'][mask, j], weak[mask, j]
            m = np.isfinite(y)
            dis.append(dict(set=name, finding=f, compared=int(m.sum()), agree=int((y[m] == w[m]).sum()),
                            gold_pos_weak_neg=int(((y[m] == 1) & (w[m] == 0)).sum()),
                            gold_neg_weak_pos=int(((y[m] == 0) & (w[m] == 1)).sum())))
    save(dis, 'gold_vs_weak_labels.csv')
    return audit


# ----------------------------------------------------------------------------- 3. training behaviour
def training(d):
    sessions = {}
    for name, folder in (('session1_train', RUN1), ('session2_resume', RES)):
        sessions[name] = [json.loads(line) for line in use(folder/'training_events.jsonl').read_text(
            encoding='utf-8').splitlines() if line.strip()]
    rows, progress = [], []
    for sess, ev in sessions.items():
        started = {e['epoch']: pd.Timestamp(e['utc']) for e in ev if e['event'] == 'epoch_started'}
        evals = [e for e in ev if e['event'] == 'evaluation']
        for e in ev:
            if e['event'] == 'training_progress':
                progress.append(dict(session=sess, utc=e['utc'], epoch=e['epoch'], batch=e['batch'],
                                     optimizer_step=e['optimizer_steps'], running_loss=e['loss'],
                                     gradient_norm_preclip=e['gradient_norm'], lr_coatnet=e['lr'][0],
                                     lr_radimagenet=e['lr'][1], lr_fusion=e['lr'][2], lr_head=e['lr'][3],
                                     loss_scale=e['scale'], nonfinite_losses=e['nonfinite_losses'],
                                     sec_per_microbatch=e['sec_per_microbatch'],
                                     data_wait_fraction=e['data_wait_fraction'], peak_gpu_gib=e['peak_gpu_gib']))
            if e['event'] != 'epoch_complete':
                continue
            ep = e['epoch']
            eval_s = sum(x['seconds'] for x in evals if x['tag'] in (f'epoch{ep}_raw', f'epoch{ep}_ema'))
            wall = (pd.Timestamp(e['utc']) - started[ep]).total_seconds()/60 if ep in started else np.nan
            auc_ = e['macro_auc']
            rows.append(dict(epoch=ep, session=sess, optimizer_steps=e['optimizer_steps'],
                             training_loss_soft_bce=e['training_loss'],
                             gold74_macro_auc_raw_logged=auc_['raw']['gold74'], gold74_macro_auc_ema_logged=auc_['ema']['gold74'],
                             validation_macro_auc_raw_logged=auc_['raw'].get('validation'),
                             validation_macro_auc_ema_logged=auc_['ema'].get('validation'),
                             wall_minutes=wall, eval_minutes=eval_s/60, train_minutes=wall - eval_s/60,
                             note='resumed: session 1 stopped at microbatch 1896/1950 (11 h guard); wall time '
                                  'covers session 2 only' if ep == 20 else ''))
    epochs = pd.DataFrame(rows).sort_values('epoch').reset_index(drop=True)
    Yg, src = d['Yg'], d['source']
    m58 = src == 'original_gold58'
    per_ep, recomputed = [], []
    for ep in range(0, 21):
        for kind in (('raw',) if ep == 0 else ('raw', 'ema')):
            key = 'parent_step0' if ep == 0 else f'epoch{ep}_{kind}'
            P = d['P'][key]
            pf = per_finding_auc(Yg, P)
            pf58 = per_finding_auc(Yg[m58], P[m58])
            pf16 = per_finding_auc(Yg[~m58], P[~m58])
            ev16 = evaluable(Yg[~m58])
            recomputed.append(dict(epoch=ep, weights=kind, gold74_macro_auc=float(np.nanmean(pf)),
                                   gold58_macro_auc=float(np.nanmean(pf58)),
                                   gold16_macro_auc=float(np.nanmean(pf16[ev16])), gold16_evaluable_findings=int(ev16.sum())))
            for j, f in enumerate(LABELS):
                per_ep.append(dict(epoch=ep, weights=kind, finding=f, gold74_auc=pf[j], gold58_auc=pf58[j]))
    rec = pd.DataFrame(recomputed)
    save(rec, 'training_per_epoch_gold_auc.csv')
    save(per_ep, 'training_per_epoch_per_finding_auc.csv')
    for kind in ('raw', 'ema'):
        logged = epochs.set_index('epoch')[f'gold74_macro_auc_{kind}_logged']
        mine = rec[rec.weights == kind].set_index('epoch').gold74_macro_auc.reindex(logged.index)
        d['checks'].setdefault('logged_vs_recomputed_gold74', {})[kind] = float((logged - mine).abs().max())
    save(epochs, 'training_epochs.csv')
    prog = save(progress, 'training_progress.csv')

    allev = sessions['session1_train'] + sessions['session2_resume']
    steady = prog[prog.optimizer_step > 20]
    ready = next(e for e in sessions['session1_train'] if e['event'] == 'model_ready')
    diag = dict(
        planned_epochs=20, completed_epochs=int(epochs.epoch.max()),
        optimizer_steps_total=int(epochs.optimizer_steps.max()), updates_per_epoch=ready['updates_per_epoch'],
        event_counts={s: pd.Series([e['event'] for e in ev]).value_counts().to_dict() for s, ev in sessions.items()},
        amp_update_skipped_events=sum(e['event'] == 'amp_update_skipped' for e in allev),
        max_nonfinite_losses_in_epoch=int(prog.nonfinite_losses.max()),
        gradient_norm_preclip=dict(logged_points=len(steady), median=float(steady.gradient_norm_preclip.median()),
                                   p95=float(steady.gradient_norm_preclip.quantile(.95)),
                                   max=float(prog.gradient_norm_preclip.max()), clip_norm=3.0,
                                   fraction_of_logged_steps_clipped=float((steady.gradient_norm_preclip > 3).mean())),
        loss_scale=dict(min=float(prog.loss_scale.min()), max=float(prog.loss_scale.max())),
        sec_per_microbatch=dict(median=float(steady.sec_per_microbatch.median()),
                                p95=float(steady.sec_per_microbatch.quantile(.95))),
        data_wait_fraction_max=float(steady.data_wait_fraction.max()), peak_gpu_gib_max=float(prog.peak_gpu_gib.max()),
        model_ready={k: ready[k] for k in ('trainable_parameters', 'frozen_batchnorm_layers', 'probe_peak_gib',
                                           'world_size', 'microbatches_per_rank', 'accumulate_per_rank',
                                           'effective_batch', 'amp', 'checkpoint_parts', 'fused_attention')},
        epoch_minutes_median_session1=float(epochs[epochs.session == 'session1_train'].wall_minutes.median()),
        train_minutes_median_session1=float(epochs[epochs.session == 'session1_train'].train_minutes.median()),
        eval_minutes_median_session1=float(epochs[epochs.session == 'session1_train'].eval_minutes.median()),
        session_span_utc={s: [ev[0]['utc'], ev[-1]['utc']] for s, ev in sessions.items()},
        paused=[e for e in allev if e['event'] == 'paused_time_budget'],
        best_updates=[e for e in allev if e['event'] == 'best_updated'],
        training_complete=[e for e in allev if e['event'] == 'training_complete'],
        train_loss_vs_ema_gold74_pearson=float(np.corrcoef(epochs.training_loss_soft_bce,
                                                           epochs.gold74_macro_auc_ema_logged)[0, 1]))
    return epochs, rec, prog, diag


# ----------------------------------------------------------------------------- 4-6. performance, operating points, comparisons
KEY_GOLD = {'parent_step0': 'Raptor V5 parent = fusion at step 0 (zero-init fusion)',
            'epoch1_ema': 'epoch 1 EMA',
            'epoch2_ema': 'epoch 2 EMA = selected by protocol (best gold74); weights NOT retained',
            'saved_fusion_best': 'fusion_best.pt as saved = raw epoch 2 (checkpoint bug); submitted',
            'epoch20_raw': 'final raw weights (fusion_last.pt)', 'epoch20_ema': 'final EMA (fusion_ema_epoch20.pt)'}
KEY_VAL = ['parent_step0', 'saved_fusion_best', 'epoch20_raw', 'epoch20_ema']


def load_prior_baselines(ids58, Y58):
    """Earlier-project gold58 predictions (different families, labels, splits); AUC comparisons only."""
    prior = {}
    files = {'Global96': 'global96/coatnet_global96_gold58_reference.npz', 'D4': 'd4/d4_gold58_reference.npz',
             'Repair': 'repair/coatnet_repairv1_gold58_top3_predictions.npz',
             'Residual': 'residual/gold58_top3_predictions.npz'}
    for name, rel in files.items():
        z = np.load(use(PRIOR/rel), allow_pickle=False)
        ix = pd.Index(z['study_uids'].astype(str)).get_indexer(ids58)
        assert (ix >= 0).all() and np.array_equal(z['truth'][ix], Y58), name
        prior[f'prior_{name}'] = z['probability_mean'][ix].astype(float)
    z = np.load(use(PRIOR/'dino/oof.npz'), allow_pickle=False)
    assert list(z['targets']) == LABELS
    ix = pd.Index(z['ids'].astype(str)).get_indexer(ids58)
    assert (ix >= 0).all()
    prior['prior_DINO_saved_OOF'] = z['pred'][ix].astype(float)
    rad = pd.read_csv(use(PRIOR/'e13/rsna_rad_e11/v52_e11_oof.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    prior['prior_Rad_E13_OOF'] = rad.loc[ids58, LABELS].to_numpy(float)
    return prior


def agreement_rows(set_name, Y, preds, base='parent_step0'):
    rows = []
    for k, P in preds.items():
        if k == base or k.startswith('prior_'):
            continue
        for j, f in enumerate(LABELS):
            m = np.isfinite(Y[:, j]); y = Y[m, j]
            a, b = preds[base][m, j], P[m, j]
            ca, cb = (a >= .5) == (y == 1), (b >= .5) == (y == 1)
            rows.append(dict(set=set_name, model=k, reference=base, finding=f, n=int(m.sum()),
                             spearman=float(spearmanr(a, b).correlation), mean_abs_prob_change=float(np.abs(a - b).mean()),
                             disagreement_rate_at_0_5=float(((a >= .5) != (b >= .5)).mean()),
                             errors_reference_at_0_5=int((~ca).sum()), errors_model_at_0_5=int((~cb).sum()),
                             fixed_by_model=int((~ca & cb).sum()), introduced_by_model=int((ca & ~cb).sum())))
    return rows


def operating_points(d):
    rows = []
    Yg, Yv = d['Yg'], d['Yv']
    for model in ('saved_fusion_best', 'parent_step0'):
        Pg, Pv = d['P'][model], d['V'][model]
        for j, f in enumerate(LABELS):
            m = np.isfinite(Yg[:, j]); yg, pg = Yg[m, j], Pg[m, j]
            yv, pv = Yv[:, j], Pv[:, j]
            specs = [('validation (weak labels)', 'fixed 0.5', .5, yv, pv),
                     ('gold74', 'fixed 0.5', .5, yg, pg)]
            for target in (.90, .95):
                tv = threshold_for_sensitivity(yv, pv, target)
                tg = threshold_for_sensitivity(yg, pg, target)
                specs += [('validation (weak labels)', f'sens>={target:.2f} chosen on validation (same set, exploratory)', tv, yv, pv),
                          ('gold74', f'sens>={target:.2f} chosen on validation weak labels (independent cases)', tv, yg, pg),
                          ('gold74', f'sens>={target:.2f} chosen on gold74 (same set, exploratory)', tg, yg, pg)]
            for eval_set, rule, t, y, p in specs:
                rows.append(dict(model=model, finding=f, evaluated_on=eval_set, threshold_rule=rule, n=len(y),
                                 positives=int(y.sum()), **confusion(y, p, t)))
    return rows


def soft_agreement(d):
    rows = []
    S = d['Sv']
    for k, P in d['V'].items():
        for j, f in enumerate(LABELS):
            s, p = S[:, j], P[:, j]
            pc = np.clip(p, EPS, 1 - EPS)
            rows.append(dict(model=k, finding=f, n=len(s), soft_bce=float(-np.mean(s*np.log(pc) + (1 - s)*np.log(1 - pc))),
                             mean_abs_error_vs_soft=float(np.abs(p - s).mean()),
                             spearman_vs_soft=float(spearmanr(p, s).correlation),
                             mean_soft_target=float(s.mean()), mean_prediction=float(p.mean())))
    return rows


# ----------------------------------------------------------------------------- 7. errors and subgroups
def error_queues(d):
    corpus = d['corpus']
    gold_weak = d['hyb_b'].loc[d['g_uids'], LABELS].to_numpy(float)
    gold_soft = d['hyb_s'].loc[d['g_uids'], LABELS].to_numpy(float)
    errors, disagreements = [], []
    sets = (('gold74', d['g_uids'], d['Yg'], d['P'], 'gold annotation', gold_weak, gold_soft),
            ('validation', d['v_uids'], d['Yv'], d['V'], 'weak HYBRID binary', d['Yv'], d['Sv']))
    for set_name, uids, Y, preds, kind, weak, soft in sets:
        dep, par = preds['saved_fusion_best'], preds['parent_step0']
        for i, u in enumerate(uids):
            info = dict(set=set_name, case_ref=ref(u),
                        subset=(d['source'][i] if set_name == 'gold74' else 'validation'),
                        slots_present=int(corpus.loc[u, 'n_slots']), triplet_windows=int(corpus.loc[u, 'windows']))
            for j, f in enumerate(LABELS):
                y = Y[i, j]
                if np.isnan(y):
                    continue
                p, q = dep[i, j], par[i, j]
                common = dict(**info, finding=f, label=int(y), label_type=kind, p_saved_fusion_best=float(p),
                              p_parent=float(q),
                              p_epoch2_ema=float(preds['epoch2_ema'][i, j]) if 'epoch2_ema' in preds else np.nan,
                              weak_binary=weak[i, j], weak_soft=soft[i, j],
                              gold_weak_disagree=bool(set_name == 'gold74' and np.isfinite(weak[i, j]) and weak[i, j] != y))
                if (p >= .5) != (y == 1):
                    errors.append(dict(**common, error='false negative' if y == 1 else 'false positive',
                                       error_margin=float(abs(p - y)), confident_error=bool(abs(p - y) >= .8),
                                       parent_also_wrong=bool((q >= .5) != (y == 1))))
                if abs(p - q) >= .3:
                    disagreements.append(dict(**common, abs_prob_change=float(abs(p - q)),
                                              fusion_correct=bool((p >= .5) == (y == 1)),
                                              parent_correct=bool((q >= .5) == (y == 1))))
    err = pd.DataFrame(errors).sort_values(['set', 'error_margin'], ascending=[True, False])
    dis = pd.DataFrame(disagreements).sort_values(['set', 'abs_prob_change'], ascending=[True, False])
    save(err, 'error_review_queue.csv')
    save(dis, 'component_disagreement_queue.csv')
    return err, dis


def subgroups(d):
    rows = []
    corpus = d['corpus']
    for set_name, uids, Y, preds in (('gold74', d['g_uids'], d['Yg'], d['P']),
                                     ('validation (weak labels)', d['v_uids'], d['Yv'], d['V'])):
        c = corpus.loc[uids]
        groups = {'slots present <= 4': c.n_slots <= 4, 'slots present = 5': c.n_slots == 5,
                  'slots present = 6': c.n_slots == 6, 'windows < 62': c.windows < 62,
                  'windows = 62 (cap)': c.windows == 62, 'no cross-series windows': c.cross_series_windows == 0,
                  'has cross-series windows': c.cross_series_windows > 0}
        for s in SLOTS:
            groups[f'{s} present'] = c[s]
            groups[f'{s} absent'] = ~c[s]
        for gname, mask in groups.items():
            mask = mask.to_numpy(bool)
            if mask.sum() < 2:
                rows.append(dict(set=set_name, subgroup=gname, studies=int(mask.sum()), note='too few studies'))
                continue
            Ys = Y[mask]
            ev = evaluable(Ys)
            minpos = int(np.nanmin(np.nansum(Ys[:, ev], 0))) if ev.any() else 0
            for model in ('parent_step0', 'saved_fusion_best'):
                pf = per_finding_auc(Ys, preds[model][mask])
                rows.append(dict(set=set_name, subgroup=gname, studies=int(mask.sum()), model=model,
                                 evaluable_findings=int(ev.sum()), min_positives_in_evaluable=minpos,
                                 macro_auc_evaluable=float(np.nanmean(pf[ev])) if ev.any() else np.nan,
                                 sparse_exploratory=bool(mask.sum() < 50 or ev.sum() < 12 or minpos < 5)))
    return save(rows, 'subgroups.csv')


# ----------------------------------------------------------------------------- 8. inference practicality
def inference(d):
    timings = []
    for folder in (RUN1, RES):
        for line in (folder/'training_events.jsonl').read_text(encoding='utf-8').splitlines():
            e = json.loads(line) if line.strip() else {}
            if e.get('event') == 'evaluation':
                timings.append(dict(tag=e['tag'], split=e['split'], studies=e['studies'], seconds=e['seconds']))
    t = pd.DataFrame(timings)
    t['seconds_per_study_wall_2xT4'] = t.seconds/t.studies
    per_split = t.groupby('split').agg(evaluations=('tag', 'size'), studies=('studies', 'first'),
                                       median_seconds=('seconds', 'median'),
                                       median_seconds_per_study_wall_2xT4=('seconds_per_study_wall_2xT4', 'median'))
    sub = []
    raw = use(SUBMIT_LOG).read_text(encoding='utf-8', errors='replace')
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        entries = [json.loads(x.lstrip(',')) for x in raw.splitlines() if x.lstrip(',').startswith('{')]
    for entry in entries:
        for line in str(entry.get('data', '')).splitlines():
            if line.startswith('{"event"'):
                sub.append(json.loads(line))
    return dict(parameters=PARAMETERS, parameters_total=sum(PARAMETERS.values()),
                parameters_provenance='component counts from constructing the model code; total equals logged trainable_parameters',
                checkpoint_bytes_fusion_best=next((e['bytes'] for e in sub if e.get('event') == 'model_loaded'), None),
                evaluation_timing_by_split=per_split.reset_index().to_dict('records'),
                evaluation_timing_definition=('wall seconds per evaluation pass on 2x T4 (DDP, studies split across GPUs), '
                                              'fp16, full pure-window bag in chunks of 16 padded windows, preprocessed '
                                              'tensors read from disk; excludes DICOM preprocessing and model loading'),
                submission_events=sub)


# ----------------------------------------------------------------------------- plots
COLORS = {'parent_step0': '#6b7280', 'epoch2_ema': '#2563eb', 'saved_fusion_best': '#dc2626',
          'epoch20_ema': '#16a34a', 'epoch20_raw': '#a3a3a3', 'epoch1_ema': '#9333ea'}


def plot_training(epochs, rec, prog, val_summary):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    ax[0].plot(epochs.epoch, epochs.training_loss_soft_bce, 'o-', color='#111827')
    ax[0].set(title='Training loss (soft-target BCE, augmented, epoch mean)', xlabel='epoch', ylabel='loss')
    for kind, style in (('raw', 's--'), ('ema', 'o-')):
        r = rec[(rec.weights == kind) | (rec.epoch == 0)]
        ax[1].plot(r.epoch, r.gold74_macro_auc, style, label=f'gold74 {kind}', ms=4)
    r = rec[(rec.weights == 'ema') | (rec.epoch == 0)]
    ax[1].plot(r.epoch, r.gold58_macro_auc, ':', color='#7c3aed', label='gold58 subset (EMA)')
    parent = rec[rec.epoch == 0].gold74_macro_auc.iloc[0]
    ax[1].axhline(parent, color='#6b7280', lw=1, label='parent (step 0)')
    e2 = rec[(rec.epoch == 2) & (rec.weights == 'ema')].gold74_macro_auc.iloc[0]
    e2r = rec[(rec.epoch == 2) & (rec.weights == 'raw')].gold74_macro_auc.iloc[0]
    ax[1].plot([2], [e2], '*', ms=16, color='#2563eb', label='selected: epoch 2 EMA (not retained)')
    ax[1].plot([2], [e2r], 'X', ms=11, color='#dc2626', label='saved fusion_best.pt = raw epoch 2')
    ax[1].set(title='Gold74 macro AUC (selection set)', xlabel='epoch (0 = parent)', ylabel='macro AUC')
    ax[1].legend(fontsize=7)
    vs = val_summary.set_index('model').macro_auc
    xs = {'parent_step0': 0, 'saved_fusion_best': 2, 'epoch20_raw': 20, 'epoch20_ema': 20.3}
    for k, x in xs.items():
        ax[2].plot([x], [vs[k]], 'o', ms=8, color=COLORS[k], label=k)
    ax[2].set(title='Validation (433) macro AUC vs weak HYBRID labels', xlabel='epoch', ylabel='macro AUC')
    ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT/'plots/learning_curves.png', dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    for col in ('lr_coatnet', 'lr_fusion', 'lr_head'):
        ax[0].plot(prog.optimizer_step, prog[col], label=col.replace('lr_', ''))
    ax[0].set(yscale='log', title='Learning rate (RadImageNet = CoAtNet)', xlabel='optimizer step')
    ax[0].legend(fontsize=8)
    ax[1].plot(prog.optimizer_step, prog.gradient_norm_preclip, '.', ms=3)
    ax[1].axhline(3.0, color='r', lw=1, label='clip norm 3.0')
    ax[1].set(title='Gradient norm before clipping (logged steps)', xlabel='optimizer step'); ax[1].legend(fontsize=8)
    ax[2].plot(prog.optimizer_step, prog.sec_per_microbatch, '.', ms=3)
    ax[2].set(title='Seconds per microbatch (1 study, per rank)', xlabel='optimizer step')
    fig.tight_layout(); fig.savefig(OUT/'plots/optimisation.png', dpi=130); plt.close(fig)


def plot_heatmap(rec_pf):
    base = rec_pf[rec_pf.epoch == 0].set_index('finding').gold74_auc
    ema = rec_pf[rec_pf.weights == 'ema'].pivot(index='finding', columns='epoch', values='gold74_auc').loc[LABELS]
    delta = ema.sub(base.loc[LABELS], axis=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(delta.to_numpy(), cmap='RdBu', vmin=-.08, vmax=.08, aspect='auto')
    ax.set_yticks(range(12), LABELS); ax.set_xticks(range(delta.shape[1]), delta.columns)
    ax.set(title='Gold74 per-finding AUC, EMA minus parent', xlabel='epoch')
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(OUT/'plots/per_finding_auc_change_heatmap.png', dpi=130)
    plt.close(fig)


def plot_curves(Y, preds, models, name, title):
    for kind in ('roc', 'pr'):
        fig, axes = plt.subplots(3, 4, figsize=(16, 11))
        for j, (f, ax) in enumerate(zip(LABELS, axes.ravel())):
            m = np.isfinite(Y[:, j]); y = Y[m, j]
            for k in models:
                p = preds[k][m, j]
                if kind == 'roc':
                    fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, color=COLORS.get(k), lw=1.3, label=k)
                else:
                    pr, rc, _ = precision_recall_curve(y, p); ax.plot(rc, pr, color=COLORS.get(k), lw=1.3, label=k)
            if kind == 'roc':
                ax.plot([0, 1], [0, 1], ':', color='k', lw=.7)
            else:
                ax.axhline(y.mean(), ls=':', color='k', lw=.7)
            ax.set_title(f'{f} (n={len(y)}, pos={int(y.sum())})', fontsize=9)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f'{title} - {"ROC" if kind == "roc" else "precision-recall (dotted = prevalence)"}')
        fig.tight_layout(); fig.savefig(OUT/f'plots/{kind}_{name}.png', dpi=110); plt.close(fig)


def plot_reliability(Y, P, name, title):
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    edges = np.linspace(0, 1, 11)
    for j, (f, ax) in enumerate(zip(LABELS, axes.ravel())):
        m = np.isfinite(Y[:, j]); y, p = Y[m, j], P[m, j]
        which = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        xs = [p[which == b].mean() for b in range(10) if (which == b).any()]
        ys = [y[which == b].mean() for b in range(10) if (which == b).any()]
        ns = [(which == b).sum() for b in range(10) if (which == b).any()]
        ax.plot([0, 1], [0, 1], ':', color='k', lw=.7)
        ax.scatter(xs, ys, s=[8 + 2*n for n in ns], color='#dc2626')
        ax.plot(xs, ys, color='#dc2626', lw=.8)
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.set_title(f'{f} (ECE {ece(y, p):.3f})', fontsize=9)
    fig.suptitle(f'{title} - reliability (10 equal-width bins; marker size ~ count)')
    fig.tight_layout(); fig.savefig(OUT/f'plots/reliability_{name}.png', dpi=110); plt.close(fig)


def plot_distribution(Y, P, name, title):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for j, (f, ax) in enumerate(zip(LABELS, axes.ravel())):
        m = np.isfinite(Y[:, j]); y, p = Y[m, j], P[m, j]
        bins = np.linspace(0, 1, 21)
        ax.hist(p[y == 0], bins=bins, alpha=.6, label='label 0', color='#2563eb')
        ax.hist(p[y == 1], bins=bins, alpha=.6, label='label 1', color='#dc2626')
        ax.set_title(f, fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'{title} - predicted probability by label')
    fig.tight_layout(); fig.savefig(OUT/f'plots/prediction_distribution_{name}.png', dpi=110); plt.close(fig)


def plot_comparison(summary58):
    s = summary58.sort_values('macro_auc')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(s.macro_auc, range(len(s)), xerr=[s.macro_auc - s.macro_auc_ci_low, s.macro_auc_ci_high - s.macro_auc],
                fmt='o', color='#111827', ecolor='#9ca3af', capsize=3)
    ax.set_yticks(range(len(s)), s.model)
    ax.set(title='Gold58 (official, complete labels) macro AUC, 95% study bootstrap CI\n'
                 'all models are selection-exposed on Gold58; prior baselines differ in training data/splits',
           xlabel='macro AUC')
    fig.tight_layout(); fig.savefig(OUT/'plots/gold58_model_comparison.png', dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    (OUT/'plots').mkdir(parents=True, exist_ok=True)
    d = load_everything()
    audit = data_audit(d)
    epochs, rec, prog, diag = training(d)

    Yg, src = d['Yg'], d['source']
    m58 = src == 'original_gold58'
    ids58 = list(np.array(d['g_uids'])[m58])
    gold_models = {k: d['P'][k] for k in KEY_GOLD}
    prior = load_prior_baselines(ids58, Yg[m58].astype(int))
    sets = {
        'gold74 (selection set; per-finding available labels)': (Yg, gold_models,
            'Selection set: chose the checkpoint; 16 report-verified studies may have been in parent training'),
        'gold58 official (complete labels)': (Yg[m58], {**{k: v[m58] for k, v in gold_models.items()}, **prior},
            'Official complete-label subset; used repeatedly for selection in this project; parent excluded it'),
        'gold16 report-verified': (Yg[~m58], {k: v[~m58] for k, v in gold_models.items()},
            'Report-derived labels; tiny; parent training exclusion unverified'),
        'validation 433 (weak HYBRID binary)': (d['Yv'], {k: d['V'][k] for k in KEY_VAL},
            'Agreement with weak labels; excluded from fusion gradients, parent probably trained on these studies'),
    }
    summary, findings, pairs_out, agree = [], [], [], []
    pairs = [('saved_fusion_best', 'parent_step0'), ('epoch2_ema', 'parent_step0'), ('epoch1_ema', 'parent_step0'),
             ('epoch20_ema', 'parent_step0'), ('epoch20_raw', 'parent_step0'), ('epoch20_ema', 'saved_fusion_best'),
             ('epoch2_ema', 'saved_fusion_best')]
    pairs58 = pairs + [(m, p) for m in ('saved_fusion_best', 'parent_step0') for p in prior]
    for i, (name, (Y, preds, note)) in enumerate(sets.items()):
        boot = bootstrap(Y, preds, len(Y), SEED + i)
        s, f = summarize(name, Y, preds, boot, note)
        summary += s; findings += f
        pairs_out += paired(name, Y, preds, boot, pairs58 if name.startswith('gold58') else pairs)
        agree += agreement_rows(name, Y, preds)
        if name.startswith('gold58'):
            plot_comparison(pd.DataFrame(s))
    summary = save(summary, 'metrics_summary.csv')
    save(findings, 'metrics_per_finding.csv')
    save(pairs_out, 'paired_comparisons.csv')
    save(agree, 'component_agreement.csv')
    save(operating_points(d), 'operating_points.csv')
    save(soft_agreement(d), 'validation_soft_target_agreement.csv')
    err, dis = error_queues(d)
    sub = subgroups(d)
    inf = inference(d)

    plot_training(epochs, rec, prog, summary[summary.set.str.startswith('validation')])
    plot_heatmap(pd.read_csv(OUT/'training_per_epoch_per_finding_auc.csv'))
    curve_models = ['parent_step0', 'epoch2_ema', 'saved_fusion_best', 'epoch20_ema']
    plot_curves(Yg, gold_models, curve_models, 'gold74', 'Gold74')
    plot_curves(d['Yv'], d['V'], ['parent_step0', 'saved_fusion_best', 'epoch20_ema'], 'validation',
                'Validation 433 vs weak labels')
    plot_reliability(Yg, d['P']['saved_fusion_best'], 'gold74_saved_fusion_best', 'Gold74, saved fusion_best')
    plot_reliability(d['Yv'], d['V']['saved_fusion_best'], 'validation_saved_fusion_best',
                     'Validation 433 (weak labels), saved fusion_best')
    plot_distribution(Yg, d['P']['saved_fusion_best'], 'gold74_saved_fusion_best', 'Gold74, saved fusion_best')

    result = dict(bootstrap=dict(replicates=B, seed=SEED, unit='study (no patient IDs available)',
                                 method='percentile; same resampled indices for all models in a set (paired); '
                                        'replicates where any full-sample-evaluable finding has one class are dropped '
                                        'and counted, never re-averaged over fewer findings'),
                  checks=d['checks'], data_audit=audit, training=diag, inference=inf,
                  models=KEY_GOLD, error_counts=err.groupby(['set', 'error']).size().to_dict() if len(err) else {},
                  confident_error_counts=err[err.confident_error].groupby('set').size().to_dict() if len(err) else {},
                  disagreement_cases=len(dis))
    result['error_counts'] = {f'{a} | {b}': int(v) for (a, b), v in result['error_counts'].items()}
    (OUT/'analysis_summary.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')

    manifest = dict(command='python scripts/analyze_fusion_posttrain.py', bootstrap_replicates=B,
                    inputs={k: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                            for k, p in sorted(INPUTS.items())},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    outputs=sorted(str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file()
                                   and p.name != 'artifact_manifest.json'))
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
    print(summary[['set', 'model', 'macro_auc', 'macro_auc_ci_low', 'macro_auc_ci_high', 'macro_auc_complete_case',
                   'micro_auc_pooled_cells', 'macro_log_loss', 'macro_brier', 'macro_ece_10bin',
                   'bootstrap_dropped_single_class']].round(4).to_string(index=False))
    print(json.dumps(d['checks'], indent=1, default=str))


if __name__ == '__main__':
    main()

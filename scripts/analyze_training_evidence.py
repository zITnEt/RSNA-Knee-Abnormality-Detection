"""Offline, resumable statistical audit of the ResNet50 fusion campaign.

Full command on the VM (uses existing inputs; no downloads):
  python scripts/analyze_training_evidence.py --run run --labels labels/gold74_eval_set.csv     --data data --comparators analysis_inputs/comparators --metadata analysis_inputs/train_series.csv     --out results/resnet50_20261013 --infer

Report-only command after collecting the evidence and inference caches:
  python scripts/analyze_training_evidence.py --run audit/resnet50_run/run     --labels audit/resnet50_run/labels/gold74_eval_set.csv     --comparators audit/resnet50_run/comparators --metadata audit/resnet50_run/train_series.csv     --manifests audit/resnet50_run/expected_manifests --out results/resnet50_20261013

Only analysis is written. Existing epoch predictions are reused; tensor-identical
snapshots share caches. All intermediate raw AND embedded EMA states are covered."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def aucs(truth, p):
    """Per-label AUROC with missing-label masks and explicit single-class NaNs."""
    truth, p = np.asarray(truth), np.asarray(p)
    if truth.shape != p.shape or truth.ndim != 2:
        raise ValueError('Truth and predictions must be matching 2D arrays')
    result = []
    for j in range(truth.shape[1]):
        mask = np.isfinite(truth[:, j])
        y, score = truth[mask, j], p[mask, j]
        result.append(roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan)
    return np.asarray(result)


def legacy_main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / 'audit/training'
    OUT = ROOT / 'results/training_analysis'
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(SRC / 'labels/train.csv').set_index('StudyInstanceUID')
    targets = list(train.columns[1:])
    gold = train.dropna(subset=targets)[targets].astype(int)
    ids = gold.index
    y = gold.to_numpy()
    pred = {}
    sources = {}
    for name, file in {
        'Global96':'global96/coatnet_global96_gold58_reference.npz',
        'D4':'d4/d4_gold58_reference.npz',
        'Repair':'repair/coatnet_repairv1_gold58_top3_predictions.npz',
        'Residual':'residual/gold58_top3_predictions.npz',
    }.items():
        z = np.load(SRC/file, allow_pickle=False)
        ix = pd.Index(z['study_uids']).get_indexer(ids)
        assert (ix >= 0).all()
        assert np.array_equal(z['truth'][ix], y), name
        pred[name] = z['probability_mean'][ix]
        sources[name] = file
    z = np.load(SRC/'dino/oof.npz', allow_pickle=False)
    assert list(z['targets']) == targets
    ix = pd.Index(z['ids']).get_indexer(ids)
    assert (ix >= 0).all()
    pred['DINO_saved_OOF'] = z['pred'][ix]
    derived = z['y_derived'][ix]
    rad = pd.read_csv(SRC/'e13/rsna_rad_e11/v52_e11_oof.csv').set_index('StudyInstanceUID')
    pred['Rad_E13_OOF'] = rad.loc[ids, targets].to_numpy()
    for p in pred.values():
        assert p.shape == y.shape and np.isfinite(p).all()


    rows=[]
    for name,p in pred.items():
        for j,t in enumerate(targets):
            rows.append(dict(model=name,finding=t,auc=roc_auc_score(y[:,j],p[:,j]),
                average_precision=average_precision_score(y[:,j],p[:,j]),
                brier=brier_score_loss(y[:,j],p[:,j]),positives=int(y[:,j].sum()),
                negatives=int((1-y[:,j]).sum()),prediction_min=float(p[:,j].min()),
                prediction_max=float(p[:,j].max()),prediction_std=float(p[:,j].std())))
    metrics=pd.DataFrame(rows)
    metrics.to_csv(OUT/'per_finding_metrics.csv',index=False)
    prevalence=pd.DataFrame({'finding':targets,'gold_positive':y.sum(0),'gold_negative':(1-y).sum(0),
        'gold_prevalence':y.mean(0),'derived_vs_gold_disagreements':(derived!=y).sum(0)})
    prevalence.to_csv(OUT/'label_support.csv',index=False)
    label_audit=[]
    for j,t in enumerate(targets):
        label_audit.append(dict(finding=t,derived_positive_gold_negative=int(((derived[:,j]==1)&(y[:,j]==0)).sum()),
            derived_negative_gold_positive=int(((derived[:,j]==0)&(y[:,j]==1)).sum())))
    pd.DataFrame(label_audit).to_csv(OUT/'derived_label_disagreements.csv',index=False)
    # Anonymous row numbers support review without exporting report text or study identifiers.
    errors=[]
    for name,p in pred.items():
        rank=rankdata(p,axis=0)/len(y)
        for i in range(len(y)):
            for j,t in enumerate(targets):
                errors.append(dict(gold_row=i,model=name,finding=t,label=int(y[i,j]),prediction=p[i,j],
                    contradiction_rank=float(1-rank[i,j] if y[i,j] else rank[i,j])))
    pd.DataFrame(errors).sort_values('contradiction_rank',ascending=False).to_csv(OUT/'error_review.csv',index=False)

    # Paired study bootstrap: reflects sample uncertainty, not checkpoint-selection bias.
    rng=np.random.default_rng(20260928)
    boots={k:[] for k in pred}
    for _ in range(1000):
        idx=rng.integers(0,len(y),len(y))
        for name,p in pred.items(): boots[name].append(float(np.nanmean(aucs(y[idx],p[idx]))))
    summary=[]
    for name,p in pred.items():
        lo,hi=np.quantile(boots[name],[.025,.975])
        summary.append({'model':name,'macro_auc':float(aucs(y,p).mean()),'bootstrap_low':lo,'bootstrap_high':hi})
    pd.DataFrame(summary).to_csv(OUT/'model_summary.csv',index=False)
    differences=[]
    for i,a in enumerate(pred):
        for b in list(pred)[i+1:]:
            delta=np.array(boots[a])-np.array(boots[b])
            lo,hi=np.quantile(delta,[.025,.975])
            differences.append(dict(model_a=a,model_b=b,auc_difference=float(aucs(y,pred[a]).mean()-aucs(y,pred[b]).mean()),
                paired_bootstrap_low=lo,paired_bootstrap_high=hi))
    pd.DataFrame(differences).to_csv(OUT/'paired_auc_differences.csv',index=False)

    cor=[]
    names=list(pred)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            for j,t in enumerate(targets):cor.append({'model_a':a,'model_b':b,'finding':t,'spearman':spearmanr(pred[a][:,j],pred[b][:,j]).statistic})
    pd.DataFrame(cor).to_csv(OUT/'prediction_correlations.csv',index=False)
    ranks={k:rankdata(p,axis=0)/len(y) for k,p in pred.items()}
    abl=[]
    # Descriptive only: these predictions do not reproduce the full submitted ensemble.
    for subset_name,subset in [('four_CoAt',names[:4]),('six_available',names)]:
        full=np.mean([ranks[k] for k in subset],axis=0)
        base=aucs(y,full).mean()
        abl.append({'group':subset_name,'removed':'none','macro_auc':base,'delta_vs_full':0})
        for k in subset:
            score=aucs(y,np.mean([ranks[n] for n in subset if n!=k],axis=0)).mean()
            abl.append({'group':subset_name,'removed':k,'macro_auc':score,'delta_vs_full':score-base})
    pd.DataFrame(abl).to_csv(OUT/'exploratory_ablations.csv',index=False)

    epochs=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,file in [('Global96','global96/training_history_global96_e24.json'),('D4','d4/training_history_d4.json'),('Repair','repair/training_history_terminal_epoch18.json'),('Residual','residual/training_history_through_epoch10.json')]:
        data=json.loads((SRC/file).read_text())
        for r in data['history']:
            epochs.append({'model':name,**{k:r.get(k) for k in ['epoch','train_loss','train_bce','gold_auc','gold_bce']}})
        h=pd.DataFrame([r for r in epochs if r['model']==name])
        axes[0].plot(h.epoch,h.gold_auc,label=name)
        axes[1].plot(h.epoch,h.train_loss,label=name)
    axes[0].set(title='Gold58 validation AUC',xlabel='Epoch',ylabel='Macro AUC')
    axes[1].set(title='Training loss (objectives differ)',xlabel='Epoch',ylabel='Loss')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=160);plt.close(fig)
    pd.DataFrame(epochs).to_csv(OUT/'epoch_metrics.csv',index=False)
    pivot=metrics.pivot(index='finding',columns='model',values='auc').loc[targets,names]
    fig,ax=plt.subplots(figsize=(10,7));im=ax.imshow(pivot,vmin=.5,vmax=1,cmap='YlGnBu')
    ax.set_xticks(range(len(names)),names,rotation=20,ha='right');ax.set_yticks(range(12),targets)
    for i in range(12):
        for j in range(len(names)):ax.text(j,i,f'{pivot.iloc[i,j]:.3f}',ha='center',va='center',color='white' if pivot.iloc[i,j]>.85 else 'black')
    ax.set_title('Gold58 AUC — descriptive, selection-biased validation')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/'per_finding_auc.png',dpi=160);plt.close(fig)

    members=json.loads((SRC/'dino/manifest.json').read_text())['members']
    pd.DataFrame([{k:m.get(k) for k in ['id','fold','seed','epochs_done','holdout','annot']} for m in members]).to_csv(OUT/'dino_member_metadata.csv',index=False)
    rad.groupby('fold')[targets].count().iloc[:,0].rename('studies').to_csv(OUT/'rad_fold_sizes.csv')
    provenance=[]
    for p in SRC.rglob('*'):
        if p.is_file():provenance.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (OUT/'artifact_inventory.json').write_text(json.dumps(provenance,indent=2))
    report='''# Training evidence and next experiments

    ## Scope and reliability

    Official train.csv: 4,407 studies; only 58 have all 12 structured labels. No MRI images, weights, or new training were needed. Raw reports remain in the ignored audit folder and are not reproduced here.

    Global96, D4, Repair and Residual reference labels were aligned by study ID and verified exactly against official labels. DINO targets were verified; its saved OOF and Rad E13 OOF were aligned to the same 58 IDs. OOF training exclusion and cross-family split compatibility have NOT been independently established. Saved DINO OOF is not proven to equal the final 20-member inference ensemble; Rad OOF is not the complete calibrated Rad branch. Therefore these are component diagnostics, not validation of the submitted 0.943 ensemble.

    Gold58 was used for epoch/architecture selection. Results are selection-biased, with only 58 studies. Bootstrap intervals below quantify study-sampling variation only, not selection bias; resamples lacking a class omit that finding. Brier scores describe the stored outputs and should not be compared as calibrated probabilities when outputs are ranks. Do not tune weights on these results and claim an independent gain.

    ## Recomputed scores

    | Component | Macro AUC | Bootstrap 95% interval |
    |---|---:|---:|
    '''
    for r in summary:report+=f"| {r['model']} | {r['macro_auc']:.5f} | {r['bootstrap_low']:.3f}–{r['bootstrap_high']:.3f} |\n"
    report+='''
    ## Files

    - full_epoch_metrics.csv: all flattened history fields, including per-label AUC, learning rates, throughput, memory and attention diagnostics where supplied.
    - training_history_summary.csv and training_configurations.json: best/final epochs, losses, timing and reported configurations (generated by scripts/summarize_training_histories.py). Timings come from authors' training hardware, sometimes different GPUs across epochs; they do not estimate Kaggle hidden-test runtime.
    - per_finding_metrics.csv: AUC, average precision, Brier, class support and prediction spread.
    - label_support.csv: official positive/negative counts and disagreements with DINO derived labels.
    - derived_label_disagreements.csv: disagreement direction, not assumed annotation errors.
    - error_review.csv: anonymous study-row/finding review queue, sorted by contradictory prediction rank; not a clinical error adjudication.
    - paired_auc_differences.csv: paired bootstrap intervals for component differences, still subject to selection bias and split compatibility limitations.
    - prediction_correlations.csv: per-finding pairwise rank correlations.
    - exploratory_ablations.csv: equal-rank mixtures and leave-one-out comparisons; NOT the submitted ensemble and NOT unbiased performance estimates.
    - epoch_metrics.csv and learning_curves.png: learning trajectories for four CoAt branches.
    - dino_member_metadata.csv: folds, seeds, epochs and author-reported scores; score definitions need provenance review.
    - rad_fold_sizes.csv: saved OOF fold sizes.
    - artifact_inventory.json: source paths, sizes and hashes.

    ## Next experiments, in priority order

    Concrete signals:

    - Global96 peaks at epoch 16 (0.92740), then ends at 0.92425. Repair peaks at epoch 12 (0.90422), then ends at 0.88681. Residual peaks at epoch 4 (0.90046), then ends at 0.88362. Declining training loss alongside worse validation is consistent with overfitting, though Gold58 is noisy. Checkpoint-average scores in the score table differ from single-epoch peaks.
    - In saved DINO OOF, Lateral Meniscus AUC is 0.642 and Lateral OA 0.708; Synovitis is 0.742. These are useful review targets, not measurements of the submitted 20-member branch.
    - Derived versus official labels disagree on Contusion in 23/58 cases (20 derived-positive/official-negative), Effusion in 17/58 (16 derived-positive/official-negative), and Synovitis in 17/58 (14 derived-negative/official-positive). Audit extraction rules, definitions and missing-value handling; disagreement alone does not establish which label is wrong.
    - Global96 and D4 macro AUC differ by only 0.00033, much smaller than study-sampling uncertainty. MCL has just 9 positives and Lateral OA 11.
    - In a four-CoAt equal-rank mixture, removing Repair changes AUC by +0.00149 and removing Residual by +0.00110. These are exploratory pruning candidates, not evidence to remove branches from the differently weighted full submission.

    1. Audit split provenance and patient grouping; recover training code and verify that each evaluated study was excluded. Establish an independent validation strategy before optimization.
    2. Inspect Synovitis, Lateral OA, PF OA and Lateral Meniscus errors and label definitions. Check class support in label_support.csv before interpreting differences. Disagreements between derived and official labels deserve label review, not automatic replacement.
    3. Create a DINO-only baseline on a trustworthy split to match the intended research focus; the saved OOF is a starting artifact, not proof of final-ensemble behavior.
    4. Test a small number of fixed ensemble ablations on that split, measuring both AUC and runtime. A weak standalone branch may add complementary information.
    5. For Global96, compare early stopping/checkpoint averaging with additional training: epoch 16 beats epoch 24 despite declining train loss. Do not assume longer training helps.
    6. Profile inference independently. Remove the unused ConvNeXt search in a new version and verify prediction parity; training metrics cannot identify hidden-run runtime bottlenecks.

    ## Public sources

    - https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights
    - https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coatnet-global96-top3
    - https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2
    - https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3
    - https://www.kaggle.com/code/sofiaanjenje/rsna-knee-e13-train
    - https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data

    Downloads reflect currently available public artifacts; histories are not all independently pinned to the submitted version. D4 selected parent hash and Rad E13 head hash match the known baseline references. Required remaining evidence: A5/Raptor training histories, full pipeline validation predictions, patient-level split proof, and held-out evaluation untouched by model selection. A5 and the inspected Raptor weight dataset expose weights but no separate training-history artifacts.
    '''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print(prevalence.to_string(index=False))
    print(pd.DataFrame(abl).to_string(index=False))
    print('Saved',OUT)


# Extended campaign analysis reuses the original AUROC helper above.
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent))


LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
SEED = 20261014
BINS = 10
WIDE = .20


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save_json(value, path, compact=False):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean(value), indent=None if compact else 2,
        separators=(',',':') if compact else None, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def inventory(args):
    """Print and persist actual artifacts before inference or metric calculation."""
    roots = [args.run, args.labels, args.comparators, args.metadata, args.manifests,
             args.out/'inference_cache', args.out/'checkpoint_audit.json']
    rows = []
    for root in filter(None, roots):
        root = Path(root)
        if not root.exists():
            rows.append(dict(path=str(root), exists=False))
            continue
        files = [root] if root.is_file() else sorted(root.rglob('*'))
        for p in files:
            if p.is_file():
                r = dict(path=str(p), exists=True, bytes=p.stat().st_size)
                if p.suffix not in ('.pt', '.pth'):
                    r['sha256'] = digest(p)
                rows.append(r)
    print('ARTIFACT INVENTORY (before computation)', flush=True)
    for r in rows:
        print(json.dumps(r), flush=True)
    save_json(rows, args.out / 'artifact_inventory.json')
    return rows


def load_npz(path, ids, truth):
    with np.load(path, allow_pickle=False) as z:
        index = pd.Index(z['study_uid'].astype(str))
        if index.has_duplicates or set(index) != set(ids):
            raise ValueError(f'{path}: study set differs')
        ix = index.get_indexer(ids)
        if not np.array_equal(z['labels'][ix], truth, equal_nan=True):
            raise ValueError(f'{path}: labels differ from Gold74')
        p = expit(z['logits'][ix])
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError(f'{path}: invalid probability')
    return p


def bootstrap_counts(n, replicates, seed=SEED):
    ix = np.random.default_rng(seed).integers(0, n, (replicates, n))
    return np.asarray([np.bincount(i, minlength=n) for i in ix], dtype=np.float64)


def probability_bins(p):
    """Equal-width rational edges; preserve stored float values at boundaries."""
    edges=np.arange(BINS+1,dtype=np.float64)/BINS
    return np.clip(np.searchsorted(edges,np.asarray(p,dtype=np.float64),side='right')-1,0,BINS-1)


def weighted_metrics(y, p, counts):
    """Exact multiplicity-weighted metrics; identical to explicitly repeated rows.

    Sorting and tie grouping once per label permits >=2000 paired study resamples
    across every snapshot without repeatedly fitting sklearn metric objects.
    Missing labels contribute zero weight, never a fabricated negative.
    """
    mask = np.isfinite(y)
    y, p, w = y[mask], p[mask], counts[:, mask]
    n = w.sum(1)
    pos = w @ y
    neg = n - pos
    order = np.argsort(-p, kind='stable')
    score, yy, ww = p[order], y[order], w[:, order]
    if not len(y):
        return {k: np.full(len(counts), np.nan) for k in
                ['auc', 'ap', 'brier', 'ece', 'mean_probability', 'prevalence']}
    starts = np.r_[0, np.flatnonzero(np.diff(score) != 0) + 1]
    wp = np.add.reduceat(ww * yy, starts, axis=1)
    wn = np.add.reduceat(ww * (1 - yy), starts, axis=1)
    tp = np.cumsum(wp, axis=1)
    fp = np.cumsum(wn, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        auc = (wp * (neg[:, None] - fp + .5 * wn)).sum(1) / (pos * neg)
        ap = (np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0) * wp).sum(1) / pos
        brier = (w @ ((p - y)**2)) / n
        mean = (w @ p) / n
        prevalence = pos / n
        ece = np.zeros(len(w))
        bins = probability_bins(p)
        for b in range(BINS):
            m = bins == b
            ece += np.abs(w[:, m] @ (p[m] - y[m])) / n
    auc[(pos == 0) | (neg == 0)] = np.nan
    ap[pos == 0] = np.nan
    return dict(auc=auc, ap=ap, brier=brier, ece=ece,
                mean_probability=mean, prevalence=prevalence)


def interval(values, point):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    lo, hi = np.quantile(v, [.025, .975]) if len(v) else (np.nan, np.nan)
    return dict(value=point, ci_low=lo, ci_high=hi, ci_width=hi-lo,
                valid_resamples=len(v), invalid_resamples=len(values)-len(v))


def macro_samples(per_label, evaluable):
    """Fixed full-sample label set; reject resamples missing either class."""
    selected = per_label[:, evaluable]
    valid = np.isfinite(selected).all(1)
    result = np.full(len(selected), np.nan)
    if selected.shape[1]:
        result[valid] = selected[valid].mean(1)
    return result


def operating(y, p, threshold, counts):
    mask = np.isfinite(y)
    yy, pp, w = y[mask], p[mask], counts[:, mask]
    hit = pp >= threshold
    pos, neg = w @ yy, w @ (1 - yy)
    tp, tn = w @ (yy * hit), w @ ((1-yy) * ~hit)
    fp, fn = neg-tn, pos-tp
    with np.errstate(divide='ignore', invalid='ignore'):
        se, sp = tp / pos, tn / neg
        f1 = 2*tp / (2*tp + fp + fn)
    base = np.ones((1, len(y)))
    if len(counts) == 1 and np.array_equal(counts, base):
        return dict(sensitivity=se[0], specificity=sp[0], f1=f1[0],
                    tp=tp[0], tn=tn[0], fp=fp[0], fn=fn[0])
    return dict(sensitivity=se, specificity=sp, f1=f1)


def reliability(y, p):
    mask = np.isfinite(y)
    yy, pp = y[mask], p[mask]
    bins = probability_bins(pp)
    return [dict(bin=b, low=b/BINS, high=(b+1)/BINS, n=int((bins == b).sum()),
                 mean_probability=float(pp[bins == b].mean()) if (bins == b).any() else None,
                 observed_prevalence=float(yy[bins == b].mean()) if (bins == b).any() else None)
            for b in range(BINS)]


def best_f1_threshold_samples(y, p, counts):
    """Re-fit F1 on the fixed full-sample candidate grid in each study resample."""
    mask=np.isfinite(y)
    yy,pp,w=y[mask],p[mask],counts[:,mask]
    candidates=np.unique(np.r_[0,pp,np.nextafter(1.,2.)])
    hit=pp[:,None]>=candidates
    tp=w@(yy[:,None]*hit)
    fp=w@((1-yy[:,None])*hit)
    pos=(w@yy)[:,None]
    denominator=tp+fp+pos
    score=np.divide(2*tp,denominator,out=np.full_like(tp,np.nan),where=denominator>0)
    selected=candidates[np.nanargmax(np.nan_to_num(score,nan=-1),axis=1)]
    selected[(w@yy)==0]=np.nan
    return selected


def state_hash(state):
    h = hashlib.sha256()
    for k, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def verify_parent(run, out):
    """Establish that the zero-fusion initial state is the actual parent baseline."""
    import torch
    cfg=json.loads((run/'training_config.json').read_text())
    initial=torch.load(run/'checkpoints/initial.pt',map_location='cpu',weights_only=True,mmap=True)['model']
    parent=torch.load(cfg['raptor'],map_location='cpu',weights_only=True,mmap=True)['model']
    differences=[]
    for key,value in parent.items():
        mapped=key.replace('backbone.','backbone.coatnet.',1) if key.startswith('backbone.') else key
        if mapped not in initial or not torch.equal(value,initial[mapped]): differences.append(key)
    zero_outputs={key:bool(torch.count_nonzero(value)==0) for key,value in initial.items()
                  if key in ('backbone.f1.up.weight','backbone.f1.up.bias','backbone.f2.up.weight','backbone.f2.up.bias')}
    passed=not differences and len(zero_outputs)==4 and all(zero_outputs.values())
    evidence=dict(parent_file=cfg['raptor'],parent_file_sha256=digest(cfg['raptor']),
        parent_tensor_sha256=state_hash(parent),initial_parent_tensors_identical=not differences,
        differing_keys=differences,zero_fusion_output_tensors=zero_outputs,baseline_verified=passed)
    save_json(evidence,out/'parent_verification.json')
    if not passed: raise ValueError('Initial snapshot does not reproduce the actual parent with zero fusion')
    return evidence


def inspect_tensor_metadata(data, ids, out):
    """Inspect every Gold study's nested audit metadata without reading pixels."""
    keys={}; discovered=[]
    canonical={'laterality':'Laterality','imagelaterality':'Laterality',
        'magneticfieldstrength':'MagneticFieldStrength','fieldstrength':'MagneticFieldStrength',
        'manufacturer':'Manufacturer','manufacturermodelname':'Scanner','scanner':'Scanner'}
    files={p.stem:p for p in Path(data).rglob('*.npz') if p.stem in set(ids)}
    if set(files)!=set(ids): raise ValueError('Missing Gold metadata tensor files')
    def walk(value,found):
        if isinstance(value,dict):
            for key,v in value.items():
                keys[key]=keys.get(key,0)+1
                normalized=''.join(c for c in key.lower() if c.isalnum())
                if normalized in canonical and isinstance(v,(str,int,float)):
                    found.setdefault(canonical[normalized],set()).add(str(v))
                walk(v,found)
        elif isinstance(value,list):
            for v in value: walk(v,found)
    for uid in ids:
        found={}
        with np.load(files[uid],allow_pickle=False) as z:
            audit=json.loads(str(z['audit_json'].item())) if 'audit_json' in z else {}
            walk(audit,found)
            native={r['series_uid']:r['native_depth'] for r in audit.get('selection',[])
                    if r.get('present') and 'native_depth' in r}
            spacing=np.asarray([v for r in audit.get('source_files',[]) for v in r.get('spacing',[])],float)
            spacing=spacing[np.isfinite(spacing)&(spacing>0)]
            row=dict(StudyInstanceUID=uid,series_count=len(np.unique(z['slice_series_uids'])),
                     native_slice_count=sum(native.values()) if native else None,
                     median_native_pixel_spacing_mm=float(np.median(spacing)) if len(spacing) else None,
                     **{k:'|'.join(sorted(v)) for k,v in found.items()})
        discovered.append(row)
    frame=pd.DataFrame(discovered).set_index('StudyInstanceUID').reindex(ids)
    path=out/'study_metadata.csv'
    if path.exists():
        prior=pd.read_csv(path,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').reindex(ids)
        for column in frame: prior[column]=frame[column]
        frame=prior
    frame.to_csv(path)
    result=dict(studies_inspected=len(ids),nested_key_occurrences=keys,
                scanner_laterality_columns_found=[c for c in canonical.values() if c in frame],
                note='Inspected nested audit JSON in all 74 tensors. No pixel values read for this inventory.')
    save_json(result,out/'tensor_metadata_inventory.json')
    return result


def collect_checkpoints(args, ids, truth):
    """Read weights, verify best snapshots, deduplicate states, infer missing caches.

    Reuses the actual model loader, verified StudyDataset and bag_logits, never
    alters training/model files. Hooks only operate in this analysis process.
    """
    import gc
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.raptor_v5_resnet50 import load_fused_checkpoint
    from scripts.train_resnet50_cloud import discover, StudyDataset, bag_logits
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    verify_parent(args.run,args.out)
    cache = args.out / 'inference_cache'
    cache.mkdir(exist_ok=True)
    entries, unique, lr = [], {}, []
    paths = sorted((args.run / 'checkpoints').glob('*.pt'))
    if not paths:
        raise ValueError('No checkpoint files found for --infer')
    for path in paths:
        ck = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        states = [('raw' if 'ema' not in path.stem else 'ema', ck['model'])]
        if 'ema' in ck:
            states.append(('embedded_ema', ck['ema']['shadow']))
        for kind, state in states:
            sha = state_hash(state)
            tag = path.stem + ('_embedded_ema' if kind == 'embedded_ema' else '')
            r = dict(checkpoint=path.name, variant=kind, tag=tag, state_sha256=sha,
                     bytes=path.stat().st_size, epoch=ck.get('epoch', ck.get('epoch_index')),
                     global_step=ck.get('global_step'), next_study=ck.get('next_study'))
            entries.append(r)
            unique.setdefault(sha, dict(path=path, kind=kind, tag=tag))
        if 'scheduler' in ck and '_raw' in path.stem:
            lr.append(dict(checkpoint=path.name, epoch=ck['epoch_index'],
                           global_step=ck['global_step'], lr=ck['scheduler'].get('_last_lr'),
                           grad_norm=None))
        del ck
        gc.collect()
        print('AUDITED', path.name, flush=True)
    history = json.loads((args.run/'history.json').read_text())
    bests, verify = [], []
    by_tag = {r['tag']: r for r in entries}
    for kind in ('raw', 'ema'):
        row = max(history, key=lambda r: r[kind]['validation']['macro_auc'])
        expected = by_tag[f"epoch_{row['epoch']:02d}_{kind}"]
        actual = by_tag[f'best_validation_{kind}']
        passed = actual['state_sha256'] == expected['state_sha256']
        verify.append(dict(kind=kind, selected_epoch=row['epoch'],
                           selection_auc=row[kind]['validation']['macro_auc'],
                           tensor_identical=passed, selected_sha256=expected['state_sha256'],
                           best_sha256=actual['state_sha256']))
        if not passed:
            raise ValueError(f'Best {kind} weights do not match the validation-selected epoch!')
        bests.append(expected['state_sha256'])
    save_json(dict(checkpoints=entries, unique_states=len(unique), best_verification=verify,
                   epoch_learning_rates=lr), args.out/'checkpoint_audit.json')
    files, _, _ = discover(args.data)
    ds = StudyDataset(files, ids, dict(zip(ids, truth)))
    items = [ds[i] for i in range(len(ds))]
    metadata = []
    for uid in ids:
        with np.load(files[uid], allow_pickle=False) as z:
            metadata.append(dict(StudyInstanceUID=uid, windows=len(z['images']),
                source_slices=len(z['slice_series_uids']),
                populated_slots=len(np.unique(z['triplet_slot_ids'])),
                **{f'slot_{s}_present': bool(np.any(z['triplet_slot_ids'] == s)) for s in range(6)}))
    pd.DataFrame(metadata).to_csv(args.out/'study_metadata.csv', index=False)
    inspect_tensor_metadata(args.data,ids,args.out)
    final_hashes = [by_tag[f'epoch_10_{k}']['state_sha256'] for k in ('raw', 'ema')]
    diagnose = set(bests + final_hashes)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def infer(model, mode, output, stats=None):
        logits = []
        with torch.inference_mode():
            for i, item in enumerate(items):
                if stats is not None:
                    stats['uid'] = ids[i]
                chunk=12 if sha in diagnose else args.inference_chunk
                out = bag_logits(model, item['images'].cuda(), chunk, amp)[0]
                logits.append(out.cpu().numpy())
                if (i+1) % 20 == 0:
                    print('INFERENCE_PROGRESS',output.name,i+1,len(items),flush=True)
        tmp = output.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, study_uid=np.asarray(ids), labels=truth, logits=np.asarray(logits),
                            inference_chunk=chunk, inference_seed=SEED, deterministic=True)
        tmp.replace(output)
        print('INFERRED', output.name, mode, flush=True)

    def describe(t):
        a = t.detach().float().cpu().numpy().ravel()
        return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                    min=float(a.min()), p01=float(np.quantile(a,.01)),
                    p10=float(np.quantile(a,.1)), p50=float(np.median(a)),
                    p90=float(np.quantile(a,.9)), p99=float(np.quantile(a,.99)), max=float(a.max()))

    for sha, source in unique.items():
        full_file = cache/f'{sha}_full.npz'
        need_diag = sha in diagnose
        # Existing epoch predictions are an evaluation of precisely these saved
        # tensors. Re-infer bests to establish numerical parity independently.
        aliases = [r for r in entries if r['state_sha256'] == sha]
        saved = [args.run/'predictions'/f"{r['tag']}_gold74.npz" for r in aliases
                 if r['tag'].startswith('epoch_') and r['variant'] in ('raw','ema')]
        saved = [p for p in saved if p.exists()]
        if not full_file.exists() and saved and not need_diag:
            full_file.write_bytes(saved[0].read_bytes())
        diag_file = cache/f'{sha}_diagnostics.json'
        if full_file.exists() and (not need_diag or (diag_file.exists() and
                all((cache/f'{sha}_{m}.npz').exists() for m in ('no_fusion','rad_mean')))):
            continue
        ck = torch.load(source['path'], map_location='cpu', weights_only=True, mmap=True)
        if source['kind'] == 'embedded_ema':
            ck = {**{k:v for k,v in ck.items() if k not in ('model','ema')},
                  'model': ck['ema']['shadow']}
        model = load_fused_checkpoint(ck).cuda().eval()
        del ck; gc.collect()
        diagnostics = dict(uid=None, windows=[], gates={'f1': [], 'f2': []})
        handles = []
        if need_diag and not diag_file.exists():
            for name in ('f1','f2'):
                block = getattr(model.backbone, name)
                def capture(module, inputs, output, name=name):
                    first, second = inputs
                    stream = second if name == 'f1' else first
                    a = module.norm_first(first.permute(0,2,3,1))
                    b = module.norm_second(second.permute(0,2,3,1))
                    wa, wb = module.down.weight.split([module.first_channels, module.second_channels], dim=1)
                    ca = torch.nn.functional.linear(a, wa).float().flatten(1).norm(dim=1)
                    cb = torch.nn.functional.linear(b, wb).float().flatten(1).norm(dim=1)
                    rn = ca if name == 'f1' else cb
                    cn = cb if name == 'f1' else ca
                    # Pre-nonlinearity additive projection contributions; NOT a
                    # causal fraction of final predictions after GELU/gating.
                    fraction = rn / (rn + cn).clamp_min(1e-12)
                    ratio = output.float().flatten(1).norm(dim=1) / stream.float().flatten(1).norm(dim=1).clamp_min(1e-12)
                    for r, f in zip(ratio.cpu().tolist(), fraction.cpu().tolist()):
                        diagnostics['windows'].append(dict(study_uid=diagnostics['uid'], layer=name,
                            residual_norm_over_stream=r, resnet_projection_norm_fraction=f))
                handles.append(block.register_forward_hook(capture))
                if block.gate is not None:
                    def gate_hook(module, inputs, output, name=name):
                        diagnostics['gates'][name].append(output.detach().float().cpu())
                    handles.append(block.gate.register_forward_hook(gate_hook))
        infer(model, 'full', full_file, diagnostics if handles else None)
        if saved:
            reference = load_npz(saved[0], ids, truth)
            recomputed = load_npz(full_file, ids, truth)
            save_json(dict(saved_file=str(saved[0]), max_probability_difference=np.max(np.abs(reference-recomputed)),
                auc_saved=aucs(truth,reference), auc_recomputed=aucs(truth,recomputed)),
                cache/f'{sha}_parity.json')
        for h in handles:
            h.remove()
        if handles:
            diagnostics['gates'] = {k: describe(torch.cat(v)) for k,v in diagnostics['gates'].items()}
            diagnostics.pop('uid')
            save_json(diagnostics, diag_file)
        if need_diag:
            import types
            modules = [model.backbone.f1, model.backbone.f2]
            # With both residuals zero, the backbone is exactly the fine-tuned
            # CoAtNet path. Bypass unused ResNet/fusion computation only here.
            original_forward=model.backbone.forward
            model.backbone.forward=types.MethodType(lambda self,image:self.coatnet(image),model.backbone)
            output = cache/f'{sha}_no_fusion.npz'
            if not output.exists():
                infer(model, 'no_fusion', output)
            model.backbone.forward=original_forward
            # Per-checkpoint window-weighted Gold74 mean feature maps, no labels.
            # Mean replacements preserve learned fusion/head weights, removing
            # study-specific ResNet information. Exploratory, transductive control.
            feature_file = cache/f'{sha}_feature_means.npz'
            if feature_file.exists():
                with np.load(feature_file) as z:
                    means = [torch.from_numpy(z[k]).cuda() for k in ('conv3','conv5')]
            else:
                sums, count = None, 0
                with torch.inference_mode(), torch.autocast('cuda', dtype=amp):
                    for item in items:
                        for images in item['images'].split(12):
                            features = model.backbone.resnet50(images.cuda())
                            if sums is None:
                                sums = [f.float().sum(0,keepdim=True) for f in features]
                            else:
                                for s,f in zip(sums,features): s.add_(f.float().sum(0,keepdim=True))
                            count += len(images)
                means = [s/count for s in sums]
                np.savez_compressed(feature_file, conv3=means[0].cpu().numpy(), conv5=means[1].cpu().numpy(), windows=count)
            original_resnet_forward=model.backbone.resnet50.forward
            def replace_resnet(self, images):
                return tuple(mean.to(amp).expand(len(images),-1,-1,-1) for mean in means)
            model.backbone.resnet50.forward=types.MethodType(replace_resnet,model.backbone.resnet50)
            output = cache/f'{sha}_rad_mean.npz'
            if not output.exists():
                infer(model, 'rad_mean', output)
            model.backbone.resnet50.forward=original_resnet_forward
        del model
        gc.collect(); torch.cuda.empty_cache()
    save_json(dict(checkpoints=entries, unique_states=len(unique), best_verification=verify,
                   epoch_learning_rates=lr, inference_complete=True), args.out/'checkpoint_audit.json')


def distribution(values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0)
    hist, edges = np.histogram(v, bins=30)
    return dict(n=len(v), mean=v.mean(), std=v.std(ddof=1), min=v.min(), max=v.max(),
                quantiles=dict(zip(['p01','p025','p05','p25','p50','p75','p95','p975','p99'],
                                  np.quantile(v,[.01,.025,.05,.25,.5,.75,.95,.975,.99]))),
                fraction_le_zero=np.mean(v <= 0), histogram_counts=hist, histogram_edges=edges)


def comparison(a, b, metrics, boot, evaluable):
    delta = boot[a]['auc'] - boot[b]['auc']
    macro = macro_samples(delta, evaluable)
    point = metrics[a]['macro_auc']['value'] - metrics[b]['macro_auc']['value']
    result = dict(model_a=a, model_b=b, macro_auc_difference=interval(macro, point),
                  distribution=distribution(macro), macro_difference_resamples=macro, per_label=[])
    for j, label in enumerate(LABELS):
        ma, mb = metrics[a]['per_label'][j], metrics[b]['per_label'][j]
        result['per_label'].append(dict(label=label, n=ma['n'], positives=ma['positives'],
            prevalence=ma['prevalence']['value'],
            auc_difference=interval(delta[:,j], ma['auc']['value'] - mb['auc']['value'])))
    return result, macro, delta


def training_dynamics(args, preds, truth, metrics):
    history = json.loads((args.run/'history.json').read_text())
    events = [json.loads(l) for l in (args.run/'events.jsonl').read_text().splitlines() if l.strip()]
    audit_file = args.out/'checkpoint_audit.json'
    audit = json.loads(audit_file.read_text()) if audit_file.exists() else {}
    lrs = {r['epoch']:r['lr'] for r in audit.get('epoch_learning_rates',[])}
    rows, checks = [], []
    for row in history:
        epoch = row['epoch']
        progress = [e for e in events if e.get('event') == 'training_progress' and e.get('epoch') == epoch]
        starts = [e for e in events if e.get('event') == 'epoch_started' and e.get('epoch') == epoch]
        ends = [e for e in events if e.get('event') == 'epoch_complete' and e.get('epoch') == epoch]
        wall = ((datetime.fromisoformat(ends[-1]['utc'].replace('Z','+00:00')) -
                 datetime.fromisoformat(starts[0]['utc'].replace('Z','+00:00'))).total_seconds()
                if starts and ends else None)
        for kind in ('raw','ema'):
            name = f'epoch_{epoch:02d}_{kind}'
            for split in ('gold74','validation'):
                file = args.run/'predictions'/f'{name}_{split}.npz'
                if file.exists():
                    with np.load(file,allow_pickle=False) as z:
                        recomputed = float(np.nanmean(aucs(z['labels'], expit(z['logits']))))
                    logged = row[kind][split]['macro_auc']
                    checks.append(dict(epoch=epoch, weights=kind, split=split, logged_auc=logged,
                                       saved_prediction_auc=recomputed, difference=recomputed-logged,
                                       matches=abs(recomputed-logged) < 1e-7))
            raw_gold, ema_gold = row['raw']['gold74']['macro_auc'], row['ema']['gold74']['macro_auc']
            raw_val, ema_val = row['raw']['validation']['macro_auc'], row['ema']['validation']['macro_auc']
            rows.append(dict(epoch=epoch, weights=kind, train_loss=row['train_loss'],
                validation_loss=row[kind]['validation']['loss'],
                validation_macro_auc=row[kind]['validation']['macro_auc'],
                gold_macro_auc=row[kind]['gold74']['macro_auc'],
                gold_loss=row[kind]['gold74']['loss'],
                ema_minus_raw_gold_auc=ema_gold-raw_gold,
                ema_minus_raw_validation_auc=ema_val-raw_val,
                epoch_wall_seconds_including_evaluation=wall,
                logged_training_seconds=progress[-1]['seconds'] if progress else None,
                lr_coatnet=lrs.get(epoch,[None]*4)[0], lr_resnet=lrs.get(epoch,[None]*4)[1],
                lr_fusion=lrs.get(epoch,[None]*4)[2], lr_head=lrs.get(epoch,[None]*4)[3],
                grad_norm=None))
    pd.DataFrame(rows).to_csv(args.out/'training_dynamics.csv',index=False)
    return dict(epochs=rows, metric_checks=checks,
                all_saved_epoch_metrics_match=all(r['matches'] for r in checks),
                best_validation={k:max(history,key=lambda r:r[k]['validation']['macro_auc'])['epoch'] for k in ('raw','ema')},
                best_gold_descriptive={k:max(history,key=lambda r:r[k]['gold74']['macro_auc'])['epoch'] for k in ('raw','ema')},
                grad_norm_note='Computed for clipping during training but never persisted. Unavailable.',
                train_auc_note='No per-epoch training AUC was logged or evaluated. Unavailable.',
                wall_time_note='UTC epoch-start to epoch-complete includes evaluation/checkpoint IO; not inferred throughput.')


def subgroup_analysis(args, gold, ids, truth, preds, selected, counts):
    frames = []
    meta_file = args.out/'study_metadata.csv'
    if meta_file.exists():
        frames.append(pd.read_csv(meta_file,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').reindex(ids))
    elif args.manifests and args.manifests.exists():
        rows = []
        for f in sorted(args.manifests.glob('*.json')):
            for r in json.loads(f.read_text()).get('completed',[]):
                if r['study_uid'] in ids:
                    rows.append(dict(StudyInstanceUID=r['study_uid'],windows=r['windows'],source_slices=r['source_slices'],
                        populated_slots=sum(r['slot_present']),
                        **{f'slot_{j}_present':v for j,v in enumerate(r['slot_present'])}))
        if rows:
            frames.append(pd.DataFrame(rows).set_index('StudyInstanceUID').reindex(ids))
    metadata = pd.concat(frames,axis=1) if frames else pd.DataFrame(index=ids)
    if 'source' in gold:
        metadata['label_source'] = gold.loc[ids,'source']
    if args.metadata and args.metadata.exists():
        frame = pd.read_csv(args.metadata,dtype={'StudyInstanceUID':str})
        frame = frame[frame.StudyInstanceUID.isin(ids)]
        for col in ('Laterality','laterality','MagneticFieldStrength','Manufacturer','Scanner','field_strength'):
            if col in frame:
                metadata[col] = frame.groupby('StudyInstanceUID')[col].agg(lambda v:'|'.join(sorted(set(v.dropna().astype(str))))).reindex(ids)
        if not frames:
            metadata['series_count'] = frame.groupby('StudyInstanceUID').size().reindex(ids)
    metadata.index.name='StudyInstanceUID'
    metadata.to_csv(args.out/'study_metadata.csv')
    groups = {}
    for col in metadata:
        if col in ('windows','source_slices','populated_slots','series_count','native_slice_count','median_native_pixel_spacing_mm'):
            cut = {'windows':60,'source_slices':150,'populated_slots':5,'series_count':5,
                   'native_slice_count':200,'median_native_pixel_spacing_mm':.5}[col]
            groups[f'{col}<{cut}'] = metadata[col].to_numpy() < cut
            groups[f'{col}>={cut}'] = metadata[col].to_numpy() >= cut
        else:
            for value in metadata[col].dropna().unique():
                groups[f'{col}={value}'] = metadata[col].to_numpy() == value
    results = []
    for group,mask in groups.items():
        n = int(mask.sum())
        support = []
        for j,label in enumerate(LABELS):
            y = truth[mask,j]; finite=np.isfinite(y); pos=int(np.nansum(y)); neg=int(finite.sum()-pos)
            support.append(dict(label=label,n=int(finite.sum()),positives=pos,negatives=neg))
        r = dict(group=group,studies=n,support=support,models={})
        if n < 20:
            r['skipped']='Fewer than 20 studies; no metric interpreted.'
        else:
            ev = np.array([s['positives']>=5 and s['negatives']>=5 for s in support])
            r['evaluable_labels']=[LABELS[j] for j in np.flatnonzero(ev)]
            for name in selected:
                point=aucs(truth[mask],preds[name][mask])
                per=np.column_stack([weighted_metrics(truth[:,j],preds[name][:,j],counts*mask)['auc'] for j in range(12)])
                r['models'][name]=dict(macro_auc=interval(macro_samples(per,ev),np.mean(point[ev]) if ev.any() else np.nan),
                    per_label=[dict(**support[j],auc=interval(per[:,j],point[j]),
                        skipped=None if ev[j] else 'Fewer than 5 positives or 5 negatives') for j in range(12) if ev[j]])
        results.append(r)
    save_json(results,args.out/'subgroups.json')
    unavailable=[c for c in ['Laterality','MagneticFieldStrength','Manufacturer','Scanner'] if c not in metadata]
    return dict(groups=results,unavailable_metadata=unavailable,minimum_studies=20,minimum_per_class=5,
                note='Exploratory subgroups; no multiple-comparison correction. No inference for small groups.')


def make_plots(args, truth, preds, metrics, selected, training, pair_samples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=args.out/'plots'; directory.mkdir(exist_ok=True)
    for kind in ('roc','pr','reliability'):
        fig,axes=plt.subplots(3,4,figsize=(18,12))
        for j,(label,ax) in enumerate(zip(LABELS,axes.flat)):
            mask=np.isfinite(truth[:,j]); y=truth[mask,j]
            for name in selected:
                p=preds[name][mask,j]
                if kind=='roc' and len(np.unique(y))==2:
                    x,v,_=roc_curve(y,p); ax.plot(x,v,label=name)
                elif kind=='pr' and y.sum()>0:
                    v,x,_=precision_recall_curve(y,p); ax.step(x,v,where='post',label=name)
                elif kind=='reliability':
                    bins=metrics[name]['per_label'][j]['reliability']
                    ax.plot([r['mean_probability'] for r in bins if r['n']],
                            [r['observed_prevalence'] for r in bins if r['n']],'-o',label=name)
            if kind in ('roc','reliability'): ax.plot([0,1],[0,1],'--',color='gray')
            else: ax.axhline(y.mean(),color='gray',ls='--')
            ax.set(title=f'{label}\npositive={int(y.sum())}/{len(y)}, prevalence={y.mean():.4f}',xlim=(0,1),ylim=(0,1),
                   xlabel={'roc':'False positive rate','pr':'Recall','reliability':'Mean probability'}[kind],
                   ylabel={'roc':'Sensitivity','pr':'Precision','reliability':'Observed prevalence'}[kind])
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7)
        fig.tight_layout();fig.savefig(directory/f'{kind}.png',dpi=160);plt.close(fig)
    frame=pd.DataFrame(training['epochs'])
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    raw=frame[frame.weights=='raw']
    axes[0,0].plot(raw.epoch,raw.train_loss,label='Training: soft HYBRID targets')
    for kind,part in frame.groupby('weights'):
        axes[0,0].plot(part.epoch,part.validation_loss,'--',label=f'{kind} validation')
        axes[0,1].plot(part.epoch,part.validation_macro_auc,label=f'{kind} validation')
        axes[0,1].plot(part.epoch,part.gold_macro_auc,'--',label=f'{kind} Gold74')
    for c in ('lr_coatnet','lr_fusion','lr_head'):
        axes[0,2].plot(raw.epoch,raw[c],label=c)
    axes[1,0].plot(raw.epoch,raw.epoch_wall_seconds_including_evaluation/60,label='Epoch wall minutes')
    axes[1,1].plot(raw.epoch,raw.ema_minus_raw_gold_auc,label='EMA - raw Gold AUC')
    axes[1,1].plot(raw.epoch,raw.ema_minus_raw_validation_auc,label='EMA - raw validation AUC')
    axes[1,1].axhline(0,color='gray',ls='--')
    axes[1,2].text(.5,.5,'Gradient norm unavailable\nTrainer did not persist it',ha='center',va='center',transform=axes[1,2].transAxes)
    axes[1,2].set_axis_off()
    for ax,title in zip(axes.flat,['Train/validation BCE','Validation and Gold AUC','Measured endpoint LR','Epoch wall time','EMA minus raw AUC','Gradient norm']):
        ax.set(title=title,xlabel='Epoch'); ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(directory/'training.png',dpi=160);plt.close(fig)
    if pair_samples:
        fig,axes=plt.subplots(len(pair_samples),1,figsize=(9,3*len(pair_samples)),squeeze=False)
        for ax,(name,values) in zip(axes.flat,pair_samples.items()):
            values=values[np.isfinite(values)]
            ax.hist(values,bins=40,alpha=.8); ax.axvline(0,color='black',ls='--')
            ax.set(title=name,xlabel='Paired macro-AUC difference',ylabel='Resamples')
        fig.tight_layout();fig.savefig(directory/'paired_difference_distributions.png',dpi=150);plt.close(fig)


def analyze(args, gold, ids, truth, artifacts):
    audit_path=args.out/'checkpoint_audit.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    preds, aliases, sources = {}, {}, {}
    grouped={}
    for r in audit.get('checkpoints',[]): grouped.setdefault(r['state_sha256'],[]).append(r)
    for sha,rows in grouped.items():
        epoch=[r['tag'] for r in rows if r['tag'].startswith('epoch_') and r['tag'].endswith(('_raw','_ema')) and 'before_eval' not in r['tag']]
        initial=[r['tag'] for r in rows if r['tag']=='initial']
        name=sorted(epoch)[0] if epoch else ('parent_initial' if initial else rows[0]['tag'])
        for r in rows: aliases[r['tag']]=name
        for mode in ('full','no_fusion','rad_mean'):
            f=args.out/'inference_cache'/f'{sha}_{mode}.npz'
            if f.exists():
                key=name if mode=='full' else name+'__'+mode
                preds[key]=load_npz(f,ids,truth); sources[key]=dict(file=str(f),state_sha256=sha,mode=mode)
                with np.load(f,allow_pickle=False) as z:
                    sources[key]['recorded_inference_settings']={k:z[k].item() for k in
                        ('inference_chunk','inference_seed','deterministic') if k in z}
    for f in sorted((args.run/'predictions').glob('epoch_*_gold74.npz')):
        tag=f.stem.removesuffix('_gold74'); name=aliases.get(tag,tag)
        if name not in preds:
            preds[name]=load_npz(f,ids,truth);sources[name]=dict(file=str(f),mode='full')
    missing=[]
    if 'parent_initial' not in preds: missing.append('Current-input parent initial evaluation missing')
    fold_predictions=[]
    if args.comparators and args.comparators.exists():
        for k in range(5):
            f=args.comparators/f'preds_cv_fold{k}_gold74.csv'
            if not f.exists(): continue
            frame=pd.read_csv(f,dtype={'StudyInstanceUID':str})
            if 'mode' in frame: frame=frame[frame['mode']=='full']
            frame=frame.set_index('StudyInstanceUID')
            if frame.index.has_duplicates or set(frame.index)!=set(ids): raise ValueError(f'{f}: mismatched comparator studies')
            p=frame.loc[ids,LABELS].to_numpy(float)
            if not np.isfinite(p).all() or np.any((p<0)|(p>1)): raise ValueError('Invalid comparator probabilities')
            fold_predictions.append(p)
            preds[f'cv_fold{k}']=p;sources[f'cv_fold{k}']=dict(file=str(f),mode='saved_full')
    if len(fold_predictions)==5:
        preds['existing_cv5_ensemble']=np.mean(fold_predictions,axis=0)
        sources['existing_cv5_ensemble']=dict(operation='Unweighted arithmetic probability mean of five saved full fold predictions',
            caveat='Historical RadImageNet models and preprocessing differ; matched study IDs, not a controlled architectural comparison. Historical Gold74 model selection induces optimism.')
    else: missing.append(f'Five-fold ensemble unavailable: only {len(fold_predictions)} fold prediction files')
    if not audit.get('inference_complete'): missing.append('Checkpoint inference audit is incomplete; coverage is limited to existing predictions')
    counts=bootstrap_counts(len(ids),args.bootstrap)
    evaluable=np.array([len(np.unique(truth[np.isfinite(truth[:,j]),j]))==2 for j in range(12)])
    point_counts=np.ones((1,len(ids)))
    metrics, boot, pred_rows, errors = {}, {}, [], []
    for name,p in preds.items():
        point_auc=aucs(truth,p)
        per_boot={k:np.empty((args.bootstrap,12)) for k in ['auc','ap','brier','ece','mean_probability','prevalence']}
        per=[]
        for j,label in enumerate(LABELS):
            mask=np.isfinite(truth[:,j]); y=truth[mask,j]; pp=p[mask,j]
            values=weighted_metrics(truth[:,j],p[:,j],counts)
            points=weighted_metrics(truth[:,j],p[:,j],point_counts)
            r=dict(label=label,n=int(mask.sum()),missing=int((~mask).sum()),positives=int(y.sum()),negatives=int(len(y)-y.sum()))
            for key in per_boot:
                per_boot[key][:,j]=values[key]
                r[key]=interval(values[key],point_auc[j] if key=='auc' else points[key][0])
            r['auc_ci_too_wide']=bool(r['auc']['ci_width']>WIDE) if np.isfinite(r['auc']['ci_width']) else True
            r['ap_ci_too_wide']=bool(r['ap']['ci_width']>WIDE) if np.isfinite(r['ap']['ci_width']) else True
            r['ci_too_wide']=r['auc_ci_too_wide'] or r['ap_ci_too_wide']
            r['wide_ci_rule']='AUROC or AP 95% CI width > 0.20, or undefined; metric-specific descriptive precision flags'
            r['reliability']=reliability(truth[:,j],p[:,j])
            thresholds=[dict(name=f'fixed_{t}',threshold=t,selection='predeclared probability threshold') for t in (.1,.25,.5,.75,.9)]
            if len(np.unique(y))==2:
                candidates=np.unique(np.r_[0,pp,np.nextafter(1.,2.)])
                f1=[operating(truth[:,j],p[:,j],t,point_counts)['f1'] for t in candidates]
                thresholds.append(dict(name='best_f1',threshold=float(candidates[int(np.nanargmax(f1))]),selection='Maximal empirical Gold74 F1; smallest threshold breaks ties. Optimistic, not deployment validation.'))
                for target in (.9,.95):
                    fpr,tpr,ts=roc_curve(y,pp,drop_intermediate=False)
                    eligible=np.flatnonzero(fpr <= 1-target+1e-12)
                    ix=eligible[np.argmax(tpr[eligible])]
                    thresholds.append(dict(name=f'specificity_at_least_{target}',threshold=float(ts[ix]),selection='Empirical Gold74 ROC; optimistic threshold selection'))
            r['operating_points']=[]
            for threshold in thresholds:
                t=threshold['threshold']; pts=operating(truth[:,j],p[:,j],t,point_counts); bs=operating(truth[:,j],p[:,j],t,counts)
                r['operating_points'].append(dict(**threshold,**{k:interval(bs[k],pts[k]) for k in bs},
                    confusion={k:pts[k] for k in ('tp','tn','fp','fn')},ci_note='Conditional on the fixed reported threshold; selection uncertainty not included.'))
                if threshold['name']=='best_f1':
                    r['operating_points'][-1]['threshold_uncertainty']=interval(
                        best_f1_threshold_samples(truth[:,j],p[:,j],counts),t)
            if len(np.unique(y))==2:
                fpr,tpr,ts=roc_curve(y,pp,drop_intermediate=False)
                precision,recall,pt=precision_recall_curve(y,pp)
                r['roc_curve']=dict(fpr=fpr,tpr=tpr,thresholds=ts)
                r['pr_curve']=dict(precision=precision,recall=recall,thresholds=pt)
            per.append(r)
            for kind,m in [('false_positive',(y==0)&(pp>=.5)),('false_negative',(y==1)&(pp<.5))]:
                selected_ix=np.flatnonzero(m)
                order=np.argsort(-pp[selected_ix] if kind=='false_positive' else pp[selected_ix],kind='stable')
                for rank,i in enumerate(selected_ix[order][:10],1):
                    errors.append(dict(model=name,label=label,error=kind,rank=rank,StudyInstanceUID=np.asarray(ids)[mask][i],truth=y[i],probability=pp[i],threshold=.5))
        macro=macro_samples(per_boot['auc'],evaluable)
        metrics[name]=dict(studies=len(ids),evaluable_labels=[LABELS[j] for j in np.flatnonzero(evaluable)],
                          macro_auc=interval(macro,np.mean(point_auc[evaluable])),per_label=per,source=sources[name])
        boot[name]=per_boot
        for i,uid in enumerate(ids):
            pred_rows.append(dict(model=name,StudyInstanceUID=uid,**dict(zip(LABELS,p[i])),
                                  **{f'truth_{label}':truth[i,j] for j,label in enumerate(LABELS)}))
        print('SCORED',name,flush=True)
    pd.DataFrame(pred_rows).to_csv(args.out/'per_study_predictions.csv',index=False,float_format='%.17g')
    pd.DataFrame(errors).to_csv(args.out/'error_queue.csv',index=False,float_format='%.17g')
    summary=pd.DataFrame([dict(model=k,**v['macro_auc']) for k,v in metrics.items()])
    summary.to_csv(args.out/'headline_metrics.csv',index=False)
    pf=[]
    for name,m in metrics.items():
        for r in m['per_label']:
            record={k:r[k] for k in ['label','n','missing','positives','negatives','ci_too_wide','auc_ci_too_wide','ap_ci_too_wide']}
            for key in ['auc','ap','brier','ece','mean_probability','prevalence']:
                record.update({f'{key}_{k}':v for k,v in r[key].items()})
            pf.append(dict(model=name,**record))
    pd.DataFrame(pf).to_csv(args.out/'per_label_metrics.csv',index=False)
    training=training_dynamics(args,preds,truth,metrics)
    selected=[aliases.get(f'best_validation_{k}',f"epoch_{training['best_validation'][k]:02d}_{k}") for k in ('raw','ema')]
    selected=list(dict.fromkeys(selected+[k for k in ('parent_initial','existing_cv5_ensemble') if k in preds]))
    pairs=[]; distributions=[]; plot_samples={}
    pair_set=set()
    for name in preds:
        if name.startswith('cv_fold') or name in ('parent_initial','existing_cv5_ensemble') or '__' in name: continue
        for reference in ('parent_initial','existing_cv5_ensemble'):
            if reference in preds: pair_set.add((name,reference))
    for name in list(preds):
        if '__' in name:
            pair_set.add((name.split('__')[0],name))
    for epoch in range(1,11):
        if all(f'epoch_{epoch:02d}_{k}' in preds for k in ('raw','ema')):
            pair_set.add((f'epoch_{epoch:02d}_ema',f'epoch_{epoch:02d}_raw'))
    for kind in ('raw','ema'):
        best=aliases.get(f'best_validation_{kind}',f"epoch_{training['best_validation'][kind]:02d}_{kind}")
        if best in preds and f'epoch_10_{kind}' in preds: pair_set.add((f'epoch_10_{kind}',best))
    for a,b in sorted(pair_set):
        result,samples,delta=comparison(a,b,metrics,boot,evaluable)
        pairs.append(result)
        for i,v in enumerate(samples): distributions.append(dict(model_a=a,model_b=b,resample=i,macro_auc_difference=v))
        if a in selected[:2] and b in ('parent_initial','existing_cv5_ensemble',a+'__no_fusion',a+'__rad_mean'):
            plot_samples[f'{a} - {b}']=samples
    pd.DataFrame(distributions).to_csv(args.out/'paired_macro_auc_distributions.csv',index=False,float_format='%.17g')
    pd.DataFrame([dict(model_a=r['model_a'],model_b=r['model_b'],**r['macro_auc_difference']) for r in pairs]).to_csv(args.out/'paired_macro_auc_deltas.csv',index=False)
    pd.DataFrame([dict(model_a=r['model_a'],model_b=r['model_b'],label=p['label'],n=p['n'],positives=p['positives'],prevalence=p['prevalence'],**p['auc_difference']) for r in pairs for p in r['per_label']]).to_csv(args.out/'paired_per_label_deltas.csv',index=False)
    correlation={}; cofail=[]
    for name in selected:
        residual=pd.DataFrame(preds[name]-truth,columns=LABELS)
        matrix=residual.corr(min_periods=10)
        correlation[name]=dict(matrix=matrix.to_numpy(),labels=LABELS,pairs=[])
        for j in range(12):
            for k in range(j+1,12):
                mask=np.isfinite(truth[:,j]) & np.isfinite(truth[:,k])
                x,y=(preds[name]-truth)[mask][:,[j,k]].T
                w=counts[:,mask]; n=w.sum(1)
                with np.errstate(divide='ignore',invalid='ignore'):
                    mx,my=w@x/n,w@y/n
                    cov=w@(x*y)/n-mx*my
                    corr=cov/np.sqrt((w@(x*x)/n-mx*mx)*(w@(y*y)/n-my*my))
                fail=(preds[name][mask]>=.5)!=truth[mask]
                row=dict(model=name,label_a=LABELS[j],label_b=LABELS[k],n=int(mask.sum()),
                         residual_correlation=interval(corr,matrix.iloc[j,k]),
                         both_wrong=int((fail[:,j]&fail[:,k]).sum()),
                         both_wrong_rate=interval((w@(fail[:,j]&fail[:,k]))/n,
                             float(np.mean(fail[:,j]&fail[:,k]))))
                correlation[name]['pairs'].append(row);cofail.append(row)
    pd.DataFrame([dict(**{k:v for k,v in r.items() if k not in ('residual_correlation','both_wrong_rate')},
        **{f'residual_correlation_{k}':v for k,v in r['residual_correlation'].items()},
        **{f'both_wrong_rate_{k}':v for k,v in r['both_wrong_rate'].items()}) for r in cofail]).to_csv(args.out/'residual_correlations.csv',index=False)
    subgroup=subgroup_analysis(args,gold,ids,truth,preds,selected,counts)
    diagnostics={}
    for sha,rows in grouped.items():
        f=args.out/'inference_cache'/f'{sha}_diagnostics.json'
        if f.exists():
            content=json.loads(f.read_text());name=aliases[rows[0]['tag']]
            frame=pd.DataFrame(content['windows'])
            layers={}
            for layer,g in frame.groupby('layer'):
                layers[layer]={}
                for field in ('residual_norm_over_stream','resnet_projection_norm_fraction'):
                    # Cluster bootstrap: sum window values and counts by study.
                    sums=g.groupby('study_uid')[field].sum().reindex(ids).to_numpy()
                    ns=g.groupby('study_uid').size().reindex(ids).to_numpy()
                    bs=(counts@sums)/(counts@ns)
                    v=g[field].to_numpy()
                    layers[layer][field]=dict(mean=interval(bs,v.mean()),median=np.median(v),
                        p01=np.quantile(v,.01),p10=np.quantile(v,.1),p90=np.quantile(v,.9),p99=np.quantile(v,.99),windows=len(v))
                layers[layer]['gate']=content['gates'][layer]
            diagnostics[name]=dict(layers=layers,
                fraction_definition='Norm of ResNet part of the additive down-projection / (ResNet norm + CoAtNet norm), before depthwise/GELU/gate. Bias excluded; not an identifiable fraction of final causal prediction signal.')
    if not diagnostics: missing.append('Fusion activation diagnostics unavailable')
    missing.extend(['Per-epoch gradient norms and training AUC not persisted',
                    'Scanner, field strength and laterality unavailable in supplied metadata'])
    parent_file=args.out/'parent_verification.json'
    readings=args.out/'source_readings.json'
    metadata_inventory=args.out/'tensor_metadata_inventory.json'
    result=dict(run_name='resnet50_20261013',orientation_head='d49b3458314ede05079a1b12609ad1326aeec227',
        seed=SEED,bootstrap=dict(resamples=args.bootstrap,unit='study',paired=True,scheme='Unstratified percentile bootstrap',
            macro_policy='Fixed full-sample evaluable label set; discard a macro replicate if any label loses a class. Per-label intervals use all valid replicates.',
            uncertainty_scope='Study sampling only. No training-seed, label-error, patient-clustering or selection uncertainty; multiplicity unadjusted.'),
        calibration=dict(ece_bins=BINS,scheme='Equal-width [0,1], left closed/right open except final bin includes 1; count-weighted absolute calibration gap'),
        threshold_note='Gold74 best-F1 and specificity-constrained thresholds are descriptive in-sample estimates; no deployment claim.',
        training_config=json.loads((args.run/'training_config.json').read_text()),
        gold_definition=dict(studies=len(ids),source_counts=gold['source'].value_counts().to_dict() if 'source' in gold else {},
                             labeled_entries=int(np.isfinite(truth).sum()),missing_entries=int((~np.isfinite(truth)).sum()),
                             missing_per_label=dict(zip(LABELS,(~np.isfinite(truth)).sum(0))),
                             disjoint_from_fusion_train_and_validation=True,
                             historical_parent_training_exclusion='Not independently re-established; historical selection may include Gold.'),
        support=[dict(label=r['label'],n=r['n'],positives=r['positives'],negatives=r['negatives'],prevalence=r['prevalence']) for r in next(iter(metrics.values()))['per_label']],
        metrics=metrics,aliases=aliases,checkpoint_audit=audit,comparisons=pairs,training=training,
        parent_verification=json.loads(parent_file.read_text()) if parent_file.exists() else None,
        source_readings=json.loads(readings.read_text()) if readings.exists() else None,
        tensor_metadata_inventory=json.loads(metadata_inventory.read_text()) if metadata_inventory.exists() else None,
        residual_correlations=correlation,subgroups=subgroup,fusion_diagnostics=diagnostics,
        missing_evidence=missing,artifact_inventory=artifacts,
        distributions_file='paired_macro_auc_distributions.csv',
        reused_tooling='Extended analyze_training_evidence.py in place: reused its AUROC helper (generalized missing-label masks), study-ID/label alignment, sklearn metric definitions, paired study-resampling design and training-curve approach. Fixed-label macro-bootstrap and 10-bin calibration conventions were also reviewed in the existing analyze_fusion_posttrain.py in the parent worktree. Added weighted bootstrap acceleration, caches, ablations, activation hooks and subgroup/error outputs. Inference directly reuses load_fused_checkpoint, StudyDataset/discover and bag_logits from the actual model/trainer. The legacy no-argument entry point remains import-safe.')
    save_json(result,args.out/'statistics.json',compact=True)
    make_plots(args,truth,preds,metrics,selected,training,plot_samples)
    write_report(args,result,selected)
    print('\nHEADLINE TABLE',flush=True)
    print(summary[summary.model.isin(selected+['epoch_10_raw','epoch_10_ema','epoch_01_raw'])].to_string(index=False),flush=True)


def fmt(metric):
    if metric.get('value') is None or not np.isfinite(metric.get('value',np.nan)): return 'unavailable'
    lo,hi=metric.get('ci_low'),metric.get('ci_high')
    if lo is None or hi is None: return f"{metric['value']:.6f} (CI unavailable)"
    return f"{metric['value']:.6f} [{lo:.6f}, {hi:.6f}]"


def write_report(args,result,selected):
    result=clean(result)
    metrics=result['metrics'];pairs=result['comparisons'];training=result['training']
    def find_pair(a,b): return next((r for r in pairs if r['model_a']==a and r['model_b']==b),None)
    def claim(pair):
        m=pair['macro_auc_difference'];lo,hi=m['ci_low'],m['ci_high']
        meaning='uncertain: interval crosses zero' if lo<=0<=hi else ('higher AUC under this bootstrap' if lo>0 else 'lower AUC under this bootstrap')
        return f"{pair['model_a']} minus {pair['model_b']}: **{fmt(m)}**; {meaning}."
    lines=['# ResNet50 fusion training: statistical report','', '## Verdict','']
    early=find_pair('epoch_01_raw','parent_initial')
    if early:
        lines.append('- Exploratory early improvement: '+claim(early)+' Epoch 1 is the observed Gold peak, identified after training; it was not selected by validation, and this interval is unadjusted for epoch inspection.')
    for ref in ('parent_initial','existing_cv5_ensemble'):
        rows=[find_pair(name,ref) for name in selected[:2]]
        if any(rows): lines.append('- Validation-selected comparisons: '+' '.join(claim(r) for r in rows if r))
    for name in selected[:2]:
        contrasts=[]
        for mode in ('no_fusion','rad_mean'):
            pair=find_pair(name,name+'__'+mode)
            if pair:
                contrasts.append(f"full minus {mode}: {fmt(pair['macro_auc_difference'])}")
        if contrasts: lines.append('- Fusion in '+name+': '+ '; '.join(contrasts)+'.')
    later=[]
    for kind in ('raw','ema'):
        best=next((n for n in selected[:2] if n.endswith('_'+kind)),None)
        pair=find_pair('epoch_10_'+kind,best)
        if pair: later.append(claim(pair))
    if later: lines.append('- Later-epoch change: '+' '.join(later))
    ablation_pairs=[r for r in pairs if '__' in r['model_b']]
    if ablation_pairs and all(r['macro_auc_difference']['ci_low']<=0<=r['macro_auc_difference']['ci_high'] for r in ablation_pairs):
        lines.append(f"- Fusion contribution beyond sampling noise is **not established**: all {len(ablation_pairs)} selected/final [ablation intervals](#paired-difference-distributions) cross zero. This does not establish zero effect.")
    lines += ['', 'The data cannot establish a new-patient or clinical benefit, a controlled architecture gain against historical CV5, or a causal fraction of ResNet prediction signal. Intervals cover study sampling only; historical Gold selection, unavailable patient grouping and holdout-fitted thresholds limit interpretation. Gradient norm, training AUC, scanner/field strength and laterality are unavailable.',
        '', '## Headline discrimination', '', '| Checkpoint / comparator | Macro ROC-AUC (95% CI) | Valid / total resamples |', '|---|---:|---:|']
    for name in list(dict.fromkeys(selected+['epoch_10_raw','epoch_10_ema','epoch_01_raw'])):
        if name in metrics:
            m=metrics[name]['macro_auc'];lines.append(f"| {name} | {fmt(m)} | {m['valid_resamples']} / {result['bootstrap']['resamples']} |")
    lines += ['', 'All checkpoints, per-label support, AUROC, average precision (step-integral PR-AUC, not trapezoidal area), calibration and thresholds are in `statistics.json` and `per_label_metrics.csv`. `aliases` maps every snapshot and embedded EMA to its evaluated tensor state. No model was selected using the new Gold74 analysis.',
        '', '## Holdout and uncertainty', '',f"Orientation HEAD: `{result['orientation_head']}`. Gold74 IDs and missing-label masks were checked against the persisted training split and label SHA256. Gold74 is disjoint from the 3,900 training and 433 validation studies. Seed {result['seed']}; {result['bootstrap']['resamples']} unstratified study resamples; identical resamples across all models and paired deltas. A macro resample is dropped if any full-sample evaluable label loses a class, so label composition never changes.",
        '', 'AUROC or AP CI width > 0.20 is flagged as too imprecise for a claim on that label/metric. This explicit heuristic is not a significance test; both widths and separate flags are retained. No multiplicity correction; per-epoch inspection and historical Gold selection can be optimistic. An empirical AUC of 1 may have a degenerate [1,1] bootstrap interval: resampling observed perfectly separated cases cannot invent unseen difficult cases, so that interval does not establish population-perfect performance. Calibration uses 10 equal-width bins; PR curves show prevalence baselines. Empty calibration bins remain empty.',
        '', '| Label | Labeled studies | Positive | Prevalence (95% CI) |', '|---|---:|---:|---:|']
    for r in result['support']: lines.append(f"| {r['label']} | {r['n']} | {r['positives']} | {fmt(r['prevalence'])} |")
    for name in selected[:2]:
        lines += ['',f'## Per-label metrics: {name}', '', '| Label | Positive / labeled | AUROC (95% CI) | AUROC CI width | Precision flag | AP (95% CI) | AP CI width | Brier (95% CI) | ECE (95% CI) |', '|---|---:|---:|---:|---|---:|---:|---:|---:|']
        for r in metrics[name]['per_label']:
            flag=', '.join(k for k,v in [('AUROC wide',r['auc_ci_too_wide']),('AP wide',r['ap_ci_too_wide'])] if v) or 'within rule'
            lines.append(f"| {r['label']} | {r['positives']} / {r['n']} | {fmt(r['auc'])} | {r['auc']['ci_width']:.6f} | {flag} | {fmt(r['ap'])} | {r['ap']['ci_width']:.6f} | {fmt(r['brier'])} | {fmt(r['ece'])} |")
        lines += ['', '| Label | Mean probability (95% CI) | Observed prevalence (95% CI) | Sensitivity at 0.5 (95% CI) | Specificity at 0.5 (95% CI) | Best-F1 threshold (95% CI) | Best F1 (conditional 95% CI) |', '|---|---:|---:|---:|---:|---:|---:|']
        for r in metrics[name]['per_label']:
            fixed=next(v for v in r['operating_points'] if v['name']=='fixed_0.5')
            best=next((v for v in r['operating_points'] if v['name']=='best_f1'),None)
            lines.append(f"| {r['label']} | {fmt(r['mean_probability'])} | {fmt(r['prevalence'])} | {fmt(fixed['sensitivity'])} | {fmt(fixed['specificity'])} | {fmt(best['threshold_uncertainty'])} | {fmt(best['f1'])} |" if best else f"| {r['label']} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |")
        lines += ['', 'Threshold CIs re-fit maximal F1 inside each study resample on the fixed full-sample grid of observed probabilities plus 0 and an always-negative threshold; smallest threshold breaks ties. Sensitivity, specificity and F1 intervals condition on the reported full-sample threshold; they do not remove holdout threshold-selection optimism. JSON also provides fixed thresholds 0.10/0.25/0.50/0.75/0.90 and empirical operating points at at least 90%/95% specificity.']
    lines += ['', '## Paired difference distributions', '', '| Comparison | Macro-AUC difference (95% CI) | Bootstrap mean | SD | Median | Fraction <= 0 |', '|---|---:|---:|---:|---:|---:|']
    for r in pairs:
        if r['model_a'] in selected[:2] or (r['model_a'] in ('epoch_10_raw','epoch_10_ema') and (r['model_b'] in selected[:2] or '__' in r['model_b'])):
            d=r['distribution'];lines.append(f"| {r['model_a']} - {r['model_b']} | {fmt(r['macro_auc_difference'])} | {d['mean']:.6f} | {d['std']:.6f} | {d['quantiles']['p50']:.6f} | {d['fraction_le_zero']:.6f} |")
    lines += ['', 'Exact macro difference draws are in `paired_macro_auc_distributions.csv` and JSON; JSON also includes quantiles and histogram counts/edges. The fraction <= 0 is a description of resampled differences, not a calibrated p-value. `paired_per_label_deltas.csv` includes positive counts, prevalence and per-label CIs for every contrast. Comparators use identical study IDs; CV5 is the equal probability mean of the five saved full fold outputs, not a newly fitted ensemble.',
        '', '## Training dynamics and best-checkpoint audit', '', '| Epoch | Train BCE | Raw val BCE | EMA val BCE | Raw val AUC | EMA val AUC | Raw Gold AUC (95% CI) | EMA Gold AUC (95% CI) | EMA - raw Gold (95% CI) | Wall seconds |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for epoch in range(1,11):
        rows=[r for r in training['epochs'] if r['epoch']==epoch];raw=next(r for r in rows if r['weights']=='raw');ema=next(r for r in rows if r['weights']=='ema')
        pair=find_pair(f'epoch_{epoch:02d}_ema',f'epoch_{epoch:02d}_raw')
        lines.append(f"| {epoch} | {raw['train_loss']:.8f} | {raw['validation_loss']:.8f} | {ema['validation_loss']:.8f} | {raw['validation_macro_auc']:.8f} | {ema['validation_macro_auc']:.8f} | {fmt(metrics[f'epoch_{epoch:02d}_raw']['macro_auc'])} | {fmt(metrics[f'epoch_{epoch:02d}_ema']['macro_auc'])} | {fmt(pair['macro_auc_difference']) if pair else 'unavailable'} | {raw['epoch_wall_seconds_including_evaluation']} |")
    lines += ['', 'Training BCE uses soft HYBRID scores, intensity augmentation and 24 sampled windows; validation BCE uses binary HYBRID labels and all windows; Gold BCE uses available structured Gold labels and all windows. These losses use different targets and protocols. Only the raw optimizer model has a logged training loss; EMA has separate evaluation metrics. Train/validation losses and validation AUC above are logged measurements, not inferential estimates; per-study train loss was not saved, so no loss CI is fabricated. Endpoint LR is read from raw checkpoint scheduler state in `training_dynamics.csv`. Wall time is UTC epoch start to completion, including evaluation and checkpoint writes. Gradient-norm and training-AUC columns are null because these values were not persisted.',
        '',f"Validation-selected epochs: raw **{training['best_validation']['raw']}**, EMA **{training['best_validation']['ema']}**. Descriptive Gold peaks: raw {training['best_gold_descriptive']['raw']}, EMA {training['best_gold_descriptive']['ema']} (not new selections). All 40 saved epoch prediction AUCs match logged metrics: **{training['all_saved_epoch_metrics_match']}**."]
    for r in result['checkpoint_audit'].get('best_verification',[]):
        lines.append(f"- Best {r['kind']}: tensor-identical to selected epoch {r['selected_epoch']}: **{r['tensor_identical']}**; validation selection AUC {r['selection_auc']:.10f}; tensor SHA256 `{r['best_sha256']}`. This proves weight/selection correspondence rather than trusting a filename.")
    if result.get('parent_verification'):
        v=result['parent_verification']
        lines.append(f"- Initial baseline verified against the actual parent checkpoint: **{v['baseline_verified']}**. Every CoAtNet/head tensor is identical and both fusion up-projection weights/biases are exactly zero. Parent file SHA256 `{v['parent_file_sha256']}`.")
    parity=[]
    for f in sorted((args.out/'inference_cache').glob('*_parity.json')):
        z=json.loads(f.read_text());parity.append(dict(file=f.name,**z))
        lines.append(f"- Independent re-inference of `{Path(z['saved_file']).name}`: maximum probability difference {z['max_probability_difference']:.10g}; per-label saved/recomputed AUCs recorded in `{f.name}`. Deterministic cuDNN settings differ from the trainer's benchmark-enabled settings, so BF16 probabilities can differ slightly despite identical weights. Headline/ablation metrics for re-inferred snapshots use the deterministic full cache; logged training trajectories retain original evaluations. This numerical difference is not evidence of mismatched checkpoint weights.")
    result['checkpoint_parity']=parity
    save_json(result,args.out/'statistics.json',compact=True)
    lines += ['', '## Fusion diagnostics and ablations', '', 'Full, no_fusion (both residual corrections forced to zero), and rad_mean (checkpoint-specific window-weighted mean ResNet feature maps from Gold74, labels unused) retain the same checkpoint weights and study windows. rad_mean is a transductive, exploratory control, not a deployment baseline. Ablations cover validation-selected raw/EMA and final raw/EMA. All available states receive full inference or reuse their saved full evaluation; aliases avoid repeated inference of identical tensors.',
        '', '| Checkpoint | Layer | Residual norm / stream norm, mean (95% CI) | Median [p10, p90] | ResNet down-projection norm fraction, mean (95% CI) | Gate mean / SD / p01 / p99 |', '|---|---|---:|---:|---:|---:|']
    for name,d in result['fusion_diagnostics'].items():
        for layer,r in d['layers'].items():
            n=r['residual_norm_over_stream'];f=r['resnet_projection_norm_fraction'];g=r['gate']
            lines.append(f"| {name} | {layer} | {fmt(n['mean'])} | {n['median']:.6f} [{n['p10']:.6f}, {n['p90']:.6f}] | {fmt(f['mean'])} | {g['mean']:.6f} / {g['std']:.6f} / {g['p01']:.6f} / {g['p99']:.6f} |")
    lines += ['', 'Residual norm is measured before adding the correction to the CoAtNet stream (S2 Z, S4 V). CIs resample studies while retaining their windows. Gate distributions are measured activation summaries. Branch fraction is the ratio of additive ResNet versus CoAtNet down-projection norms before nonlinear mixing, excluding bias; it is **not an identifiable causal percentage of final model signal**. Functional ablations supply the evidence about contribution to discrimination.',
        '', '## Error review and labels that fail together', '', '`error_queue.csv` contains the top 10 actual false positives and false negatives per model/label at threshold 0.5, ranked by contradiction, with study IDs and full-precision probabilities. An empty queue means no such threshold error, not a fabricated example. Residuals are probability minus observed label; correlations use pairwise available labels. The table is descriptive with bootstrap CIs; it does not establish a shared cause.',
        '', '| Model | Labels | Labeled pairs | Residual correlation (95% CI) | Both threshold decisions wrong |', '|---|---|---:|---:|---:|']
    for name,d in result['residual_correlations'].items():
        rows=sorted(d['pairs'],key=lambda r:abs(r['residual_correlation']['value'] or 0),reverse=True)[:5]
        for r in rows:lines.append(f"| {name} | {r['label_a']} / {r['label_b']} | {r['n']} | {fmt(r['residual_correlation'])} | {r['both_wrong']} |")
    lines += ['', 'The pairs below have the largest observed joint error rates at 0.5. These measure failure together, separately from signed residual correlation; the ordering is exploratory.',
        '', '| Model | Labels | Both wrong / labeled pairs | Joint error rate (95% CI) |', '|---|---|---:|---:|']
    for name,d in result['residual_correlations'].items():
        rows=sorted(d['pairs'],key=lambda r:r['both_wrong_rate']['value'],reverse=True)[:5]
        for r in rows:lines.append(f"| {name} | {r['label_a']} / {r['label_b']} | {r['both_wrong']} / {r['n']} | {fmt(r['both_wrong_rate'])} |")
    lines += ['', '## Subgroups', '', 'Groups below 20 studies are skipped. Within retained groups, a label needs at least five positives and five negatives. Macro averages therefore may use different labels and must not be compared across groups as if composition were identical. All label counts and CIs are in `subgroups.json`. Bins were declared in the script; no cutpoint search was performed. Source slices count exported selected slices; native slice count sums native depths of distinct selected series; pixel spacing is the median of native row/column spacing components across selected source slices. Nested audit metadata were inspected in every Gold tensor; scanner and laterality were not inferred from pixels.',
        '', '| Group | Studies | Status / evaluable labels | Selected raw macro AUC (95% CI) |', '|---|---:|---|---:|']
    for r in result['subgroups']['groups']:
        if 'skipped' in r:lines.append(f"| {r['group']} | {r['studies']} | {r['skipped']} | skipped |")
        else:lines.append(f"| {r['group']} | {r['studies']} | {', '.join(r['evaluable_labels'])} | {fmt(r['models'][selected[0]]['macro_auc'])} |")
    lines += ['', 'Unavailable metadata: '+', '.join(result['subgroups']['unavailable_metadata'])+'. These subgroups were not estimated.',
        '', '## Artifacts, reuse and reproduction', '', result['reused_tooling'], '',f"Checkpoint audit: {len(result['checkpoint_audit'].get('checkpoints',[]))} snapshot/variant entries, {result['checkpoint_audit'].get('unique_states','unavailable')} unique tensor states. Inventory is printed before computation and preserved in `artifact_inventory.json`. File hashes pin input predictions, labels/config and metadata; checkpoint tensor hashes pin actual weights. Saved CSV predictions avoid inference recomputation. Cache files are local, offline inputs; no network calls exist in the analysis script.",
        '', 'Missing evidence:', '']
    lines += ['- '+r for r in result['missing_evidence']]
    lines += ['', 'The published branch contains analysis code and aggregate results only. Study-level predictions, error queues, metadata, holdout/split evidence and inference caches remain local and are excluded from the published commit. A fresh clone needs those retained local inputs before reproduction; the command below makes no downloads.',
        '', 'Run the report from the locally retained cached evidence with:', '', '```powershell',
        'python scripts/analyze_training_evidence.py --run results/resnet50_20261013/evidence/run --labels results/resnet50_20261013/evidence/labels/gold74_eval_set.csv --comparators results/resnet50_20261013/evidence/comparators --metadata results/resnet50_20261013/evidence/train_series.csv --manifests results/resnet50_20261013/evidence/expected_manifests --out results/resnet50_20261013',
        '```', '', 'For checkpoint inference on the VM, add `--infer --data data` and use `--run run --labels labels/gold74_eval_set.csv`. Caches resume automatically. Analysis dependencies: numpy, pandas, scipy, scikit-learn, matplotlib; inference also uses the original installed torch/torchvision/timm runtime. Unit tests: `python -m unittest discover -s tests -p test_analyze_resnet50_run.py`.',
        '', 'Plots: `plots/roc.png`, `plots/pr.png`, `plots/reliability.png`, `plots/training.png`, `plots/paired_difference_distributions.png`. CSVs retain full precision; displayed report values keep six decimal places and both interval endpoints. Machine-readable JSON uses null for unavailable/undefined values.']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--comparators', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--manifests', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results/resnet50_20261013'))
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--inference-chunk', type=int, default=48,
                        help='Window batch for newly inferred intermediate states; selected/final diagnostic re-inference uses original batch 12')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--infer-only', action='store_true')
    args = parser.parse_args()
    if args.bootstrap < 2000:
        parser.error('At least 2000 bootstrap resamples are required')
    if args.inference_chunk < 1:
        parser.error('Inference window batch must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    artifacts = inventory(args)
    gold = pd.read_csv(args.labels, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').sort_index()
    if gold.index.has_duplicates or len(gold) != 74:
        raise ValueError('Gold74 must contain exactly 74 unique studies')
    ids, truth = gold.index.tolist(), gold[LABELS].to_numpy(float)
    if not np.isin(truth[np.isfinite(truth)], [0,1]).all():
        raise ValueError('Gold labels must be binary where available')
    split = json.loads((args.run/'training_split.json').read_text())
    if set(ids) != set(split['heldout_gold']) or set(ids) & (set(split['train']) | set(split['validation'])):
        raise ValueError('Gold holdout differs or overlaps training/validation')
    cfg = json.loads((args.run/'training_config.json').read_text())
    if cfg['labels_sha256']['gold74_eval_set.csv'] != digest(args.labels):
        raise ValueError('Gold definition hash differs from training config')
    if args.infer or args.infer_only:
        if args.data is None:
            parser.error('--data is required for inference')
        collect_checkpoints(args, ids, truth)
    if not args.infer_only:
        analyze(args, gold, ids, truth, artifacts)



if __name__ == '__main__':
    main() if len(sys.argv) > 1 else legacy_main()

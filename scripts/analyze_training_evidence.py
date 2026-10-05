"""Analyze public artifacts; no training or checkpoint deserialization."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

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

def aucs(truth, p):
    return np.array([roc_auc_score(truth[:,j],p[:,j]) if len(np.unique(truth[:,j]))==2 else np.nan for j in range(12)])

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

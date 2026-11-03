# ResNet50 fusion training: statistical report

## Verdict

- Exploratory early improvement: epoch_01_raw minus parent_initial: **0.009409 [0.000180, 0.018153]**; higher AUC under this bootstrap. Epoch 1 is the observed Gold peak, identified after training; it was not selected by validation, and this interval is unadjusted for epoch inspection.
- Validation-selected comparisons: epoch_03_raw minus parent_initial: **0.002211 [-0.010376, 0.013846]**; uncertain: interval crosses zero. epoch_03_ema minus parent_initial: **0.002494 [-0.009237, 0.012958]**; uncertain: interval crosses zero.
- Validation-selected comparisons: epoch_03_raw minus existing_cv5_ensemble: **0.000093 [-0.010011, 0.009920]**; uncertain: interval crosses zero. epoch_03_ema minus existing_cv5_ensemble: **0.000376 [-0.008207, 0.009139]**; uncertain: interval crosses zero.
- Fusion in epoch_03_raw: full minus no_fusion: 0.000010 [-0.001038, 0.000959]; full minus rad_mean: 0.000177 [-0.000628, 0.001015].
- Fusion in epoch_03_ema: full minus no_fusion: -0.000092 [-0.001378, 0.001223]; full minus rad_mean: 0.000154 [-0.000789, 0.001235].
- Later-epoch change: epoch_10_raw minus epoch_03_raw: **-0.011049 [-0.020012, -0.002141]**; lower AUC under this bootstrap. epoch_10_ema minus epoch_03_ema: **-0.011015 [-0.020172, -0.001205]**; lower AUC under this bootstrap.
- Fusion contribution beyond sampling noise is **not established**: all 8 selected/final [ablation intervals](#paired-difference-distributions) cross zero. This does not establish zero effect.

The data cannot establish a new-patient or clinical benefit, a controlled architecture gain against historical CV5, or a causal fraction of ResNet prediction signal. Intervals cover study sampling only; historical Gold selection, unavailable patient grouping and holdout-fitted thresholds limit interpretation. Gradient norm, training AUC, scanner/field strength and laterality are unavailable.

## Headline discrimination

| Checkpoint / comparator | Macro ROC-AUC (95% CI) | Valid / total resamples |
|---|---:|---:|
| epoch_03_raw | 0.924824 [0.900362, 0.944714] | 2000 / 2000 |
| epoch_03_ema | 0.925106 [0.900750, 0.946106] | 2000 / 2000 |
| parent_initial | 0.922612 [0.900635, 0.942124] | 2000 / 2000 |
| existing_cv5_ensemble | 0.924730 [0.900259, 0.945342] | 2000 / 2000 |
| epoch_10_raw | 0.913774 [0.891913, 0.933996] | 2000 / 2000 |
| epoch_10_ema | 0.914091 [0.891985, 0.934305] | 2000 / 2000 |
| epoch_01_raw | 0.932021 [0.909867, 0.950962] | 2000 / 2000 |

All checkpoints, per-label support, AUROC, average precision (step-integral PR-AUC, not trapezoidal area), calibration and thresholds are in `statistics.json` and `per_label_metrics.csv`. `aliases` maps every snapshot and embedded EMA to its evaluated tensor state. No model was selected using the new Gold74 analysis.

## Holdout and uncertainty

Orientation HEAD: `d49b3458314ede05079a1b12609ad1326aeec227`. Gold74 IDs and missing-label masks were checked against the persisted training split and label SHA256. Gold74 is disjoint from the 3,900 training and 433 validation studies. Seed 20261014; 2000 unstratified study resamples; identical resamples across all models and paired deltas. A macro resample is dropped if any full-sample evaluable label loses a class, so label composition never changes.

AUROC or AP CI width > 0.20 is flagged as too imprecise for a claim on that label/metric. This explicit heuristic is not a significance test; both widths and separate flags are retained. No multiplicity correction; per-epoch inspection and historical Gold selection can be optimistic. An empirical AUC of 1 may have a degenerate [1,1] bootstrap interval: resampling observed perfectly separated cases cannot invent unseen difficult cases, so that interval does not establish population-perfect performance. Calibration uses 10 equal-width bins; PR curves show prevalence baselines. Empty calibration bins remain empty.

| Label | Labeled studies | Positive | Prevalence (95% CI) |
|---|---:|---:|---:|
| ACL | 74 | 29 | 0.391892 [0.283784, 0.500000] |
| MCL | 74 | 11 | 0.148649 [0.081081, 0.229730] |
| Medial Meniscus | 73 | 35 | 0.479452 [0.364865, 0.594595] |
| Lateral Meniscus | 74 | 30 | 0.405405 [0.297297, 0.527027] |
| Medial OA | 72 | 22 | 0.305556 [0.200000, 0.416667] |
| Lateral OA | 71 | 14 | 0.197183 [0.112676, 0.300000] |
| PF OA | 73 | 31 | 0.424658 [0.315068, 0.540541] |
| Effusion | 74 | 48 | 0.648649 [0.540541, 0.756757] |
| Synovitis | 73 | 36 | 0.493151 [0.380282, 0.611111] |
| Baker's | 67 | 18 | 0.268657 [0.166604, 0.385714] |
| Contusion | 73 | 26 | 0.356164 [0.246575, 0.465753] |
| Fracture | 72 | 22 | 0.305556 [0.205415, 0.410959] |

## Per-label metrics: epoch_03_raw

| Label | Positive / labeled | AUROC (95% CI) | AUROC CI width | Precision flag | AP (95% CI) | AP CI width | Brier (95% CI) | ECE (95% CI) |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| ACL | 29 / 74 | 0.986973 [0.965544, 0.999265] | 0.033721 | within rule | 0.980561 [0.947506, 0.999162] | 0.051656 | 0.095476 [0.064503, 0.134183] | 0.199755 [0.157582, 0.258017] |
| MCL | 11 / 74 | 1.000000 [1.000000, 1.000000] | 0.000000 | within rule | 1.000000 [1.000000, 1.000000] | 0.000000 | 0.073595 [0.060089, 0.088253] | 0.236358 [0.215539, 0.268810] |
| Medial Meniscus | 35 / 73 | 0.970677 [0.930982, 0.996142] | 0.065160 | within rule | 0.967967 [0.920524, 0.996341] | 0.075818 | 0.082250 [0.052032, 0.118749] | 0.115418 [0.097242, 0.185101] |
| Lateral Meniscus | 30 / 74 | 0.912879 [0.843216, 0.967768] | 0.124552 | within rule | 0.885205 [0.787809, 0.959952] | 0.172143 | 0.120871 [0.084718, 0.162145] | 0.108902 [0.096585, 0.201494] |
| Medial OA | 22 / 72 | 0.947273 [0.897104, 0.985003] | 0.087899 | AP wide | 0.886576 [0.761327, 0.969246] | 0.207919 | 0.116344 [0.078878, 0.159409] | 0.156460 [0.134354, 0.240267] |
| Lateral OA | 14 / 71 | 0.845865 [0.688204, 0.960662] | 0.272457 | AUROC wide, AP wide | 0.726467 [0.488362, 0.905681] | 0.417319 | 0.179487 [0.127275, 0.237367] | 0.239632 [0.178962, 0.324633] |
| PF OA | 31 / 73 | 0.869432 [0.767718, 0.951211] | 0.183493 | within rule | 0.872326 [0.761186, 0.954064] | 0.192878 | 0.138209 [0.094257, 0.186367] | 0.102656 [0.085512, 0.206460] |
| Effusion | 48 / 74 | 0.939904 [0.873460, 0.983500] | 0.110040 | within rule | 0.965658 [0.921027, 0.992193] | 0.071166 | 0.175637 [0.120476, 0.237490] | 0.179628 [0.111415, 0.269166] |
| Synovitis | 36 / 73 | 0.819820 [0.708759, 0.910916] | 0.202157 | AUROC wide, AP wide | 0.798891 [0.661568, 0.917684] | 0.256116 | 0.199022 [0.176010, 0.224601] | 0.136519 [0.081031, 0.236539] |
| Baker's | 18 / 67 | 0.985261 [0.958326, 1.000000] | 0.041674 | within rule | 0.959709 [0.872220, 1.000000] | 0.127780 | 0.062738 [0.039638, 0.093149] | 0.163353 [0.146350, 0.215609] |
| Contusion | 26 / 73 | 0.903437 [0.822656, 0.966361] | 0.143705 | AP wide | 0.834721 [0.686550, 0.947686] | 0.261136 | 0.165997 [0.123554, 0.213568] | 0.209288 [0.152670, 0.292482] |
| Fracture | 22 / 72 | 0.916364 [0.842050, 0.975057] | 0.133007 | AP wide | 0.853796 [0.717848, 0.953868] | 0.236021 | 0.137517 [0.103354, 0.174518] | 0.215562 [0.158727, 0.283459] |

| Label | Mean probability (95% CI) | Observed prevalence (95% CI) | Sensitivity at 0.5 (95% CI) | Specificity at 0.5 (95% CI) | Best-F1 threshold (95% CI) | Best F1 (conditional 95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 0.553353 [0.476890, 0.626305] | 0.391892 [0.283784, 0.500000] | 1.000000 [1.000000, 1.000000] | 0.800000 [0.680851, 0.911159] | 0.592212 [0.532221, 0.880111] | 0.935484 [0.857143, 0.985915] |
| MCL | 0.302074 [0.260114, 0.347561] | 0.148649 [0.081081, 0.229730] | 1.000000 [1.000000, 1.000000] | 0.984127 [0.948276, 1.000000] | 0.564163 [0.462280, 0.564163] | 1.000000 [1.000000, 1.000000] |
| Medial Meniscus | 0.542412 [0.461805, 0.622933] | 0.479452 [0.364865, 0.594595] | 0.914286 [0.812458, 1.000000] | 0.868421 [0.750000, 0.970588] | 0.434653 [0.406670, 0.767737] | 0.918919 [0.845055, 0.975000] |
| Lateral Meniscus | 0.411810 [0.347762, 0.484077] | 0.405405 [0.297297, 0.527027] | 0.700000 [0.527741, 0.861135] | 0.931818 [0.846111, 1.000000] | 0.460846 [0.208919, 0.460846] | 0.800000 [0.666667, 0.904110] |
| Medial OA | 0.460154 [0.388664, 0.536517] | 0.305556 [0.200000, 0.416667] | 0.954545 [0.850000, 1.000000] | 0.820000 [0.705828, 0.920000] | 0.553105 [0.449997, 0.860530] | 0.823529 [0.684178, 0.923077] |
| Lateral OA | 0.436815 [0.371139, 0.506548] | 0.197183 [0.112676, 0.300000] | 0.785714 [0.538462, 1.000000] | 0.736842 [0.619048, 0.842105] | 0.825091 [0.438793, 0.968612] | 0.692308 [0.444444, 0.875000] |
| PF OA | 0.461561 [0.392248, 0.538729] | 0.424658 [0.315068, 0.540541] | 0.774194 [0.617647, 0.909091] | 0.809524 [0.681818, 0.921103] | 0.594523 [0.276544, 0.660147] | 0.827586 [0.701754, 0.920000] |
| Effusion | 0.810899 [0.772310, W0.847130] | 0.648649 [0.540541, 0.756757] | 1.000000 [1.000000, 1.000000] | 0.230769 [0.076852, 0.391304] | 0.828620 [0.702333, 0.868983] | 0.909091 [0.842105, 0.961538] |
| Synovitis | 0.448884 [0.420254, 0.479224] | 0.493151 [0.380282, 0.611111] | 0.611111 [0.437396, 0.766667] | 0.837838 [0.709583, 0.944444] | 0.420199 [0.335279, 0.480389] | 0.794872 [0.685695, 0.881755] |
| Baker's | 0.390315 [0.318208, 0.467677] | 0.268657 [0.166604, 0.385714] | 0.944444 [0.812500, 1.000000] | 0.959184 [0.891304, 1.000000] | 0.600339 [0.327112, 0.600339] | 0.944444 [0.846154, 1.000000] |
| Contusion | 0.550284 [0.485098, 0.610069] | 0.356164 [0.246575, 0.465753] | 0.923077 [0.807692, 1.000000] | 0.723404 [0.590842, 0.840000] | 0.717598 [0.565175, 0.816069] | 0.814815 [0.680851, 0.913075] |
| Fracture | 0.318947 [0.282463, 0.358984] | 0.305556 [0.205415, 0.410959] | 0.363636 [0.181818, 0.565217] | 1.000000 [1.000000, 1.000000] | 0.313702 [0.264779, 0.313702] | 0.844444 [0.717949, 0.944444] |

Threshold CIs re-fit maximal F1 inside each study resample on the fixed full-sample grid of observed probabilities plus 0 and an always-negative threshold; smallest threshold breaks ties. Sensitivity, specificity and F1 intervals condition on the reported full-sample threshold; they do not remove holdout threshold-selection optimism. JSON also provides fixed thresholds 0.10/0.25/0.50/0.75/0.90 and empirical operating points at at least 90%/95% specificity.

## Per-label metrics: epoch_03_ema

| Label | Positive / labeled | AUROC (95% CI) | AUROC CI width | Precision flag | AP (95% CI) | AP CI width | Brier (95% CI) | ECE (95% CI) |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| ACL | 29 / 74 | 0.986973 [0.963984, 1.000000] | 0.036016 | within rule | 0.979542 [0.939873, 1.000000] | 0.060127 | 0.081182 [0.054761, 0.114774] | 0.162612 [0.149682, 0.226559] |
| MCL | 11 / 74 | 0.997114 [0.986325, 1.000000] | 0.013675 | within rule | 0.984160 [0.919444, 1.000000] | 0.080556 | 0.080456 [0.064210, 0.099911] | 0.239826 [0.206816, 0.273501] |
| Medial Meniscus | 35 / 73 | 0.968421 [0.926087, 0.995367] | 0.069280 | within rule | 0.965221 [0.914938, 0.995903] | 0.080965 | 0.084602 [0.053275, 0.122894] | 0.117702 [0.099921, 0.187838] |
| Lateral Meniscus | 30 / 74 | 0.907576 [0.834955, 0.962063] | 0.127108 | within rule | 0.879913 [0.780220, 0.956033] | 0.175813 | 0.122920 [0.084925, 0.166446] | 0.104355 [0.094710, 0.200569] |
| Medial OA | 22 / 72 | 0.953636 [0.906122, 0.987579] | 0.081457 | within rule | 0.905551 [0.797228, 0.974806] | 0.177578 | 0.117994 [0.077287, 0.165486] | 0.168746 [0.139195, 0.249872] |
| Lateral OA | 14 / 71 | 0.840852 [0.682005, 0.960622] | 0.278617 | AUROC wide, AP wide | 0.728946 [0.487535, 0.906605] | 0.419070 | 0.170870 [0.117205, 0.229883] | 0.215600 [0.153531, 0.300719] |
| PF OA | 31 / 73 | 0.874040 [0.775807, 0.954553] | 0.178746 | within rule | 0.880685 [0.773724, 0.955599] | 0.181875 | 0.143054 [0.097079, 0.194409] | 0.144447 [0.107236, 0.236401] |
| Effusion | 48 / 74 | 0.955929 [0.900702, 0.991015] | 0.090314 | within rule | 0.974956 [0.937667, 0.995853] | 0.058186 | 0.163143 [0.110550, 0.221979] | 0.193543 [0.123068, 0.277049] |
| Synovitis | 36 / 73 | 0.822072 [0.714695, 0.909110] | 0.194414 | AP wide | 0.809386 [0.678687, 0.920371] | 0.241684 | 0.191088 [0.166438, 0.218983] | 0.129771 [0.093195, 0.238194] |
| Baker's | 18 / 67 | 0.984127 [0.953236, 1.000000] | 0.046764 | within rule | 0.954659 [0.856595, 1.000000] | 0.143405 | 0.066541 [0.041753, 0.098027] | 0.169376 [0.148370, 0.221473] |
| Contusion | 26 / 73 | 0.896890 [0.816516, 0.961455] | 0.144939 | AP wide | 0.815139 [0.658240, 0.939371] | 0.281131 | 0.151072 [0.112280, 0.195114] | 0.151676 [0.119296, 0.247143] |
| Fracture | 22 / 72 | 0.913636 [0.826414, 0.980393] | 0.153979 | AP wide | 0.863982 [0.731203, 0.961139] | 0.229936 | 0.131054 [0.096861, 0.167886] | 0.183283 [0.124515, 0.258606] |

| Label | Mean probability (95% CI) | Observed prevalence (95% CI) | Sensitivity at 0.5 (95% CI) | Specificity at 0.5 (95% CI) | Best-F1 threshold (95% CI) | Best F1 (conditional 95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 0.527696 [0.450616, 0.602148] | 0.391892 [0.283784, 0.500000] | 1.000000 [1.000000, 1.000000] | 0.888889 [0.790698, 0.975000] | 0.606655 [0.445173, 0.830670] | 0.933333 [0.851806, 0.985507] |
| MCL | 0.324295 [0.278098, 0.373336] | 0.148649 [0.081081, 0.229730] | 1.000000 [1.000000, 1.000000] | 0.936508 [0.865672, 0.984848] | 0.624783 [0.510792, 0.645519] | 0.956522 [0.833333, 1.000000] |
| Medial Meniscus | 0.544476 [0.463484, 0.624196] | 0.479452 [0.364865, 0.594595] | 0.942857 [0.848485, 1.000000] | 0.868421 [0.750000, 0.970588] | 0.682404 [0.336279, 0.760116] | 0.914286 [0.835759, 0.974359] |
| Lateral Meniscus | 0.399105 [0.333908, 0.471459] | 0.405405 [0.297297, 0.527027] | 0.700000 [0.527741, 0.861135] | 0.909091 [0.808490, 0.977778] | 0.217795 [0.204613, 0.567990] | 0.800000 [0.688525, 0.891566] |
| Medial OA | 0.474302 [0.400962, 0.552866] | 0.305556 [0.200000, 0.416667] | 1.000000 [1.000000, 1.000000] | 0.800000 [0.685185, 0.903869] | 0.626447 [0.449590, 0.851747] | 0.823529 [0.684178, 0.923077] |
| Lateral OA | 0.412783 [0.344533, 0.486001] | 0.197183 [0.112676, 0.300000] | 0.714286 [0.454545, 0.933333] | 0.771930 [0.666532, 0.870385] | 0.832621 [0.378431, 0.965159] | 0.692308 [0.444444, 0.875000] |
| PF OA | 0.495206 [0.423529, 0.573108] | 0.424658 [0.315068, 0.540541] | 0.838710 [0.700000, 0.965546] | 0.714286 [0.577727, 0.846154] | 0.680960 [0.325548, 0.705550] | 0.827586 [0.701754, 0.920000] |
| Effusion | 0.808540 [0.765672, 0.849447] | 0.648649 [0.540541, 0.756757] | 1.000000 [1.000000, 1.000000] | 0.230769 [0.076852, 0.391304] | 0.835541 [0.700108, 0.859363] | 0.927835 [0.865965, 0.973913] |
| Synovitis | 0.458261 [0.425711, 0.493352] | 0.493151 [0.380282, 0.611111] | 0.666667 [0.500000, 0.818240] | 0.837838 [0.709583, 0.944444] | 0.403637 [0.334367, 0.505929] | 0.771084 [0.658220, 0.863158] |
| Baker's | 0.402848 [0.329470, 0.480564] | 0.268657 [0.166604, 0.385714] | 0.944444 [0.812500, 1.000000] | 0.918367 [0.836288, 0.980769] | 0.635972 [0.359009, 0.635972] | 0.944444 [0.846154, 1.000000] |
| Contusion | 0.507840 [0.441470, 0.569044] | 0.356164 [0.246575, 0.465753] | 0.923077 [0.807692, 1.000000] | 0.723404 [0.590842, 0.840000] | 0.550600 [0.462394, 0.799698] | 0.793103 [0.666667, 0.892308] |
| Fracture | 0.317211 [0.277312, 0.360386] | 0.305556 [0.205415, 0.410959] | 0.409091 [0.214286, 0.625000] | 1.000000 [1.000000, 1.000000] | 0.291301 [0.274944, 0.333959] | 0.851064 [0.731707, 0.944470] |

Threshold CIs re-fit maximal F1 inside each study resample on the fixed full-sample grid of observed probabilities plus 0 and an always-negative threshold; smallest threshold breaks ties. Sensitivity, specificity and F1 intervals condition on the reported full-sample threshold; they do not remove holdout threshold-selection optimism. JSON also provides fixed thresholds 0.10/0.25/0.50/0.75/0.90 and empirical operating points at at least 90%/95% specificity.

## Paired difference distributions

| Comparison | Macro-AUC difference (95% CI) | Bootstrap mean | SD | Median | Fraction <= 0 |
|---|---:|---:|---:|---:|---:|
| epoch_03_ema - epoch_03_ema__no_fusion | -0.000092 [-0.001378, 0.001223] | -0.000098 | 0.000651 | -0.000110 | 0.575000 |
| epoch_03_ema - epoch_03_ema__rad_mean | 0.000154 [-0.000789, 0.001235] | 0.000161 | 0.000505 | 0.000136 | 0.374500 |
| epoch_03_ema - epoch_03_raw | 0.000282 [-0.004823, 0.005246] | 0.000303 | 0.002566 | 0.000373 | 0.449000 |
| epoch_03_ema - existing_cv5_ensemble | 0.000376 [-0.008207, 0.009139] | 0.000303 | 0.004514 | 0.000232 | 0.479500 |
| epoch_03_ema - parent_initial | 0.002494 [-0.009237, 0.012958] | 0.002234 | 0.005606 | 0.002280 | 0.337500 |
| epoch_03_raw - epoch_03_raw__no_fusion | 0.000010 [-0.001038, 0.000959] | -0.000014 | 0.000490 | -0.000015 | 0.515000 |
| epoch_03_raw - epoch_03_raw__rad_mean | 0.000177 [-0.000628, 0.001015] | 0.000152 | 0.000408 | 0.000134 | 0.360000 |
| epoch_03_raw - existing_cv5_ensemble | 0.000093 [-0.010011, 0.009920] | -0.000000 | 0.005107 | -0.000011 | 0.501500 |
| epoch_03_raw - parent_initial | 0.002211 [-0.010376, 0.013846] | 0.001931 | 0.006138 | 0.001955 | 0.374500 |
| epoch_10_ema - epoch_03_ema | -0.011015 [-0.020172, -0.001205] | -0.010869 | 0.004788 | -0.010869 | 0.985000 |
| epoch_10_ema - epoch_10_ema__no_fusion | -0.000925 [-0.003969, 0.001957] | -0.000911 | 0.001452 | -0.000878 | 0.743500 |
| epoch_10_ema - epoch_10_ema__rad_mean | -0.000069 [-0.002692, 0.002617] | -0.000035 | 0.001359 | -0.000036 | 0.511000 |
| epoch_10_raw - epoch_03_raw | -0.011049 [-0.020012, -0.002141] | -0.010887 | 0.004506 | -0.010896 | 0.989500 |
| epoch_10_raw - epoch_10_raw__no_fusion | -0.001173 [-0.004233, 0.001651] | -0.001153 | 0.001487 | -0.001079 | 0.777000 |
| epoch_10_raw - epoch_10_raw__rad_mean | -0.000163 [-0.002840, 0.002694] | -0.000139 | 0.001426 | -0.000138 | 0.540500 |

Exact macro difference draws are in `paired_macro_auc_distributions.csv` and JSON; JSON also includes quantiles and histogram counts/edges. The fraction <= 0 is a description of resampled differences, not a calibrated p-value. `paired_per_label_deltas.csv` includes positive counts, prevalence and per-label CIs for every contrast. Comparators use identical study IDs; CV5 is the equal probability mean of the five saved full fold outputs, not a newly fitted ensemble.

## Training dynamics and best-checkpoint audit

| Epoch | Train BCE | Raw val BCE | EMA val BCE | Raw val AUC | EMA val AUC | Raw Gold AUC (95% CI) | EMA Gold AUC (95% CI) | EMA - raw Gold (95% CI) | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.48369576 | 0.34839812 | 0.34663680 | 0.92546949 | 0.92445349 | 0.932021 [0.909867, 0.950962] | 0.929746 [0.907348, 0.948504] | -0.002275 [-0.006982, 0.001843] | 5692.0 |
| 2 | 0.47011421 | 0.37795216 | 0.34835362 | 0.92323821 | 0.92440403 | 0.928265 [0.905472, 0.947901] | 0.927671 [0.903611, 0.947632] | -0.000594 [-0.005845, 0.004643] | 5691.0 |
| 3 | 0.45965627 | 0.35014644 | 0.34290066 | 0.92624561 | 0.92571519 | 0.924824 [0.900362, 0.944714] | 0.925106 [0.900750, 0.946106] | 0.000282 [-0.004823, 0.005246] | 5699.0 |
| 4 | 0.45129543 | 0.35064393 | 0.34265813 | 0.92486673 | 0.92520098 | 0.926322 [0.903420, 0.945810] | 0.925610 [0.902709, 0.944884] | -0.000712 [-0.005126, 0.003206] | 5691.0 |
| 5 | 0.44203527 | 0.35596225 | 0.34162861 | 0.92297052 | 0.92520994 | 0.919761 [0.895311, 0.940501] | 0.921794 [0.899239, 0.941469] | 0.002033 [-0.002087, 0.006338] | 5700.0 |
| 6 | 0.43308368 | 0.34246692 | 0.34046063 | 0.92240577 | 0.92433234 | 0.920768 [0.898491, 0.940433] | 0.920500 [0.898257, 0.939921] | -0.000268 [-0.005060, 0.004270] | 5686.0 |
| 7 | 0.42669483 | 0.34374997 | 0.34112215 | 0.92509539 | 0.92385824 | 0.917255 [0.895432, 0.937301] | 0.918329 [0.895987, 0.938321] | 0.001074 [-0.002373, 0.004547] | 5703.0 |
| 8 | 0.41981786 | 0.33427241 | 0.34181085 | 0.92354731 | 0.92409575 | 0.917255 [0.896044, 0.936791] | 0.916882 [0.895547, 0.936549] | -0.000373 [-0.003183, 0.002493] | 5695.0 |
| 9 | 0.41506785 | 0.33972788 | 0.34213701 | 0.92336706 | 0.92358437 | 0.914918 [0.892837, 0.934963] | 0.914947 [0.893052, 0.934856] | 0.000029 [-0.002182, 0.002186] | 5700.0 |
| 10 | 0.41407061 | 0.34217358 | 0.34291816 | 0.92310129 | 0.92334382 | 0.913774 [0.891913, 0.933996] | 0.914091 [0.891985, 0.934305] | 0.000316 [-0.000972, 0.001540] | 5711.0 |

Training BCE uses soft HYBRID scores, intensity augmentation and 24 sampled windows; validation BCE uses binary HYBRID labels and all windows; Gold BCE uses available structured Gold labels and all windows. These losses use different targets and protocols. Only the raw optimizer model has a logged training loss; EMA has separate evaluation metrics. Train/validation losses and validation AUC above are logged measurements, not inferential estimates; per-study train loss was not saved, so no loss CI is fabricated. Endpoint LR is read from raw checkpoint scheduler state in `training_dynamics.csv`. Wall time is UTC epoch start to completion, including evaluation and checkpoint writes. Gradient-norm and training-AUC columns are null because these values were not persisted.

Validation-selected epochs: raw **3**, EMA **3**. Descriptive Gold peaks: raw 1, EMA 1 (not new selections). All 40 saved epoch prediction AUCs match logged metrics: **True**.
- Best raw: tensor-identical to selected epoch 3: **True**; validation selection AUC 0.9262456078; tensor SHA256 `1613955116825dd771bcf2dbf0f1fd2370f3c8db92f5e50b6c907b4308fe0e2e`. This proves weight/selection correspondence rather than trusting a filename.
- Best ema: tensor-identical to selected epoch 3: **True**; validation selection AUC 0.9257151929; tensor SHA256 `e9b05632358a930d3bb4e8ac9ed0f053c11b6d5c8b8f776183b48665d2213ea8`. This proves weight/selection correspondence rather than trusting a filename.
- Initial baseline verified against the actual parent checkpoint: **True**. Every CoAtNet/head tensor is identical and both fusion up-projection weights/biases are exactly zero. Parent file SHA256 `20d45e0732a0e9120025d0172a4d2dd0170900bc250b06742dfb7ae08a63786a`.
- Independent re-inference of `epoch_03_raw_gold74.npz`: maximum probability difference 0.00491592288; per-label saved/recomputed AUCs recorded in `1613955116825dd771bcf2dbf0f1fd2370f3c8db92f5e50b6c907b4308fe0e2e_parity.json`. Deterministic cuDNN settings differ from the trainer's benchmark-enabled settings, so BF16 probabilities can differ slightly despite identical weights. Headline/ablation metrics for re-inferred snapshots use the deterministic full cache; logged training trajectories retain original evaluations. This numerical difference is not evidence of mismatched checkpoint weights.
- Independent re-inference of `epoch_10_raw_gold74.npz`: maximum probability difference 0.006865218282; per-label saved/recomputed AUCs recorded in `224500cef17428576f965875f3c4bc3513895d0cc37b2a4cd4b8970de387e5bc_parity.json`. Deterministic cuDNN settings differ from the trainer's benchmark-enabled settings, so BF16 probabilities can differ slightly despite identical weights. Headline/ablation metrics for re-inferred snapshots use the deterministic full cache; logged training trajectories retain original evaluations. This numerical difference is not evidence of mismatched checkpoint weights.
- Independent re-inference of `epoch_10_ema_gold74.npz`: maximum probability difference 0.006214231253; per-label saved/recomputed AUCs recorded in `a1595dec5581aff447b282de8f1cc3e6bf62c0cf5cbb166849979a1ae780682a_parity.json`. Deterministic cuDNN settings differ from the trainer's benchmark-enabled settings, so BF16 probabilities can differ slightly despite identical weights. Headline/ablation metrics for re-inferred snapshots use the deterministic full cache; logged training trajectories retain original evaluations. This numerical difference is not evidence of mismatched checkpoint weights.
- Independent re-inference of `epoch_03_ema_gold74.npz`: maximum probability difference 0.003077745438; per-label saved/recomputed AUCs recorded in `e9b05632358a930d3bb4e8ac9ed0f053c11b6d5c8b8f776183b48665d2213ea8_parity.json`. Deterministic cuDNN settings differ from the trainer's benchmark-enabled settings, so BF16 probabilities can differ slightly despite identical weights. Headline/ablation metrics for re-inferred snapshots use the deterministic full cache; logged training trajectories retain original evaluations. This numerical difference is not evidence of mismatched checkpoint weights.

## Fusion diagnostics and ablations

Full, no_fusion (both residual corrections forced to zero), and rad_mean (checkpoint-specific window-weighted mean ResNet feature maps from Gold74, labels unused) retain the same checkpoint weights and study windows. rad_mean is a transductive, exploratory control, not a deployment baseline. Ablations cover validation-selected raw/EMA and final raw/EMA. All available states receive full inference or reuse their saved full evaluation; aliases avoid repeated inference of identical tensors.

| Checkpoint | Layer | Residual norm / stream norm, mean (95% CI) | Median [p10, p90] | ResNet down-projection norm fraction, mean (95% CI) | Gate mean / SD / p01 / p99 |
|---|---|---:|---:|---:|---:|
| epoch_03_ema | f1 | 0.004903 [0.004886, 0.004923] | 0.004889 [0.004715, 0.005112] | 0.694617 [0.693852, 0.695341] | 0.434633 / 0.098930 / 0.226562 / 0.656250 |
| epoch_03_ema | f2 | 0.003869 [0.003785, 0.003954] | 0.003802 [0.002993, 0.004835] | 0.577952 [0.576125, 0.579854] | 0.495057 / 0.040112 / 0.398438 / 0.589844 |
| epoch_03_raw | f1 | 0.005265 [0.005245, 0.005290] | 0.005285 [0.004984, 0.005514] | 0.707029 [0.706288, 0.707729] | 0.426817 / 0.106282 / 0.214844 / 0.664062 |
| epoch_03_raw | f2 | 0.003273 [0.003197, 0.003348] | 0.003189 [0.002473, 0.004177] | 0.579423 [0.577623, 0.581426] | 0.494425 / 0.039977 / 0.398438 / 0.585938 |
| epoch_10_raw | f1 | 0.013242 [0.013174, 0.013316] | 0.013308 [0.012416, 0.013931] | 0.698998 [0.698270, 0.699695] | 0.507005 / 0.106897 / 0.255859 / 0.722656 |
| epoch_10_raw | f2 | 0.003718 [0.003558, 0.003872] | 0.003420 [0.002544, 0.005344] | 0.580379 [0.576819, 0.584112] | 0.494658 / 0.060971 / 0.351562 / 0.636719 |
| epoch_10_ema | f1 | 0.012812 [0.012748, 0.012884] | 0.012874 [0.012027, 0.013472] | 0.698923 [0.698197, 0.699616] | 0.502942 / 0.104263 / 0.255859 / 0.714844 |
| epoch_10_ema | f2 | 0.003586 [0.003428, 0.003737] | 0.003297 [0.002447, 0.005163] | 0.580239 [0.576688, 0.584001] | 0.494333 / 0.059603 / 0.353516 / 0.632812 |

Residual norm is measured before adding the correction to the CoAtNet stream (S2 Z, S4 V). CIs resample studies while retaining their windows. Gate distributions are measured activation summaries. Branch fraction is the ratio of additive ResNet versus CoAtNet down-projection norms before nonlinear mixing, excluding bias; it is **not an identifiable causal percentage of final model signal**. Functional ablations supply the evidence about contribution to discrimination.

## Error review and labels that fail together

`error_queue.csv` contains the top 10 actual false positives and false negatives per model/label at threshold 0.5, ranked by contradiction, with study IDs and full-precision probabilities. An empty queue means no such threshold error, not a fabricated example. Residuals are probability minus observed label; correlations use pairwise available labels. The table is descriptive with bootstrap CIs; it does not establish a shared cause.

| Model | Labels | Labeled pairs | Residual correlation (95% CI) | Both threshold decisions wrong |
|---|---|---:|---:|---:|
| epoch_03_raw | Effusion / Synovitis | 73 | 0.258700 [0.032698, 0.481982] | 3 |
| epoch_03_raw | MCL / Lateral OA | 71 | 0.249014 [0.074525, 0.441138] | 0 |
| epoch_03_raw | Synovitis / Contusion | 72 | 0.218539 [0.000063, 0.420442] | 5 |
| epoch_03_raw | Lateral Meniscus / Medial OA | 72 | 0.214067 [0.036815, 0.373576] | 0 |
| epoch_03_raw | ACL / Contusion | 73 | 0.212696 [-0.012496, 0.459981] | 4 |
| epoch_03_ema | MCL / Lateral OA | 71 | 0.251953 [0.071391, 0.448505] | 1 |
| epoch_03_ema | ACL / Contusion | 73 | 0.235386 [-0.003206, 0.496032] | 2 |
| epoch_03_ema | Medial OA / Baker's | 66 | 0.230457 [-0.013066, 0.434546] | 1 |
| epoch_03_ema | Effusion / Synovitis | 73 | 0.225431 [-0.001881, 0.452465] | 2 |
| epoch_03_ema | MCL / Medial Meniscus | 73 | -0.219113 [-0.498374, 0.072455] | 0 |
| parent_initial | ACL / Contusion | 73 | 0.297887 [0.046705, 0.548778] | 5 |
| parent_initial | Medial OA / Baker's | 66 | 0.255414 [0.014132, 0.474121] | 1 |
| parent_initial | Lateral Meniscus / Medial OA | 72 | 0.243286 [0.046135, 0.424924] | 4 |
| parent_initial | ACL / Lateral Meniscus | 74 | -0.211033 [-0.436117, 0.012248] | 1 |
| parent_initial | Lateral OA / Baker's | 65 | 0.186582 [-0.017622, 0.385129] | 1 |
| existing_cv5_ensemble | ACL / Contusion | 73 | 0.267421 [0.003078, 0.527905] | 2 |
| existing_cv5_ensemble | Medial OA / Baker's | 66 | 0.231320 [-0.026740, 0.463142] | 1 |
| existing_cv5_ensemble | Lateral Meniscus / Medial OA | 72 | 0.222213 [0.029781, 0.392725] | 0 |
| existing_cv5_ensemble | MCL / Medial Meniscus | 73 | -0.217303 [-0.498173, 0.113210] | 1 |
| existing_cv5_ensemble | ACL / Lateral OA | 71 | 0.217044 [-0.006976, 0.453162] | 2 |

The pairs below have the largest observed joint error rates at 0.5. These measure failure together, separately from signed residual correlation; the ordering is exploratory.

| Model | Labels | Both wrong / labeled pairs | Joint error rate (95% CI) |
|---|---|---:|---:|
| epoch_03_raw | Lateral OA / Synovitis | 7 / 70 | 0.100000 [0.029851, 0.176511] |
| epoch_03_raw | Lateral OA / Contusion | 7 / 70 | 0.100000 [0.029412, 0.178082] |
| epoch_03_raw | Synovitis / Fracture | 6 / 71 | 0.084507 [0.028169, 0.157143] |
| epoch_03_raw | Synovitis / Contusion | 5 / 72 | 0.069444 [0.014085, 0.136986] |
| epoch_03_raw | Contusion / Fracture | 5 / 72 | 0.069444 [0.014080, 0.126806] |
| epoch_03_ema | Lateral OA / Contusion | 6 / 70 | 0.085714 [0.028169, 0.159420] |
| epoch_03_ema | Lateral OA / Synovitis | 5 / 70 | 0.071429 [0.014286, 0.138938] |
| epoch_03_ema | Lateral Meniscus / Lateral OA | 5 / 71 | 0.070423 [0.014085, 0.136986] |
| epoch_03_ema | Medial OA / Lateral OA | 5 / 71 | 0.070423 [0.014085, 0.132353] |
| epoch_03_ema | Synovitis / Contusion | 5 / 72 | 0.069444 [0.014085, 0.136986] |
| parent_initial | MCL / Contusion | 12 / 73 | 0.164384 [0.082192, 0.256757] |
| parent_initial | Lateral OA / Contusion | 10 / 70 | 0.142857 [0.071403, 0.231884] |
| parent_initial | Synovitis / Contusion | 10 / 72 | 0.138889 [0.068493, 0.222222] |
| parent_initial | MCL / Lateral Meniscus | 10 / 74 | 0.135135 [0.067568, 0.216216] |
| parent_initial | MCL / Lateral OA | 9 / 71 | 0.126761 [0.055556, 0.211268] |
| existing_cv5_ensemble | Synovitis / Fracture | 8 / 71 | 0.112676 [0.042857, 0.191781] |
| existing_cv5_ensemble | Lateral OA / Synovitis | 7 / 70 | 0.100000 [0.029851, 0.176511] |
| existing_cv5_ensemble | Contusion / Fracture | 6 / 72 | 0.083333 [0.027397, 0.143002] |
| existing_cv5_ensemble | Lateral OA / PF OA | 5 / 71 | 0.070423 [0.014085, 0.136986] |
| existing_cv5_ensemble | Synovitis / Contusion | 5 / 72 | 0.069444 [0.013889, 0.128618] |

## Subgroups

Groups below 20 studies are skipped. Within retained groups, a label needs at least five positives and five negatives. Macro averages therefore may use different labels and must not be compared across groups as if composition were identical. All label counts and CIs are in `subgroups.json`. Bins were declared in the script; no cutpoint search was performed. Source slices count exported selected slices; native slice count sums native depths of distinct selected series; pixel spacing is the median of native row/column spacing components across selected source slices. Nested audit metadata were inspected in every Gold tensor; scanner and laterality were not inferred from pixels.

| Group | Studies | Status / evaluable labels | Selected raw macro AUC (95% CI) |
|---|---:|---|---:|
| windows<60 | 1 | Fewer than 20 studies; no metric interpreted. | skipped |
| windows>=60 | 73 | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.924628 [0.900313, 0.944726] |
| source_slices<150 | 41 | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.925036 [0.882600, 0.956561] |
| source_slices>=150 | 33 | ACL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.904104 [0.860394, 0.937912] |
| populated_slots<5 | 30 | ACL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.917708 [0.868478, 0.957328] |
| populated_slots>=5 | 44 | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.933345 [0.906028, 0.954569] |
| slot_0_present=True | 71 | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.928875 [0.905009, 0.948555] |
| slot_0_present=False | 3 | Fewer than 20 studies; no metric interpreted. | skipped |
| slot_1_present=False | 25 | ACL, Medial Meniscus, Lateral Meniscus, Medial OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.936577 [0.896084, 0.970126] |
| slot_1_present=True | 49 | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.912449 [0.881179, 0.939854] |
| slot_2_present=True | 71 | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.932833 [0.910418, 0.950449] |
| slot_2_present=False | 3 | Fewer than 20 studies; no metric interpreted. | skipped |
| slot_3_present=False | 28 | ACL, Medial Meniscus, Lateral Meniscus, Medial OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.918939 [0.873210, 0.957409] |
| slot_3_present=True | 46 | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.933742 [0.908404, 0.953578] |
| slot_4_present=True | 74 | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.924824 [0.900362, 0.944714] |
| slot_5_present=True | 33 | ACL, Medial Meniscus, Lateral Meniscus, Medial OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.937526 [0.909987, 0.961905] |
| slot_5_present=False | 41 | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.908538 [0.873585, 0.938335] |
| series_count<5 | 30 | ACL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.917708 [0.868478, 0.957328] |
| series_count>=5 | 44 | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.933345 [0.906028, 0.954569] |
| native_slice_count<200 | 67 | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.934831 [0.911755, 0.953120] |
| native_slice_count>=200 | 7 | Fewer than 20 studies; no metric interpreted. | skipped |
| median_native_pixel_spacing_mm<0.5 | 66 | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.923000 [0.895872, 0.945415] |
| median_native_pixel_spacing_mm>=0.5 | 8 | Fewer than 20 studies; no metric interpreted. | skipped |
| label_source=manual_verified_r2 | 7 | Fewer than 20 studies; no metric interpreted. | skipped |
| label_source=manual_verified | 9 | Fewer than 20 studies; no metric interpreted. | skipped |
| label_source=original_gold58 | 58 | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | 0.927452 [0.899138, 0.950143] |

Unavailable metadata: Laterality, MagneticFieldStrength, Manufacturer, Scanner. These subgroups were not estimated.

## Artifacts, reuse and reproduction

Extended analyze_training_evidence.py in place: reused its AUROC helper (generalized missing-label masks), study-ID/label alignment, sklearn metric definitions, paired study-resampling design and training-curve approach. Fixed-label macro-bootstrap and 10-bin calibration conventions were also reviewed in the existing analyze_fusion_posttrain.py in the parent worktree. Added weighted bootstrap acceleration, caches, ablations, activation hooks and subgroup/error outputs. Inference directly reuses load_fused_checkpoint, StudyDataset/discover and bag_logits from the actual model/trainer. The legacy no-argument entry point remains import-safe.

Checkpoint audit: 102 snapshot/variant entries, 69 unique tensor states. Inventory is printed before computation and preserved in `artifact_inventory.json`. File hashes pin input predictions, labels/config and metadata; checkpoint tensor hashes pin actual weights. Saved CSV predictions avoid inference recomputation. Cache files are local, offline inputs; no network calls exist in the analysis script.

Missing evidence:

- Per-epoch gradient norms and training AUC not persisted
- Scanner, field strength and laterality unavailable in supplied metadata

The published branch contains analysis code and aggregate results only. Study-level predictions, error queues, metadata, holdout/split evidence and inference caches remain local and are excluded from the published commit. A fresh clone needs those retained local inputs before reproduction; the command below makes no downloads.

Run the report from the locally retained cached evidence with:

```powershell
python scripts/analyze_training_evidence.py --run results/resnet50_20261013/evidence/run --labels results/resnet50_20261013/evidence/labels/gold74_eval_set.csv --comparators results/resnet50_20261013/evidence/comparators --metadata results/resnet50_20261013/evidence/train_series.csv --manifests results/resnet50_20261013/evidence/expected_manifests --out results/resnet50_20261013
```

For checkpoint inference on the VM, add `--infer --data data` and use `--run run --labels labels/gold74_eval_set.csv`. Caches resume automatically. Analysis dependencies: numpy, pandas, scipy, scikit-learn, matplotlib; inference also uses the original installed torch/torchvision/timm runtime. Unit tests: `python -m unittest discover -s tests -p test_analyze_resnet50_run.py`.

Plots: `plots/roc.png`, `plots/pr.png`, `plots/reliability.png`, `plots/training.png`, `plots/paired_difference_distributions.png`. CSVs retain full precision; displayed report values keep six decimal places and both interval endpoints. Machine-readable JSON uses null for unavailable/undefined values.

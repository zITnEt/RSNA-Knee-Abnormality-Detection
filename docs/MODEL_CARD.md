# Model card

## Intended use

Research on study-level, multi-label knee MRI classification and review of the author's engineering work. Outputs cover ACL, MCL, medial/lateral meniscus, medial/lateral/patellofemoral osteoarthritis, effusion, synovitis, Baker's cyst, contusion, and fracture. The models are not validated for diagnosis, screening, treatment decisions, or clinical deployment.

## Models and assets

The repository contains three source variants: CoAtNet with RadImageNet residual fusion, CoAtNet with gated ImageNet ResNet50 fusion, and CoAtNet only. External pretrained parents and checkpoints are required for meaningful inference. No trained model weights are distributed in this publication.

The original public baseline uses a broader ensemble. Its historical 0.943 anchor and the component diagnostics in `results/training_analysis` describe upstream artifacts; they do not validate the custom model as a replacement for the complete ensemble.

## Recorded gated ResNet50 evaluation

- Task: 12 study-level binary findings, with missing evaluation labels masked.
- Recorded split: 3,900 training studies, 433 validation studies, and Gold74 separated from this fine-tuning split.
- Selection: raw and EMA checkpoints chosen by validation macro ROC-AUC, both at epoch 3.
- Gold74 estimate for selected EMA: 0.925106 macro ROC-AUC; 95% study-bootstrap interval 0.900750–0.946106.
- Parent comparator: 0.922612; paired selected-EMA minus parent interval −0.009237 to +0.012958.
- Bootstrap: 2,000 unstratified study resamples, shared across paired comparisons, seed 20261014; label composition is retained in valid macro resamples.
- The exact interval and metric tables are in the [recorded report](../results/resnet50_20261013/REPORT.md).

## Limitations

Gold74 is small, label support varies, and some findings have wide intervals. Studies lack a demonstrated patient-level grouping. Parent pretraining may overlap later fine-tuning validation studies, making those validation results optimistic. Historical Gold inspection and thresholds fitted on the holdout create selection effects that study-bootstrap intervals do not remove. No multiplicity correction establishes a new architecture benefit.

The report's epoch-1 peak is exploratory and was not the validation-selected checkpoint. Ablation intervals cross zero, so useful fusion contribution is not established. Study-level discrimination does not prove calibration, generalization to a new hospital/scanner/population, clinical utility, or patient-level safety. Scanner/field strength and laterality were unavailable in the recorded analysis.

Full reports include references to private/raw evaluation products used during research. Those per-study products are intentionally absent from this public snapshot; selected aggregate tables and figures remain available.

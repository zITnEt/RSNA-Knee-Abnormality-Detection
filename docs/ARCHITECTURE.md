# Architecture and experiment boundaries

## Study representation

The model input is a study bag of three-channel slice windows, typically shaped `[B, K, 3, 384, 384]`. Each window uses three sampled slices. K can vary by study; training and evaluation sampling settings are part of the experiment configuration.

The preprocessing families serve different purposes:

| Recipe | Where to inspect it | Important distinction |
| --- | --- | --- |
| Historical six-slot Raptor export | `scripts/preprocess_six_slot_raptor.py` | Preserves its documented concatenation/boundary-window behavior. |
| Series-safe 384 | `scripts/preprocess_series_safe_384.py` | Forms windows within the source MRI series. |
| Minimum-overlap 384 | `scripts/preprocess_min_overlap_384.py` | Uses series-aware, minimum-overlap sampling and explicit capacity rules. |
| Raw series-aware sampling | `raptor_data/series_triplets.py` | Allocates windows to series; can repeat complete safe windows when capacity is insufficient. |

Do not substitute recipes silently. Input normalization, crop geometry, slice allocation, pretrained parent, and checkpoint format must agree with the run configuration.

## Model variants

**Root: RadImageNet residual MLP fusion.** `models/raptor_v5_radimagenet.py` inserts feature fusion at CoAtNet stages S2/S4 using RadImageNet ResNet50 feature maps. The model preserves the parent CoAtNet attention-MIL head and zero-initializes fusion output projections. Later run configurations support a different RadImageNet input convention; inspect the checkpoint configuration.

**ResNet50: gated bottleneck fusion.** `experiments/resnet50_fusion/models/raptor_v5_resnet50.py` uses ImageNet ResNet50 with a smaller gated residual fusion design. Both backbones consume the shared triplet. Its training/inference tools and checkpoint format belong to this directory.

**CoAtNet only.** `experiments/coatnet_only/models/raptor_v5.py` removes the extra encoder and fusion blocks, retaining the CoAtNet trunk and per-finding attention-MIL head. The migration tool strictly extracts supported CoAtNet/head tensors from original or compatible fusion checkpoints.

## Training and evaluation

Study-level window encoding can be chunked to control activation memory. Gradient checkpointing trades recomputation for memory; BatchNorm handling is tested because recomputation can otherwise update running statistics twice. DDP, mixed precision, accumulation, and EMA are implemented in the relevant trainer variants.

Resumable checkpoints retain the optimizer, scheduler, scaler where applicable, RNG state, split, configuration, training cursor, and raw/EMA state. Evaluation masks missing labels and distinguishes single-class findings from evaluable ROC-AUC results. The model produces 12 logits; sigmoid scores are exported only after enforcing the inference/submission contract.

Operational launchers are historical examples. Configure your own paths, cloud identifiers, storage, and credentials before using them. Source snapshot provenance is recorded in `manifests/SOURCE_SNAPSHOT.json`.

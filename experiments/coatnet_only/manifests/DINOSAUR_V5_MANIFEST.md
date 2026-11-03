# DINOsaur V5 reproduction manifest

Audit date: 2026-09-27 (Asia/Tashkent)

## Source identity

- Source notebook: [RSNA Knee | DINOsaur V5? 🦖](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5?scriptVersionId=351856876)
- Author: Roman Tamrazov (`romantamrazov`)
- Exact audited version: V10 of 20; Kaggle script version ID `351856876`
- V10 run timestamp: 2026-09-22 12:26–12:46 UTC
- V10 source SHA-256: `271c6188b9d2dcd5a231631e82f3b2d241d7c864586a5e1fe1c743389fd5060f`
- Reported Public LB on the historical page: `0.943`
- Accelerator/runtime: 2 × NVIDIA T4; 20m12s public/example-data run
- Internet: disabled
- Competition: `rsna-knee-abnormality-detection` (competition source ID `154281`)
- Competition input path: `/kaggle/input/competitions/rsna-knee-abnormality-detection`
- Competition input size displayed by Kaggle: 576.42 GB; it must stay on Kaggle.

Important version finding: the notebook has continued changing after the original brief was written. V10 remains the requested historical object, but the current notebook has newer versions. The V10 execution output says its optional ConvNeXt arm was skipped because `dinov5_convnext_manifest.json` was not found, so `submission.csv` remained the exact parent 0.943 anchor. Exact V10 reproduction must preserve that fail-closed behavior; attaching a replacement ConvNeXt bundle would change predictions.

## Reproduction-critical inputs

Availability was checked through the authenticated Kaggle API and the historical V10 Inputs page. Dataset sizes below are metadata-only sums; no model files were downloaded locally. `Version` is the immutable Kaggle source ID recorded in the V10 notebook metadata.

### Official competition data

Name: RSNA Knee Abnormality Detection  
Owner: Radiological Society of North America / Kaggle  
Kaggle slug: `rsna-knee-abnormality-detection`  
Version: competition source ID `154281`  
Expected path: `/kaggle/input/competitions/rsna-knee-abnormality-detection`  
Approximate size: 576.42 GB  
Publicly accessible: YES (competition rules accepted)  
Available now: YES  
Required for exact reproduction: YES

### DINOv2 pretrained model

Name: DINO v2, small, PyTorch  
Owner: Meta Research  
Kaggle slug: `metaresearch/dinov2/PyTorch/small/1`  
Version: model instance version ID `4533` / V1  
Expected path: `/kaggle/input/models/metaresearch/dinov2/pytorch/small/1`  
Approximate size: about 85 MB  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES

### DINOv2 fine-tuned ensemble weights

Name: RSNA Knee weights  
Owner: Pilkwang  
Kaggle slug: `pilkwang/rsna-knee-weights`  
Version: dataset version ID `18716507`  
Expected path: `/kaggle/input/datasets/pilkwang/rsna-knee-weights`  
Approximate size: 1.66 GiB (23 files)  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES  
Notes: `manifest.json` must contain exactly 20 members; V10 fingerprint-validated all 20. It also looks for `rsna_20260807_v1.pt` when present.

### A5 five-fold weights

Name: knee mri fold weights  
Owner: Mattia Angeli  
Kaggle slug: `mattiaangeli/knee-mri-fold-weights`  
Version: dataset version ID `18229736`  
Expected path: `/kaggle/input/datasets/mattiaangeli/knee-mri-fold-weights`  
Approximate size: 451 MiB (6 files)  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES  
Notes: requires `m_f0.pt` through `m_f4.pt`; V10 strict-loaded all five.

### Raptor checkpoints

Name: Knee MRI - Max-Span Dense Corpus  
Owner: Dread Development  
Kaggle slug: `dreaddevelopment/raptor-knee-maxspan`  
Version: dataset version ID `19120128`  
Expected path: `/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan`  
Approximate size: 279 MiB  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES  
Checkpoint: `raptor_ft_coatnet_v5_full_swa.pt`; used twice (forward/reverse), combined Raptor weights 0.60 and 0.10.

Name: Knee MRI - Native 384 Corpus  
Owner: Dread Development  
Kaggle slug: `dreaddevelopment/raptor-knee-native384`  
Version: dataset version ID `19122845`  
Expected path: `/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384`  
Approximate size: 279 MiB  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES  
Checkpoint: `raptor_ft_coatnet_v8_full_swa.pt`; Raptor weight 0.20.

Name: Knee MRI - Native 384 Dense Corpus  
Owner: Dread Development  
Kaggle slug: `dreaddevelopment/raptor-knee-native384dense`  
Version: dataset version ID `19134209`  
Expected path: `/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384dense`  
Approximate size: 279 MiB  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES  
Checkpoint: `raptor_ft_coatnet_v10_full.pt`; Raptor weight 0.10.

### CoAtNet inference families and offline wheels

Name: RSNA Knee CoAt Residual Gated Epoch10 Top3  
Owner: Mattia Angeli  
Kaggle slug: `mattiaangeli/rsna-knee-coat-resgated-ep10-top3`  
Version: dataset version ID `18956429`  
Expected path: `/kaggle/input/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3`  
Approximate size: 842 MiB (21 files)  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES  
Notes: epochs 4/6/8; manifest SHA-256 `98511a8fdeb9da0e6e70c78d013ff636e1476f31c80b5dc134d294b18c3f284e`; runtime source SHA-256 `b11e58f8d7cabe9e264ac70b01e811dc6a846aa01248ace1f0e6536d0d4094d9`.

Name: RSNA Knee CoAtNet Global96 Gold Top3  
Owner: Mattia Angeli  
Kaggle slug: `mattiaangeli/rsna-knee-coatnet-global96-top3`  
Version: pinned V10 dataset source; immutable ID preserved in the notebook metadata.  
Expected path: `/kaggle/input/datasets/mattiaangeli/rsna-knee-coatnet-global96-top3`  
Approximate size: 863 MiB (19 files)  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES  
Notes: epochs 16/23/18; manifest SHA-256 `015f09030e76f86274cadae40777b3c3cf5c19f8c835d5ad88e15db41903779e`.

Name: RSNA Knee CoAtNet D4 Depth-Zone SWA3 B2  
Owner: Mattia Angeli  
Kaggle slug: `mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2`  
Version: pinned in V10 dataset sources (historical Inputs page)  
Expected path: `/kaggle/input/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2`  
Approximate size: 1.11 GiB (74 files)  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES  
Notes: parent checkpoint `coatnet_d4_parent_rank8_swa2.pt`, parent SHA-256 `5de38e333e3184d7b52fb068183f0cc198d6cbabb6ffbe36b26338622e82d297`, parent epochs 15/16; adapter epochs 11/9/5, adapter SHA-256 `b7c48b19997bc7ecb7e997f8dcf491ec5701ff917d1dac4b369eb85ac97ae89a`; manifest SHA-256 `7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d`.

Name: Repair-v1 CoAt top-3 artifacts  
Owner: Roman Tamrazov / upstream public asset lineage  
Kaggle slug: supplied through V10's pinned public inputs; manifest located by filename  
Version: V10 source includes dataset version ID `19862359` and notebook output version ID `346320098`; the exact historical attachments are preserved when V10 is forked.  
Expected path: discovered inside the pinned Kaggle inputs by exact manifest filename  
Approximate size: included in the attached asset bundle  
Publicly accessible: YES (V10 historical Inputs page exposes it and V10 ran it successfully)  
Available now: YES  
Required for exact reproduction: YES  
Notes: `coatnet_repairv1_top3_manifest.json`, manifest SHA-256 `99288a9a42f36ba544d7b0ad2fe645e3b1fc9c507c23802c16ca0d2e5085f684`; epochs 12/7/11. Do not accept the notebook's soft fallback for an exact baseline—preflight must require this member.

Name: OpenCV Python Headless 4.12.0.88 x86_64 wheel  
Owner: Mattia Angeli  
Kaggle slug: `mattiaangeli/opencv-python-headless-4120088-x86`  
Version: pinned V10 dataset source  
Expected path: `/kaggle/input/datasets/mattiaangeli/opencv-python-headless-4120088-x86`  
Approximate size: 51.5 MiB  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES  
Notes: wheel glob `opencv_python_headless-4.12.0.88-*.whl`; SHA-256 `236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9`.

Name: timm 1.0.22 offline wheel  
Owner: packaged with the CoAtNet artifacts  
Kaggle slug: D4 and Global96 datasets above  
Version: pinned with those artifact versions  
Expected path: artifact root / `timm-1.0.22-py3-none-any.whl`  
Approximate size: small wheel, included above  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES  
SHA-256: `888981753e65cbaacfc07494370138b1700a27b1f0af587f4f9b47bc024161d0`.

### RadImageNet assets and notebook outputs

Name: resnet-50-RadImageNet-Marwan  
Owner: Marwan Mahmoud  
Kaggle slug: `marwanmath/resnet-50-radimagenet-marwan`  
Version: pinned V10 dataset source  
Expected path: `/kaggle/input/datasets/marwanmath/resnet-50-radimagenet-marwan/ResNet50.pt`  
Approximate size: 90 MiB  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES  
SHA-256: `08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734`.

Name: RSNA Knee V52 RadImageNet Heads 20260812  
Owner: prvsiyan  
Kaggle slug: `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`  
Version: pinned V10 dataset source  
Expected path: `/kaggle/input/datasets/prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`  
Approximate size: 60.6 MiB  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES  
Notes: `v52_radimagenet_heads.pt`; accepted fallback path also exists in the E9 asset. Reference-head SHA-256 `0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa`.

Name: RSNA Knee E11 Diverse RadImageNet Heads V20  
Owner: Antoine G.  
Kaggle slug: `antoinegg1/rsna-knee-e11-diverse-heads-v20`  
Version: pinned V10 dataset source  
Expected path: `/kaggle/input/datasets/antoinegg1/rsna-knee-e11-diverse-heads-v20`  
Approximate size: 61.5 MiB  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES

Name: RSNA Knee E9 RadImageNet Heads V15  
Owner: Antoine G.  
Kaggle slug: `antoinegg1/rsna-knee-e9-radimagenet-heads-v15`  
Version: pinned V10 dataset source  
Expected path: `/kaggle/input/datasets/antoinegg1/rsna-knee-e9-radimagenet-heads-v15`  
Approximate size: 61.5 MiB  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: YES

Name: RSNA Knee E11 train output  
Owner: Sofia Anjenje  
Kaggle slug: `sofiaanjenje/rsna-knee-e11-train`  
Version: notebook output version ID `342671664`  
Expected path: `/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e11-train/rsna_rad_e11`  
Approximate size: checkpoint output only  
Publicly accessible: YES  
Available now: YES (kernel COMPLETE)  
Required for exact reproduction: YES

Name: RSNA Knee E13 train output  
Owner: Sofia Anjenje  
Kaggle slug: `sofiaanjenje/rsna-knee-e13-train`  
Version: notebook output version ID `342849430`  
Expected path: `/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e13-train/rsna_rad_e11`  
Approximate size: checkpoint output only  
Publicly accessible: YES  
Available now: YES (kernel COMPLETE)  
Required for exact reproduction: YES  
Notes: `v52_e11_heads.pt`; expected E13/second-head SHA-256 `ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a`.

Name: RSNA Knee LLM-read report labels  
Owner: Pilkwang  
Kaggle slug: `pilkwang/rsna-knee-llm-labels`  
Version: dataset version ID `18673450`  
Expected path: `/kaggle/input/datasets/pilkwang/rsna-knee-llm-labels`  
Approximate size: 1.0 MB  
Publicly accessible: YES  
Available now: YES  
Required for exact reproduction: retained upstream input; not read by the final inference cells on hidden test.

### Optional V10 ConvNeXt training output

Name: `# RSNA Knee | DINOsaur V5? train 🦖` output  
Owner: Roman Tamrazov  
Kaggle slug: `romantamrazov/rsna-knee-dinosaur-v5-train`  
Version: notebook output version ID `346320098`  
Expected path: Kaggle notebook-output mount; V10 searches `/kaggle/input/**/dinov5_convnext_manifest.json`  
Approximate size: unknown  
Publicly accessible: YES as a historical V10 input  
Available now: historical output is attachable; current source kernel status is ERROR  
Required for exact reproduction: NO  
Notes: in the published V10 run, `dinov5_convnext_manifest.json` was absent and the arm intentionally fell back to the exact anchor. Adding any alternative bundle would change predictions and is prohibited for BASELINE-001.

## Model graph and weights

- DINOv2: 20 fingerprint-validated fine-tuned members; six-block shared prefix optimization; public-frontier no-jitter output retained.
- A5: 5 folds, `m_f0.pt`–`m_f4.pt`, strict loading.
- RadImageNet: ResNet50 encoder plus E10/E13/E11 head/rank branches and the embedded matched calibrator.
- Raptor: four views with weights `[0.60, 0.10, 0.10, 0.20]` for MaxSpan V5, Native384Dense V10, MaxSpan V5 reverse, and Native384 V8.
- CoAt family: residual-gated top3, Global96 top3, D4 SWA3, Repair-v1 top3. V10 reduces these by rank of their probability mean, equal weight within the four-member family.
- Inner Raptor/CoAt blend: public Raptor 0.60, private CoAt family 0.40; no inner rerank.
- Outer target weights (CoAt/Raptor branch): default 0.60; ACL 0.75; Medial Meniscus 0.80; Lateral Meniscus 1.00; Lateral OA 0.75; Fracture 0.75.
- Final transform: percentile-rank blending and final rerank, exactly as V10.
- Optional ConvNeXt branch: skipped in the scored V10 execution; must remain skipped for exact reproduction.

## Preprocessing and TTA

- DICOM reader: `pydicom`; mixed transfer syntaxes are expected.
- DINO/A5/Rad paths use series selection by plane and fluid/fat-suppression metadata, physical crops, laterality normalization, percentile intensity windows, and fixed slice layouts.
- Raptor uses 140 mm crops; 336/384 resolution by arm; capacity-aware dense sampling up to 96 unique slices / 94 adjacent windows.
- DINO public-frontier target pooling uses max/top-2 pooling for selected targets; the retained submission path is the original no-jitter public-frontier ensemble.
- Full inference requires exactly two NVIDIA T4 GPUs in the reference wrappers.

## Python/runtime requirements

Required packages are supplied by the Kaggle image or attached offline wheels: Python 3.12, PyTorch/CUDA, torchvision, transformers, timm (including pinned child runtime 1.0.22), numpy, pandas, pydicom, OpenCV 4.12.0, and standard-library modules. Final inference makes no network downloads; Hugging Face offline flags are set.

## Checkpoint conclusion

**Are all model weights required to reproduce the V10 ~0.943 submission publicly available? YES.**

Evidence:

1. The exact V10 historical Inputs page exposes every attachment.
2. Authenticated metadata calls successfully listed all 14 named public datasets without downloading them.
3. The V10 run output shows 20/20 DINO fingerprints, all five A5 folds, all four Raptor views, and all four CoAt family members passing.
4. The optional ConvNeXt files are not required because the scored V10 run itself did not find the manifest and retained the anchor.

Exactness caveat: fork the historical V10 so Kaggle preserves its immutable dataset/notebook/model version IDs. Reattaching only the latest versions by slug is not considered exact, even when filenames match.

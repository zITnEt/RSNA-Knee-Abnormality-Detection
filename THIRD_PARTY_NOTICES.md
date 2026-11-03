# Attribution and external assets

The project extends public competition work. Its portfolio contribution is the engineering and evaluation described in the case study; the underlying competition, public baseline, attention-MIL design, and pretrained architectures are credited below.

| Source | Credit and use |
| --- | --- |
| [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) | RSNA / Kaggle competition and dataset. Obtain access and accept applicable terms separately. No MRI dataset or report corpus is distributed here. |
| [RSNA Knee — DINOsaur V5](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5?scriptVersionId=351856876) | Roman Tamrazov. Historical V10 baseline, including embedded upstream model/inference code and the historical 0.943 anchor. The published notebook has outputs removed; it is not byte-identical to the audited download. |
| [Raptor knee max-span](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan) | Dread Development. Raptor pretrained parent, CoAtNet configuration, study-level attention-MIL design, and preprocessing conventions. |
| [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) | Ross Wightman and contributors. `timm` CoAtNet implementation; this project records `timm==1.0.22`. |
| [torchvision](https://github.com/pytorch/vision) | PyTorch contributors. ResNet50 architecture and ImageNet-1K V2 pretrained checkpoint used in the gated ResNet experiment. |
| [RadImageNet](https://github.com/BMEII-AI/RadImageNet) | RadImageNet authors. Medical-image pretrained ResNet50 used in the earlier fusion experiment. The recorded checkpoint source is [marwanmath/resnet-50-radimagenet-marwan](https://www.kaggle.com/datasets/marwanmath/resnet-50-radimagenet-marwan). |
| [JEV knee labels](https://www.kaggle.com/datasets/riadmohamed42/jev-knee-labels) | Dataset author and contributing label sources. HYBRID soft/binary labels and Gold74 evaluation inputs used in recorded experiments. Label files are not bundled. |

The baseline additionally references DINOv2 (Meta Research), Pilkwang's fine-tuned DINOv2 ensemble, Mattia Angeli's knee MRI weights, and other public competition inputs. Their versions, hashes, and roles are recorded in [the historical manifest](manifests/DINOSAUR_V5_MANIFEST.md) and the original notebook source. Retain their attribution when using that pipeline.

The rights granted by each source must be checked at that source before reuse or redistribution. A public URL or this notice does not itself grant a license. No blanket license is claimed over third-party code, images, data, reports, or pretrained weights.

## Snapshot transformations

The publication includes source code from the root worktree and the ResNet50 and CoAtNet-only worktrees. Notebook outputs, execution counts, and runtime widget metadata were cleared. Two historical cloud deployment identifiers were replaced by explicit placeholders. Text encoding/line endings were normalized where necessary; the pinned CV fold CSV retains its exact original bytes. `manifests/SOURCE_SNAPSHOT.json` records original and published SHA-256 values and these transformations. Original worktrees were preserved.

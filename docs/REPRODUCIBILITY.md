# Setup and reproduction

## 1. Install dependencies

Python 3.12 is the GitHub Actions target. Create a virtual environment and activate it, then install a mutually compatible PyTorch/torchvision pair for your CPU/CUDA runtime and the remaining requirements:

```bash
python -m pip install -r requirements.txt
```

For CPU-only review, the workflow in `.github/workflows/checks.yml` installs PyTorch from its CPU wheel index before the remaining requirements. `timm==1.0.22` is pinned because the model depends on that implementation. Other minimum versions describe a compatible family; this is not a fully locked historical environment.

## 2. Verify the data-free path

From the repository root:

```bash
python scripts/portfolio_demo.py
python scripts/run_portfolio_checks.py
```

The demo creates eight clearly named synthetic study rows with 12 generated scores. It validates a correctly formatted CSV and verifies rejection of an out-of-range score. Outputs appear under ignored `demo-output/`. No checkpoint, MRI data, or medical inference is involved.

The check runner runs existing CPU tests in separate processes for each variant, with GPU visibility disabled. It covers model routing, checkpoint/EMA behavior, samplers, synthetic DICOM preprocessing, fold utilities, and statistical helpers. Integration tests that need private label/audit inputs skip explicitly. Add `--distributed` to enable the two-process Gloo integration test, which requires local socket access. Full real-backbone 384 × 384 tests remain available separately and require more RAM/time.

## 3. Select one experiment

| Working directory | Model source | Additional dependencies/assets |
| --- | --- | --- |
| Repository root | `models/raptor_v5_radimagenet.py` | Authorized input tensors/DICOMs, Raptor parent, RadImageNet parent. |
| `experiments/resnet50_fusion` | `models/raptor_v5_resnet50.py` | Minimum-overlap V2 shards, labels, Raptor parent, ImageNet ResNet50 V2 parent. |
| `experiments/coatnet_only` | `models/raptor_v5.py` | Supported original/migrated Raptor checkpoint and matching preprocessed input. |

Keep the selected directory at the front of the Python import path. Prefer `python -m scripts.<name>` when a script imports `models` or `scripts`; do not mix modules from different variants.

## 4. Obtain real assets separately

Use your own authorized Kaggle account and the original source pages in `THIRD_PARTY_NOTICES.md` and `manifests/DINOSAUR_V5_MANIFEST.md`. The manifest describes the historical baseline; it does not guarantee present availability. Some preprocessing inputs used in the recorded runs are private Kaggle outputs and are not distributed here. You may need to rebuild them with the included preprocessing code.

Kaggle orchestration additionally requires the optional dependencies in `requirements-kaggle.txt` and an authenticated Kaggle CLI/API environment. Cloud deployment examples require the Google Cloud CLI and a configured project. These are separate from the data-free review path.

Do not run cloud launchers or Kaggle publication/monitor scripts as a setup check: they can start compute jobs or publish datasets. Configure your own account, storage, paths, credentials, and runtime deliberately. Historical infrastructure identifiers in this snapshot are placeholders.

## 5. Example: gated ResNet50 single-split training

After preparing all 32 pinned minimum-overlap V2 shards and the required label files, use a CUDA runtime and run from `experiments/resnet50_fusion`:

```bash
python -m scripts.train_resnet50_cloud --help
python -m scripts.train_resnet50_cloud \
  --data /path/to/preprocessed-shards \
  --labels /path/to/labels \
  --raptor /path/to/raptor_ft_coatnet_v5_full_swa.pt \
  --resnet /path/to/resnet50-11ad3fa6.pth \
  --output /path/to/run \
  --epochs 10
```

The trainer checks the expected recipe and input manifests. Preserve the complete output directory for resume and audit. Use `--preflight-only` to perform the implemented GPU preflight before committing to a run. The historical cloud guides in this variant explain the original training protocol.

## 6. Checkpoints, inference, and evaluation

The variants have different checkpoint formats. Use each model's loader and the matching inference/export script; loading a similarly named `.pt` file into a different variant is not a supported substitution. `experiments/coatnet_only/scripts/strip_raptor_fusion.py` is the explicit migration route for compatible checkpoints.

Validate a real submission against the exact official sample:

```bash
python scripts/validate_submission.py /path/to/submission.csv /path/to/sample_submission.csv
```

The selected aggregate reports are readable without external inputs. Recomputing the original metrics requires the recorded model states and per-study predictions/labels, which are not bundled. Read the model card before comparing numbers from different splits or label sources.

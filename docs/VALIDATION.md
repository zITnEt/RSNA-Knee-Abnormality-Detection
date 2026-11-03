# Portfolio publication checks

Date: November 4, 2026 (Asia/Tashkent).

## Executed locally

| Check | Result |
| --- | --- |
| Data-free submission demo | Passed: 8 synthetic study rows, 12 score columns, valid CSV accepted, out-of-range example rejected. |
| Root CPU suite | 84 tests reported; successful, 2 optional/integration skips. |
| Gated ResNet50 CPU suite | 49 tests reported; successful, 1 skipped class requiring private label inputs. |
| CoAtNet-only CPU suite | 33 tests reported; successful. |
| Python syntax | 144 Python source files parsed successfully. |
| Notebook hygiene | 166 notebooks checked; code-cell outputs and execution counts cleared. |
| Source provenance | Published source hashes checked against the snapshot ledger. |
| Recorded CV fold file | Exact bytes retained, SHA-256 `5e899b9bd07c55759d4a8cd2425d2095bba94605196f88849f6f18c16dacc869`. |
| Secret-pattern scan | No matches in the publication source or 1,273 locally reachable historical text blobs. |
| New documentation links | Relative targets resolved. |
| Portfolio visuals | Cover and architecture PNGs rendered and visually inspected. |
| Upwork text lengths | Title 53 / 70; role 64 / 100; description 505 / 600 characters. |

The lightweight runner disables the two-process Gloo integration test unless `--distributed` is supplied; this local environment restricts socket access. A local pilot audit test and a class requiring the private label package also skip explicitly. Real-backbone suites remain available separately. The local test temp directory was placed inside the writable workspace.

Local environment: Windows, Python 3.14, torch 2.12.1+cu126, timm 1.0.22, pydicom 3.0.2, numpy 2.4.4, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, OpenCV 4.13.0. GPU visibility was disabled for the three test suites. The GitHub workflow targets Python 3.12 on Ubuntu with CPU PyTorch; its status is shown in the repository's Actions tab.

These checks validate the publication and selected code contracts. They do not constitute a new medical-image inference/training run, a complete reproduction of the historical metric report, a clinical validation, or proof that a pattern scanner can detect every secret.

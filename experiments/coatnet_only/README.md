# CoAtNet-only Raptor experiment

This directory preserves the CoAtNet-only source worktree as an independent runnable variant. Run it from this directory so that `models` resolves to the correct architecture.

[models/raptor_v5.py](models/raptor_v5.py) removes the ResNet encoder and fusion blocks while retaining the original CoAtNet trunk, per-finding attention-MIL head, 12-label order, and `[B, K, 3, 384, 384]` input contract. The strict checkpoint migration code can extract supported CoAtNet/head weights from compatible original or raw/EMA fusion states.

- [Model guide](docs/RAPTOR_V5_COATNET.md)
- [Offline checkpoint export](scripts/strip_raptor_fusion.py)
- [Self-contained notebook](kaggle/raptor_v5_coatnet.ipynb)
- [Regression tests](tests/test_raptor_v5.py)

Install the root repository's `requirements.txt`. The source demonstrates model simplification and migration; it does not claim a newly measured performance result. See the [reproduction guide](../../docs/REPRODUCIBILITY.md) for shared setup and [third-party credits](../../THIRD_PARTY_NOTICES.md).

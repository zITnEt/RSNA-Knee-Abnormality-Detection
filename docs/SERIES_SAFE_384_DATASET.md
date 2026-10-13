# Series-safe six-slot MRI dataset

This new recipe is separate from the earlier V5 336-to-384 cache. The user
specified direct 384×384 resize and prohibited every cross-series triplet.

Actual completed export: 4,407 studies, 253,788 sampled slices and 213,428
triplets. Study bags contain 18–60 triplets (median 48, mean 48.43). The
4,333 non-gold studies contain 209,786 triplets; Gold74 contains 3,642.
Compressed NPZ files total 119,514,388,101 bytes (119.51 GB / 111.31 GiB),
plus small metadata files. There are zero cross-series triplets.

| Slot order | Category | Reference and sampled-slice allocation |
|---|---|---:|
| 0 | Sagittal PD/T2 fat-suppressed | 18 |
| 1 | Sagittal PD/T2 without fat suppression | 14 |
| 2 | Coronal PD/T2 fat-suppressed | 12 |
| 3 | Coronal T1 without fat suppression | 8 |
| 4 | Axial PD/T2 fat-suppressed | 12 |
| 5 | Sagittal T1 without fat suppression | 8 |

One matching acquired series is selected per slot, preserving competition
metadata row priority and checking MRI weighting/fat suppression headers.
This means **up to six selected series per study**, rather than exporting
every series in the competition. A series UID cannot be selected twice.
The anatomical plane comes from `train_series.csv`. Weighting is determined
from SeriesDescription/SequenceName (T1, T2, PD), then scanning sequence and
TR/TE when necessary: TR below 800 ms indicates T1; otherwise TE above 60 ms
indicates T2; otherwise TR at least 800 ms indicates PD. Recognized gradient
echo sequences are excluded from these T1/PD/T2 slots. Explicit fat-saturation
description or ScanOptions evidence takes precedence; otherwise the CSV
Fat_Suppression flag is used, defaulting to non-FS when unknown. The CSV's
Fluid_Sensitive flag is only a fallback for unknown weighting. The first
matching metadata row wins; the pipeline does not rank by series length.
Slices are ordered by a common IOP normal dotted with IPP. Missing/inconsistent
physical geometry, invalid spacing, duplicate physical positions or duplicate
SOP identifiers cause an explicit failure; there is no filename ordering fallback.

Within the 2–98% depth band (`lo=floor(.02*N)`,
`hi=max(floor(.98*N)-1,lo)`, inclusive), sample unique evenly spaced slices
up to the slot allocation. Missing/short series are not padded, duplicated or
replaced, and unused allocation is not redistributed. Maximum 72 sampled slices.
Specifically, for native depth N, the eligible capacity is hi-lo+1 and the
sample count is `m=min(slot_allocation,hi-lo+1)`. The native indices are
`round(linspace(lo,hi,m))`, using NumPy rounding. For N=100, the band is native
indices 2 through 97 (zero based). A short series can contribute fewer than
three slices and therefore no triplet. A study with no eligible triplets
fails explicitly; it is not silently removed from the complete corpus.
All six populated slots with sufficient depth produce 60 triplets, from
`sum(max(sampled_count-2,0))`, never 70 cross-boundary windows.

For each selected series, apply modality LUT/rescaling and MONOCHROME1
inversion (`max-rescaled_pixel`). Compute p2/p98 over its actual reference
slices, scale/clip to [0,1], center crop 140 mm using the row and column pixel
spacing separately (clamp to acquired field), resize directly to 384×384 with
OpenCV INTER_AREA, and quantize `(scaled_resized*255).astype(uint8)`.
The exact scaling equation is `clip((pixel-p2)/(p98-p2+1e-6),0,1)`.
Percentiles are computed over all pixels of the sampled reference slices,
before cropping, independently for each selected series. References and
exported samples are the same unique slice set. MONOCHROME1 uses the maximum
of each modality-rescaled slice. Crop dimensions are
`round(140/row_spacing)` by `round(140/column_spacing)` pixels, clamped to
each acquired image's dimensions; the crop is centered with integer floor
offsets. A field of view smaller than 140 mm is retained without padding.
The crop is resized directly to a square, so there is no aspect-preserving
letterbox. Quantization truncates to uint8; there is no intermediate 336
resize, augmentation, denoising, registration, or cross-series interpolation.
Form windows of three neighboring **sampled** slices inside that series only.
Divide by 255 and apply ImageNet mean [0.485,0.456,0.406] and standard deviation
[0.229,0.224,0.225]. Store the normalized tensors as float16.
Sampled neighbors can have gaps in native slice numbers.
For sampled slices `[s0,s1,s2,s3]` in series A, windows are
`[s0,s1,s2]` and `[s1,s2,s3]`. The final samples of A never combine with the
first samples of B. All overlapping windows are retained; K is not capped
at the former 12-triplet training setting. The six slot maxima yield
16+12+10+6+10+6=60 windows. Normalization uses float32 arithmetic before the
ready tensor is cast to float16. Channel means/std differ even though each
channel represents a different neighboring grayscale slice.

Each study NPZ includes:

- `slices_uint8`: [N,384,384] preprocessed source planes.
- `images`: [K,3,384,384] normalized float16 training inputs.
- `triplet_indices`, `slice_series_uids`, `triplet_series_uids`: explicit ownership.
- `slice_slot_ids`, `triplet_slot_ids`, `same_series`, `slot_present`.
- `audit_json`: original filenames, SOP/series IDs, native sampled indices,
  physical positions, pixel spacing and per-series p2/p98.

Every saved study is reopened and checked against the uint8 source planes;
every triplet must have one SeriesInstanceUID. The manifest records SHA-256,
byte size, slice count, triplet count and split for each study.

The complete corpus has 4,407 studies. **Only the 4,333 `train` studies may
be used for training**; all 74 gold studies are marked `held_out_gold` and
are excluded from `labels_hybrid_train.csv`. Gold pixels are retained for
separate evaluation. Published datasets are copied directly from the Kaggle
notebook outputs and retain `series_safe_384_shard_XX/train/` and
`series_safe_384_shard_XX/held_out_gold/` folders, along with the explicit
split column. Never treat every NPZ in a shard as training data.
HYBRID soft labels remain separate from image arrays. Gold annotations and
sparse official annotations preserve their missing values.
There is no new label estimation or model-generated relabeling. The supplied
HYBRID values are copied for the 12 findings: ACL, MCL, medial meniscus,
lateral meniscus, medial OA, lateral OA, PF OA, effusion, synovitis, Baker's,
contusion, fracture. This export's `train` split denotes the non-gold
training pool. An 80/20 development split is not baked into the pixel export;
it belongs to the later training configuration and must be study based.

Load `images` directly using `SeriesSafe384Dataset`; no decoding, crop, resize,
division by 255 or normalization belongs in training after that loader.
`k=None` keeps every stored triplet; batch size one or a ragged collate is
required for variable K. Private datasets and notebook outputs retain the
original RSNA competition data terms; this export grants no additional rights.

The 32 private shards are named `zafarnematov/rsna-series-safe-384-shard-00`
through `zafarnematov/rsna-series-safe-384-shard-31`. Attach all shards to use
the complete corpus. The companion `series_safe_384_training_inputs.ipynb`
discovers each nested recipe directory automatically. The root helper
`preprocess_six_slot_raptor.py` is an imported legacy utility; its old recipe
is **not** the recipe used for these arrays. The nested `recipe.json` and
each NPZ's `audit_json` are the authoritative configuration.

Production uses 32 deterministic partitions of sorted StudyInstanceUIDs:
shard i gets `sorted_uids[i::32]`. These are storage/work partitions, not
training/validation folds. CPU notebooks process up to five shards at a
time, using four per-study worker threads in each notebook. Every study is
checked before saving, re-opened and checked again afterward, then hashed.
Failures are recorded explicitly and prevent the affected shard from being
accepted as complete. The monitor checks completion receipts and requires
4,407 unique study IDs, 4,333 non-gold studies, 74 held-out gold studies,
identical recipe hashes, and zero cross-series windows across all 32 shards.
Publication checks every remote NPZ path and byte size. The final loader
notebook additionally reads the mounted published manifests, verifies full
coverage, exercises direct loading, and hashes/checks one study per shard.

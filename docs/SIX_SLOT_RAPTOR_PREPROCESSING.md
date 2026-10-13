# Six-slot training tensors

Created on 2026-10-09 for the user's Kaggle training task. The raw MRI files
stay on Kaggle. The self-contained notebooks live in
`kaggle/six_slot_preprocessing/shard_00` through `shard_31`.

## Image contract

Physical slice order comes from `dot(ImagePositionPatient,
cross(ImageOrientationPatient[:3], ImageOrientationPatient[3:]))`, with an
InstanceNumber fallback. DICOM modality LUT/rescaling and MONOCHROME1 inversion
are applied before contrast scaling.

Within each selected series, the retained source band is 2–98%. A fixed
evenly spaced reference sample establishes its 2nd and 98th intensity
percentiles; all selected slices from that series share those bounds.
Intensity scaling is `clip((x-p2)/(p98-p2+1e-6),0,1)`.

Each slice gets a 140 mm center crop, using PixelSpacing[0], clamped to the
available square image. OpenCV INTER_AREA resizes it to 336, then quantization
stores `(image*255).astype(uint8)`. Conversion back to float32 divides by 255;
PyTorch bilinear interpolation with align_corners=False produces 384×384.
Three adjacent sampled planes become three channels. ImageNet normalization
uses mean 0.485/0.456/0.406 and std 0.229/0.224/0.225. Export converts the final
tensor to float16. This introduces at most the normal float16 rounding error;
the parent pixel-math parity test checks that conversion explicitly.

The final training export retains V5's nominal 64-source-slice budget, yielding
at most 62 windows. The initial storage pilot used the dense inference override
of 96 slices; its files and receipts are separate and are not production shards.

## Six-slot extension

| Slot | Plane | Fluid-sensitive | Fat-suppressed | Reference slices |
|---|---|---|---|---|
| SAG_FLUID_FS | Sagittal | Yes | Yes | 18 |
| COR_FLUID_FS | Coronal | Yes | Yes | 12 |
| AX_FLUID_FS | Axial | Yes | Yes | 12 |
| SAG_FLUID_NOFS | Sagittal | Yes | No | 14 |
| COR_T1 | Coronal | No; header identified as T1 | No | 8 |
| SAG_T1 | Sagittal | No; header identified as T1 | No | 8 |

This extends V5's five slots; it is not a byte-identical V5 dataset. The
reference counts above are explicit engineering choices for the six-category
extension. The 64 real slices are allocated proportionally to these counts,
with capacity limits and redistribution to available series. Source picks are
unique and evenly spread within each band's native slice indices. Missing
categories remain absent and are recorded in a six-element slot_present mask.
Different categories are never substituted to conceal missing acquisitions.
The official CSV couples Fluid_Sensitive to Fat_Suppression. Sequence weighting
therefore comes from DICOM descriptions, sequence names, TR/TE and GRE flags,
as in the recovered six-slot classifier. Positive header fat-sat evidence also
corrects metadata, including Dixon water protocols. The regression test includes
a PD-without-FS acquisition whose CSV Fluid_Sensitive flag is zero.

As in the saved V5 inference path, slots are concatenated before sliding
triplets are formed. Some windows therefore cross series boundaries.
triplet_slot_ids and same_series make these windows explicit. Adjacent here
means adjacent selected planes, not necessarily consecutive native slices.

## Files and use

Each study is an NPZ containing images `[K,3,384,384]` float16, triplet_slot_ids
`[K,3]`, same_series `[K]`, slot_present `[6]`, and audit_json. Loading images
requires no DICOM decoding, crop, resize, intensity scaling, or normalization.
All NPZs can be opened with allow_pickle=False.

```python
import numpy as np
import torch

with np.load(study_file, allow_pickle=False) as data:
    x = torch.from_numpy(data['images'])  # already model-ready
with torch.autocast('cuda', dtype=torch.float16):
    logits = model(x[None].cuda())
```

ReadyTripletDataset in raptor_data/ready_tensors.py loads attached shard roots.
Its default k=12 chooses a fixed-size training bag; this is sampling, with no
further image preprocessing. k=None returns the complete exported bag.
The dictionary also includes slot metadata. Pass batch['images'] to the model.

Official sparse annotations and JEV FINAL2 soft labels are separate CSVs.
They are not interchangeable. splits.csv marks the official fully labeled
studies and all JEV gold74 IDs as held_out_gold, leaving the other IDs as train.
Publisher exports gold74 annotations separately, retaining missing cells.
Evaluation losses must mask missing gold annotations. No random patient split
or claim of patient-disjoint validation is made.

## Validation and ongoing work

The numerical parity test reproduces V5's crop, uint8 quantization,
interpolation and normalization independently. The synthetic DICOM test
exercises all six categories, reversed file names, physical ordering,
modality rescaling, MONOCHROME1, cross-series metadata, and NPZ round trips.
Each real study is re-opened after saving. Manifests contain per-file SHA-256.

The queue monitor checks every 60 seconds and allows five concurrent CPU
notebooks. It records live progress, launches subsequent shards, retries delayed
output receipts, and stops launching after a failed or incomplete shard. The
launched monitor is bounded to 24h to cover preprocessing and publication.
The separate publisher stages one completed shard locally, verifies its
checksums, uploads a private Kaggle dataset, checks its privacy/readiness, and
cleans only that verified staging folder. It stops on quota or upload errors.

Both processes require this computer to remain running and connected. Their
bounded processes temporarily inhibit idle sleep when started on AC power;
they change no persistent power setting and do not keep the display on. Their
state, PID files, events, live logs, and verified receipts are under
results/six_slot_preprocessing. The full dataset is ready only when every
production shard's manifest is complete and all 4,407 study IDs are covered.

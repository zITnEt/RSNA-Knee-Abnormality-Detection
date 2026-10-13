# Minimum-overlap 384 MRI dataset

The completed preprocessing contains 264,323 triplets and 609,494 distinct
source slices across 4,407 studies. There are 670 studies with no overlap
between triplets. Fifteen studies have fewer than 60 triplets (30–59);
the remaining 4,392 have 60. All triplets are series-safe. Training-pool
studies contain 259,895 triplets; Gold74 contains 4,428 and remains held out.

Publication uses **V2** of the 32 private
`zafarnematov/rsna-series-safe-384-shard-00` through `-31` datasets.
The original arrays remain available in dataset V1. Kaggle counts the latest
private dataset versions toward the storage quota; versioning avoids keeping
two separate full dataset collections. Original preprocessing notebook runs
remain in version history; their V2 runs losslessly prepare the new publication.

Same DICOM ordering, six-slot series selection, modality rescaling,
MONOCHROME1 inversion, original percentile references, 140 mm crop, direct
384x384 INTER_AREA resize, uint8 quantization, ImageNet normalization, HYBRID
labels and Gold74 exclusion as the completed series-safe 384 dataset.

One first matching series per slot remains selected. The six slot order is
SAG fluid FS / SAG fluid non-FS / COR fluid FS / COR T1 / AX fluid FS / SAG T1.
The per-series intensity reference allocations remain 18/14/12/8/12/8, using
the exact original unique linspace indices within the native 2-98% band.
References can differ from the newly exported slice set. Pixel preprocessing
for a shared native slice is unchanged from the previous export.

Triplet sampling changes:

1. Let c_i be each selected series' eligible native slice count. A series
   supports max(c_i-2,0) distinct neighboring three-slice windows.
2. Set K=min(60,sum(max(c_i-2,0))). Never pad or duplicate complete windows.
3. Allocate K triplets by exact dynamic programming, in this priority order:
   maximize sum(min(c_i,3*k_i)) distinct used slices; maximize
   sum(min(k_i,floor(c_i/3))) mutually disjoint primary windows; then minimize
   squared deviation from the former full-study triplet weights
   16/12/10/6/10/6, renormalized over series with at least three slices.
4. For k_i triplets, sample min(c_i,3*k_i) unique slices evenly over that
   series' entire eligible native-index band. Group disjoint triples first.
   If overlap is necessary, cover remaining tail slices first, then add
   distinct neighboring windows while balancing slice usage.
5. Save every used slice once, with explicit triplet indices. All three
   indices in a triplet are consecutive in that series' sampled list and
   belong to the same SeriesInstanceUID. Native slice indices may have gaps.

At the attainable K, repeated source-slice uses are minimized globally over
the selected series. The exact minimum is 3*K minus the maximum distinct
used-slice count. If 60 disjoint windows are possible, all 180 source slices
are unique. If they are impossible, overlap is allowed only to the extent
needed by the selected-series capacities. If fewer than 60 distinct windows
exist, the study has a smaller bag. No study is excluded for being short.
Triplet allocation can shift between available slots to minimize reuse.

Arrays retain the previous names and shape contract: slices_uint8 [N,384,384],
images float16 [K,3,384,384], triplet_indices [K,3], slice/triplet series UIDs,
slot IDs, slot_present, same_series and audit_json. Added arrays:
slice_usage_counts and primary_disjoint_subset. The primary subset is
mutually disjoint internally; fallback windows may share its slices.

Every study is validated before saving and after reopening. Checks include
exact normalization from stored uint8 slices, unique source SOPs, no duplicate
windows, no cross-series triplets, optimal unique-slice coverage, minimum
repeated uses and a maximum-size mutually disjoint primary subset.
The audit records original percentile references, selected-series capacities,
allocation, unique-slice coverage, repeated uses and source provenance.

Gold74 remains held_out_gold and excluded from train and HYBRID training
labels. All 4,407 studies are retained: 4,333 training-pool studies and 74
held-out studies. The 32 processing shards are storage partitions, not
training/validation folds. No training is started by this campaign.

Load the stored images directly with MinimumOverlap384Dataset, k=None.
Do not repeat pixel scaling, crop, resize or normalization. Variable bag
sizes require study batch size one or a ragged collate. The original RSNA
competition terms continue to apply; this private export grants no new rights.

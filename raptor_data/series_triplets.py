"""Form Raptor triplets within SeriesInstanceUID boundaries, preserving K.

Adjacent here means adjacent in the physically ordered *sampled* series, as in
the parent Raptor loader; sampling may skip native DICOM slices. Never infer
series boundaries from pixels or an anonymous concatenated corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


SLOTS = (("Sagittal", 1, 18), ("Sagittal", 0, 14),
         ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12))


@dataclass(frozen=True)
class TripletSpec:
    slot: int
    series_uid: str
    indices: tuple[int, int, int]


@dataclass(frozen=True)
class SamplingPlan:
    triplets: tuple[TripletSpec, ...]
    sampled_indices: tuple[tuple[int, ...], ...]
    target_triplets: int
    unique_triplets: int
    repeated_triplets: int

    @property
    def source_count(self) -> int:
        return sum(len(indices) for indices in self.sampled_indices)


def _allocate(capacities, weights, target):
    """Weighted largest-remainder allocation with capacity redistribution."""
    capacities = np.asarray(capacities, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    ideal = target * weights / weights.sum()
    quotas = np.floor(ideal).astype(np.int64)
    remainder = target - int(quotas.sum())
    quotas[np.argsort(-(ideal - quotas), kind="stable")[:remainder]] += 1
    quotas = np.minimum(quotas, capacities)
    wanted = min(target, int(capacities.sum()))
    deficit = np.zeros(len(quotas))
    while int(quotas.sum()) < wanted:
        available = quotas < capacities
        active_weights = np.where(available, weights, 0)
        deficit[available] += active_weights[available] / active_weights.sum()
        choice = int(np.argmax(np.where(available, deficit, -np.inf)))
        quotas[choice] += 1
        deficit[choice] -= 1
    return quotas


def plan_triplets(capacities, series_uids, weights=None, *,
                  target_triplets: int | None = None, legacy_slice_budget: int = 96,
                  shortage: str = "repeat") -> SamplingPlan:
    """Allocate real slices separately in each series and preserve the old K.

    Default K = min(96, total eligible source slices) - 2, matching dense
    parent inference. A series assigned q triplets needs q+2 source slices.
    If physical capacity is insufficient, repeat valid complete triplets
    (audited), or use shortage='error'. Never repeat a channel to fabricate
    a three-slice triplet and never join slices across series.
    """
    capacities = np.asarray(capacities)
    if (capacities.ndim != 1 or len(capacities) == 0
            or not np.issubdtype(capacities.dtype, np.integer)
            or np.any(capacities < 0) or len(series_uids) != len(capacities)):
        raise ValueError("Expected nonnegative integer capacity and UID per series slot")
    weights = np.ones(len(capacities)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != capacities.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Expected positive finite allocation weights")
    if shortage not in ("repeat", "error"):
        raise ValueError("shortage must be 'repeat' or 'error'")
    present = [str(uid) for uid, n in zip(series_uids, capacities) if n > 0]
    if any(uid.strip().lower() in ("", "none", "nan", "<na>") for uid in present) or len(present) != len(set(present)):
        raise ValueError("Each populated slot must have its own known SeriesInstanceUID")
    if not isinstance(legacy_slice_budget, int) or legacy_slice_budget < 3:
        raise ValueError("legacy_slice_budget must be at least three")
    target = (min(legacy_slice_budget, int(capacities.sum())) - 2
              if target_triplets is None else target_triplets)
    if not isinstance(target, int) or isinstance(target, bool) or target < 1:
        raise ValueError("Triplet count must be a positive integer")
    available = np.maximum(capacities - 2, 0)
    if int(available.sum()) == 0:
        raise ValueError("No series contains three distinct acquired slices")
    if shortage == "error" and int(available.sum()) < target:
        raise ValueError(f"Need {target} unique same-series triplets; only {int(available.sum())} possible")
    quotas = _allocate(available, weights, target)
    sampled, specs = [], []
    for slot, (capacity, quota, uid) in enumerate(zip(capacities, quotas, series_uids)):
        indices = (np.linspace(0, int(capacity) - 1, int(quota) + 2).round().astype(int)
                   if quota else np.empty(0, dtype=int))
        if len(indices) != len(np.unique(indices)):
            raise RuntimeError("Source allocation selected duplicate slices")
        sampled.append(tuple(int(i) for i in indices))
        for start in range(int(quota)):
            specs.append(TripletSpec(slot, str(uid), tuple(int(i) for i in indices[start:start+3])))
    unique = len(specs)
    if unique < target:
        # Preserve every unique window, then fill the deficit with evenly spread
        # safe windows. The repetition count is returned for experiment audits.
        extra = np.linspace(0, unique - 1, target - unique).round().astype(int)
        specs.extend(specs[int(i)] for i in extra)
    return SamplingPlan(tuple(specs), tuple(sampled), target, unique, target - unique)


def _protocol_flag(value):
    try:
        number = float(value)
        return int(number) if np.isfinite(number) and number in (0, 1) else None
    except (ValueError, TypeError):
        text = str(value).strip().lower()
        return 1 if text in ("true", "yes", "y", "t") else 0 if text in ("false", "no", "n", "f") else None


def _pick_series(rows, plane, fluid, used):
    candidates = [row for row in rows if row["Anatomical_Plane"] == plane
                  and str(row["SeriesInstanceUID"]) not in used]
    preferred = [row for row in candidates if _protocol_flag(row.get("Fluid_Sensitive")) == fluid]
    return (preferred or candidates)[0] if candidates else None


def dicom_reader():
    """Order each individual series in patient coordinates and decode pixels."""
    import pydicom
    from pydicom.pixels import apply_modality_lut

    def order_and_meta(directory):
        directory = Path(directory)
        records = []
        for file in directory.glob("*.dcm"):
            header = pydicom.dcmread(file, stop_before_pixels=True)
            if str(header.SeriesInstanceUID) != directory.name:
                raise ValueError(f"DICOM SeriesInstanceUID does not match its directory: {file}")
            orientation = getattr(header, "ImageOrientationPatient", None)
            position = getattr(header, "ImagePositionPatient", None)
            if orientation is not None and position is not None:
                normal = np.cross(np.asarray(orientation[:3], float), np.asarray(orientation[3:], float))
                coordinate = float(np.dot(np.asarray(position, float), normal))
            else:
                coordinate = float(header.InstanceNumber)
            if not np.isfinite(coordinate):
                raise ValueError(f"Nonfinite slice coordinate: {file}")
            spacing = float(getattr(header, "PixelSpacing", [0.5])[0])
            records.append((coordinate, str(file), spacing, str(header.SOPInstanceUID)))
        records.sort(key=lambda record: (record[0], record[1]))
        if len({record[3] for record in records}) != len(records):
            raise ValueError(f"Duplicate acquired SOPInstanceUID in {directory}")
        median = float(np.median([record[2] for record in records])) if records else 0.5
        return [(file, spacing) for _, file, spacing, _ in records], median

    def read_px(path):
        dataset = pydicom.dcmread(path)
        image = apply_modality_lut(dataset.pixel_array, dataset).astype(np.float32)
        if image.ndim != 2 or not np.isfinite(image).all():
            raise ValueError(f"Invalid MRI slice pixels: {path}")
        if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
            image = image.max() - image
        return image

    return order_and_meta, read_px


def prepare_study_triplets(study_uid: str, series_rows, series_root,
                           *, target_triplets=None, slots=SLOTS, span=(0.02, 0.98),
                           norm="imagenet", shortage="repeat", reader=None):
    """Raw-DICOM replacement for the parent's concatenated global stack.

    Return ([K,3,384,384], audit). Keep the parent five-slot selection, source
    depth band, series percentile reference, 140mm center crop and normalization.
    Every window comes from exactly one series, including shortage repeats.
    A reader=(order_and_meta, read_px) can reuse an existing DICOM cache.
    """
    import cv2

    if norm not in ("imagenet", "none"):
        raise ValueError("norm must be 'imagenet' or 'none'")
    if not 0 <= span[0] < span[1] <= 1:
        raise ValueError("Invalid source depth span")
    order_and_meta, read_px = dicom_reader() if reader is None else reader[:2]
    used, pools, capacities, uids = set(), [], [], []
    for plane, fluid, reference_count in slots:
        record = _pick_series(series_rows, plane, fluid, used)
        if record is None:
            pools.append(None); capacities.append(0); uids.append(None)
            continue
        uid = str(record["SeriesInstanceUID"])
        used.add(uid)
        files, spacing = order_and_meta(Path(series_root) / str(study_uid) / uid)
        lo = int(len(files) * span[0])
        hi = min(len(files)-1, max(int(len(files) * span[1])-1, lo))
        capacity = max(0, hi-lo+1)
        reference = np.linspace(lo, hi, reference_count).round().astype(int) if capacity else []
        pools.append((files, spacing, lo, reference))
        capacities.append(capacity); uids.append(uid)
    plan = plan_triplets(capacities, uids, [slot[2] for slot in slots],
                         target_triplets=target_triplets, shortage=shortage)
    planes, source_files, decoded_files = {}, [], []
    for slot, (pool, selected) in enumerate(zip(pools, plan.sampled_indices)):
        if pool is None or not selected:
            continue
        files, median_spacing, lo, reference = pool
        picks = [lo+i for i in selected]
        arrays = {i: read_px(files[i][0]) for i in sorted(set(picks) | set(reference))}
        decoded_files.extend(str(files[i][0]) for i in arrays)
        low, high = np.percentile(np.concatenate([arrays[int(i)].ravel() for i in reference]), [2, 98])
        for local, index in zip(selected, picks):
            path, spacing = files[index]
            image = np.clip((arrays[index]-low)/(high-low+1e-6), 0, 1)
            spacing = spacing if spacing > 0 else median_spacing
            h, w = image.shape
            size = min(int(round(140.0/max(spacing, .001))), min(h, w))
            y, x = (h-size)//2, (w-size)//2
            image = cv2.resize(image[y:y+size, x:x+size], (384, 384), interpolation=cv2.INTER_AREA)
            # Retain the parent's uint8 quantization before tensor conversion.
            planes[(slot, local)] = (image*255).astype(np.uint8)
            source_files.append(str(path))
    if len(source_files) != len(set(source_files)):
        raise ValueError("Selected source file occurs in multiple series slots")
    windows = np.stack([np.stack([planes[(spec.slot, index)] for index in spec.indices])
                        for spec in plan.triplets]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(windows)
    if norm == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor-mean)/std
    audit = {"study_uid": str(study_uid), "capacities": capacities,
             "triplets": len(plan.triplets), "unique_triplets": plan.unique_triplets,
             "repeated_triplets": plan.repeated_triplets,
             "sampled_slices": plan.source_count, "decoded_slices": len(set(decoded_files)),
             "series_uids": uids, "sampled_slices_per_series": [len(x) for x in plan.sampled_indices],
             "window_series_uids": [spec.series_uid for spec in plan.triplets],
             "window_source_files": [[str(pools[spec.slot][0][pools[spec.slot][2]+i][0])
                                      for i in spec.indices] for spec in plan.triplets]}
    return tensor, audit


class SeriesTripletDataset(Dataset):
    """Training K stays fixed; eval defaults to the parent's per-study dense K.

    `series` maps study UIDs to rows from train_series.csv/test_series.csv.
    Anonymous all_vols.npy stacks cannot enforce SeriesInstanceUID ownership;
    use raw series, or rebuild a cache retaining equivalent source metadata.
    """

    def __init__(self, ids, series, series_root, labels=None, *, train=False,
                 k=12, eval_k=None, norm="imagenet", aug=True, shortage="repeat"):
        self.ids, self.series, self.series_root, self.labels = ids, series, series_root, labels
        self.train, self.k, self.eval_k = train, k, eval_k
        self.norm, self.aug, self.shortage = norm, aug, shortage
        if not isinstance(k, int) or k < 1:
            raise ValueError("Training K must be positive")
        self.reader = None

    def __len__(self):
        return len(self.ids)

    def __getstate__(self):
        # DICOM reader closures are constructed inside each spawned worker.
        state = self.__dict__.copy()
        state["reader"] = None
        return state

    def load(self, index):
        uid = str(self.ids[index])
        if self.reader is None:
            self.reader = dicom_reader()
        # Plan the same dense candidate count as inference, then select K for
        # training. This keeps sparse studies' eval counts equal to the parent.
        x, audit = prepare_study_triplets(uid, self.series[uid], self.series_root,
                                          target_triplets=self.eval_k, norm="none",
                                          shortage=self.shortage, reader=self.reader)
        if self.train:
            first_positions = {}
            for i, paths in enumerate(audit["window_source_files"]):
                first_positions.setdefault(tuple(paths), i)
            candidates = torch.tensor(list(first_positions.values()))
            if self.shortage == "error" and self.k > len(candidates):
                raise ValueError("Insufficient training triplets without repetition")
            if self.k <= len(candidates):
                selected = candidates[torch.randperm(len(candidates))[:self.k]]
            else:
                selected = torch.cat([candidates[torch.randperm(len(candidates))]
                                      for _ in range((self.k+len(candidates)-1)//len(candidates))])[:self.k]
            x = x[selected]
            for key in ("window_series_uids", "window_source_files"):
                audit[key] = [audit[key][i] for i in selected.tolist()]
            audit["candidate_triplets"] = audit["triplets"]
            audit["triplets"] = self.k
            keys = [tuple(paths) for paths in audit["window_source_files"]]
            audit["unique_triplets"] = len(set(keys))
            audit["repeated_triplets"] = self.k-len(set(keys))
            if self.aug:
                x = (x * (0.9+0.2*torch.rand(()))).clamp(0, 1)
        if self.norm == "imagenet":
            x = (x-torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)) / torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
        elif self.norm != "none":
            raise ValueError("norm must be 'imagenet' or 'none'")
        return x, audit

    def __getitem__(self, index):
        x, _ = self.load(index)
        return x if self.labels is None else (x, torch.as_tensor(self.labels[str(self.ids[index])], dtype=torch.float32))

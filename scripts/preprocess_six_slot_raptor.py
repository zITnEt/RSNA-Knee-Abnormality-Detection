"""Six MRI slots with the saved Raptor V5 image preprocessing contract.

Runs on Kaggle. Outputs fully normalized float16 [K,3,384,384] study bags.
No DICOM data is downloaded to the local development computer.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from pydicom.pixels import apply_modality_lut

# Six-slot extension; original V5 used five slots with 18/14/12/8/12 references.
SLOTS = [
    ("SAG_FLUID_FS", "Sagittal", 1, 1, 18),
    ("COR_FLUID_FS", "Coronal", 1, 1, 12),
    ("AX_FLUID_FS", "Axial", 1, 1, 12),
    ("SAG_FLUID_NOFS", "Sagittal", 1, 0, 14),
    ("COR_T1", "Coronal", 0, 0, 8),
    ("SAG_T1", "Sagittal", 0, 0, 8),
]
MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
RECIPE = {
    "format": "rsna_six_slot_raptor_v5_v1", "slots": SLOTS,
    "source_span": [.02, .98], "intensity_percentiles": [2, 98],
    "intensity_reference": "original evenly spaced per-series reference slices",
    "crop_mm": 140, "cache_resolution": 336, "model_resolution": 384,
    "resize": "cv2.INTER_AREA to 336; torch bilinear align_corners=False to 384",
    "quantization": "uint8 after 336 resize, before /255 and 384 interpolation",
    "mean": [.485, .456, .406], "std": [.229, .224, .225],
    "dtype": "float16", "normalized": True, "source_budget": 64,
    "triplets": "sliding adjacent sampled slices in concatenated slot order, like parent V5",
    "missing_slots": "absent, never substitute a different category or fabricate pixels",
    "selector": "dicom_weight_type; positive fat-sat header evidence overrides metadata",
    "labels": "annotations are separate from pixels; gold74 excluded from training split",
}
RECIPE_SHA = hashlib.sha256(json.dumps(RECIPE, sort_keys=True).encode()).hexdigest()
LOCK = threading.Lock()


def json_write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def allocate(capacities, weights, target=64):
    capacities = np.asarray(capacities, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    ideal = target * weights / weights.sum()
    quotas = np.floor(ideal).astype(np.int64)
    for i in np.argsort(-(ideal-quotas), kind="stable")[:target-int(quotas.sum())]:
        quotas[i] += 1
    quotas = np.minimum(quotas, capacities)
    deficit = np.zeros(len(quotas))
    wanted = min(target, int(capacities.sum()))
    while quotas.sum() < wanted:
        available = quotas < capacities
        active = np.where(available, weights, 0)
        deficit[available] += active[available]/active.sum()
        choice = int(np.argmax(np.where(available, deficit, -np.inf)))
        quotas[choice] += 1
        deficit[choice] -= 1
    return quotas


def flag(value):
    try:
        value = float(value)
        return int(value) if value in (0, 1) else None
    except (ValueError, TypeError):
        return None


def weight_type(h):
    desc = re.sub(r"[_\-.]", " ", f"{getattr(h, 'SeriesDescription', '')} {getattr(h, 'SequenceName', '')}".lower())
    t1 = bool(re.search(r"\bt1\b|\bt1w\b", desc))
    t2 = bool(re.search(r"\bt2\b|\bt2w\b", desc))
    pdw = bool(re.search(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens", desc))
    tr, te = float(getattr(h, 'RepetitionTime', 'nan')), float(getattr(h, 'EchoTime', 'nan'))
    if t1 and not t2 and not pdw:
        return "T1"
    if t2 and not pdw:
        return "T2"
    if pdw:
        return "PD"
    if "GR" in str(getattr(h, "ScanningSequence", "")).upper():
        return "GRE"
    return "T1" if tr < 800 else "T2" if te > 60 else "PD" if tr >= 800 else "UNK"


def classify(row, directory):
    files = sorted(directory.glob("*.dcm"))
    if not files:
        return None
    h = pydicom.dcmread(files[len(files)//2], stop_before_pixels=True)
    kind = weight_type(h)
    # In this competition CSV, Fluid_Sensitive and Fat_Suppression are
    # effectively coupled. PD/T2 without FS must be recovered from the header,
    # as in the original recovered six-slot classifier.
    fluid = int(kind in ("PD", "T2")) if kind != "UNK" else flag(row.get("Fluid_Sensitive"))
    fs = flag(row.get("Fat_Suppression"))
    desc = re.sub(r"[_\-.]", " ", f"{getattr(h, 'SeriesDescription', '')} {getattr(h, 'SequenceName', '')}".lower())
    options = getattr(h, "ScanOptions", [])
    options = options.split('\\') if isinstance(options, str) else options
    opts = set(str(v).upper() for v in options)
    header_fs = bool(re.search(r"\bfs\b|fat\s*sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|water excit|\btirm\b|\bsting\b|\bfatsup\b|smart fat|\bwater\b", desc)) or bool(opts & {"FS", "FATSAT", "FAT_SAT", "FSAT"})
    if header_fs:
        fs = 1
    elif fs is None:
        fs = 0
    return {"uid": str(row["SeriesInstanceUID"]), "plane": row["Anatomical_Plane"],
            "fluid": fluid, "fs": fs, "weight": kind, "files": files,
            "description": str(getattr(h, "SeriesDescription", ""))}


def order_files(series):
    records, sop_uids = [], set()
    for file in series["files"]:
        h = pydicom.dcmread(file, stop_before_pixels=True)
        sop = str(h.SOPInstanceUID)
        if sop in sop_uids:
            raise ValueError(f"Duplicate SOPInstanceUID: {file}")
        sop_uids.add(sop)
        if str(h.SeriesInstanceUID) != series["uid"]:
            raise ValueError(f"Series UID mismatch: {file}")
        iop, ipp = getattr(h, "ImageOrientationPatient", None), getattr(h, "ImagePositionPatient", None)
        if iop is not None and ipp is not None and len(iop) == 6:
            position = float(np.dot(np.asarray(ipp, float), np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))))
        else:
            position = float(h.InstanceNumber)
        if not np.isfinite(position):
            raise ValueError(f"Nonfinite slice coordinate: {file}")
        spacing = float(getattr(h, "PixelSpacing", [.5])[0])
        records.append((position, str(file), spacing))
    records.sort(key=lambda r: (r[0], r[1]))
    good_spacing = [r[2] for r in records if np.isfinite(r[2]) and r[2] > 0]
    median = float(np.median(good_spacing)) if good_spacing else .5
    return [(file, spacing if np.isfinite(spacing) and spacing > 0 else median) for _, file, spacing in records]


def pixels(file):
    d = pydicom.dcmread(file)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(f"Invalid acquired pixels: {file}")
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max()-a
    return a


def crop_plane(a, spacing, low, high):
    a = np.clip((a-low)/(high-low+1e-6), 0, 1)
    h, w = a.shape
    size = min(int(round(140/max(spacing, .001))), min(h, w))
    y, x = (h-size)//2, (w-size)//2
    a = cv2.resize(a[y:y+size, x:x+size], (336, 336), interpolation=cv2.INTER_AREA)
    return (a*255).astype(np.uint8)


def model_ready(planes):
    # Preserve parent's /255, interpolation, then channel-specific normalization.
    volume = np.stack(planes)
    source = torch.from_numpy(volume.astype(np.float32)/255)
    resized = F.interpolate(source[:, None], size=(384, 384), mode="bilinear", align_corners=False)[:, 0]
    x = torch.stack([resized[:-2], resized[1:-1], resized[2:]], dim=1)
    return ((x-MEAN)/STD).numpy().astype(np.float16)


def prepare(uid, rows, series_root):
    candidates, warnings = [], []
    for row in rows:
        s = classify(row, series_root/uid/str(row["SeriesInstanceUID"]))
        if s is not None:
            candidates.append(s)
    used, pools, selection = set(), [], []
    for name, plane, fluid, fs, reference_count in SLOTS:
        matching = [s for s in candidates if s["uid"] not in used and s["plane"] == plane
                    and s["fluid"] == fluid and s["fs"] == fs
                    and (fluid == 1 or s["weight"] == "T1")]
        # Retain metadata row priority, matching the Raptor selection convention.
        if not matching:
            pools.append(None)
            selection.append({"slot": name, "present": False})
            continue
        series = matching[0]
        used.add(series["uid"])
        files = order_files(series)
        lo = int(len(files)*.02)
        hi = max(int(len(files)*.98)-1, lo)
        refs = np.linspace(lo, hi, reference_count).round().astype(int)
        pools.append((files, lo, hi, refs))
        selection.append({"slot": name, "present": True, "series_uid": series["uid"],
                          "description": series["description"], "weight": series["weight"]})
    capacities = [0 if p is None else p[2]-p[1]+1 for p in pools]
    quotas = allocate(capacities, [s[4] for s in SLOTS])
    planes, slot_ids, sources, intensity = [], [], [], []
    for slot, (pool, count) in enumerate(zip(pools, quotas)):
        if pool is None or not count:
            continue
        files, lo, hi, refs = pool
        picks = lo + np.linspace(0, hi-lo, int(count)).round().astype(int)
        arrays = {int(i): pixels(files[int(i)][0]) for i in sorted(set(refs)|set(picks))}
        low, high = np.percentile(np.concatenate([arrays[int(i)].ravel() for i in refs]), [2, 98])
        intensity.append({"slot": slot, "p2": float(low), "p98": float(high), "reference_indices": refs.tolist()})
        for i in picks:
            file, spacing = files[int(i)]
            planes.append(crop_plane(arrays[int(i)], spacing, low, high))
            slot_ids.append(slot)
            sources.append({"slot": slot, "file": Path(file).name, "series_uid": selection[slot]["series_uid"], "native_index": int(i)})
    if len(planes) < 3:
        raise ValueError("Fewer than three acquired slices across the six categories")
    x = model_ready(planes)
    ids = np.asarray(slot_ids, dtype=np.int8)
    triplet_slots = np.stack([ids[:-2], ids[1:-1], ids[2:]], 1)
    pure = (triplet_slots[:, 0] == triplet_slots[:, 1]) & (triplet_slots[:, 1] == triplet_slots[:, 2])
    audit = {"study_uid": uid, "recipe_sha256": RECIPE_SHA, "selection": selection,
             "capacities": capacities, "quotas": quotas.tolist(), "source_slices": len(planes),
             "windows": len(x), "cross_series_windows": int((~pure).sum()),
             "source_files": sources, "intensity": intensity, "warnings": warnings}
    if not np.isfinite(x).all() or tuple(x.shape[1:]) != (3, 384, 384):
        raise ValueError("Nonfinite or invalid model input")
    return {"images": x, "triplet_slot_ids": triplet_slots, "same_series": pure,
            "slot_present": np.asarray([s["present"] for s in selection], bool),
            "audit_json": np.asarray(json.dumps(audit, allow_nan=False))}, audit


def save_npz(path, data):
    tmp = path.with_suffix(".npz.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name, array in data.items():
            with z.open(name+".npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
    tmp.replace(path)


def find_root():
    for p in ("/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection"):
        if (Path(p)/"train_series.csv").is_file():
            return Path(p)
    raise FileNotFoundError("Attach rsna-knee-abnormality-detection competition data")


def run(shard=0, shards=32, limit=None, root=None, output=None, workers=2):
    print(json.dumps({'phase':'input_setup','shard':shard}), flush=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    root = find_root() if root is None else Path(root)
    output = Path(output or f"/kaggle/working/six_slot_raptor_shard_{shard:02d}")
    output.mkdir(parents=True, exist_ok=True)
    series = pd.read_csv(root/"train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    labels = pd.read_csv(root/"train.csv", dtype={"StudyInstanceUID": str})
    ids = sorted(labels.StudyInstanceUID.unique())[shard::shards]
    if limit:
        ids = ids[:limit]
    groups = {str(uid): g.to_dict("records") for uid, g in series.groupby("StudyInstanceUID", sort=False)}
    series_root = next((root/p for p in ("train_series", "train_images", "train") if (root/p).is_dir()), None)
    if series_root is None:
        raise FileNotFoundError("No competition training DICOM directory")
    json_write(output/"recipe.json", dict(RECIPE, sha256=RECIPE_SHA))
    labels[labels.StudyInstanceUID.isin(ids)].to_csv(output/"labels_official.csv", index=False)
    gold = set(labels.loc[labels[LABELS].notna().all(axis=1), "StudyInstanceUID"])
    # Never recursively scan the competition's millions of DICOM files to find
    # a small annotation CSV; use the two supported attached-dataset layouts.
    jev = next((p/'gold74_eval_set.csv' for p in (
        Path('/kaggle/input/datasets/riadmohamed42/jev-knee-labels'),
        Path('/kaggle/input/jev-knee-labels')) if (p/'gold74_eval_set.csv').is_file()), None)
    if jev is None:
        raise FileNotFoundError('Attach riadmohamed42/jev-knee-labels: gold74 metadata is required')
    if jev is not None:
        gold |= set(pd.read_csv(jev, dtype={"StudyInstanceUID": str}).StudyInstanceUID)
        final2 = jev.parent/"FINAL2_labels_scores.csv"
        if final2.is_file():
            table = pd.read_csv(final2, dtype={"StudyInstanceUID": str})
            table[table.StudyInstanceUID.isin(ids)].to_csv(output/"labels_jev_final2.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ids, "split": ["held_out_gold" if uid in gold else "train" for uid in ids]}).to_csv(output/"splits.csv", index=False)
    print(json.dumps({'phase':'metadata_ready','shard':shard,'studies':len(ids),'series_root':str(series_root)}), flush=True)
    start = time.time()
    results, failures = [], []
    def heartbeat(state="running"):
        value = {"state": state, "shard": shard, "shards": shards, "expected_studies": len(ids),
                 "completed_studies": len(results), "failed_studies": len(failures),
                 "elapsed_seconds": round(time.time()-start, 1), "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "output_bytes": sum(r["bytes"] for r in results), "recipe_sha256": RECIPE_SHA}
        json_write(output/"progress.json", value)
        print(json.dumps(value), flush=True)
    def one(uid):
        file = output/(uid+".npz")
        # Reserve worst-case remaining uncompressed bag size before processing.
        if shutil.disk_usage(output).free < 512*1024**2:
            raise OSError("Less than 512 MiB free: stop before corrupting output")
        data, audit = prepare(uid, groups[uid], series_root)
        save_npz(file, data)
        # Re-open and check the exact saved model tensor.
        with np.load(file, allow_pickle=False) as saved:
            if not np.array_equal(saved["images"], data["images"]):
                raise ValueError("Saved tensor round-trip mismatch")
        digest = hashlib.sha256()
        with file.open("rb") as f:
            for block in iter(lambda: f.read(8*1024**2), b""):
                digest.update(block)
        return {"study_uid": uid, "file": file.name, "windows": audit["windows"],
                "cross_series_windows": audit["cross_series_windows"], "slot_present": data["slot_present"].tolist(),
                "bytes": file.stat().st_size, "sha256": digest.hexdigest()}
    heartbeat()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, uid): uid for uid in ids}
        for future in as_completed(futures):
            uid = futures[future]
            try:
                row = future.result()
                results.append(row)
                with (output/"index.jsonl").open("a") as f:
                    f.write(json.dumps(row)+"\n")
            except Exception as exc:
                failure = {"study_uid": uid, "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                with (output/"failures.jsonl").open("a") as f:
                    f.write(json.dumps(failure)+"\n")
                print("STUDY_FAILED "+json.dumps(failure), flush=True)
            heartbeat()
    manifest = {"recipe_sha256": RECIPE_SHA, "shard": shard, "shards": shards,
                "expected_uids": ids, "completed": results, "failures": failures,
                "complete": len(results) == len(ids) and not failures}
    json_write(output/"manifest.json", manifest)
    heartbeat("complete" if manifest["complete"] else "incomplete")
    if failures:
        raise RuntimeError(f"Shard incomplete: {len(failures)} failed studies; see failures.jsonl")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=32)
    p.add_argument("--limit", type=int)
    p.add_argument("--root")
    p.add_argument("--output")
    a = p.parse_args()
    run(**vars(a))

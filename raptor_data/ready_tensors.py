"""Load exported tensors directly; performs no image preprocessing."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def per_slot_indices(slot_of_window, k, generator=None):
    """Choose k window indices, spreading them as evenly as possible over slots.

    slot_of_window: 1-D integer tensor giving each candidate window's slot.
    Slots are filled round-robin in a random order, one window at a time, up to
    each slot's capacity, so six present slots with k=12 get two windows each.
    Within a slot, windows are drawn without replacement. Only when the study
    has fewer than k candidate windows are extra windows repeated.
    """
    slot_of_window = torch.as_tensor(slot_of_window).long()
    if slot_of_window.ndim != 1 or not len(slot_of_window):
        raise ValueError('Need a nonempty 1-D slot assignment')
    if k < 1:
        raise ValueError('k must be positive')
    slots = torch.unique(slot_of_window).tolist()
    members = {s: torch.nonzero(slot_of_window == s).flatten() for s in slots}
    quota = {s: 0 for s in slots}
    remaining = k
    while remaining:
        order = torch.randperm(len(slots), generator=generator).tolist()
        progressed = False
        for position in order:
            s = slots[position]
            if quota[s] < len(members[s]) and remaining:
                quota[s] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    chosen = []
    for s in slots:
        if quota[s]:
            pool = members[s]
            chosen.append(pool[torch.randperm(len(pool), generator=generator)[:quota[s]]])
    chosen = torch.cat(chosen)
    if remaining:
        extra = torch.randint(len(slot_of_window), (remaining,), generator=generator)
        chosen = torch.cat([chosen, extra])
    return chosen[torch.randperm(len(chosen), generator=generator)]


class ReadyTripletDataset(Dataset):
    """Use K=12 for fixed-size training bags, or k=None for full study bags.

Attach the preprocessing notebook outputs or published private shard datasets.
Images are already cropped, resized, intensity-scaled, and ImageNet-normalized.
The stored float16 tensor is suitable for training under torch.autocast.

windows='all' keeps every exported window; windows='pure' keeps only windows
whose three slices come from one series (same_series), falling back to all
windows for a study that has none. sampling='uniform' draws k windows
uniformly; sampling='per_slot' spreads k windows evenly over present slots.
"""
    def __init__(self, roots, *, split='train', k=12, labels=None,
                 sampling='uniform', windows='all'):
        self.k = k
        if k is not None and (not isinstance(k, int) or k < 1):
            raise ValueError('k must be positive or None')
        if sampling not in ('uniform', 'per_slot'):
            raise ValueError(f'Unknown sampling: {sampling}')
        if windows not in ('all', 'pure'):
            raise ValueError(f'Unknown window policy: {windows}')
        self.sampling, self.windows = sampling, windows
        roots = [Path(r) for r in roots]
        tables = [pd.read_csv(p, dtype={'StudyInstanceUID': str})
                  for root in roots for p in root.rglob('splits.csv')]
        if not tables:
            raise ValueError('No preprocessing splits.csv found')
        splits = pd.concat(tables, ignore_index=True)
        if splits.StudyInstanceUID.duplicated().any():
            raise ValueError('Attach each shard once; duplicate study IDs found')
        wanted = set(splits.loc[splits['split']==split, 'StudyInstanceUID'])
        self.files = sorted(p for root in roots for p in root.rglob('*.npz') if p.stem in wanted)
        if {p.stem for p in self.files} != wanted:
            raise ValueError('Attached shard metadata and tensor files disagree')
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        with np.load(file, allow_pickle=False) as saved:
            slots = torch.from_numpy(saved['triplet_slot_ids'])
            same_series = torch.from_numpy(saved['same_series']).bool()
            present = torch.from_numpy(saved['slot_present'])
            x = torch.from_numpy(saved['images'])
        if x.dtype != torch.float16 or x.ndim != 4 or tuple(x.shape[1:]) != (3, 384, 384) or not len(x):
            raise ValueError(f'Invalid preprocessed image contract: {file}')
        if len(slots) != len(x) or len(same_series) != len(x):
            raise ValueError(f'Window metadata length mismatch: {file}')
        candidates = torch.arange(len(x))
        if self.windows == 'pure' and same_series.any():
            candidates = candidates[same_series]
        if self.k is not None:
            if self.sampling == 'per_slot':
                # A window's slot is the slot of its middle slice.
                local = per_slot_indices(slots[candidates, 1], self.k)
            else:
                local = torch.randperm(len(candidates))[:self.k]
                if len(local) < self.k:
                    local = torch.cat([local, torch.randint(len(candidates), (self.k-len(local),))])
            indices = candidates[local]
        else:
            indices = candidates
        x, slots, same_series = x[indices], slots[indices], same_series[indices]
        item = {'images': x, 'triplet_slot_ids': slots, 'same_series': same_series,
                'slot_present': present, 'study_uid': file.stem}
        if self.labels is not None:
            item['targets'] = torch.as_tensor(self.labels[file.stem], dtype=torch.float32)
        return item

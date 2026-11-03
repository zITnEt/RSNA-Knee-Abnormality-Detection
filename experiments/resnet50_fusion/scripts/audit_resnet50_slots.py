"""Check V2 triplet metadata while a detached cloud training run continues."""
import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[1]
files = sorted((root / 'data').rglob('*.npz'))
output = root / 'run' / 'slot_audit.json'
errors = []
for index, file in enumerate(files, 1):
    try:
        with np.load(file, allow_pickle=False) as saved:
            slots = saved['triplet_slot_ids']
            if slots.ndim != 2 or slots.shape[1] != 3 or not np.all(slots == slots[:, :1]):
                errors.append([file.name, 'inconsistent_slots'])
            if not np.all(saved['same_series']):
                errors.append([file.name, 'cross_series'])
    except Exception as exc:
        errors.append([file.name, type(exc).__name__])
    if index % 500 == 0 or index == len(files):
        output.write_text(json.dumps({'checked': index, 'total': len(files), 'errors': errors[:10],
                                      'error_count': len(errors), 'complete': index == len(files)}))
if errors:
    raise SystemExit(1)

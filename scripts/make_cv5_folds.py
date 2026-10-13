"""Write configs/cv5_folds.csv with the same function the trainer uses, and print fold balance.

    python scripts/make_cv5_folds.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from raptor_data.cv_folds import stratified_folds  # noqa: E402

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
split = json.loads((ROOT/'results/fusion_ddp_run1/training_split.json').read_text())
ids = sorted(split['train'] + split['validation'])
gold = set(split['heldout_gold'])
assert len(ids) == 4333 and len(set(ids)) == 4333 and not (set(ids) & gold) and len(gold) == 74
table = pd.read_csv(ROOT/'data/external/jev_knee/original/HYBRID_labels_binary.csv',
                    dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
y = table.loc[ids, LABELS].to_numpy()
folds = stratified_folds(ids, y, 5, 20261010)
frame = pd.DataFrame(sorted(folds.items()), columns=['StudyInstanceUID', 'fold'])
out = ROOT/'configs/cv5_folds.csv'
frame.to_csv(out, index=False)
f = frame.set_index('StudyInstanceUID').loc[ids, 'fold'].to_numpy()
print('studies per fold', np.bincount(f).tolist(), 'train per fold', (len(ids) - np.bincount(f)).tolist())
print(pd.DataFrame({name: np.bincount(f[y[:, j] == 1], minlength=5) for j, name in enumerate(LABELS)}).to_string())
print('gold74 overlap', len(set(folds) & gold))

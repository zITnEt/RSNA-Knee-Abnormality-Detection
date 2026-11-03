"""Export all available history fields without loading model checkpoints."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'audit/training'
OUT = ROOT / 'results/training_analysis'
files = {
    'Global96': 'global96/training_history_global96_e24.json',
    'D4': 'd4/training_history_d4.json',
    'Repair': 'repair/training_history_terminal_epoch18.json',
    'Residual': 'residual/training_history_through_epoch10.json',
}
histories, summaries, configs = [], [], {}
for name, file in files.items():
    obj = json.loads((SRC / file).read_text())
    h = pd.json_normalize(obj['history'], sep='.')
    h.insert(0, 'model', name)
    histories.append(h)
    configs[name] = obj.get('config', {})
    best = h.loc[h.gold_auc.idxmax()]
    last = h.iloc[-1]
    summaries.append(dict(model=name, epochs=len(h), best_epoch=int(best.epoch),
        best_single_epoch_auc=best.gold_auc, final_epoch_auc=last.gold_auc,
        best_minus_final_auc=best.gold_auc-last.gold_auc,
        initial_train_loss=h.iloc[0].train_loss, final_train_loss=last.train_loss,
        median_train_seconds=h.train_seconds.median() if 'train_seconds' in h else None,
        median_gold_seconds=h.gold_seconds.median() if 'gold_seconds' in h else None,
        max_allocated_gib=h.peak_allocated_gib.max() if 'peak_allocated_gib' in h else None))
pd.concat(histories, ignore_index=True).to_csv(OUT / 'full_epoch_metrics.csv', index=False)
pd.DataFrame(summaries).to_csv(OUT / 'training_history_summary.csv', index=False)
(OUT / 'training_configurations.json').write_text(json.dumps(configs, indent=2), encoding='utf-8')
print(pd.DataFrame(summaries).to_string(index=False))

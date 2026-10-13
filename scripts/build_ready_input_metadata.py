"""Prepare the direct-input notebook with all 32 completed output sources."""
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'kaggle/six_slot_training_inputs'
folder.mkdir(parents=True,exist_ok=True)
shutil.copyfile(ROOT/'kaggle/six_slot_training_inputs.ipynb',folder/'training_inputs.ipynb')
metadata={'id':'zafarnematov/rsna-six-slot-training-inputs',
          'title':'RSNA Six Slot Training Inputs','code_file':'training_inputs.ipynb',
          'language':'python','kernel_type':'notebook','is_private':True,
          'enable_gpu':False,'enable_internet':False,'dataset_sources':[],
          'competition_sources':[],'model_sources':[],
          'kernel_sources':[f'zafarnematov/rsna-six-slot-raptor-v5-preprocessing-{i:02d}' for i in range(32)]}
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(str(folder))

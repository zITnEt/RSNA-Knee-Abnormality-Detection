"""Lossless re-encoding of a Kaggle-processed pilot, never pixel preprocessing."""
import json
import time
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=next((ROOT/'data/min_overlap_384_validation').rglob('*.npz'))
results=[]
for level in (3,6):
    target=source.with_name('compression_level_'+str(level)+'.npz');start=time.time()
    with zipfile.ZipFile(source) as src,zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=level) as dst:
        for info in src.infolist():dst.writestr(info.filename,src.read(info.filename))
    with zipfile.ZipFile(source) as src,zipfile.ZipFile(target) as dst:
        assert src.namelist()==dst.namelist()
        assert all(src.read(name)==dst.read(name) for name in src.namelist())
    results.append(dict(level=level,bytes=target.stat().st_size,original_bytes=source.stat().st_size,
        reduction_percent=100*(1-target.stat().st_size/source.stat().st_size),elapsed_seconds=time.time()-start,
        identical_array_bytes=True))
(ROOT/'results/min_overlap_384/compression_benchmark.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))

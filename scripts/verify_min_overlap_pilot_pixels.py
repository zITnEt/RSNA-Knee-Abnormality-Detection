"""Compare existing preprocessed arrays; never preprocess source data locally."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
old_base=next((ROOT/'data/series_safe_384_staging/pilot').rglob('recipe.json')).parent
files=list((ROOT/'data/min_overlap_384_validation').rglob('*.npz'))
assert len(files)==1
new_file=files[0];old_file=old_base/(new_file.stem+'.npz')
with np.load(old_file,allow_pickle=False) as old,np.load(new_file,allow_pickle=False) as new:
    first=json.loads(str(old['audit_json']));second=json.loads(str(new['audit_json']))
    refs={r['slot']:r for r in first['intensity']}
    for reference in second['intensity']:
        assert reference==refs[reference['slot']], 'Intensity references or percentiles changed'
    old_sources={r['sop_uid']:i for i,r in enumerate(first['source_files'])}
    shared=0
    for i,source in enumerate(second['source_files']):
        if source['sop_uid'] in old_sources:
            np.testing.assert_array_equal(new['slices_uint8'][i],old['slices_uint8'][old_sources[source['sop_uid']]])
            shared+=1
    assert shared>0
    report=dict(study_uid=new_file.stem,matching_shared_slices=shared,reference_percentiles_identical=True,
        images_shape=list(new['images'].shape),unique_slices=len(new['slices_uint8']),
        repeated_slice_uses=second['repeated_slice_uses'],optimal_unique_slices=second['optimal_unique_slices'])
(ROOT/'results/min_overlap_384/pilot_pixel_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

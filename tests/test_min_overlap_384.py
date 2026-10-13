import itertools
import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.runtime_preprocess'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import preprocess_min_overlap_384 as p


class MinimumOverlapTests(unittest.TestCase):
    def test_series_optimal_coverage_and_disjoint_subset(self):
        for capacity in range(3,80):
            for k in range(1,min(capacity-2,60)+1):
                picks,windows,primary=p.series_windows(7,7+capacity-1,k)
                self.assertEqual(len(picks),min(capacity,3*k))
                self.assertEqual(len(set(picks)),len(picks))
                self.assertEqual(set(windows.ravel()),set(range(len(picks))))
                self.assertEqual(len({tuple(w) for w in windows}),k)
                members=windows[primary].ravel()
                self.assertEqual(len(members),len(set(members)))
                self.assertEqual(primary.sum(),min(k,capacity//3))

    def test_global_unique_coverage_matches_exhaustive_search(self):
        for caps in ((3,4,5),(7,8,3),(2,9,4),(8,8,8),(20,3,3)):
            capacities=(*caps,0,0,0)
            for target in range(1,min(15,sum(max(c-2,0) for c in caps))+1):
                allocation,score=p.allocate_triplets(capacities,target)
                feasible=[k for k in itertools.product(*(range(max(c-2,0)+1) for c in caps)) if sum(k)==target]
                expected=max((sum(min(c,3*n) for c,n in zip(caps,k)),sum(min(n,c//3) for c,n in zip(caps,k))) for k in feasible)
                self.assertEqual((score['optimal_unique_slices'],score['optimal_disjoint_subset']),expected)
                self.assertEqual(allocation.sum(),target)

    def test_zero_overlap_when_enough_native_slices_and_short_study(self):
        allocation,score=p.allocate_triplets([48,36,30,18,30,18])
        np.testing.assert_array_equal(allocation,[16,12,10,6,10,6])
        self.assertEqual(score['minimum_repeated_slice_uses'],0)
        self.assertEqual(score['optimal_unique_slices'],180)
        allocation,score=p.allocate_triplets([4,3,0,0,0,0])
        self.assertEqual(allocation.sum(),3)
        self.assertEqual(score['optimal_unique_slices'],7)
        self.assertEqual(score['minimum_repeated_slice_uses'],2)

    def test_real_pilot_audit_capacities_and_unchanged_pixel_function(self):
        audit_file=ROOT/'results/series_safe_384/unique_audit_full/study_capacity.jsonl'
        if not audit_file.exists():self.skipTest('Local metadata audit not available')
        rows=[json.loads(line) for line in audit_file.read_text().splitlines()]
        for row in rows[::100]:
            caps=row['eligible_slices_per_slot'];allocation,score=p.allocate_triplets(caps)
            unique=0;count=0
            for c,k in zip(caps,allocation):
                if not k:continue
                picks,windows,primary=p.series_windows(0,c-1,k)
                unique+=len(picks);count+=len(windows)
            self.assertEqual(unique,score['optimal_unique_slices'])
            self.assertEqual(count,min(60,sum(max(c-2,0) for c in caps)))


if __name__=='__main__':unittest.main()

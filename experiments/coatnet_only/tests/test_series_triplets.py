import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from raptor_data.series_triplets import (
    SLOTS, SeriesTripletDataset, balanced_slot_quotas, plan_triplets,
    prepare_study_triplets,
)


class SeriesTripletTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def assert_valid(self, plan, capacities, uids):
        self.assertEqual(len(plan.triplets), plan.target_triplets)
        for spec in plan.triplets:
            self.assertEqual(spec.series_uid, uids[spec.slot])
            self.assertEqual(len(set(spec.indices)), 3)
            self.assertEqual(tuple(sorted(spec.indices)), spec.indices)
            self.assertTrue(all(0 <= i < capacities[spec.slot] for i in spec.indices))

    def test_five_series_preserve_94_with_104_slices(self):
        capacities = [100]*5
        uids = [f"series-{i}" for i in range(5)]
        plan = plan_triplets(capacities, uids, [s[2] for s in SLOTS])
        self.assert_valid(plan, capacities, uids)
        self.assertEqual(plan.target_triplets, 94)
        self.assertEqual(plan.source_count, 104)
        self.assertEqual(plan.unique_triplets, 94)
        self.assertEqual(plan.repeated_triplets, 0)

    def test_missing_series_and_saturated_quotas_redistribute(self):
        capacities, uids = [3, 5, 200, 0, 0], ["a", "b", "c", None, None]
        plan = plan_triplets(capacities, uids, [18, 14, 12, 8, 12])
        self.assert_valid(plan, capacities, uids)
        self.assertEqual(plan.target_triplets, 94)
        self.assertEqual(plan.source_count, 100)  # 94 + 2 per used series
        self.assertEqual(len(plan.sampled_indices[0]), 3)
        self.assertEqual(len(plan.sampled_indices[1]), 5)
        self.assertEqual(plan.repeated_triplets, 0)

    def test_low_capacity_repeats_whole_safe_triplets_keeps_parent_count(self):
        capacities, uids = [4, 4, 2], ["a", "b", "short"]
        plan = plan_triplets(capacities, uids)
        self.assert_valid(plan, capacities, uids)
        self.assertEqual(plan.target_triplets, 8)  # parent's min(96,10)-2
        self.assertEqual(plan.unique_triplets, 4)
        self.assertEqual(plan.repeated_triplets, 4)
        self.assertEqual(plan.sampled_indices[2], ())
        with self.assertRaises(ValueError):
            plan_triplets(capacities, uids, shortage="error")

    def test_one_series_matches_original_sliding_window_count(self):
        plan = plan_triplets([120], ["a"])
        self.assertEqual(plan.source_count, 96)
        self.assertEqual(plan.target_triplets, 94)
        self.assertEqual(plan.repeated_triplets, 0)

    def test_fixed_count_deterministic_plan_and_impossible_study(self):
        a = plan_triplets([30, 30], ["a", "b"], target_triplets=12)
        self.assertEqual(a, plan_triplets([30, 30], ["a", "b"], target_triplets=12))
        self.assertEqual(a.source_count, 16)
        for capacities, uids in [([2, 2, 2], ["a", "b", "c"]),
                                 ([30, 30], ["a", "a"]), ([30], [None]),
                                 ([-1], ["a"]), ([3.5], ["a"])]:
            with self.assertRaises(ValueError):
                plan_triplets(capacities, uids)

    def synthetic_reader(self):
        def order(directory):
            return [(str(Path(directory)/f"{i:03}.dcm"), 1.0) for i in range(30)], 1.0
        def pixel(path):
            return np.full((8, 8), float(Path(path).stem), dtype=np.float32)
        return order, pixel

    def test_preprocessing_checks_all_channel_series_and_384_output(self):
        rows = [dict(SeriesInstanceUID="sag", Anatomical_Plane="Sagittal", Fluid_Sensitive=1),
                dict(SeriesInstanceUID="cor", Anatomical_Plane="Coronal", Fluid_Sensitive=1)]
        windows, audit = prepare_study_triplets("study", rows, "root", target_triplets=8,
                                                norm="none", reader=self.synthetic_reader())
        self.assertEqual(tuple(windows.shape), (8, 3, 384, 384))
        self.assertEqual(audit["sampled_slices"], 12)
        self.assertEqual(audit["triplets"], 8)
        for uid, files in zip(audit["window_series_uids"], audit["window_source_files"]):
            self.assertEqual({Path(f).parent.name for f in files}, {uid})
            self.assertEqual(len(set(files)), 3)
        # Channels retain physical sampled order, and the exact same normalized
        # tensor is subsequently passed to both model backbones.
        self.assertTrue(torch.all(windows[:, 0] < windows[:, 1]))
        self.assertTrue(torch.all(windows[:, 1] < windows[:, 2]))

    def test_training_dataset_preserves_requested_k_and_safe_metadata(self):
        rows = [dict(SeriesInstanceUID="sag", Anatomical_Plane="Sagittal", Fluid_Sensitive=1)]
        dataset = SeriesTripletDataset(["study"], {"study": rows}, "root",
                                       {"study": [0.0]*12}, train=True, k=12, norm="none", aug=False)
        with patch("raptor_data.series_triplets.dicom_reader", return_value=self.synthetic_reader()):
            windows, audit = dataset.load(0)
            x, labels = dataset[0]
        self.assertEqual(tuple(windows.shape), (12, 3, 384, 384))
        self.assertEqual(tuple(x.shape), (12, 3, 384, 384))
        self.assertEqual(labels.shape, (12,))
        self.assertEqual(audit["triplets"], 12)
        self.assertEqual(len(audit["window_source_files"]), 12)
        self.assertEqual(set(audit["window_series_uids"]), {"sag"})
        self.assertIsNone(dataset.__getstate__()["reader"])

    def test_randomized_capacities_always_preserve_cardinality_and_boundaries(self):
        rng = np.random.default_rng(5)
        uids = [f"s{i}" for i in range(5)]
        for _ in range(100):
            capacities = rng.integers(0, 100, 5).tolist()
            plan = plan_triplets(capacities, uids, [18, 14, 12, 8, 12])
            self.assert_valid(plan, capacities, uids)
            used_series = sum(bool(indices) for indices in plan.sampled_indices)
            self.assertEqual(plan.source_count, plan.unique_triplets+2*used_series)
            self.assertEqual(plan.target_triplets, min(96, sum(capacities))-2)

    def test_balanced_quotas_split_k_evenly_with_random_remainder(self):
        gen = torch.Generator().manual_seed(0)
        extras = [0]*5
        for _ in range(200):
            quotas = balanced_slot_quotas([20, 18, 15, 12, 10], 24, gen)
            self.assertEqual(sum(quotas), 24)
            self.assertTrue(all(q in (4, 5) for q in quotas))
            for s, q in enumerate(quotas):
                extras[s] += q == 5
        self.assertTrue(all(e > 0 for e in extras))  # remainder rotates over slots

    def test_balanced_quotas_skip_empty_and_redistribute_small_slots(self):
        self.assertEqual(balanced_slot_quotas([0, 30, 30, 0], 24), [0, 12, 12, 0])
        self.assertEqual(balanced_slot_quotas([2, 100, 100], 24), [2, 11, 11])
        quotas = balanced_slot_quotas([1, 3, 100], 24)
        self.assertEqual(quotas[:2], [1, 3])
        self.assertEqual(quotas[2], 20)

    def test_balanced_quotas_spread_repeats_when_study_is_short(self):
        quotas = balanced_slot_quotas([3, 2], 24)
        self.assertEqual(sum(quotas), 24)
        self.assertEqual(sorted(quotas), [12, 12])
        # A slot holding more unique windows than K/n keeps all of them.
        self.assertEqual(balanced_slot_quotas([15, 1], 24), [15, 9])
        for bad in [([0, 0], 24), ([5], 0), ([-1, 5], 4)]:
            with self.assertRaises(ValueError):
                balanced_slot_quotas(*bad)

    def test_training_k24_is_balanced_across_series_slots(self):
        rows = [dict(SeriesInstanceUID="sag", Anatomical_Plane="Sagittal", Fluid_Sensitive=1),
                dict(SeriesInstanceUID="cor", Anatomical_Plane="Coronal", Fluid_Sensitive=1),
                dict(SeriesInstanceUID="ax", Anatomical_Plane="Axial", Fluid_Sensitive=0)]
        dataset = SeriesTripletDataset(["study"], {"study": rows}, "root",
                                       {"study": [0.0]*12}, train=True, norm="none", aug=False)
        self.assertEqual(dataset.k, 24)
        with patch("raptor_data.series_triplets.dicom_reader", return_value=self.synthetic_reader()):
            for _ in range(5):
                windows, audit = dataset.load(0)
                self.assertEqual(tuple(windows.shape), (24, 3, 384, 384))
                counts = {uid: audit["window_series_uids"].count(uid) for uid in ("sag", "cor", "ax")}
                self.assertEqual(counts, {"sag": 8, "cor": 8, "ax": 8})
                self.assertEqual(sorted(audit["slot_quotas"].values()), [8, 8, 8])
                self.assertEqual(audit["repeated_triplets"], 0)
                for slot, uid, files in zip(audit["window_slots"], audit["window_series_uids"],
                                            audit["window_source_files"]):
                    self.assertEqual(audit["series_uids"][slot], uid)
                    self.assertEqual({Path(f).parent.name for f in files}, {uid})


if __name__ == "__main__":
    unittest.main()

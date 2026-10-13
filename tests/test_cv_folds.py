"""5-fold split and CV metric bookkeeping (pure numpy; no GPU or timm needed)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raptor_data.cv_folds import combined_score, metric_rows, stratified_folds  # noqa: E402


def synthetic(n=4333, seed=0):
    rng = np.random.default_rng(seed)
    rates = np.array([.05, .03, .45, .15, .40, .12, .30, .55, .20, .10, .08, .02])
    y = (rng.random((n, 12)) < rates).astype(int)
    ids = [f'1.2.3.{i}' for i in rng.permutation(n)]
    return ids, y


class Folds(unittest.TestCase):
    def test_partition_sizes_and_balance(self):
        ids, y = synthetic()
        folds = stratified_folds(ids, y)
        f = np.array([folds[u] for u in ids])
        self.assertEqual(set(f.tolist()), {0, 1, 2, 3, 4})
        sizes = np.bincount(f)
        self.assertLessEqual(sizes.max() - sizes.min(), 3)
        for j in range(12):
            pos = np.bincount(f[y[:, j] == 1], minlength=5)
            self.assertLessEqual(pos.max() - pos.min(), 2, j)

    def test_deterministic_and_order_independent(self):
        ids, y = synthetic(800)
        a = stratified_folds(ids, y)
        perm = np.random.default_rng(5).permutation(len(ids))
        b = stratified_folds([ids[i] for i in perm], y[perm])
        self.assertEqual(a, b)
        self.assertNotEqual(a, stratified_folds(ids, y, seed=1))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            stratified_folds(['a', 'a'], np.zeros((2, 12)))
        with self.assertRaises(ValueError):
            stratified_folds(['a', 'b'], np.full((2, 12), 0.5))


class Metrics(unittest.TestCase):
    def test_combined_score_is_mean(self):
        by = {'gold74': {'macro_auc': 0.92}, 'validation': {'macro_auc': 0.96}}
        self.assertAlmostEqual(combined_score(by), 0.94)
        self.assertEqual(combined_score({'gold74': {'macro_auc': 0.9}}), 0.9)

    def test_rows_for_raw_and_ema_every_epoch(self):
        ev = lambda g, v: {'gold74': {'macro_auc': g}, 'validation': {'macro_auc': v}}
        history = [dict(step=0, epoch=0, evaluation={'raw': ev(.92, .95)}),
                   dict(step=100, epoch=1, training_loss=.5, evaluation={'raw': ev(.93, .94), 'ema': ev(.91, .93)})]
        rows = metric_rows(history, fold=2)
        self.assertEqual([(r['epoch'], r['weights']) for r in rows], [(0, 'raw'), (0, 'ema'), (1, 'raw'), (1, 'ema')])
        self.assertAlmostEqual(rows[3]['validation_accuracy'], .92)
        self.assertEqual(rows[1]['gold74_macro_auc'], .92)
        self.assertTrue(rows[1]['note'])
        self.assertEqual({r['fold'] for r in rows}, {2})


if __name__ == '__main__':
    unittest.main()

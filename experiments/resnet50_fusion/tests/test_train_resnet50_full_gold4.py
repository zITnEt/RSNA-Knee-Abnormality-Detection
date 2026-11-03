import math
import unittest

from scripts.train_resnet50_full_gold4 import make_split, reference_lr_factor


class FullGoldTrainingTests(unittest.TestCase):
    def test_all_non_gold_studies_train_and_gold_is_only_evaluation_set(self):
        ids = [f'train-{i:04d}' for i in range(4333)]
        gold = [f'gold-{i:02d}' for i in range(74)]
        split = make_split(ids[::-1], gold[::-1], 20261013)
        self.assertEqual(set(split), {'train', 'heldout_gold'})
        self.assertEqual(split['train'], sorted(ids))
        self.assertEqual(split['heldout_gold'], sorted(gold))
        self.assertFalse(set(split['train']) & set(split['heldout_gold']))
        with self.assertRaises(ValueError):
            make_split(ids, [ids[0]], 20261013)

    def test_each_new_epoch_replays_exact_prior_epoch_learning_rates(self):
        def old_factor(step):
            if step < 100:
                return .01 + .99 * (step + 1) / 100
            return .01 + .99 * .5 * (1 + math.cos(math.pi * min(1, (step - 100) / 2340)))

        for epoch in range(4):
            new = [reference_lr_factor(epoch * 271 + i, 271) for i in range(271)]
            old = [old_factor(epoch * 244 + i) for i in range(244)]
            self.assertEqual(new[0], old[0])
            self.assertEqual(new[-1], old[-1])
            self.assertEqual(set(new), set(old))
            self.assertTrue(all(new[i] >= new[i + 1] for i in range(271 - 1)) if epoch else True)


if __name__ == '__main__':
    unittest.main()

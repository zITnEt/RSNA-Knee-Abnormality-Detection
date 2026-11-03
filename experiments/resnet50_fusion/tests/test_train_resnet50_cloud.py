import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.train_resnet50_cloud import EMA, atomic_save, make_split, metrics, train_mode


class CloudTrainingTests(unittest.TestCase):
    def test_split_is_reproducible_and_gold_is_excluded(self):
        ids = [f'train-{i:04d}' for i in range(4333)]
        gold = [f'gold-{i:02d}' for i in range(74)]
        split = make_split(ids, gold, 20261013)
        self.assertEqual(split, make_split(ids[::-1], gold[::-1], 20261013))
        self.assertEqual([len(split[k]) for k in ('train', 'validation', 'heldout_gold')], [3900, 433, 74])
        self.assertFalse(set(split['train']) & set(split['validation']))
        self.assertEqual(set(split['train']) | set(split['validation']), set(ids))
        self.assertFalse(set(ids) & set(split['heldout_gold']))
        with self.assertRaises(ValueError):
            make_split(ids, [ids[0]], 1)

    def test_accuracy_masks_unknown_gold_and_auc_skips_single_class(self):
        truth = np.zeros((3, 12), np.float32)
        truth[:, 0] = [0, 1, np.nan]
        logits = np.full_like(truth, -5.)
        logits[:, 0] = [-5, 5, 100]
        result = metrics(truth, logits)
        self.assertEqual(result['accuracy'], 1.)
        self.assertEqual(result['macro_auc'], 1.)
        self.assertEqual(result['labeled_entries'], 35)
        self.assertEqual(result['per_finding']['ACL']['available'], 2)
        self.assertIsNone(result['per_finding']['MCL']['auc'])
        self.assertLess(result['loss'], .01)
        truth[0, 1] = .2
        with self.assertRaises(ValueError):
            metrics(truth, logits)

    def make_model(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        model.register_buffer('counter', torch.tensor(0, dtype=torch.int64))
        return model

    def test_ema_math_float32_and_integer_buffer_copy(self):
        model = self.make_model().half()
        ema = EMA(model, .998)
        initial = copy.deepcopy(ema.shadow)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(2)
            model.counter.add_(3)
        ema.update(model)
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.assertEqual(ema.shadow[name].dtype, torch.float32)
                torch.testing.assert_close(ema.shadow[name], initial[name]*(2/11)+value.float()*(9/11))
            else:
                torch.testing.assert_close(ema.shadow[name], value)
        self.assertEqual(ema.updates, 1)

    def test_ema_swap_restores_all_raw_state_even_after_evaluation_error(self):
        model = self.make_model()
        ema = EMA(model)
        with torch.no_grad():
            model[0].weight.add_(3)
            model[1].running_mean.add_(2)
            model.counter.add_(4)
        raw = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(RuntimeError, 'evaluation'):
            with ema.installed(model):
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, ema.shadow[name])
                raise RuntimeError('evaluation failed')
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, raw[name])

    def test_checkpoint_ema_resume_matches_uninterrupted_updates(self):
        model = self.make_model()
        ema = EMA(model)
        for i in range(3):
            with torch.no_grad():
                model[0].weight.add_(i+1)
            ema.update(model)
        # Keep test artifacts under the writable workspace on restricted Windows hosts.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory)/'checkpoint.pt'
            atomic_save(dict(model=model.state_dict(), ema=ema.state_dict()), path)
            saved = torch.load(path, weights_only=True)
        restored = self.make_model()
        restored.load_state_dict(saved['model'])
        resumed = EMA(restored)
        resumed.load_state_dict(saved['ema'], restored)
        with torch.no_grad():
            model[0].weight.add_(4)
            restored[0].weight.add_(4)
        ema.update(model); resumed.update(restored)
        self.assertEqual(ema.updates, resumed.updates)
        for name in ema.shadow:
            torch.testing.assert_close(ema.shadow[name], resumed.shadow[name], rtol=0, atol=0)

    def test_batchnorm_statistics_frozen_but_affine_remains_trainable(self):
        model = self.make_model()
        before = model[1].running_mean.clone()
        train_mode(model)
        model(torch.randn(3, 2)).sum().backward()
        torch.testing.assert_close(model[1].running_mean, before)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertIsNotNone(model[1].weight.grad)


if __name__ == '__main__':
    unittest.main()

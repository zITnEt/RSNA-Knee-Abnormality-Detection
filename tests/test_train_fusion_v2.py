"""CPU checks for the v2 fine-tuning recipe (no medical images or pretrained weights)."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

from raptor_data.ready_tensors import ReadyTripletDataset, per_slot_indices  # noqa: E402
import train_fusion_v2 as T  # noqa: E402


class PerSlotSampling(unittest.TestCase):
    def test_six_slots_two_each(self):
        slots = torch.tensor(sum([[s]*n for s, n in enumerate([16, 10, 9, 12, 6, 5])], []))
        for _ in range(50):
            idx = per_slot_indices(slots, 12)
            self.assertEqual(len(idx), 12)
            self.assertEqual(len(set(idx.tolist())), 12)
            self.assertEqual(torch.bincount(slots[idx], minlength=6).tolist(), [2]*6)

    def test_four_slots_three_each(self):
        slots = torch.tensor(sum([[s]*n for s, n in zip([0, 2, 3, 5], [20, 15, 10, 8])], []))
        idx = per_slot_indices(slots, 12)
        counts = torch.bincount(slots[idx], minlength=6)
        self.assertEqual(counts[[0, 2, 3, 5]].tolist(), [3, 3, 3, 3])

    def test_capacity_redistribution(self):
        slots = torch.tensor([0]*30 + [1] + [2]*2)
        idx = per_slot_indices(slots, 12)
        counts = torch.bincount(slots[idx], minlength=3).tolist()
        self.assertEqual(counts[1], 1)
        self.assertEqual(counts[2], 2)
        self.assertEqual(counts[0], 9)
        self.assertEqual(len(set(idx.tolist())), 12)

    def test_shortage_repeats_only_when_needed(self):
        slots = torch.tensor([0, 0, 1, 1, 2])
        idx = per_slot_indices(slots, 12)
        self.assertEqual(len(idx), 12)
        self.assertEqual(set(idx.tolist()), {0, 1, 2, 3, 4})


def write_study(directory, uid, slot_ids, same_series):
    n = len(slot_ids)
    images = np.random.default_rng(abs(hash(uid)) % 2**32).standard_normal((n, 3, 384, 384)).astype(np.float16)
    images[:, :, 0, 0] = np.arange(n)[:, None]   # window index marker
    triplet = np.stack([slot_ids, slot_ids, slot_ids], 1).astype(np.int8)
    np.savez(directory/f'{uid}.npz', images=images, triplet_slot_ids=triplet,
             same_series=np.asarray(same_series, bool), slot_present=np.ones(6, bool),
             audit_json=np.asarray('{}'))


class Dataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)/'six_slot_raptor_shard_00'
        root.mkdir()
        slots = [0]*6 + [1]*4 + [2]*4 + [3]*3 + [4]*3 + [5]*3
        same = [True]*len(slots)
        for i in (5, 9, 13):   # boundary windows
            same[i] = False
        for uid in ('a', 'b', 'g'):
            write_study(root, uid, slots, same)
        pd.DataFrame(dict(StudyInstanceUID=['a', 'b', 'g'], split=['train', 'train', 'held_out_gold'])
                     ).to_csv(root/'splits.csv', index=False)
        self.root = root
        self.same = np.asarray(same)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pure_full_bag_drops_boundary_windows(self):
        data = ReadyTripletDataset([self.root], split='train', k=None, windows='pure')
        item = data[0]
        kept = item['images'][:, 0, 0, 0].float().round().long().tolist()
        self.assertEqual(kept, [i for i in range(len(self.same)) if self.same[i]])
        self.assertTrue(item['same_series'].all())

    def test_per_slot_training_bag(self):
        labels = {'a': np.zeros(12, np.float32), 'b': np.ones(12, np.float32)}
        data = ReadyTripletDataset([self.root], split='train', k=12, labels=labels,
                                   sampling='per_slot', windows='pure')
        for _ in range(10):
            item = data[1]
            self.assertEqual(tuple(item['images'].shape), (12, 3, 384, 384))
            self.assertTrue(item['same_series'].all())
            counts = torch.bincount(item['triplet_slot_ids'][:, 1].long(), minlength=6).tolist()
            self.assertEqual(counts, [2]*6)
            self.assertEqual(item['targets'].tolist(), [1.0]*12)

    def test_default_behaviour_unchanged(self):
        data = ReadyTripletDataset([self.root], split='train', k=None)
        self.assertEqual(len(data[0]['images']), len(self.same))


class Augmentation(unittest.TestCase):
    def cfg(self, **kw):
        base = {k: 0.0 for k in ('aug_rotate_deg', 'aug_scale', 'aug_shift', 'aug_contrast',
                                 'aug_gamma', 'aug_brightness', 'aug_reverse_p')}
        base.update(kw)
        return base

    def bag(self):
        torch.manual_seed(0)
        u = torch.rand(2, 3, 3, 384, 384)
        mean = torch.tensor(T.MEAN).view(1, 1, 3, 1, 1)
        std = torch.tensor(T.STD).view(1, 1, 3, 1, 1)
        return ((u-mean)/std), u

    def test_identity_when_disabled(self):
        x, _ = self.bag()
        y = T.gpu_augment(x, self.cfg())
        self.assertEqual(y.shape, x.shape)
        self.assertLess((y-x).abs().max().item(), 1e-3)

    def test_reversal_in_intensity_space(self):
        x, u = self.bag()
        y = T.gpu_augment(x, self.cfg(aug_reverse_p=1.0))
        mean = torch.tensor(T.MEAN).view(1, 1, 3, 1, 1)
        std = torch.tensor(T.STD).view(1, 1, 3, 1, 1)
        back = y*std + mean
        self.assertLess((back - u.flip(2)).abs().max().item(), 1e-4)

    def test_full_augmentation_keeps_contract(self):
        x, _ = self.bag()
        cfg = {k: v for k, v in T.DEFAULTS.items() if k.startswith('aug_')}
        y = T.gpu_augment(x.half(), cfg)
        self.assertEqual(y.dtype, torch.float16)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        mean = torch.tensor(T.MEAN).view(1, 1, 3, 1, 1)
        std = torch.tensor(T.STD).view(1, 1, 3, 1, 1)
        back = y.float()*std + mean
        self.assertGreaterEqual(back.min().item(), -1e-2)
        self.assertLessEqual(back.max().item(), 1+1e-2)
        self.assertGreater((y.float()-x).abs().mean().item(), 1e-3)


class TrainingPieces(unittest.TestCase):
    def test_bn_statistics_frozen_but_affine_trainable(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.ReLU())
        before = net[1].running_mean.clone()
        frozen = T.set_train_mode(net, dict(freeze_bn_stats=True))
        self.assertEqual(frozen, 1)
        self.assertTrue(net[0].training)
        out = net(torch.randn(4, 3, 8, 8)*5+3).sum()
        out.backward()
        self.assertTrue(torch.equal(before, net[1].running_mean))
        self.assertIsNotNone(net[1].weight.grad)

    def test_lr_schedule(self):
        f = T.lr_lambda_factory(10, 100, 0.01)
        warm = [f(s) for s in range(10)]
        self.assertTrue(all(b > a for a, b in zip(warm, warm[1:])))
        self.assertAlmostEqual(f(9), 1.0)
        self.assertAlmostEqual(f(10), 1.0)
        self.assertAlmostEqual(f(100), 0.01)
        self.assertTrue(all(f(s) >= f(s+1) for s in range(10, 100)))

    def test_ema_update_swap_restore(self):
        net = torch.nn.Linear(3, 2)
        ema = T.EMA(net, 0.9)
        original = [p.detach().clone() for p in net.parameters()]
        with torch.no_grad():
            for p in net.parameters():
                p.add_(1.0)
        ema.update(net, step=1000)
        for e, o in zip(ema.params, original):
            self.assertTrue(torch.allclose(e, o+0.1, atol=1e-6))
        trained = [p.detach().clone() for p in net.parameters()]
        backup = ema.swap_in(net)
        for p, e in zip(net.parameters(), ema.params):
            self.assertTrue(torch.equal(p, e))
        ema.restore(net, backup)
        for p, t in zip(net.parameters(), trained):
            self.assertTrue(torch.equal(p, t))

    def test_macro_auc_masks_missing_and_single_class(self):
        y = np.full((6, 12), np.nan, np.float32)
        y[:, 0] = [0, 0, 1, 1, 0, 1]
        y[:4, 1] = [0, 1, 0, 1]
        y[:, 2] = 0
        s = np.zeros((6, 12), np.float32)
        s[:, 0] = [0.1, 0.2, 0.9, 0.8, 0.3, 0.7]
        s[:, 1] = [0.9, 0.1, 0.8, 0.2, 0.0, 0.0]
        auc, per = T.macro_auc(y, s)
        self.assertEqual(set(per), {'ACL', 'MCL'})
        self.assertAlmostEqual(per['ACL'], 1.0)
        self.assertAlmostEqual(per['MCL'], 0.0)
        self.assertAlmostEqual(auc, 0.5)


class StageCheckpointing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import timm  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(str(error))

    def test_checkpoint_parts_preserve_outputs_and_gradients(self):
        import timm
        from models.raptor_v5_radimagenet import (COATNET_ARCH, FusionConfig, FusedCoAtNet,
                                                  RadImageNetEncoder, RaptorClassifier)
        torch.manual_seed(0)
        coat = timm.create_model(COATNET_ARCH, pretrained=False, num_classes=0, global_pool='avg')
        backbone = FusedCoAtNet(coat, RadImageNetEncoder(), FusionConfig())
        model = RaptorClassifier(backbone, backbone.num_features, drop=0.0)
        with torch.no_grad():   # open the residual gates so fusion gradients are non-trivial
            backbone.f1.mlp[-1].weight.normal_(0, 1e-3)
            backbone.f2.mlp[-1].weight.normal_(0, 1e-3)
        T.set_train_mode(model, dict(freeze_bn_stats=True))
        x = torch.randn(1, 2, 3, 384, 384)

        def run(parts):
            backbone.set_checkpoint_parts(parts)
            model.zero_grad(set_to_none=True)
            out = model(x)
            out.square().sum().backward()
            grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
            return out.detach(), grads

        ref_out, ref_grads = run(())
        out, grads = run(T.CHECKPOINT_LADDER[-1])
        self.assertTrue(torch.allclose(out, ref_out, atol=1e-5, rtol=1e-4))
        self.assertEqual(set(grads), set(ref_grads))
        # Tolerance relative to the global gradient scale: tensors whose gradient is ~0
        # (e.g. softmax biases) differ only by float noise, as they do between two plain runs.
        scale = max(g.abs().max().item() for g in ref_grads.values())
        for name in ref_grads:
            self.assertTrue(torch.allclose(grads[name], ref_grads[name], atol=1e-4*scale, rtol=1e-3), name)
        with self.assertRaises(ValueError):
            backbone.set_checkpoint_parts(['bogus'])

    def test_fused_and_explicit_attention_agree(self):
        import timm
        from models.raptor_v5_radimagenet import COATNET_ARCH
        torch.manual_seed(0)
        coat = timm.create_model(COATNET_ARCH, pretrained=False, num_classes=0, global_pool='avg').eval()
        x = torch.randn(2, 3, 384, 384)
        with torch.no_grad():
            self.assertGreater(T.set_fused_attention(coat, True), 0)
            fused = coat(x)
            T.set_fused_attention(coat, False)
            explicit = coat(x)
        self.assertTrue(torch.allclose(fused, explicit, atol=1e-4, rtol=1e-4))


if __name__ == '__main__':
    unittest.main()

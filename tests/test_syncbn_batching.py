"""Small CPU regression checks for study batching and timm-aware SyncBatchNorm.

The CPU SyncBatchNorm checks cover conversion and local fallback semantics; an
actual two-GPU smoke run must verify its distributed collectives and memory use.
"""
import copy
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import train_fusion_ddp as D  # noqa: E402
from train_fusion_v2 import set_train_mode  # noqa: E402

try:
    from timm.layers.norm_act import BatchNormAct2d, SyncBatchNormAct
except ImportError:
    BatchNormAct2d = SyncBatchNormAct = None


class StudyBatching(unittest.TestCase):
    def test_requested_recipe_and_short_final_update(self):
        accumulate = D.accumulation_steps(32, 2, 2)
        self.assertEqual(accumulate, 8)
        studies_per_rank = 3900 // 2
        microbatches = math.ceil(studies_per_rank / 2)
        self.assertEqual(microbatches, 975)
        self.assertEqual(math.ceil(microbatches / accumulate), 122)
        final_start = (microbatches // accumulate) * accumulate
        self.assertEqual(microbatches - final_start, 7)
        self.assertEqual((microbatches - final_start) * 2 * 2, 28)
        weights = [D.microbatch_loss_weight(i, studies_per_rank, 2, accumulate)
                   for i in range(final_start, microbatches)]
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertTrue(all(abs(weight - 1 / 7) < 1e-12 for weight in weights))

    def test_reject_invalid_batch_configurations(self):
        invalid = [(30, 2, 2), (1, 2, 2), (0, 2, 2), (32, 0, 2),
                   (32, 2, 0), (-32, 2, 2), (32.0, 2, 2),
                   (32, True, 2), (32, 2, False)]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                D.accumulation_steps(*values)
        with self.assertRaises(ValueError):
            D.microbatch_loss_weight(6, 11, 2, 3)

    def test_accumulated_rank_gradients_match_full_group_with_short_batch(self):
        # Two equally sized rank shards, each ending with a one-study batch.
        # The final group has batches of 2, 2, 1 studies on each rank. Merely
        # dividing each batch's mean loss by three would overweight its last study.
        generator = torch.Generator().manual_seed(41)
        shards = [(torch.randn(11, 3, generator=generator, dtype=torch.float64),
                   torch.randn(11, 2, generator=generator, dtype=torch.float64))
                  for _ in range(2)]
        base = torch.nn.Linear(3, 2, dtype=torch.float64)
        with torch.no_grad():
            base.weight.fill_(0.25)
            base.bias.fill_(-0.1)
        for group_start in (0, 3):
            with self.subTest(group_start=group_start):
                rank_gradients = []
                for inputs, targets in shards:
                    net = copy.deepcopy(base)
                    for batch_index in range(group_start, group_start + 3):
                        start, stop = batch_index * 2, (batch_index + 1) * 2
                        loss = F.mse_loss(net(inputs[start:stop]), targets[start:stop])
                        weight = D.microbatch_loss_weight(batch_index, 11, 2, 3)
                        (loss * weight).backward()
                    rank_gradients.append([p.grad.clone() for p in net.parameters()])
                start, stop = group_start * 2, min((group_start + 3) * 2, 11)
                inputs = torch.cat([x[start:stop] for x, _ in shards])
                targets = torch.cat([y[start:stop] for _, y in shards])
                reference = copy.deepcopy(base)
                F.mse_loss(reference(inputs), targets).backward()
                for index, parameter in enumerate(reference.parameters()):
                    averaged = torch.stack([g[index] for g in rank_gradients]).mean(0)
                    torch.testing.assert_close(averaged, parameter.grad, rtol=1e-12, atol=1e-12)


class BatchNormConfiguration(unittest.TestCase):
    def test_reject_contradictory_and_unsynchronized_distributed_modes(self):
        net = torch.nn.Sequential(torch.nn.BatchNorm2d(3))
        invalid = [dict(sync_batchnorm=True, freeze_bn_stats=True),
                   dict(sync_batchnorm=False, freeze_bn_stats=False)]
        for cfg in invalid:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                D.configure_batchnorm(net, cfg, world=2)

    def test_frozen_mode_preserves_module_and_buffers(self):
        net = torch.nn.Sequential(torch.nn.BatchNorm2d(3))
        cfg = dict(sync_batchnorm=False, freeze_bn_stats=True)
        self.assertIs(D.configure_batchnorm(net, cfg, world=2), net)
        set_train_mode(net, cfg)
        previous = {name: value.clone() for name, value in net.named_buffers()}
        net(torch.ones(24, 3, 2, 2))
        for name, value in net.named_buffers():
            torch.testing.assert_close(value, previous[name])
        self.assertTrue(all(p.requires_grad for p in net.parameters()))

    @unittest.skipUnless(BatchNormAct2d is not None, 'timm is required for conversion checks')
    def test_conversion_preserves_timm_activation_dropout_buffers_and_eval_output(self):
        net = torch.nn.Sequential(
            BatchNormAct2d(3, act_layer=torch.nn.LeakyReLU,
                           drop_layer=lambda: torch.nn.Dropout2d(p=0.3)),
            torch.nn.BatchNorm2d(3),
        ).eval()
        with torch.no_grad():
            for bn in net:
                bn.weight.copy_(torch.tensor([-1.0, 0.5, 2.0]))
                bn.bias.copy_(torch.tensor([0.2, -0.3, 0.1]))
                bn.running_mean.copy_(torch.tensor([1.0, -2.0, 0.5]))
                bn.running_var.copy_(torch.tensor([0.5, 2.0, 1.5]))
                bn.num_batches_tracked.fill_(9)
        inputs = torch.linspace(-5, 5, 72).reshape(2, 3, 3, 4)
        expected = net(inputs).detach()
        previous = {name: value.clone() for name, value in net.state_dict().items()}
        original_act, original_drop = net[0].act, net[0].drop
        net = D.configure_batchnorm(net, dict(sync_batchnorm=True, freeze_bn_stats=False), world=2)
        self.assertIsInstance(net[0], SyncBatchNormAct)
        self.assertIsInstance(net[1], torch.nn.SyncBatchNorm)
        self.assertIs(net[0].act, original_act)
        self.assertIs(net[0].drop, original_drop)
        self.assertFalse(net[0].training)
        self.assertEqual(set(net.state_dict()), set(previous))
        for name, value in net.state_dict().items():
            torch.testing.assert_close(value, previous[name])
        torch.testing.assert_close(net(inputs), expected)

    @unittest.skipUnless(BatchNormAct2d is not None, 'timm is required for conversion checks')
    def test_unfrozen_running_stats_update_each_microstep_without_optimizer_step(self):
        # No process group is initialized: SyncBatchNorm uses its local CPU
        # fallback here. This test does not claim to test cross-GPU statistics.
        cfg = dict(sync_batchnorm=True, freeze_bn_stats=False)
        net = D.configure_batchnorm(torch.nn.Sequential(torch.nn.BatchNorm2d(3)), cfg, world=2)
        set_train_mode(net, cfg)
        bn = net[0]
        first = torch.ones(24, 3, 2, 2)
        net(first)
        torch.testing.assert_close(bn.running_mean, torch.full((3,), 0.1))
        self.assertEqual(bn.num_batches_tracked.item(), 1)
        net(first * 3)
        torch.testing.assert_close(bn.running_mean, torch.full((3,), 0.39))
        self.assertEqual(bn.num_batches_tracked.item(), 2)
        net.eval()
        previous = bn.running_mean.clone()
        net(first * 9)
        torch.testing.assert_close(bn.running_mean, previous)
        self.assertEqual(bn.num_batches_tracked.item(), 2)


if __name__ == '__main__':
    unittest.main()

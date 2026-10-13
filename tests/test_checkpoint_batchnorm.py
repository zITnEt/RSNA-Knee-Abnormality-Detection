"""CPU regression checks for one BatchNorm update per checkpointed forward."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.raptor_v5_radimagenet import FusedCoAtNet, preserve_batchnorm_buffers


class TinySection(nn.Module):
    """Exercise the production checkpoint wrapper without large MRI backbones."""
    _run = FusedCoAtNet._run

    def __init__(self, norm, checkpointed):
        super().__init__()
        self.section = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), norm,
                                     nn.GELU(), nn.Conv2d(4, 2, 1))
        self.checkpoint_parts = frozenset({"rad"} if checkpointed else ())

    def forward(self, x):
        return self._run("rad", self.section, x)


class TinyBoundSection(TinySection):
    @property
    def coatnet(self):
        return self.section

    def _stem_s1(self, x):
        return self.coatnet(x)

    def forward(self, x):
        return self._run("s1", self._stem_s1, x)


class CheckpointBatchNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def check_norm(self, norm, section_type=TinySection, part="rad"):
        reference = section_type(norm, checkpointed=False).train()
        checked = copy.deepcopy(reference)
        checked.checkpoint_parts = frozenset({part})
        # Two iterations catch cumulative-momentum and accumulated-counter drift.
        for expected_updates in (1, 2):
            reference.zero_grad(set_to_none=True)
            checked.zero_grad(set_to_none=True)
            x = (torch.randn(4, 3, 7, 7) + expected_updates).requires_grad_()
            xc = x.detach().clone().requires_grad_()
            ordinary, recomputed = reference(x), checked(xc)
            torch.testing.assert_close(recomputed, ordinary)
            ordinary.square().mean().backward()
            recomputed.square().mean().backward()
            torch.testing.assert_close(xc.grad, x.grad)
            for name, param in reference.named_parameters():
                other = dict(checked.named_parameters())[name]
                self.assertIsNotNone(param.grad, name)
                torch.testing.assert_close(other.grad, param.grad)
            for name, buffer in reference.named_buffers():
                other = dict(checked.named_buffers())[name]
                torch.testing.assert_close(other, buffer, rtol=0, atol=0)
                if name.endswith("num_batches_tracked"):
                    self.assertEqual(int(other), expected_updates)
            self.assertTrue(checked.section[1].training)
            self.assertTrue(checked.section[1].weight.requires_grad)

    def test_batchnorm_exponential_and_cumulative_momentum(self):
        for momentum in (0.1, None):
            with self.subTest(momentum=momentum):
                self.check_norm(nn.BatchNorm2d(4, momentum=momentum))

    def test_sync_batchnorm_local_fallback_keeps_training_mode(self):
        # CPU tests cannot exercise distributed CUDA collectives. SyncBatchNorm
        # without an initialized process group runs its ordinary BN fallback.
        self.check_norm(nn.SyncBatchNorm(4))

    @unittest.skipUnless(importlib.util.find_spec("timm"), "timm is not installed")
    def test_timm_batchnorm_act(self):
        from timm.layers import BatchNormAct2d
        self.check_norm(BatchNormAct2d(4, act_layer=nn.ReLU))

    def test_bound_stem_stage_scope(self):
        self.check_norm(nn.BatchNorm2d(4), TinyBoundSection, "s1")

    def test_delayed_backward_keeps_both_original_updates(self):
        reference = TinySection(nn.BatchNorm2d(4), checkpointed=False).train()
        checked = copy.deepcopy(reference)
        checked.checkpoint_parts = frozenset({"rad"})
        images = [torch.randn(4, 3, 7, 7) + shift for shift in (0, 2)]
        # Images need no gradients; parameters must still receive matching grads.
        ordinary = [reference(image).square().mean() for image in images]
        recomputed = [checked(image).square().mean() for image in images]
        sum(ordinary).backward()
        sum(recomputed).backward()
        self.assertEqual(int(checked.section[1].num_batches_tracked), 2)
        for name, buffer in reference.named_buffers():
            torch.testing.assert_close(dict(checked.named_buffers())[name], buffer, rtol=0, atol=0)
        for name, param in reference.named_parameters():
            torch.testing.assert_close(dict(checked.named_parameters())[name].grad, param.grad)

    def test_restore_after_exception_without_replacing_buffers(self):
        norm = nn.BatchNorm2d(3).train()
        original = {name: value.clone() for name, value in norm.named_buffers()}
        identities = {name: id(value) for name, value in norm.named_buffers()}
        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            with preserve_batchnorm_buffers(norm):
                norm(torch.randn(4, 3, 5, 5))
                self.assertEqual(int(norm.num_batches_tracked), 1)
                raise RuntimeError("probe failed")
        for name, value in norm.named_buffers():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
            self.assertEqual(id(value), identities[name])
        self.assertTrue(norm.training)

    def test_no_running_statistics(self):
        self.check_norm(nn.BatchNorm2d(4, track_running_stats=False))


if __name__ == "__main__":
    unittest.main()

"""Codex point 1 (checkpoint copies / EMA best) and point 6 (RadImageNet input convention)."""
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

from models.raptor_v5_radimagenet import (FusionConfig, IMAGENET_MEAN, IMAGENET_STD, RAD_INPUTS,  # noqa: E402
                                          fused_checkpoint, rad_input_transform)
import train_fusion_ddp as D  # noqa: E402


def tiny_model():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.ReLU(), torch.nn.Linear(5, 3))
    net.fusion_config = FusionConfig()
    return net


class CheckpointCopies(unittest.TestCase):
    def test_checkpoint_is_independent_of_live_parameters(self):
        net = tiny_model()
        ck = fused_checkpoint(net)
        before = {k: v.clone() for k, v in ck['model'].items()}
        with torch.no_grad():
            for p in net.parameters():
                p.add_(1.0)
        for k, v in ck['model'].items():
            self.assertTrue(torch.equal(v, before[k]), k)
            self.assertEqual(v.device.type, 'cpu')

    def test_ema_best_pattern_saves_ema_weights(self):
        net = tiny_model()
        ema = D.HostEMA(net, 0.9)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(1.0)                     # raw weights now differ from the EMA
        ema.update(net, 1000)
        raw = [p.detach().clone() for p in net.parameters()]
        backup = ema.swap_in(net)               # exactly the consider_best sequence
        try:
            ck = fused_checkpoint(net)
        finally:
            ema.restore(net, backup)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'best.pt'
            torch.save(ck, path)
            saved = torch.load(path, map_location='cpu', weights_only=True)['model']
        names = [n for n, _ in net.named_parameters()]
        for n, e, r in zip(names, ema.params, raw):
            self.assertTrue(torch.equal(saved[n], e), n)          # EMA weights were saved
            self.assertFalse(torch.equal(saved[n], r), n)         # not the raw ones
        for p, r in zip(net.parameters(), raw):
            self.assertTrue(torch.equal(p, r))                    # raw weights restored in the model

    def test_verification_rejects_wrong_weights(self):
        net = tiny_model()
        ema = D.HostEMA(net, 0.9)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(1.0)
        ema.update(net, 1000)
        wrong = fused_checkpoint(net)['model']   # raw weights, as the old bug produced
        names = [n for n, _ in net.named_parameters()]
        ok = all(torch.equal(wrong[n].float(), s.float().cpu()) for n, s in zip(names, ema.params))
        self.assertFalse(ok)


class RadInputConvention(unittest.TestCase):
    def triplet(self, seed=0):
        rng = np.random.default_rng(seed)
        planes = rng.integers(0, 256, size=(2, 3, 32, 32), dtype=np.uint8)   # 3 different uint8 slices
        u = torch.from_numpy(planes.astype(np.float32)/255)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return planes, (u - mean)/std

    def test_shared_is_identity(self):
        _, x = self.triplet()
        self.assertIs(rad_input_transform(x, 'shared'), x)

    def test_pm1_middle_matches_baseline_formula(self):
        planes, x = self.triplet()
        got = rad_input_transform(x, 'pm1_middle')
        # Baseline Rad path: uint8 / 127.5 - 1, single slice expanded to 3 channels.
        middle = torch.from_numpy(planes[:, 1].astype(np.float32)/127.5 - 1)
        expected = middle[:, None].expand(-1, 3, -1, -1)
        self.assertEqual(got.shape, x.shape)
        self.assertLess((got - expected).abs().max().item(), 1e-5)

    def test_pm1_triplet_keeps_three_slices(self):
        planes, x = self.triplet()
        got = rad_input_transform(x, 'pm1_triplet')
        expected = torch.from_numpy(planes.astype(np.float32)/127.5 - 1)
        self.assertLess((got - expected).abs().max().item(), 1e-5)
        self.assertGreaterEqual(got.min().item(), -1 - 1e-5)
        self.assertLessEqual(got.max().item(), 1 + 1e-5)

    def test_gradients_flow_through_conversion(self):
        _, x = self.triplet()
        x.requires_grad_(True)
        rad_input_transform(x, 'pm1_middle').sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(x.grad[:, 0].abs().sum().item(), 0.0)   # only the middle slice is used
        self.assertGreater(x.grad[:, 1].abs().sum().item(), 0.0)

    def test_unknown_mode_rejected(self):
        _, x = self.triplet()
        with self.assertRaises(ValueError):
            rad_input_transform(x, 'bogus')

    def test_old_checkpoint_config_defaults_to_previous_behaviour(self):
        old = {k: v for k, v in asdict(FusionConfig()).items() if k != 'rad_input'}
        self.assertEqual(FusionConfig(**old).rad_input, 'shared')
        self.assertEqual(RAD_INPUTS, ('shared', 'pm1_triplet', 'pm1_middle'))

    def test_training_defaults_use_baseline_convention(self):
        import train_fusion_v2 as T
        self.assertEqual(D.DEFAULTS['rad_input'], 'pm1_middle')
        self.assertEqual(T.DEFAULTS['rad_input'], 'pm1_middle')


class FusedModelRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import timm  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(str(error))

    def test_rad_trunk_receives_converted_input_and_coatnet_the_original(self):
        import timm
        from models.raptor_v5_radimagenet import COATNET_ARCH, FusedCoAtNet, RadImageNetEncoder
        torch.manual_seed(0)
        coat = timm.create_model(COATNET_ARCH, pretrained=False, num_classes=0, global_pool='avg').eval()
        seen = {}
        rad = RadImageNetEncoder().eval()
        rad.backbone[0].register_forward_hook(lambda m, i, o: seen.__setitem__('rad', i[0].detach()))
        coat.stem.register_forward_hook(lambda m, i, o: seen.__setitem__('coat', i[0].detach()))
        backbone = FusedCoAtNet(coat, rad, FusionConfig(rad_input='pm1_middle')).eval()
        _, x = RadInputConvention().triplet()
        x = torch.nn.functional.interpolate(x, size=(384, 384))
        with torch.no_grad():
            backbone(x)
        self.assertTrue(torch.equal(seen['coat'], x))
        self.assertTrue(torch.allclose(seen['rad'], rad_input_transform(x, 'pm1_middle')))
        with self.assertRaises(ValueError):
            FusedCoAtNet(coat, rad, FusionConfig(rad_input='bogus'))


if __name__ == '__main__':
    unittest.main()

"""Architecture tests; run with unittest (no pytest dependency)."""

import copy
import unittest

import torch
from torch import nn

from models.raptor_v5_resnet50 import (
    BottleneckFusion, FusedCoAtNet, FusionConfig, RaptorClassifier,
    optimizer_groups,
)


class SmallCoAtNet(nn.Module):
    """Cheap stage substitute for testing fusion routing and gradient topology."""

    def __init__(self):
        super().__init__()
        self.num_features = 1024
        self.feature_info = [{"num_chs": c} for c in (128, 256, 512, 1024)]
        self.stem = nn.Conv2d(3, 8, 1, stride=32)
        self.stages = nn.Sequential(
            nn.Conv2d(8, 128, 1), nn.Conv2d(128, 256, 1),
            nn.Conv2d(256, 512, 1), nn.Conv2d(512, 1024, 1),
        )
        self.norm = nn.Identity()

    def set_grad_checkpointing(self, enabled):
        self.checkpointing = enabled

    def forward_head(self, x):
        return x.mean((2, 3))

    def forward(self, x):
        return self.forward_head(self.norm(self.stages(self.stem(x))))


class SmallResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(3, 512, 1, stride=32)
        self.conv5 = nn.Conv2d(512, 2048, 1)

    def forward(self, x):
        x = self.conv3(x)
        return x, self.conv5(x)


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(2)

    def make_model(self, checkpointing=False):
        config = FusionConfig(stage2_hidden=32, stage4_hidden=32,
                              gradient_checkpointing=checkpointing)
        backbone = FusedCoAtNet(SmallCoAtNet(), SmallResNet50(), config)
        model = RaptorClassifier(backbone, 1024, drop=0.0)
        return model

    def test_concat_order_separate_norms_and_zero_initialization(self):
        fusion = BottleneckFusion(2, 3, 3, 8)
        a, b = torch.randn(2, 2, 4, 4), torch.randn(2, 3, 4, 4)
        seen = []
        handle = fusion.down.register_forward_pre_hook(lambda m, args: seen.append(args[0]))
        out = fusion(a, b)
        fusion(a, 100 * b)  # rescaling one input must not touch the other's half
        handle.remove()
        expected = torch.cat((nn.functional.layer_norm(a.permute(0, 2, 3, 1), (2,)),
                              nn.functional.layer_norm(b.permute(0, 2, 3, 1), (3,))), -1)
        torch.testing.assert_close(seen[0], expected)
        torch.testing.assert_close(seen[1][..., :2], seen[0][..., :2], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(out).item(), 0)
        self.assertGreater(torch.count_nonzero(fusion.down.weight).item(), 0)
        self.assertGreater(torch.count_nonzero(fusion.dw.weight).item(), 0)
        self.assertEqual(out.shape, b.shape)

    def test_depthwise_mixing_reaches_exactly_the_3x3_neighbourhood(self):
        fusion = BottleneckFusion(6, 3, 4, 8).eval()
        nn.init.normal_(fusion.up.weight)
        a, b = torch.randn(1, 6, 7, 7), torch.randn(1, 3, 7, 7)
        moved = a.clone()
        # A non-constant change: a per-position LayerNorm removes constant shifts.
        moved[0, :, 3, 3] = 3 * torch.randn(6)
        with torch.no_grad():
            changed = (fusion(moved, b) - fusion(a, b)).abs().sum(1)[0] > 1e-6
        expected = torch.zeros(7, 7, dtype=torch.bool)
        expected[2:5, 2:5] = True
        self.assertTrue(torch.equal(changed, expected))

    def test_default_fusion_sizes(self):
        count = lambda m: sum(p.numel() for p in m.parameters())
        f1, f2 = BottleneckFusion(512, 256, 256, 128), BottleneckFusion(1024, 2048, 1024, 256)
        # LNs + 1x1 down + depthwise 3x3 + 1x1 up
        self.assertEqual(count(f1), 1536 + 98432 + 1280 + 33024)      # 134,272
        self.assertEqual(count(f2), 6144 + 786688 + 2560 + 263168)    # 1,058,560
        g1 = BottleneckFusion(512, 256, 256, 128, gate_hidden=64)
        g2 = BottleneckFusion(1024, 2048, 1024, 256, gate_hidden=128)
        # Gate MLPs: 768 -> 64 -> 256 and 3072 -> 128 -> 1024
        self.assertEqual(count(g1.gate), 49216 + 16640)               # 65,856
        self.assertEqual(count(g2.gate), 393344 + 132096)             # 525,440
        self.assertEqual(count(g1) + count(g2), 1784128)
        config = FusionConfig()
        self.assertEqual((config.stage2_hidden, config.stage4_hidden), (128, 256))
        self.assertEqual((config.stage2_gate_hidden, config.stage4_gate_hidden), (64, 128))

    def test_channel_gate_scales_each_channel_of_the_residual(self):
        fusion = BottleneckFusion(6, 3, 4, 8, gate_hidden=5).eval()
        nn.init.normal_(fusion.up.weight)
        a, b = torch.randn(2, 6, 5, 5), torch.randn(2, 3, 5, 5)
        gates, ungated = [], []
        handles = [fusion.gate.register_forward_hook(lambda m, i, out: gates.append(out)),
                   fusion.up.register_forward_hook(lambda m, i, out: ungated.append(out))]
        with torch.no_grad():
            out = fusion(a, b)
        for handle in handles:
            handle.remove()
        g = gates[0]
        self.assertEqual(tuple(g.shape), (2, 4))  # one gate per sample and channel
        self.assertTrue(bool(((g > 0) & (g < 1)).all()))
        torch.testing.assert_close(out, ungated[0] * g[:, :, None, None])
        self.assertFalse(torch.allclose(g[0], g[1]))  # conditioned on the input
        # The gate sees spatially pooled, separately normalized inputs.
        pooled = torch.cat((nn.functional.layer_norm(a.permute(0, 2, 3, 1), (6,)),
                            nn.functional.layer_norm(b.permute(0, 2, 3, 1), (3,))), -1).mean((1, 2))
        torch.testing.assert_close(g, fusion.gate(pooled))

    def test_gate_hidden_zero_disables_gating(self):
        gated = BottleneckFusion(6, 3, 4, 8, gate_hidden=5).eval()
        nogate = BottleneckFusion(6, 3, 4, 8, gate_hidden=0).eval()
        self.assertIsNone(nogate.gate)
        nn.init.normal_(gated.up.weight)
        branch = {k: v for k, v in gated.state_dict().items() if not k.startswith("gate.")}
        nogate.load_state_dict(branch, strict=True)
        ungated = []
        handle = gated.up.register_forward_hook(lambda m, i, out: ungated.append(out))
        a, b = torch.randn(1, 6, 5, 5), torch.randn(1, 3, 5, 5)
        with torch.no_grad():
            gated(a, b)
            out = nogate(a, b)
        handle.remove()
        torch.testing.assert_close(out, ungated[0], rtol=0, atol=0)  # g = 1
        with self.assertRaises(ValueError):
            BottleneckFusion(6, 3, 4, 8, gate_hidden=-1)

    def test_zero_residuals_preserve_parent_logits(self):
        model = self.make_model().eval()
        parent = copy.deepcopy(model)
        parent.backbone = parent.backbone.coatnet
        x = torch.randn(2, 3, 3, 384, 384)
        with torch.no_grad():
            torch.testing.assert_close(model(x), parent(x), rtol=0, atol=0)

    def test_shared_input_and_stage2_continuation(self):
        model = self.make_model().eval()
        bb = model.backbone
        inputs, stage2, stage3, fusion_inputs, gates = [], [], [], [], []
        handles = [
            bb.resnet50.register_forward_pre_hook(lambda m, a: inputs.append(a[0])),
            bb.coatnet.stem.register_forward_pre_hook(lambda m, a: inputs.append(a[0])),
            bb.coatnet.stages[1].register_forward_hook(lambda m, a, out: stage2.append(out)),
            bb.coatnet.stages[2].register_forward_pre_hook(lambda m, a: stage3.append(a[0])),
            bb.f1.register_forward_pre_hook(lambda m, a: fusion_inputs.append(a)),
            bb.f1.gate.register_forward_hook(lambda m, a, out: gates.append(out)),
        ]
        # Three distinct slice channels must reach BOTH trunks unchanged.
        triplets = torch.stack([torch.full((384, 384), float(i))
                                for i in (1, 2, 3)]).reshape(1, 1, 3, 384, 384)
        # A nonzero correction must be visible in the *next* CoAtNet stage.
        with torch.no_grad():
            bb.f1.up.bias.fill_(0.25)
            model(triplets)
        for handle in handles:
            handle.remove()
        self.assertIs(inputs[0], inputs[1])
        torch.testing.assert_close(inputs[0], triplets.flatten(0, 1), rtol=0, atol=0)
        # The gated correction g1 * 0.25 (per channel) reaches stage 3.
        torch.testing.assert_close(stage3[0], stage2[0] + 0.25 * gates[0][:, :, None, None])
        self.assertEqual(fusion_inputs[0][0].shape[1], 512)  # X first
        self.assertIs(fusion_inputs[0][1], stage2[0])  # Z second

    def test_stage4_continues_through_norm_pool_and_head(self):
        model = self.make_model().eval()
        bb = model.backbone
        v, normalized_input, gates = [], [], []
        handles = [
            bb.coatnet.stages[3].register_forward_hook(lambda m, a, out: v.append(out)),
            bb.coatnet.norm.register_forward_pre_hook(lambda m, a: normalized_input.append(a[0])),
            bb.f2.gate.register_forward_hook(lambda m, a, out: gates.append(out)),
        ]
        with torch.no_grad():
            bb.f2.up.bias.fill_(0.5)
            feats = model.encode(torch.randn(2, 3, 3, 384, 384))
        for handle in handles:
            handle.remove()
        expected = v[0] + 0.5 * gates[0][:, :, None, None]
        torch.testing.assert_close(normalized_input[0], expected)
        torch.testing.assert_close(feats.flatten(0, 1), expected.mean((2, 3)))
        self.assertEqual(model.head(feats).shape, (2, 12))

    def test_gradients_open_after_first_update_and_all_parameters_are_unfrozen(self):
        model = self.make_model()
        bb = model.backbone
        groups = optimizer_groups(model)
        parameters = [p for g in groups for p in g["params"]]
        self.assertEqual(len(parameters), len(set(id(p) for p in parameters)))
        self.assertEqual(set(id(p) for p in parameters), set(id(p) for p in model.parameters()))
        self.assertTrue(all(p.requires_grad for p in parameters))
        optimizer = torch.optim.SGD(groups, lr=0.01)
        x = torch.randn(1, 2, 3, 384, 384)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            nn.functional.binary_cross_entropy_with_logits(model(x), torch.ones(1, 12)).backward()
            for fusion in (bb.f1, bb.f2):
                self.assertGreater(fusion.up.weight.grad.abs().sum().item(), 0)
            if step == 0:
                # Exact zero projections temporarily block gradients to ResNet50
                # and to the fusion layers before the up projection.
                self.assertEqual(bb.resnet50.conv3.weight.grad.abs().sum().item(), 0)
                for fusion in (bb.f1, bb.f2):
                    self.assertEqual(fusion.down.weight.grad.abs().sum().item(), 0)
                    self.assertEqual(fusion.dw.weight.grad.abs().sum().item(), 0)
                    # The gate multiplies a zero residual, so it learns from step 1.
                    self.assertEqual(fusion.gate[0].weight.grad.abs().sum().item(), 0)
                    self.assertEqual(fusion.gate[2].weight.grad.abs().sum().item(), 0)
            else:
                for module in (bb.coatnet.stem, bb.resnet50.conv3, bb.resnet50.conv5,
                               bb.f1.down, bb.f2.down, bb.f1.dw, bb.f2.dw,
                               bb.f1.gate[0], bb.f2.gate[0], bb.f1.gate[2], bb.f2.gate[2]):
                    self.assertGreater(module.weight.grad.abs().sum().item(), 0)
            optimizer.step()

    def test_checkpointed_fusion_has_same_gradients(self):
        plain = self.make_model()
        checked = self.make_model(checkpointing=True)
        checked.load_state_dict(plain.state_dict(), strict=True)
        x = torch.randn(1, 1, 3, 384, 384)
        plain(x).sum().backward()
        checked(x).sum().backward()
        for p, q in zip(plain.parameters(), checked.parameters()):
            torch.testing.assert_close(p.grad, q.grad)

    def test_invalid_shapes_and_frozen_parameters_fail(self):
        model = self.make_model()
        for shape in ((1, 1, 3, 336, 336), (1, 0, 3, 384, 384), (1, 3, 384, 384)):
            with self.assertRaises(ValueError):
                model(torch.zeros(shape))
        model.backbone.resnet50.requires_grad_(False)
        with self.assertRaises(ValueError):
            optimizer_groups(model)
        with self.assertRaises(ValueError):
            BottleneckFusion(2, 3, 3, 8)(torch.ones(1, 2, 4, 4), torch.ones(1, 3, 2, 2))


if __name__ == "__main__":
    unittest.main()

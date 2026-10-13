"""Architecture tests; run with unittest (no pytest dependency)."""

import copy
import unittest

import torch
from torch import nn

from models.raptor_v5_resnet50 import (
    ConcatenationMLP, FusedCoAtNet, FusionConfig, RaptorClassifier,
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

    def test_concat_order_and_zero_initialization(self):
        mlp = ConcatenationMLP(2, 3, 3, 8)
        a, b = torch.randn(2, 2, 4, 4), torch.randn(2, 3, 4, 4)
        seen = []
        handle = mlp.mlp[0].register_forward_pre_hook(lambda m, args: seen.append(args[0]))
        out = mlp(a, b)
        handle.remove()
        torch.testing.assert_close(seen[0], torch.cat((a, b), 1).permute(0, 2, 3, 1))
        self.assertEqual(torch.count_nonzero(out).item(), 0)
        self.assertGreater(torch.count_nonzero(mlp.mlp[1].weight).item(), 0)
        self.assertEqual(out.shape, b.shape)

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
        inputs, stage2, stage3, fusion_inputs = [], [], [], []
        handles = [
            bb.resnet50.register_forward_pre_hook(lambda m, a: inputs.append(a[0])),
            bb.coatnet.stem.register_forward_pre_hook(lambda m, a: inputs.append(a[0])),
            bb.coatnet.stages[1].register_forward_hook(lambda m, a, out: stage2.append(out)),
            bb.coatnet.stages[2].register_forward_pre_hook(lambda m, a: stage3.append(a[0])),
            bb.f1.register_forward_pre_hook(lambda m, a: fusion_inputs.append(a)),
        ]
        # Three distinct slice channels must reach BOTH trunks unchanged.
        triplets = torch.stack([torch.full((384, 384), float(i))
                                for i in (1, 2, 3)]).reshape(1, 1, 3, 384, 384)
        # A nonzero correction must be visible in the *next* CoAtNet stage.
        with torch.no_grad():
            bb.f1.mlp[-1].bias.fill_(0.25)
            model(triplets)
        for handle in handles:
            handle.remove()
        self.assertIs(inputs[0], inputs[1])
        torch.testing.assert_close(inputs[0], triplets.flatten(0, 1), rtol=0, atol=0)
        torch.testing.assert_close(stage3[0], stage2[0] + 0.25)
        self.assertEqual(fusion_inputs[0][0].shape[1], 512)  # X first
        self.assertIs(fusion_inputs[0][1], stage2[0])  # Z second

    def test_stage4_continues_through_norm_pool_and_head(self):
        model = self.make_model().eval()
        bb = model.backbone
        v, normalized_input = [], []
        handles = [
            bb.coatnet.stages[3].register_forward_hook(lambda m, a, out: v.append(out)),
            bb.coatnet.norm.register_forward_pre_hook(lambda m, a: normalized_input.append(a[0])),
        ]
        with torch.no_grad():
            bb.f2.mlp[-1].bias.fill_(0.5)
            feats = model.encode(torch.randn(2, 3, 3, 384, 384))
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(normalized_input[0], v[0] + 0.5)
        torch.testing.assert_close(feats.flatten(0, 1), (v[0] + 0.5).mean((2, 3)))
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
                self.assertGreater(fusion.mlp[-1].weight.grad.abs().sum().item(), 0)
            if step == 0:
                # Exact zero projections temporarily block gradients to ResNet50.
                self.assertEqual(bb.resnet50.conv3.weight.grad.abs().sum().item(), 0)
            else:
                for module in (bb.coatnet.stem, bb.resnet50.conv3, bb.resnet50.conv5,
                               bb.f1.mlp[1], bb.f2.mlp[1]):
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
            ConcatenationMLP(2, 3, 3, 8)(torch.ones(1, 2, 4, 4), torch.ones(1, 3, 2, 2))


if __name__ == "__main__":
    unittest.main()

"""Real timm/torchvision integration; random weights, no checkpoint downloads.

Run separately: python -m unittest discover -s tests -p test_raptor_real_backbones.py -v
"""

import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import timm
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from models.raptor_v5_resnet50 import (
    COATNET_ARCH, FusionConfig, ResNet50Encoder, RaptorClassifier,
    build_raptor_v5_resnet50, fused_checkpoint, load_fused_checkpoint,
)


class RealBackboneTests(unittest.TestCase):
    def test_default_uses_imagenet_v2_and_local_checkpoint_is_strict(self):
        network = resnet50(weights=None)
        with patch('torchvision.models.resnet50', return_value=network) as factory:
            ResNet50Encoder()
            factory.assert_called_once_with(weights=ResNet50_Weights.IMAGENET1K_V2)
        state = network.state_dict()
        state['conv1.weight'] = torch.full_like(state['conv1.weight'], 0.125)
        with patch('torchvision.models.resnet50', return_value=network) as factory:
            encoder = ResNet50Encoder({'state_dict': state})
            factory.assert_called_once_with(weights=None)
        torch.testing.assert_close(encoder.backbone[0].weight, state['conv1.weight'])
        broken = dict(state)
        del broken['conv1.weight']
        with self.assertRaises(RuntimeError):
            ResNet50Encoder(broken)

    def test_real_shapes_parent_equivalence_checkpoint_roundtrip_and_backward(self):
        torch.manual_seed(11)
        torch.set_num_threads(4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        coat = timm.create_model(COATNET_ARCH, pretrained=False, num_classes=0, global_pool="avg")
        parent = RaptorClassifier(coat, coat.num_features).eval()
        resnet = resnet50(weights=None)
        # Strict loading uses the Raptor and torchvision layouts, with synthetic
        # tensors for validation only. This is not a pretrained experiment.
        model = build_raptor_v5_resnet50(
            {"arch": COATNET_ARCH, "res": 384, "model": parent.state_dict()},
            resnet.state_dict(), FusionConfig(),
        ).eval().to(device)
        parent.to(device)
        x = torch.randn(1, 2, 3, 384, 384, device=device)
        # Distinct slice channels expose middle-slice replication or selection.
        x[:, :, 0].sub_(2)
        x[:, :, 2].add_(2)
        shared_inputs = []
        shapes = {}
        def record(name):
            def hook(module, args):
                shapes[name] = [list(t.shape) for t in args]
            return hook
        handles = [model.backbone.f1.register_forward_pre_hook(record("f1")),
                   model.backbone.f2.register_forward_pre_hook(record("f2")),
                   model.backbone.resnet50.backbone[0].register_forward_pre_hook(
                       lambda m, a: shared_inputs.append(a[0])),
                   model.backbone.coatnet.stem.register_forward_pre_hook(
                       lambda m, a: shared_inputs.append(a[0]))]
        with torch.no_grad():
            expected = parent(x)
            shared_inputs.clear()
            actual = model(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for handle in handles:
            handle.remove()
        self.assertEqual(shapes["f1"], [[2, 512, 48, 48], [2, 256, 48, 48]])
        self.assertEqual(shapes["f2"], [[2, 1024, 12, 12], [2, 2048, 12, 12]])
        self.assertIs(shared_inputs[0], shared_inputs[1])
        torch.testing.assert_close(shared_inputs[0], x.flatten(0, 1), rtol=0, atol=0)
        self.assertEqual(shared_inputs[0].shape, (2, 3, 384, 384))
        # Verify the retained head against the class extracted from the notebook.
        nb = json.loads(Path("kaggle/dinosaur_v5_reproduction.ipynb").read_text(encoding="utf-8"))
        outer = ast.parse("".join(nb["cells"][22]["source"]))
        source = next(n.value.value for n in outer.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "_KE_SRC" for t in n.targets))
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                   and n.name == "RaptorClassifier")
        scope = {"torch": torch, "nn": nn}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "baseline_head", "exec"), scope)
        original = scope["RaptorClassifier"](nn.Identity(), F_dim=1024).to(device).eval()
        head_state = {k: v for k, v in parent.state_dict().items() if not k.startswith("backbone.")}
        original.load_state_dict(head_state, strict=True)
        feats = torch.randn(2, 5, 1024, device=device)
        with torch.no_grad():
            torch.testing.assert_close(model.head(feats), original.head(feats), rtol=0, atol=0)
        del parent, coat, resnet, original, head_state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Gate-opening update, then real gradients through BOTH full backbones.
        optimizer = torch.optim.SGD(list(model.backbone.f1.parameters())
                                    + list(model.backbone.f2.parameters()), lr=0.01)
        model.train()
        self.assertTrue(model.backbone.coatnet.training)
        self.assertTrue(model.backbone.resnet50.training)
        self.assertTrue(all(module.training for module in model.modules()
                            if isinstance(module, nn.BatchNorm2d)))
        for step in range(2):
            model.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = nn.functional.binary_cross_entropy_with_logits(model(x[:, :1]), torch.ones(1, 12, device=device))
            loss.backward()
            for mlp in (model.backbone.f1, model.backbone.f2):
                self.assertGreater(mlp.mlp[-1].weight.grad.abs().sum().item(), 0)
            if step == 1:
                for trunk in (model.backbone.coatnet, model.backbone.resnet50):
                    self.assertTrue(all(p.requires_grad for p in trunk.parameters()))
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in trunk.parameters()
                                           if p.grad is not None), 0)
            optimizer.step()
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            updated = model(x)
        saved = fused_checkpoint(model)
        restored = load_fused_checkpoint(saved).eval().to(device)
        with torch.no_grad():
            torch.testing.assert_close(restored(x), updated, rtol=0, atol=0)
        print(json.dumps({"device": str(device), "fusion_shapes": shapes,
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "default_fusion_config": saved["fusion_config"]}))


if __name__ == "__main__":
    unittest.main()

"""Real timm CoAtNet and original notebook head; random weights, offline."""

import ast
import json
from pathlib import Path
import unittest

import timm
import torch
from torch import nn

from models.raptor_v5 import (
    COATNET_ARCH, RaptorClassifier, load_raptor_checkpoint, raptor_checkpoint,
)


class RealBackboneTests(unittest.TestCase):
    def test_original_head_real_backbone_checkpoint_roundtrip_and_backward(self):
        torch.manual_seed(11)
        torch.set_num_threads(4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        coat = timm.create_model(COATNET_ARCH, pretrained=False, num_classes=0,
                                 in_chans=3, global_pool="avg")
        parent = RaptorClassifier(coat, coat.num_features).eval().to(device)
        model = load_raptor_checkpoint({"arch": COATNET_ARCH, "res": 384,
            "model": parent.state_dict()}).eval().to(device)
        x = torch.randn(1, 1, 3, 384, 384, device=device)
        x[:, :, 0].sub_(2)
        x[:, :, 2].add_(2)
        with torch.no_grad():
            torch.testing.assert_close(model(x), parent(x), rtol=0, atol=0)
        # Compare with the actual original head, not a copied test formula.
        nb = json.loads(Path("kaggle/dinosaur_v5_reproduction.ipynb").read_text(encoding="utf-8"))
        outer = ast.parse("".join(nb["cells"][22]["source"]))
        source = next(n.value.value for n in outer.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "_KE_SRC" for t in n.targets))
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                   and n.name == "RaptorClassifier")
        scope = {"torch": torch, "nn": nn}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "baseline_head", "exec"), scope)
        original = scope["RaptorClassifier"](nn.Identity(), F_dim=coat.num_features).to(device).eval()
        original.load_state_dict({k: v for k, v in parent.state_dict().items()
                                  if not k.startswith("backbone.")}, strict=True)
        feats = torch.randn(2, 5, coat.num_features, device=device)
        with torch.no_grad():
            torch.testing.assert_close(model.head(feats), original.head(feats), rtol=0, atol=0)
        parameters = sum(p.numel() for p in model.parameters())
        self.assertEqual(parameters, sum(p.numel() for p in parent.parameters()))
        self.assertFalse(any(s in n for n in model.state_dict()
                             for s in ("resnet", "fusion", "backbone.f1.", "backbone.f2.")))
        del parent, coat, original
        model.train()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), torch.ones(1, 12, device=device))
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.backbone.stem.parameters()
                               if p.grad is not None), 0)
        self.assertGreater(model.clsW.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        model.eval()
        restored = load_raptor_checkpoint(raptor_checkpoint(model)).eval().to(device)
        with torch.no_grad():
            torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
        print(json.dumps({"device": str(device), "parameters": parameters,
                          "output_shape": [1, 12], "resnet_or_fusion_parameters": 0}))


if __name__ == "__main__":
    unittest.main()

"""Offline CoAtNet-only routing, strict migration, head and optimizer tests."""

import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock

import torch
from torch import nn

from models.raptor_v5 import (
    COATNET_ARCH, LABELS, LEGACY_FUSED_FORMAT, MODEL_FORMAT,
    RaptorClassifier, RaptorConfig, load_raptor_checkpoint, optimizer_groups,
    raptor_checkpoint,
)
from scripts.strip_raptor_fusion import export


class SmallCoAtNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 8
        self.stem = nn.Conv2d(3, 8, 1, stride=96)

    def set_grad_checkpointing(self, enabled):
        self.checkpointing = enabled

    def forward(self, x):
        return self.stem(x).mean((2, 3))


class RaptorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(2)
        self.parent = RaptorClassifier(SmallCoAtNet(), 8, drop=0.0)
        self.original = {"arch": COATNET_ARCH, "res": 384,
                         "model": copy.deepcopy(self.parent.state_dict())}
        self.factory = Mock(side_effect=lambda *a, **kw: SmallCoAtNet())
        # Restore only this entry: patch.dict(sys.modules) also removes lazily
        # imported torch modules, whose global operator registrations persist.
        previous = sys.modules.get("timm")
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = self.factory
        sys.modules["timm"] = fake_timm
        def restore_timm():
            if previous is None:
                sys.modules.pop("timm", None)
            else:
                sys.modules["timm"] = previous
        self.addCleanup(restore_timm)

    def legacy(self):
        state = {}
        for name, value in self.original["model"].items():
            key = "backbone.coatnet." + name.removeprefix("backbone.") if name.startswith("backbone.") else name
            state[key] = value.clone()
        # Nonzero legacy tensors must be discarded, not added or zeroed in place.
        for name in ("backbone.resnet50.conv.weight", "backbone.f1.up.weight", "backbone.f2.up.bias"):
            state[name] = torch.ones(3)
        return {"format": LEGACY_FUSED_FORMAT, "arch": COATNET_ARCH, "res": 384,
                "labels": list(LABELS), "fusion_config": {"head_dropout": 0.0}, "model": state}

    def test_parent_logits_and_input_channels_are_preserved_exactly(self):
        model = load_raptor_checkpoint(self.original).eval()
        self.factory.assert_called_once_with(COATNET_ARCH, pretrained=False,
            num_classes=0, in_chans=3, global_pool="avg")
        seen = []
        handle = model.backbone.register_forward_pre_hook(lambda m, a: seen.append(a[0]))
        x = torch.randn(2, 3, 3, 384, 384)
        x[:, :, 0].sub_(2)
        x[:, :, 2].add_(2)
        with torch.no_grad():
            torch.testing.assert_close(model(x), self.parent.eval()(x), rtol=0, atol=0)
        handle.remove()
        torch.testing.assert_close(seen[0], x.flatten(0, 1), rtol=0, atol=0)
        self.assertEqual(model(x).shape, (2, 12))

    def test_legacy_migration_copies_only_coatnet_and_head(self):
        ck = self.legacy()
        before = copy.deepcopy(ck)
        model = load_raptor_checkpoint(ck)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, self.original["model"][name], rtol=0, atol=0)
        self.assertFalse(any(token in name for name, _ in model.named_modules()
                             for token in ("resnet", "f1", "f2", "fusion", "gate")))
        for name, value in ck["model"].items():
            torch.testing.assert_close(value, before["model"][name], rtol=0, atol=0)

    def test_embedded_ema_is_selected_explicitly(self):
        ck = self.legacy()
        ck["ema"] = {"shadow": {name: value + 0.25 for name, value in ck["model"].items()}}
        ema = load_raptor_checkpoint(ck, weights="ema")
        for name, value in ema.state_dict().items():
            torch.testing.assert_close(value, self.original["model"][name] + 0.25, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "no embedded EMA"):
            load_raptor_checkpoint(self.original, weights="ema")
        with self.assertRaises(ValueError):
            load_raptor_checkpoint(self.original, weights="unknown")

    def test_checkpoint_roundtrip_and_dropout_configuration(self):
        model = load_raptor_checkpoint(self.original, config=RaptorConfig(.35, True))
        saved = raptor_checkpoint(model)
        restored = load_raptor_checkpoint(saved)
        self.assertEqual(saved["format"], MODEL_FORMAT)
        self.assertNotIn("fusion_config", saved)
        self.assertEqual(restored.att[2].p, .35)
        self.assertTrue(restored.backbone.checkpointing)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(saved, path)
            loaded = load_raptor_checkpoint(path).eval()
        x = torch.randn(1, 2, 3, 384, 384)
        with torch.no_grad():
            torch.testing.assert_close(model.eval()(x), loaded(x), rtol=0, atol=0)

    def test_optimizer_covers_only_backbone_and_head_and_gradients_flow(self):
        model = load_raptor_checkpoint(self.original)
        groups = optimizer_groups(model, backbone_lr=2e-5, head_lr=3e-4)
        self.assertEqual([g["name"] for g in groups], ["coatnet", "head"])
        self.assertEqual([g["lr"] for g in groups], [2e-5, 3e-4])
        parameters = [p for g in groups for p in g["params"]]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.parameters()})
        nn.functional.binary_cross_entropy_with_logits(
            model(torch.randn(1, 2, 3, 384, 384)), torch.ones(1, 12)).backward()
        self.assertGreater(model.backbone.stem.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.clsW.grad.abs().sum().item(), 0)
        model.backbone.requires_grad_(False)
        with self.assertRaises(ValueError):
            optimizer_groups(model)

    def test_contract_and_strict_loading_reject_bad_checkpoints(self):
        for changes in ({"arch": "wrong"}, {"res": 336}, {"format": "other"},
                        {"labels": list(reversed(LABELS))}, {"model": {}}):
            with self.assertRaises(ValueError):
                load_raptor_checkpoint({**self.original, **changes})
        missing = copy.deepcopy(self.original)
        del missing["model"]["clsW"]
        with self.assertRaises(RuntimeError):
            load_raptor_checkpoint(missing)
        legacy = self.legacy()
        legacy["model"]["backbone.unexpected.weight"] = torch.ones(1)
        with self.assertRaises(ValueError):
            load_raptor_checkpoint(legacy)
        del legacy["model"]["backbone.unexpected.weight"]
        legacy["model"]["clsb"] = torch.zeros(11)
        with self.assertRaises(RuntimeError):
            load_raptor_checkpoint(legacy)

    def test_invalid_input_shapes_are_rejected(self):
        model = load_raptor_checkpoint(self.original)
        for shape in ((1, 1, 3, 336, 336), (1, 0, 3, 384, 384),
                      (1, 2, 1, 384, 384), (1, 3, 384, 384)):
            with self.assertRaises(ValueError):
                model(torch.zeros(shape))

    def test_export_keeps_selected_weights_and_excludes_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "fusion.pt", Path(directory) / "coatnet.pt"
            ck = self.legacy()
            ck["optimizer"] = {"unrelated": "old parameter layout"}
            torch.save(ck, source)
            result = export(source, output)
            saved = torch.load(output, weights_only=True)
            self.assertEqual(result["parameters"], sum(p.numel() for p in self.parent.parameters()))
            self.assertNotIn("optimizer", saved)
            self.assertTrue(saved["conversion"]["optimizer_reset_required"])
            self.assertEqual(saved["conversion"]["source_sha256"], result["source_sha256"])
            for name, value in saved["model"].items():
                torch.testing.assert_close(value, self.original["model"][name], rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                export(source, output)
            with self.assertRaises(FileExistsError):
                export(source, source)

    def test_notebook_uses_identical_model_and_whole_bag_attention(self):
        import json
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "kaggle/raptor_v5_coatnet.ipynb").read_text(encoding="utf-8"))
        self.assertEqual("".join(notebook["cells"][2]["source"]),
                         (root / "models/raptor_v5.py").read_text(encoding="utf-8"))
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook_cell_{i}", "exec")
        model = load_raptor_checkpoint(self.original, config=RaptorConfig(head_dropout=0)).eval()
        scope = {"torch": torch, "model": model, "device": torch.device("cpu"),
                 "optimizer": torch.optim.AdamW(optimizer_groups(model))}
        exec("".join(notebook["cells"][-1]["source"]), scope)
        x = torch.randn(1, 5, 3, 384, 384)
        with torch.no_grad():
            torch.testing.assert_close(scope["bag_logits"](x, chunk=2), model(x))
        before = model.clsW.detach().clone()
        loss = scope["train_step"](x, torch.full((1, 12), .7))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model.clsW))


if __name__ == "__main__":
    unittest.main()

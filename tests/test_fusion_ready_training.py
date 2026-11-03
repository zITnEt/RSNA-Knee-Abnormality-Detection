"""Verify the saved-input training contract, holdout split and memory-safe eval."""
import ast
import json
import unittest
from contextlib import nullcontext
import copy
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.train_fusion_ready import split_indices, evaluate, FullBagClassifier, collate_studies, load_hybrid_labels, DEFAULTS, batch_settings, resume_batch_position, accumulation_weight
from models.raptor_v5_radimagenet import RaptorClassifier, LABELS
import pandas as pd
import numpy as np
import tempfile


class ReadyTrainingTests(unittest.TestCase):
    def test_deterministic_disjoint_training_validation(self):
        files = [Path(str(i)+'.npz') for i in range(4333)]
        a, b = split_indices(files, 20261009, .2)
        self.assertEqual((len(a), len(b)), (3466, 867))
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a) | set(b), set(range(4333)))
        self.assertEqual((a, b), split_indices(files, 20261009, .2))

    def test_full_bag_training_gradients_and_bn_statistics(self):
        torch.manual_seed(7)
        backbone = torch.nn.Sequential(torch.nn.Conv2d(3,8,1), torch.nn.BatchNorm2d(8),
                                       torch.nn.GELU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        model = RaptorClassifier(backbone, 8, drop=0.)
        reference = copy.deepcopy(model)
        images = torch.randn(17,3,4,4, requires_grad=True)
        reference_images = images.detach().clone().requires_grad_(True)
        expected = reference.head(torch.cat([reference.backbone(x) for x in reference_images.split(4)])[None])
        actual = FullBagClassifier(model,4)(images)
        torch.testing.assert_close(actual, expected)
        expected.sum().backward(); actual.sum().backward()
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a.grad,b.grad)
        torch.testing.assert_close(images.grad, reference_images.grad)
        self.assertTrue((images.grad.flatten(1).abs().sum(1)>0).all())
        for a,b in zip(model.buffers(),reference.buffers()):
            torch.testing.assert_close(a,b)
        self.assertEqual(model.backbone[1].num_batches_tracked.item(),5)

    def test_ragged_collation_keeps_every_triplet(self):
        images = [torch.randn(k,3,4,4) for k in (32,62)]
        items = [dict(images=x,targets=torch.zeros(12),study_uid=str(i)) for i,x in enumerate(images)]
        batch = collate_studies(items)
        self.assertEqual([len(x) for x in batch['images']], [32,62])
        for a,b in zip(images,batch['images']):
            self.assertIs(a,b)
        self.assertEqual(batch['targets'].shape,(2,12))

    def test_worker_sets_requested_amp_dtype(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*4*4,8))
            def head(self,x):return x.mean(1)
        model=Model()
        result=FullBagClassifier(model,4,torch.bfloat16)(torch.randn(7,3,4,4))
        self.assertEqual(result.dtype,torch.bfloat16)
        result.sum().backward()
        self.assertTrue(torch.isfinite(model.backbone[1].weight.grad).all())

    def test_two_full_bags_have_separate_heads_and_complete_gradients(self):
        torch.manual_seed(11)
        backbone=torch.nn.Sequential(torch.nn.Conv2d(3,8,1),torch.nn.BatchNorm2d(8),
                                      torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
        model=RaptorClassifier(backbone,8,drop=0.)
        reference=copy.deepcopy(model)
        bags=[torch.randn(k,3,4,4,requires_grad=True) for k in (5,7)]
        other=[x.detach().clone().requires_grad_(True) for x in bags]
        features=torch.cat([reference.backbone(x) for x in torch.cat(other).split(3)])
        expected=torch.cat([reference.head(x[None]) for x in features.split([5,7])])
        actual=FullBagClassifier(model,3)(bags)
        torch.testing.assert_close(actual,expected)
        self.assertEqual(actual.shape,(2,12))
        actual.sum().backward(); expected.sum().backward()
        for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a.grad,b.grad)
        for a,b in zip(bags,other):
            torch.testing.assert_close(a.grad,b.grad)
            self.assertTrue((a.grad.flatten(1).abs().sum(1)>0).all())
        for a,b in zip(model.buffers(),reference.buffers()):torch.testing.assert_close(a,b)

    def test_larger_batch_resume_keeps_study_cursor_and_effective_batch(self):
        self.assertEqual(batch_settings(2,2,16),(2,4,4))
        self.assertEqual(batch_settings(1,2,16),(1,2,8))
        self.assertEqual(resume_batch_position({'next_batch':800,'batch_size':2},4,3466),400)
        self.assertEqual(resume_batch_position({'next_batch':1733,'batch_size':2},4,3466),867)
        with self.assertRaises(ValueError):
            resume_batch_position({'next_batch':1,'batch_size':2},4,3466)
        weights=[accumulation_weight(i,4,4,3466,n) for i,n in zip((864,865,866),(4,4,2))]
        self.assertEqual(weights,[.4,.4,.2])
        self.assertAlmostEqual(sum(weights),1.)

    def test_hybrid_only_checks_required_non_gold_rows(self):
        frame = pd.DataFrame(dict(StudyInstanceUID=['train','val','gold']))
        for label in LABELS:
            frame[label] = [.25,.75,np.nan]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'HYBRID_labels_scores.csv'
            frame.to_csv(path,index=False)
            labels = load_hybrid_labels(path,['train','val'])
            self.assertEqual(set(labels),{'train','val'})
            np.testing.assert_array_equal(labels['val'],np.full(12,.75))
            with self.assertRaises(ValueError):
                load_hybrid_labels(path,['train','gold'])
        self.assertEqual(DEFAULTS['epochs'],50)
        self.assertIsNone(DEFAULTS['k'])

    def test_chunked_eval_keeps_one_full_bag_and_pixels(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.seen = []
            def backbone(self, x):
                self.seen.append(x.clone())
                return x.mean((2,3))
            def head(self, f):
                return f.mean(1).mean(1, keepdim=True).expand(-1,12)
        model = Model()
        images = torch.randn(1, 17, 3, 4, 4)
        targets = torch.rand(1,12)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            model.head(model.backbone(images[0])[None]), targets).item()
        model.seen.clear()
        with patch.object(torch.Tensor, 'cuda', lambda x, **kwargs:x), \
             patch('scripts.train_fusion_ready.torch.autocast', lambda *a, **k:nullcontext()):
            actual = evaluate(model, [dict(images=images, targets=targets)], torch.float16, chunk=5)
        self.assertAlmostEqual(actual, expected)
        torch.testing.assert_close(torch.cat(model.seen), images[0], rtol=0, atol=0)
        self.assertEqual([len(x) for x in model.seen], [5,5,5,2])

    def test_generated_notebook_bypasses_dicom_loader(self):
        notebook = json.loads(Path('kaggle/raptor_v5_radimagenet_fusion.ipynb').read_text())
        sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
        for source in sources:
            ast.parse(source)
        source = '\n'.join(sources)
        self.assertNotIn('SeriesTripletDataset', source)
        self.assertNotIn('pydicom', source)
        self.assertIn("split='held_out_gold'", source)
        self.assertIn('ReadyTripletDataset', source)
        meta = json.loads(Path('kaggle/raptor_v5_radimagenet_training/kernel-metadata.json').read_text())
        self.assertEqual(len(meta['kernel_sources']), 32)
        self.assertTrue(meta['is_private'])
        self.assertTrue(meta['enable_gpu'])
        self.assertIn('riadmohamed42/jev-knee-labels',meta['dataset_sources'])


if __name__ == '__main__':
    unittest.main()

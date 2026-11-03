"""Contracts for study-level CV fine-tuning on the pinned V2 corpus."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from scripts import train_resnet50_cv5 as cv
from scripts.train_resnet50_full_gold4 import reference_lr_factor


ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / 'configs' / 'cv5_folds.csv'
# The cloud package is deliberately kept outside the isolated code worktree.
# Split/hash integration tests use those exact staged source labels when present.
LABELS = ROOT.parents[1] / 'results' / 'resnet50_cloud' / 'package' / 'labels'


class CrossValidationSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not LABELS.is_dir():
            raise unittest.SkipTest('Pinned local cloud label package is unavailable')
        cls.assignment = pd.read_csv(FOLDS, dtype={'StudyInstanceUID': str})
        cls.ids = cls.assignment.StudyInstanceUID.tolist()
        cls.gold = pd.read_csv(LABELS / 'gold74_eval_set.csv',
                               dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()

    def test_every_non_gold_study_is_held_out_once(self):
        self.assertEqual(len(self.ids), 4333)
        self.assertEqual(len(set(self.ids)), 4333)
        self.assertEqual(len(self.gold), 74)
        self.assertFalse(set(self.ids) & set(self.gold))
        labeled = pd.read_csv(LABELS / 'HYBRID_labels_binary.csv',
                              dtype={'StudyInstanceUID': str})
        self.assertEqual(len(labeled), 4407)
        self.assertEqual(set(labeled.StudyInstanceUID), set(self.ids) | set(self.gold))
        self.assertFalse(labeled.StudyInstanceUID.duplicated().any())

        validations = []
        for fold in range(5):
            split = cv.load_cv_split(self.ids[::-1], self.gold[::-1], FOLDS, fold)
            train, validation, gold = (set(split[key]) for key in
                                       ('train', 'validation', 'heldout_gold'))
            self.assertEqual(set(split), {'train', 'validation', 'heldout_gold'})
            self.assertEqual(train | validation, set(self.ids))
            self.assertFalse(train & validation)
            self.assertFalse(gold & (train | validation))
            self.assertEqual(gold, set(self.gold))
            self.assertIn(len(validation), (866, 867))
            self.assertEqual(len(train), 4333 - len(validation))
            validations.extend(validation)

        self.assertEqual(len(validations), 4333)
        self.assertEqual(set(validations), set(self.ids))

    def test_split_rejects_wrong_corpus_gold_overlap_or_changed_csv(self):
        with self.assertRaises(ValueError):
            cv.load_cv_split(self.ids[:-1], self.gold, FOLDS, 0)
        with self.assertRaises(ValueError):
            cv.load_cv_split(self.ids, self.gold + [self.ids[0]], FOLDS, 0)
        with self.assertRaises(ValueError):
            cv.load_cv_split(self.ids, self.gold, FOLDS, 5)

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            changed = Path(directory) / 'cv5_folds.csv'
            text = FOLDS.read_text(encoding='utf-8')
            changed.write_text(text.replace(',2\n', ',1\n', 1), encoding='utf-8')
            with self.assertRaises(ValueError):
                cv.load_cv_split(self.ids, self.gold, changed, 0)

    def test_each_fold_has_both_classes_for_every_finding(self):
        labels = pd.read_csv(LABELS / 'HYBRID_labels_binary.csv',
                             dtype={'StudyInstanceUID': str})
        findings = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
                    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
                    "Baker's", 'Contusion', 'Fracture']
        merged = self.assignment.merge(labels[['StudyInstanceUID', *findings]],
                                       on='StudyInstanceUID', validate='one_to_one')
        for name in findings:
            positives = merged.groupby('fold')[name].sum()
            self.assertTrue((positives > 0).all(), name)
            self.assertTrue((positives < merged.groupby('fold').size()).all(), name)
            self.assertLessEqual(int(positives.max() - positives.min()), 1, name)

    def test_label_sources_are_pinned(self):
        hashes = cv.verify_label_hashes(LABELS)
        self.assertIn('HYBRID_labels_scores.csv', hashes)
        self.assertIn('HYBRID_labels_binary.csv', hashes)
        self.assertIn('gold74_eval_set.csv', hashes)

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            copied = Path(directory)
            for name in hashes:
                (copied / name).write_bytes((LABELS / name).read_bytes())
            target = copied / 'HYBRID_labels_binary.csv'
            target.write_bytes(target.read_bytes() + b'\n')
            with self.assertRaises(ValueError):
                cv.verify_label_hashes(copied)


class CrossValidationTrainingTests(unittest.TestCase):
    def test_five_fold_epochs_match_prior_lr_curve_at_relative_progress(self):
        # Each fold has 3,466 or 3,467 studies, giving 217 effective-batch-16
        # optimizer updates per epoch. Endpoint interpolation prevents a short
        # fold epoch from truncating the old 244-step epoch's final LR value.
        for epoch in range(5):
            mapped = [round(i * 243 / 216) for i in range(217)]
            expected = [reference_lr_factor(epoch * 244 + old_i, 244)
                        for old_i in mapped]
            actual = [cv.cv_lr_factor(epoch * 217 + i, 217) for i in range(217)]
            self.assertEqual(actual, expected)
            self.assertEqual(actual[0], reference_lr_factor(epoch * 244, 244))
            self.assertEqual(actual[-1], reference_lr_factor(epoch * 244 + 243, 244))
            self.assertEqual(len(set(mapped)), 217)

    def test_checkpoint_selection_uses_fold_only_for_raw_and_ema(self):
        best = {kind: {'score': -1., 'epoch': None} for kind in ('raw', 'ema')}
        first = {
            'raw': {'validation': {'macro_auc': .83}, 'gold74': {'macro_auc': .99}},
            'ema': {'validation': {'macro_auc': .84}, 'gold74': {'macro_auc': .40}},
        }
        self.assertEqual(set(cv.select_best(best, first, 1)), {'raw', 'ema'})
        self.assertEqual(best['raw']['score'], .83)
        self.assertEqual(best['ema']['score'], .84)

        before = copy.deepcopy(best)
        worse_fold_better_gold = {
            'raw': {'validation': {'macro_auc': .82}, 'gold74': {'macro_auc': 1.}},
            'ema': {'validation': {'macro_auc': .82}, 'gold74': {'macro_auc': 1.}},
        }
        self.assertFalse(cv.select_best(best, worse_fold_better_gold, 2))
        self.assertEqual(best, before)

        improved_ema = {
            'raw': {'validation': {'macro_auc': .82}, 'gold74': {'macro_auc': 1.}},
            'ema': {'validation': {'macro_auc': .85}, 'gold74': {'macro_auc': .01}},
        }
        self.assertEqual(set(cv.select_best(best, improved_ema, 3)), {'ema'})
        self.assertEqual(best['raw']['epoch'], 1)
        self.assertEqual(best['ema']['epoch'], 3)

    def test_new_fold_initializes_from_parents_instead_of_previous_fold(self):
        first, second, resumed = object(), object(), object()
        cfg = {'raptor': 'raptor.pt', 'resnet': 'resnet50.pth'}
        with patch.object(cv, 'build_raptor_v5_resnet50', side_effect=[first, second]) as fresh, \
             patch.object(cv, 'load_fused_checkpoint', return_value=resumed) as restore:
            self.assertIs(cv.initialize_model(cfg, None), first)
            self.assertIs(cv.initialize_model(cfg, None), second)
            self.assertEqual(fresh.call_count, 2)
            self.assertFalse(restore.called)
            self.assertIs(cv.initialize_model(cfg, {'checkpoint': 'fold-only'}), resumed)
            self.assertEqual(fresh.call_count, 2)
            self.assertEqual(restore.call_count, 1)

    def test_only_rolling_and_best_raw_ema_checkpoint_files_are_retained(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                Path('checkpoints').mkdir()
                Path('predictions').mkdir()

                cv.save_rolling_checkpoint({'global_step': 100, 'ema': {'updates': 100}})
                latest = Path('checkpoints/latest.pt')
                self.assertTrue(latest.is_file())
                self.assertEqual(Path(json.loads(Path('latest_checkpoint.json').read_text())['path']),
                                 latest)
                cv.save_rolling_checkpoint({'global_step': 200, 'ema': {'updates': 200}})
                self.assertEqual(torch.load(latest, weights_only=True)['global_step'], 200)
                self.assertEqual(torch.load(latest, weights_only=True)['ema']['updates'], 200)

                cv.save_best_checkpoint({'epoch': 1, 'weight_kind': 'raw'}, 'raw')
                cv.save_best_checkpoint({'epoch': 1, 'weight_kind': 'ema'}, 'ema')
                cv.save_best_checkpoint({'epoch': 2, 'weight_kind': 'ema'}, 'ema')
                self.assertEqual(torch.load('checkpoints/best_validation_raw.pt',
                                            weights_only=True)['epoch'], 1)
                self.assertEqual(torch.load('checkpoints/best_validation_ema.pt',
                                            weights_only=True)['epoch'], 2)
                self.assertEqual({path.name for path in Path('checkpoints').glob('*.pt')},
                                 {'latest.pt', 'best_validation_raw.pt',
                                  'best_validation_ema.pt'})

                best = {kind: {'score': -1., 'epoch': None} for kind in ('raw', 'ema')}
                result = {
                    'raw': {'validation': {'macro_auc': .90}, 'gold74': {'macro_auc': .20}},
                    'ema': {'validation': {'macro_auc': .91}, 'gold74': {'macro_auc': .10}},
                }
                cv.select_best(best, result, 1)
                for kind in ('raw', 'ema'):
                    Path(best[kind]['predictions']).touch()
                selected = cv.selection_payload(best, 2)
                self.assertEqual(selected['raw']['checkpoint'],
                                 'checkpoints/best_validation_raw.pt')
                self.assertEqual(selected['ema']['checkpoint'],
                                 'checkpoints/best_validation_ema.pt')
                self.assertEqual(selected['selected']['weight_kind'], 'ema')
                self.assertEqual(selected['selection_metric'], 'validation_macro_auc')
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()

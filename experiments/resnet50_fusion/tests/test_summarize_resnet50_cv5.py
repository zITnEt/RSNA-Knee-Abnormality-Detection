"""Synthetic five-fold integration checks for CV prediction aggregation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.summarize_resnet50_cv5 import summarize


ROOT = Path(__file__).resolve().parents[1]
N_LABELS = 12
VARIANTS = ('selected', 'raw_best', 'ema_best', 'raw_epoch05', 'ema_epoch05')


def truth_for(ids, *, gold=False):
    positions = np.array([int(uid.split('.')[-1]) for uid in ids])[:, None]
    findings = np.arange(N_LABELS)[None, :]
    labels = ((positions + 2 * findings) % 3 == 0).astype(np.float32)
    if gold:
        labels[positions[:, 0] == 0, 0] = np.nan
    return labels


def logits_for(ids, fold, kind, epoch):
    positions = np.array([int(uid.split('.')[-1]) for uid in ids])[:, None]
    findings = np.arange(N_LABELS)[None, :]
    offset = .23 if kind == 'ema' else -.17
    return (np.sin(positions * .17 + findings * .31) + fold * .11
            + offset + epoch * .07).astype(np.float32)


def chosen_for(fold, variant):
    raw_epoch, ema_epoch = 1 + fold % 3, 2 + fold % 3
    if variant == 'raw_best':
        return 'raw', raw_epoch
    if variant == 'ema_best':
        return 'ema', ema_epoch
    if variant == 'raw_epoch05':
        return 'raw', 5
    if variant == 'ema_epoch05':
        return 'ema', 5
    return ('raw', raw_epoch) if fold % 2 == 0 else ('ema', ema_epoch)


class SyntheticCVSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = [f'1.2.3.{i:04d}' for i in range(4333)]
        self.gold_ids = [f'gold.{i:02d}' for i in range(74)]
        sizes = (866, 866, 867, 867, 867)
        offsets = np.cumsum((0, *sizes))

        for fold, size in enumerate(sizes):
            directory = self.root / f'fold_{fold}'
            (directory / 'predictions').mkdir(parents=True)
            (directory / 'checkpoints').mkdir()
            validation = self.ids[offsets[fold]:offsets[fold + 1]]
            heldout = self.gold_ids
            train = self.ids[:offsets[fold]] + self.ids[offsets[fold + 1]:]
            split = dict(train=train, validation=validation, heldout_gold=heldout)
            (directory / 'training_split.json').write_text(json.dumps(split))
            (directory / 'complete.json').write_text(json.dumps(dict(fold=fold, epochs=5)))

            raw_epoch, ema_epoch = chosen_for(fold, 'raw_best')[1], chosen_for(fold, 'ema_best')[1]
            raw = dict(epoch=raw_epoch, macro_auc=(.9 if fold % 2 == 0 else .8) + fold * .001,
                       checkpoint='checkpoints/best_validation_raw.pt')
            ema = dict(epoch=ema_epoch, macro_auc=(.8 if fold % 2 == 0 else .9) + fold * .001,
                       checkpoint='checkpoints/best_validation_ema.pt')
            kind, epoch = chosen_for(fold, 'selected')
            selected = dict(weight_kind=kind, **(raw if kind == 'raw' else ema))
            selection = dict(fold=fold, selection_metric='validation_macro_auc',
                             raw=raw, ema=ema, selected=selected)
            (directory / 'best_checkpoint.json').write_text(json.dumps(selection))
            for name in ('best_validation_raw.pt', 'best_validation_ema.pt'):
                (directory / 'checkpoints' / name).touch()

            pairs = {chosen_for(fold, variant) for variant in VARIANTS}
            for kind, epoch in pairs:
                prefix = f'epoch_{epoch:02d}_{kind}'
                # Reverse OOF rows and rotate Gold rows, exercising UID alignment.
                val_order = validation[::-1]
                gold_order = self.gold_ids[fold:] + self.gold_ids[:fold]
                np.savez_compressed(directory / 'predictions' / f'{prefix}_validation.npz',
                                    study_uid=np.asarray(val_order), labels=truth_for(val_order),
                                    logits=logits_for(val_order, fold, kind, epoch))
                np.savez_compressed(directory / 'predictions' / f'{prefix}_gold74.npz',
                                    study_uid=np.asarray(gold_order),
                                    labels=truth_for(gold_order, gold=True),
                                    logits=logits_for(gold_order, fold, kind, epoch))

    def test_all_variants_cover_oof_once_and_ensemble_gold_probabilities(self):
        report = summarize(self.root)
        self.assertEqual((report['folds'], report['non_gold_studies'],
                          report['gold74_studies']), (5, 4333, 74))
        self.assertEqual(report['selection_metric'], 'per-fold validation macro AUROC')
        self.assertEqual(set(report['variants']), set(VARIANTS))
        self.assertTrue((self.root / 'cv5_summary.json').is_file())

        sizes = (866, 866, 867, 867, 867)
        offsets = np.cumsum((0, *sizes))
        for variant in VARIANTS:
            with np.load(self.root / f'{variant}_oof.npz', allow_pickle=False) as saved:
                ids, labels, logits = saved['study_uid'].astype(str), saved['labels'], saved['logits']
            self.assertEqual(ids.tolist(), self.ids)
            self.assertEqual(len(set(ids)), 4333)
            np.testing.assert_array_equal(labels, truth_for(self.ids))
            expected_logits = np.empty((4333, N_LABELS), np.float32)
            expected_gold = []
            for fold in range(5):
                kind, epoch = chosen_for(fold, variant)
                fold_ids = self.ids[offsets[fold]:offsets[fold + 1]]
                expected_logits[offsets[fold]:offsets[fold + 1]] = logits_for(
                    fold_ids, fold, kind, epoch)
                expected_gold.append(1 / (1 + np.exp(-logits_for(
                    self.gold_ids, fold, kind, epoch))))
            np.testing.assert_array_equal(logits, expected_logits)

            with np.load(self.root / f'{variant}_gold74_ensemble.npz',
                         allow_pickle=False) as saved:
                self.assertEqual(saved['study_uid'].astype(str).tolist(), self.gold_ids)
                np.testing.assert_array_equal(saved['labels'],
                                              truth_for(self.gold_ids, gold=True))
                np.testing.assert_allclose(saved['probabilities'],
                                           np.mean(expected_gold, axis=0), rtol=1e-6, atol=1e-7)
            self.assertEqual(report['variants'][variant]['fold_models'], [
                dict(fold=fold, weight_kind=chosen_for(fold, variant)[0],
                     epoch=chosen_for(fold, variant)[1]) for fold in range(5)])
            self.assertEqual(report['variants'][variant]['oof']['labeled_entries'], 4333 * 12)
            self.assertEqual(report['variants'][variant]['gold74_ensemble']['labeled_entries'],
                             74 * 12 - 1)

        self.assertNotEqual(report['variants']['selected']['fold_models'],
                            report['variants']['raw_epoch05']['fold_models'])

    def test_incomplete_fold_prediction_is_rejected(self):
        path = self.root / 'fold_2' / 'predictions' / 'epoch_05_raw_validation.npz'
        with np.load(path, allow_pickle=False) as saved:
            ids = saved['study_uid'][:-1]
            labels = saved['labels'][:-1]
            logits = saved['logits'][:-1]
        np.savez_compressed(path, study_uid=ids, labels=labels, logits=logits)
        with self.assertRaisesRegex(ValueError, 'Incomplete or invalid predictions'):
            summarize(self.root)


if __name__ == '__main__':
    unittest.main()

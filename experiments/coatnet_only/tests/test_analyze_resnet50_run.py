"""Synthetic checks against explicitly repeated sklearn inputs."""
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_training_evidence import (aucs, bootstrap_counts, interval, macro_samples,
                                  operating, reliability, weighted_metrics,
                                  best_f1_threshold_samples)


class MetricHelpersTest(unittest.TestCase):
    def test_bootstrap_metrics_match_explicit_repetition_with_ties_and_missing(self):
        truth=np.array([0.,1.,np.nan,0.,1.,1.,0.])
        score=np.array([.1,.5,.99,.5,1.,.5,0.])
        counts=bootstrap_counts(len(truth),2000,123)
        measured=weighted_metrics(truth,score,counts)
        for i in range(80):
            ix=np.repeat(np.arange(len(truth)),counts[i].astype(int))
            ix=ix[np.isfinite(truth[ix])]
            y,p=truth[ix],score[ix]
            if len(np.unique(y))==2:
                self.assertAlmostEqual(measured['auc'][i],roc_auc_score(y,p),places=12)
            else:
                self.assertTrue(np.isnan(measured['auc'][i]))
            if y.sum(): self.assertAlmostEqual(measured['ap'][i],average_precision_score(y,p),places=12)
            else: self.assertTrue(np.isnan(measured['ap'][i]))
            self.assertAlmostEqual(measured['brier'][i],np.mean((p-y)**2),places=12)
            self.assertAlmostEqual(measured['mean_probability'][i],p.mean(),places=12)
            self.assertAlmostEqual(measured['prevalence'][i],y.mean(),places=12)
            bins=np.minimum((p*10).astype(int),9)
            ece=sum(abs(np.sum(p[bins==b]-y[bins==b]))/len(y) for b in range(10))
            self.assertAlmostEqual(measured['ece'][i],ece,places=12)

    def test_point_auc_masks_missing_and_single_class(self):
        y=np.array([[0.,1.],[1.,np.nan],[np.nan,1.]])
        p=np.array([[.1,.2],[.9,.4],[.5,.7]])
        values=aucs(y,p)
        self.assertEqual(values[0],1.)
        self.assertTrue(np.isnan(values[1]))

    def test_tied_auc_and_average_precision(self):
        y=np.array([0.,1.,0.,1.]);p=np.ones(4)*.5
        metrics=weighted_metrics(y,p,np.ones((1,4)))
        self.assertEqual(metrics['auc'][0],.5)
        self.assertEqual(metrics['ap'][0],.5)

    def test_macro_keeps_fixed_label_set(self):
        values=np.array([[.9,.7],[.8,np.nan],[.6,.4]])
        measured=macro_samples(values,np.array([True,True]))
        self.assertAlmostEqual(measured[0],.8)
        self.assertTrue(np.isnan(measured[1]))
        self.assertEqual(measured[2],.5)

    def test_paired_identical_predictions_have_zero_interval(self):
        y=np.array([0.,1.,0.,1.]);p=np.array([.1,.9,.4,.7])
        counts=bootstrap_counts(4,2000,123)
        a=weighted_metrics(y,p,counts)['auc']
        b=weighted_metrics(y,p,counts)['auc']
        m=interval(a-b,0.)
        self.assertEqual(m['ci_low'],0.)
        self.assertEqual(m['ci_high'],0.)
        self.assertEqual(m['valid_resamples']+m['invalid_resamples'],2000)

    def test_operating_points_missing_labels_and_boundary(self):
        y=np.array([0.,1.,0.,1.,np.nan]);p=np.array([.1,.5,.7,.4,.9])
        result=operating(y,p,.5,np.ones((1,5)))
        self.assertEqual(result['tp'],1)
        self.assertEqual(result['fp'],1)
        self.assertEqual(result['tn'],1)
        self.assertEqual(result['fn'],1)
        self.assertEqual(result['sensitivity'],.5)
        self.assertEqual(result['specificity'],.5)
        self.assertEqual(result['f1'],.5)

    def test_ece_probability_one_stays_in_final_bin(self):
        rows=reliability(np.array([0.,1.]),np.array([0.,1.]))
        self.assertEqual(rows[0]['n'],1)
        self.assertEqual(rows[9]['n'],1)
        self.assertEqual(rows[9]['mean_probability'],1.)
        self.assertIsNone(rows[5]['mean_probability'])

    def test_calibration_bins_preserve_float32_boundary_values(self):
        # The stored float32 value nearest .7 is slightly below .7; rounding
        # p*10 in float32 would incorrectly push it into the next bin.
        p=np.array([np.float32(.7),.7,1.],dtype=np.float64)
        rows=reliability(np.ones(3),p)
        self.assertEqual(rows[6]['n'],1)
        self.assertEqual(rows[7]['n'],1)
        self.assertEqual(rows[9]['n'],1)

    def test_deterministic_study_resampling(self):
        first=bootstrap_counts(74,2000,123)
        second=bootstrap_counts(74,2000,123)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(first.sum(1),np.full(2000,74))

    def test_f1_threshold_refits_inside_each_resample(self):
        y=np.array([0.,1.,0.,1.,np.nan]);p=np.array([.1,.8,.4,.7,.95])
        samples=best_f1_threshold_samples(y,p,np.array([[1,1,1,1,1],[1,1,0,0,1]],float))
        self.assertEqual(samples[0],.7)
        # When the .4 negative and .7 positive are absent, the smallest equally
        # good threshold is .4. The missing label never changes selection.
        self.assertEqual(samples[1],.4)


if __name__=='__main__':
    unittest.main()

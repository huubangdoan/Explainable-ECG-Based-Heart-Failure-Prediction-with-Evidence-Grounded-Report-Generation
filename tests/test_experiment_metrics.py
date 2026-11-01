"""Small, hand-checkable metric fixtures; no model imports or real data."""

import unittest

import numpy as np

from ptbxl_experiments.metrics import fit_thresholds, multilabel_metrics, patient_bootstrap


def five(values):
    return np.repeat(np.asarray(values)[:, None], 5, axis=1)


class MetricTests(unittest.TestCase):
    def test_tied_ranks_and_noninterpolated_average_precision(self):
        y = five([0, 1, 1, 0])
        p = five([0.1, 0.4, 0.4, 0.8])
        result = multilabel_metrics(y, p)
        self.assertAlmostEqual(result["macro_auroc"], 0.5)
        self.assertAlmostEqual(result["macro_average_precision"], 2 / 3)
        self.assertEqual(result["valid_auroc_classes"], 5)
        self.assertNotIn("macro_f1", result)

    def test_all_tied_scores_are_chance_auc_and_prevalence_ap(self):
        result = multilabel_metrics(five([0, 1, 0, 1]), five([0.5] * 4))
        self.assertEqual(result["macro_auroc"], 0.5)
        self.assertEqual(result["macro_average_precision"], 0.5)

    def test_single_class_metrics_remain_explicitly_undefined(self):
        negative = multilabel_metrics(five([0, 0]), five([0.1, 0.2]))
        self.assertIsNone(negative["macro_auroc"])
        self.assertIsNone(negative["macro_average_precision"])
        self.assertEqual(negative["valid_auroc_classes"], 0)
        positive = multilabel_metrics(five([1, 1]), five([0.1, 0.2]))
        self.assertIsNone(positive["macro_auroc"])
        self.assertEqual(positive["macro_average_precision"], 1.0)

    def test_f1_threshold_ties_choose_largest_threshold(self):
        # At thresholds .9 and .6 F1=2/3, so use .9 (fewer positive calls).
        y = five([1, 0, 0, 1])
        p = five([0.9, 0.8, 0.7, 0.6])
        thresholds = fit_thresholds(y, p)
        np.testing.assert_array_equal(thresholds, [0.9] * 5)
        result = multilabel_metrics(y, p, thresholds)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(result["per_class"]["NORM"]["precision"], 1.0)
        self.assertAlmostEqual(result["per_class"]["NORM"]["recall"], 0.5)

    def test_no_positives_threshold_predicts_none_even_for_probability_one(self):
        y, p = five([0, 0]), five([1.0, 0.1])
        thresholds = fit_thresholds(y, p)
        self.assertTrue(np.all(thresholds > 1))
        result = multilabel_metrics(y, p, thresholds)
        self.assertIsNone(result["macro_f1"])
        self.assertEqual(result["per_class"]["NORM"]["predicted_positive"], 0)

    def test_patient_bootstrap_is_clustered_deterministic_and_counts_undefined_draws(self):
        y, p = five([0, 0, 1, 1]), five([0.1, 0.2, 0.8, 0.9])
        patients = np.array(["negative", "negative", "positive", "positive"])
        first = patient_bootstrap(y, p, patients, n_bootstrap=40, seed=7)
        self.assertEqual(first, patient_bootstrap(y, p, patients, n_bootstrap=40, seed=7))
        self.assertEqual(first["n_patients"], 2)
        self.assertLess(first["intervals"]["macro_auroc"]["valid_replicates"], 40)
        self.assertEqual(first["intervals"]["macro_auroc"]["low"], 1.0)
        self.assertEqual(first["intervals"]["macro_auroc"]["high"], 1.0)

    def test_invalid_shapes_nan_labels_and_probabilities_fail(self):
        for y, p in [(five([0, 2]), five([0.1, 0.2])),
                     (five([0, 1]), five([0.1, np.nan])),
                     (five([0, 1]), five([0.1, 1.1])),
                     (np.zeros((0, 5)), np.zeros((0, 5))),
                     (np.zeros((2, 4)), np.zeros((2, 4)))]:
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                multilabel_metrics(y, p)


if __name__ == "__main__":
    unittest.main()

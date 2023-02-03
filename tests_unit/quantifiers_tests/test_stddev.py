from unittest import TestCase

import numpy as np

from uncertainty_wizard import ProblemType
from uncertainty_wizard.quantifiers import QuantifierRegistry, StandardDeviation


class TestStandardDeviation(TestCase):

    # =================
    # Test Class Methods
    # =================

    def test_string_representation(self):
        self.assertTrue(
            isinstance(QuantifierRegistry.find("standard_deviation"), StandardDeviation)
        )
        self.assertTrue(isinstance(QuantifierRegistry.find("std"), StandardDeviation))
        self.assertTrue(
            isinstance(QuantifierRegistry.find("StandardDeviation"), StandardDeviation)
        )
        self.assertTrue(
            isinstance(QuantifierRegistry.find("stddev"), StandardDeviation)
        )

    def test_is_confidence(self):
        self.assertFalse(StandardDeviation.is_confidence())
        self.assertFalse(StandardDeviation().is_confidence())

    def test_samples_type_declaration(self):
        self.assertTrue(StandardDeviation.takes_samples())

    def test_problem_type(self):
        self.assertEqual(StandardDeviation.problem_type(), ProblemType.REGRESSION)

    # ==================
    # Test Logic
    # =================

    def test_happy_path_single(self):
        values = np.array(
            [[[1.1, 0.8, 0.08, 0.02], [0.2, 0.7, 0.08, 0.02], [0.5, 0.4, 0.08, 0.02]]]
        )
        predictions, std = StandardDeviation.calculate(values)
        self.assertEqual(2, len(predictions.shape))
        self.assertEqual(1, predictions.shape[0])
        self.assertEqual(4, predictions.shape[1])
        self.assertEqual(2, len(std.shape))
        self.assertEqual(4, std.shape[1])
        self.assertEqual(1, std.shape[0])
        self.assertAlmostEqual(1.8 / 3, predictions[0][0], delta=0.0001)
        self.assertAlmostEqual(1.9 / 3, predictions[0][1], delta=0.0001)
        self.assertAlmostEqual(0.24 / 3, predictions[0][2], delta=0.0001)
        self.assertAlmostEqual(0.06 / 3, predictions[0][3], delta=0.0001)
        self.assertAlmostEqual(
            np.std(np.array([1.1, 0.2, 0.5])), std[0][0], delta=0.0001
        )
        self.assertAlmostEqual(
            np.std(np.array([0.8, 0.7, 0.4])), std[0][1], delta=0.0001
        )
        self.assertAlmostEqual(0, std[0][2], delta=0.0001)
        self.assertAlmostEqual(0, std[0][3], delta=0.0001)

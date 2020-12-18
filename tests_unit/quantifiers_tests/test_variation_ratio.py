from unittest import TestCase

import numpy as np

from uncertainty_wizard import ProblemType
from uncertainty_wizard.quantifiers import QuantifierRegistry, VariationRatio


class TestVariationRatio(TestCase):

    # =================
    # Test Class Methods
    # =================

    def test_string_representation(self):
        self.assertTrue(
            isinstance(QuantifierRegistry.find("variation_ratio"), VariationRatio)
        )
        self.assertTrue(
            isinstance(QuantifierRegistry.find("var_ratio"), VariationRatio)
        )
        self.assertTrue(isinstance(QuantifierRegistry.find("VR"), VariationRatio))

    def test_is_confidence(self):
        self.assertFalse(VariationRatio.is_confidence())
        self.assertFalse(VariationRatio().is_confidence())

    def test_samples_type_declaration(self):
        self.assertTrue(VariationRatio.takes_samples())

    def test_problem_type(self):
        self.assertEqual(VariationRatio.problem_type(), ProblemType.CLASSIFICATION)

    # ==================
    # Test Logic
    # =================

    def test_happy_path_single(self):
        softmax_values = np.array(
            [[[0.1, 0.8, 0.08, 0.02], [0.2, 0.7, 0.08, 0.02], [0.5, 0.4, 0.08, 0.02]]]
        )
        predictions, vr = VariationRatio.calculate(softmax_values)
        self.assertEqual(1, len(predictions.shape))
        self.assertEqual(1, predictions.shape[0])
        self.assertEqual(1, len(vr.shape))
        self.assertEqual(1, vr.shape[0])
        self.assertEqual(1, predictions[0])
        self.assertAlmostEqual(0.33333, vr[0], delta=0.0001)

    def test_happy_path_batch(self):
        softmax_values = np.array(
            [
                [
                    [0.1, 0.8, 0.08, 0.02],
                    [0.2, 0.7, 0.08, 0.02],
                    [0.5, 0.4, 0.08, 0.02],
                ],
                [
                    [0.1, 0.08, 0.8, 0.02],
                    [0.02, 0.16, 0.8, 0.02],
                    [0.01, 0.17, 0.8, 0.02],
                ],
            ]
        )
        predictions, vr = VariationRatio.calculate(softmax_values)
        self.assertEqual(1, len(predictions.shape))
        self.assertEqual(2, predictions.shape[0])
        self.assertEqual(1, len(vr.shape))
        self.assertEqual(2, vr.shape[0])
        self.assertEqual(1, predictions[0])
        self.assertAlmostEqual(0.33333, vr[0], delta=0.0001)
        self.assertEqual(2, predictions[1])
        self.assertAlmostEqual(0, vr[1], delta=0.0001)

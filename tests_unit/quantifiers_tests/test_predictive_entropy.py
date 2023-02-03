from unittest import TestCase

import numpy as np

from uncertainty_wizard import ProblemType
from uncertainty_wizard.quantifiers import PredictiveEntropy, QuantifierRegistry


class TestPredictiveEntropy(TestCase):

    # =================
    # Test Class Methods
    # =================

    def test_string_representation(self):
        self.assertTrue(
            isinstance(QuantifierRegistry.find("predictive_entropy"), PredictiveEntropy)
        )
        self.assertTrue(
            isinstance(QuantifierRegistry.find("pred_entropy"), PredictiveEntropy)
        )
        self.assertTrue(isinstance(QuantifierRegistry.find("PE"), PredictiveEntropy))
        self.assertTrue(
            isinstance(QuantifierRegistry.find("PredictiveEntropy"), PredictiveEntropy)
        )

    def test_is_confidence(self):
        self.assertFalse(PredictiveEntropy.is_confidence())
        self.assertFalse(PredictiveEntropy().is_confidence())

    def test_samples_type_declaration(self):
        self.assertTrue(PredictiveEntropy.takes_samples())

    def test_problem_type(self):
        self.assertEqual(PredictiveEntropy.problem_type(), ProblemType.CLASSIFICATION)

    # ==================
    # Test Logic
    # =================

    def test_single_input_no_entropy(self):
        softmax_values = np.array([[[1, 0], [1, 0], [1, 0]]])
        predictions, predictive_entropy = PredictiveEntropy.calculate(softmax_values)
        self.assertEqual(1, len(predictions.shape))
        self.assertEqual(1, predictions.shape[0])
        self.assertEqual(1, len(predictive_entropy.shape))
        self.assertEqual(1, predictive_entropy.shape[0])
        self.assertEqual(0, predictions[0])
        self.assertAlmostEqual(0, predictive_entropy[0], delta=0.0001)

    def test_two_inputs_high_pred_entropy(self):
        softmax_values = np.array(
            [
                [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                [[1, 0], [0, 1], [1, 0], [0, 1]],
            ]
        )
        predictions, predictive_entropy = PredictiveEntropy.calculate(softmax_values)
        self.assertEqual(1, len(predictions.shape))
        self.assertEqual(2, predictions.shape[0])
        self.assertEqual(1, len(predictive_entropy.shape))
        self.assertEqual(2, predictive_entropy.shape[0])
        self.assertAlmostEqual(1, predictive_entropy[0], delta=0.0001)
        self.assertAlmostEqual(1, predictive_entropy[1], delta=0.0001)

from unittest import TestCase

import numpy as np

from uncertainty_wizard import ProblemType
from uncertainty_wizard.quantifiers import MutualInformation, QuantifierRegistry


class TestMutualInformation(TestCase):

    # =================
    # Test Class Methods
    # =================

    def test_string_representation(self):
        self.assertTrue(
            isinstance(QuantifierRegistry.find("mutual_information"), MutualInformation)
        )
        self.assertTrue(
            isinstance(QuantifierRegistry.find("mutu_info"), MutualInformation)
        )

    def test_is_confidence(self):
        self.assertFalse(MutualInformation.is_confidence())
        self.assertFalse(MutualInformation().is_confidence())

    def test_samples_type_declaration(self):
        self.assertTrue(MutualInformation.takes_samples())

    def test_problem_type(self):
        self.assertEqual(MutualInformation.problem_type(), ProblemType.CLASSIFICATION)

    # ==================
    # Test Logic
    # =================

    def test_single_input_no_entropy(self):
        softmax_values = np.array([[[1, 0], [1, 0], [1, 0]]])
        predictions, predictive_entropy = MutualInformation.calculate(softmax_values)
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
        predictions, predictive_entropy = MutualInformation.calculate(softmax_values)
        self.assertEqual(1, len(predictions.shape))
        self.assertEqual(2, predictions.shape[0])
        self.assertEqual(1, len(predictive_entropy.shape))
        self.assertEqual(2, predictive_entropy.shape[0])
        self.assertAlmostEqual(0, predictive_entropy[0], delta=0.0001)
        self.assertAlmostEqual(1, predictive_entropy[1], delta=0.0001)

    def test_as_confidence_flag(self):
        # Some hypothetical entropies
        inputs = np.ones(10)

        # Cast to uncertainty (which it already is, thus no change)
        as_uncertainty = MutualInformation.cast_conf_or_unc(
            as_confidence=False, superv_scores=inputs
        )
        np.testing.assert_equal(as_uncertainty, inputs)

        # Cast to confidence
        as_confidence = MutualInformation.cast_conf_or_unc(
            as_confidence=True, superv_scores=inputs
        )
        np.testing.assert_equal(as_confidence, inputs * (-1))

        # No casting whatsoever
        as_confidence_2 = MutualInformation.cast_conf_or_unc(
            as_confidence=None, superv_scores=inputs
        )
        np.testing.assert_equal(as_confidence_2, inputs)

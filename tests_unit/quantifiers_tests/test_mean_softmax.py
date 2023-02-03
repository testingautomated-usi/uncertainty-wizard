from unittest import TestCase

import numpy as np

from uncertainty_wizard import ProblemType
from uncertainty_wizard.quantifiers import MeanSoftmax, QuantifierRegistry


class TestMeanSoftmax(TestCase):

    # =================
    # Test Class Methods
    # =================

    def test_string_representation(self):
        self.assertTrue(
            isinstance(QuantifierRegistry.find("mean_softmax"), MeanSoftmax)
        )
        self.assertTrue(isinstance(QuantifierRegistry.find("ensembling"), MeanSoftmax))
        self.assertTrue(isinstance(QuantifierRegistry.find("MS"), MeanSoftmax))
        self.assertTrue(isinstance(QuantifierRegistry.find("MeanSoftmax"), MeanSoftmax))

    def test_is_confidence(self):
        self.assertTrue(MeanSoftmax.is_confidence())
        self.assertTrue(MeanSoftmax().is_confidence())

    def test_samples_type_declaration(self):
        self.assertTrue(MeanSoftmax.takes_samples())

    def test_problem_type(self):
        self.assertEqual(MeanSoftmax.problem_type(), ProblemType.CLASSIFICATION)

    # ==================
    # Test Logic
    # =================

    def test_single_input_no_entropy(self):
        softmax_values = np.array([[[1, 0], [1, 0], [1, 0]]])
        predictions, sm_value = MeanSoftmax.calculate(softmax_values)
        self.assertEqual(1, len(predictions.shape))
        self.assertEqual(1, predictions.shape[0])
        self.assertEqual(1, len(sm_value.shape))
        self.assertEqual(1, sm_value.shape[0])
        self.assertEqual(0, predictions[0])
        self.assertAlmostEqual(1, sm_value[0], delta=0.0005)

    def test_two_inputs_high_pred_entropy(self):
        softmax_values = np.array(
            [
                [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                [[1, 0], [0, 1], [1, 0], [0, 1]],
            ]
        )
        predictions, sm_value = MeanSoftmax.calculate(softmax_values)
        self.assertEqual(1, len(predictions.shape))
        self.assertEqual(2, predictions.shape[0])
        self.assertEqual(1, len(sm_value.shape))
        self.assertEqual(2, sm_value.shape[0])
        self.assertAlmostEqual(0.5, sm_value[0], delta=0.0005)
        self.assertAlmostEqual(0.5, sm_value[1], delta=0.0005)

    def test_as_confidence_flag(self):
        # Some hypothetical mean softmax values
        inputs = np.ones(10)

        # Cast to uncertainty
        as_uncertainty = MeanSoftmax.cast_conf_or_unc(
            as_confidence=False, superv_scores=inputs
        )
        np.testing.assert_equal(as_uncertainty, inputs * (-1))

        # Cast to confidence (which it already is, thus no change)
        as_confidence = MeanSoftmax.cast_conf_or_unc(
            as_confidence=True, superv_scores=inputs
        )
        np.testing.assert_equal(as_confidence, inputs)

        # No casting whatsoever
        as_confidence_2 = MeanSoftmax.cast_conf_or_unc(
            as_confidence=None, superv_scores=inputs
        )
        np.testing.assert_equal(as_confidence_2, inputs)

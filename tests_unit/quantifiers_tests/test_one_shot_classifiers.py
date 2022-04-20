from unittest import TestCase

import numpy as np

from uncertainty_wizard.quantifiers import (
    MaxSoftmax,
    PredictionConfidenceScore,
    QuantifierRegistry,
    SoftmaxEntropy,
)
from uncertainty_wizard.quantifiers.one_shot_classifiers import DeepGini
from uncertainty_wizard.quantifiers.quantifier import ProblemType


class TestDeepGini(TestCase):

    def test_string_representation(self):
        self.assertTrue(
            isinstance(QuantifierRegistry.find("DeepGini"), DeepGini)
        )
        self.assertTrue(
            isinstance(QuantifierRegistry.find("deep_gini"), DeepGini)
        )

    def test_is_confidence(self):
        self.assertFalse(PredictionConfidenceScore.is_confidence())
        self.assertFalse(PredictionConfidenceScore().is_confidence())

    def test_samples_type_declaration(self):
        self.assertFalse(PredictionConfidenceScore.takes_samples())

    def test_problem_type(self):
        self.assertEqual(
            DeepGini.problem_type(), ProblemType.CLASSIFICATION
        )

    def test_quantification(self):
        input_batch = np.array([
            [.1, .2, .3, .4],
            [.5, .1, .1, .3],
            [.25, .25, .25, .25],
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
        ])

        expected = np.array([
            0.7,  # https://bit.ly/301vmQ3
            0.64,  # https://bit.ly/3qkHuGm
            0.75,  # https://bit.ly/3wrPI0h
            0,  # Trivial
            0  # Re-Ordering of previous
        ])

        pred, unc = DeepGini.calculate(input_batch)
        assert np.all(pred == np.array([3, 0, 0, 0, 1]))
        assert np.all(unc == expected)


class TestPCS(TestCase):

    # =================
    # Test Class Methods
    # =================

    def test_string_representation(self):
        self.assertTrue(
            isinstance(QuantifierRegistry.find("PCS"), PredictionConfidenceScore)
        )
        self.assertTrue(
            isinstance(
                QuantifierRegistry.find("prediction_confidence_score"),
                PredictionConfidenceScore,
            )
        )

    def test_is_confidence(self):
        self.assertTrue(PredictionConfidenceScore.is_confidence())
        self.assertTrue(PredictionConfidenceScore().is_confidence())

    def test_samples_type_declaration(self):
        self.assertFalse(PredictionConfidenceScore.takes_samples())

    def test_problem_type(self):
        self.assertEqual(
            PredictionConfidenceScore.problem_type(), ProblemType.CLASSIFICATION
        )

    # ==================
    # Test Logic
    # =================

    def test_happy_path_single(self):
        softmax_values = np.array([0.1, 0.8, 0.08, 0.02])
        softmax_values = np.expand_dims(softmax_values, 0)
        prediction, pcs = PredictionConfidenceScore.calculate(softmax_values)
        self.assertEqual(1, prediction[0])
        self.assertAlmostEqual(0.7, pcs[0])

    def test_happy_path_batch(self):
        softmax_values = np.array(
            [[0.1, 0.8, 0.08, 0.02], [0.1, 0.8, 0.08, 0.02], [0.2, 0.09, 0.7, 0.01]]
        )
        prediction, pcs = PredictionConfidenceScore.calculate(softmax_values)
        self.assertEqual((3,), prediction.shape)
        self.assertEqual((3,), pcs.shape)
        self.assertEqual(1, prediction[0])
        self.assertAlmostEqual(0.7, pcs[0])
        self.assertEqual(1, prediction[1])
        self.assertAlmostEqual(0.7, pcs[1])
        self.assertEqual(2, prediction[2])
        self.assertAlmostEqual(0.5, pcs[2])

    def test_duplicate_non_winner(self):
        softmax_values = np.array([[0.1, 0.8, 0.05, 0.05], [0.2, 0.09, 0.7, 0.01]])
        prediction, pcs = PredictionConfidenceScore.calculate(softmax_values)
        self.assertEqual((2,), prediction.shape)
        self.assertEqual((2,), pcs.shape)
        self.assertEqual(1, prediction[0])
        self.assertAlmostEqual(0.7, pcs[0])
        self.assertEqual(2, prediction[1])
        self.assertAlmostEqual(0.5, pcs[1])

    def test_duplicate_winner(self):
        softmax_values = np.array([[0.4, 0.4, 0.1, 0.1], [0.2, 0.09, 0.7, 0.01]])
        prediction, pcs = PredictionConfidenceScore.calculate(softmax_values)
        self.assertEqual((2,), prediction.shape)
        self.assertEqual((2,), pcs.shape)
        self.assertTrue(
            0 == prediction[0] or 1 == prediction[0],
            "Prediction must be index 0 or 1, but was {0}".format(prediction[0]),
        )
        self.assertAlmostEqual(0, pcs[0])
        self.assertEqual(2, prediction[1])
        self.assertAlmostEqual(0.5, pcs[1])


class TestSoftmax(TestCase):

    # =================
    # Test Class Methods
    # =================

    def test_string_representation(self):
        self.assertTrue(isinstance(QuantifierRegistry.find("softmax"), MaxSoftmax))
        self.assertTrue(isinstance(QuantifierRegistry.find("max_softmax"), MaxSoftmax))

    def test_is_confidence(self):
        self.assertTrue(MaxSoftmax.is_confidence())
        self.assertTrue(MaxSoftmax().is_confidence())

    def test_samples_type_declaration(self):
        self.assertFalse(MaxSoftmax.takes_samples())

    def test_problem_type(self):
        self.assertEqual(MaxSoftmax.problem_type(), ProblemType.CLASSIFICATION)

    # ==================
    # Test Logic
    # =================

    def test_happy_path_single(self):
        softmax_values = np.array([0.1, 0.8, 0.08, 0.02])
        softmax_values = np.expand_dims(softmax_values, 0)
        prediction, softmax = MaxSoftmax.calculate(softmax_values)
        self.assertEqual(1, prediction[0])
        self.assertAlmostEqual(0.8, softmax[0])

    def test_happy_path_batch(self):
        softmax_values = np.array(
            [[0.1, 0.8, 0.08, 0.02], [0.1, 0.8, 0.08, 0.02], [0.2, 0.09, 0.7, 0.01]]
        )
        prediction, softmax = MaxSoftmax.calculate(softmax_values)
        self.assertEqual((3,), prediction.shape)
        self.assertEqual((3,), softmax.shape)
        self.assertEqual(1, prediction[0])
        self.assertAlmostEqual(0.8, softmax[0])
        self.assertEqual(1, prediction[1])
        self.assertAlmostEqual(0.8, softmax[1])
        self.assertEqual(2, prediction[2])
        self.assertAlmostEqual(0.7, softmax[2])

    def test_duplicate_non_winner(self):
        softmax_values = np.array([[0.1, 0.8, 0.05, 0.05], [0.2, 0.09, 0.7, 0.01]])
        prediction, softmax = MaxSoftmax.calculate(softmax_values)
        self.assertEqual((2,), prediction.shape)
        self.assertEqual((2,), softmax.shape)
        self.assertEqual(1, prediction[0])
        self.assertAlmostEqual(0.8, softmax[0])
        self.assertEqual(2, prediction[1])
        self.assertAlmostEqual(0.7, softmax[1])

    def test_duplicate_winner(self):
        softmax_values = np.array([[0.4, 0.4, 0.1, 0.1], [0.2, 0.09, 0.7, 0.01]])
        prediction, softmax = MaxSoftmax.calculate(softmax_values)
        self.assertEqual((2,), prediction.shape)
        self.assertEqual((2,), softmax.shape)
        self.assertTrue(
            0 == prediction[0] or 1 == prediction[0],
            "Prediction must be index 0 or 1, but was {0}".format(prediction[0]),
        )
        self.assertAlmostEqual(0.4, softmax[0])
        self.assertEqual(2, prediction[1])
        self.assertAlmostEqual(0.7, softmax[1])


class TestSoftmaxEntropy(TestCase):

    # =================
    # Test Class Methods
    # =================

    def test_string_representation(self):
        self.assertTrue(
            isinstance(QuantifierRegistry.find("softmax_entropy"), SoftmaxEntropy)
        )
        self.assertTrue(
            isinstance(QuantifierRegistry.find("SoftmaxEntropy"), SoftmaxEntropy)
        )

    def test_is_confidence(self):
        self.assertFalse(SoftmaxEntropy.is_confidence())
        self.assertFalse(SoftmaxEntropy().is_confidence())

    def test_samples_type_declaration(self):
        self.assertFalse(SoftmaxEntropy.takes_samples())

    def test_problem_type(self):
        self.assertEqual(SoftmaxEntropy.problem_type(), ProblemType.CLASSIFICATION)

    # ==================
    # Test Logic
    # =================

    def test_happy_path_single(self):
        softmax_values = np.array([0.1, 0.8, 0.08, 0.02])
        softmax_values = np.expand_dims(softmax_values, 0)
        prediction, entropies = SoftmaxEntropy.calculate(softmax_values)
        self.assertEqual(1, prediction[0])
        # https://www.wolframalpha.com/input/?i=-+%280.1*log2%280.1%29+%2B+0.8  [url continued in line below]
        #   *log2%280.8%29%2B0.08*log2%280.08%29%2B0.02*log2%280.02%29%29
        self.assertAlmostEqual(0.9941209043760985826573513724, entropies[0])

    def test_happy_path_batch(self):
        softmax_values = np.array(
            [[0.1, 0.8, 0.08, 0.02], [0.1, 0.8, 0.08, 0.02], [0.2, 0.09, 0.7, 0.01]]
        )
        prediction, softmax = SoftmaxEntropy.calculate(softmax_values)
        self.assertEqual((3,), prediction.shape)
        self.assertEqual((3,), softmax.shape)
        self.assertEqual(1, prediction[0])
        # Calc: See test above
        self.assertAlmostEqual(0.9941209043760985826573513724, softmax[0])
        self.assertEqual(1, prediction[1])
        # Calc: See test above
        self.assertAlmostEqual(0.9941209043760985826573513724, softmax[1])
        self.assertEqual(2, prediction[2])
        # https://www.wolframalpha.com/input/?i=-+%280.2*log2%280.2%29+%2B+0.09  [url continued in line below]
        # *log2%280.09%29%2B0.7*log2%280.7%29%2B0.01*log2%280.01%29%29
        self.assertAlmostEqual(1.203679208805967594, softmax[2])

    def test_duplicate_non_winner(self):
        softmax_values = np.array([[0.1, 0.8, 0.05, 0.05], [0.2, 0.09, 0.7, 0.01]])
        prediction, softmax = SoftmaxEntropy.calculate(softmax_values)
        self.assertEqual((2,), prediction.shape)
        self.assertEqual((2,), softmax.shape)
        self.assertEqual(1, prediction[0])
        # https://www.wolframalpha.com/input/?i=-+%280.1*log2%280.1%29+%2B+0.08  [url continued in line below]
        #   *log2%280.08%29%2B0.5*log2%280.5%29%2B0.05*log2%280.05%29%29
        self.assertAlmostEqual(1.0219280948873623, softmax[0])
        self.assertEqual(2, prediction[1])
        # Calculation: See test above
        self.assertAlmostEqual(1.203679208805967594, softmax[1])

    def test_duplicate_winner(self):
        softmax_values = np.array([[0.4, 0.4, 0.1, 0.1], [0.2, 0.09, 0.7, 0.01]])
        prediction, softmax = SoftmaxEntropy.calculate(softmax_values)
        self.assertEqual((2,), prediction.shape)
        self.assertEqual((2,), softmax.shape)
        self.assertTrue(
            0 == prediction[0] or 1 == prediction[0],
            "Prediction must be index 0 or 1, but was {0}".format(prediction[0]),
        )
        # https://www.wolframalpha.com/input/?i=-+%280.4*log2%280.4%29+%2B+0.4  [url continued in line below]
        #   *log2%280.4%29%2B0.1*log2%280.1%29%2B0.1*log2%280.1%29%29
        self.assertAlmostEqual(1.7219280948873623, softmax[0])
        self.assertEqual(2, prediction[1])
        # Calculation: See test above
        self.assertAlmostEqual(1.203679208805967594, softmax[1])

from typing import List

import numpy as np

import uncertainty_wizard as uwiz

from .quantifier import ConfidenceQuantifier, ProblemType, UncertaintyQuantifier


def _check_inputs_array(inputs, quantifier_name):
    """
    Checks if the input array has the right shape and properties.
    Returns the array as well as a flag indicating whether the input was batched or a single sample
    """
    assert inputs.ndim == 2, (
        "The input to calculate {0} must be two"
        "dimensional (num_inputs x num_classes)".format(quantifier_name)
    )

    # Check that all values are between 0 and 1
    assert np.all(inputs >= 0) and np.all(inputs <= 1), (
        "{0} is built on softmax outputs, but the input array does not represent softmax outputs: "
        "There are entries which are not in the interval [0,1]".format(quantifier_name)
    )

    # Check that all softmax values sum up to one
    assert np.all(
        np.isclose(np.sum(inputs, axis=1), np.ones(shape=inputs.shape[0]), 0.00001)
    ), (
        "{0} is built on softmax outputs, but the input array does not represent softmax outputs: "
        "The per-sample values do not sum up to 1".format(quantifier_name)
    )


class PredictionConfidenceScore(ConfidenceQuantifier):
    """
    The Prediction Confidence Score is a confidence metric in one-shot classification.
    Inputs/activations have to be normalized using the softmax function over all classes.
    The class with the highest activation is chosen as prediction,
    the difference between the two highest activations is used as confidence quantification.
    """

    # docstr-coverage:inherited
    @classmethod
    def takes_samples(cls) -> bool:
        return False

    # docstr-coverage:inherited
    @classmethod
    def problem_type(cls) -> ProblemType:
        return ProblemType.CLASSIFICATION

    # docstr-coverage:inherited
    @classmethod
    def aliases(cls) -> List[str]:
        return ["pcs", "prediction_confidence_score", "PredictionConfidenceScore"]

    # docstr-coverage:inherited
    @classmethod
    def calculate(cls, nn_outputs: np.ndarray):
        _check_inputs_array(nn_outputs, quantifier_name="prediction_confidence_score")

        num_samples = nn_outputs.shape[0]
        calculated_predictions = np.argmax(nn_outputs, axis=1)
        max_values = nn_outputs[np.arange(num_samples), calculated_predictions]
        values_copy = nn_outputs.copy()
        values_copy[np.arange(num_samples), calculated_predictions] = -np.inf
        second_highest_values = np.max(values_copy, axis=1)

        pcs = max_values - second_highest_values
        return calculated_predictions, pcs


class MaxSoftmax(ConfidenceQuantifier):
    """
    The MaxSoftmax is a confidence metric in one-shot classification.
    It is the defaults in most simple use cases and sometimes also referred to
    as 'Vanilla Confidence Metric'.

    Inputs/activations have to be normalized using the softmax function over all classes.
    The class with the highest activation is chosen as prediction,
    the activation of this highest activation is used as confidence quantification.
    """

    # docstr-coverage:inherited
    @classmethod
    def aliases(cls) -> List[str]:
        return ["softmax", "MaxSoftmax", "max_softmax", "sm"]

    # docstr-coverage:inherited
    @classmethod
    def takes_samples(cls) -> bool:
        return False

    # docstr-coverage:inherited
    @classmethod
    def problem_type(cls) -> ProblemType:
        return ProblemType.CLASSIFICATION

    # docstr-coverage:inherited
    @classmethod
    def calculate(cls, nn_outputs: np.ndarray):
        _check_inputs_array(nn_outputs, quantifier_name="softmax")

        num_samples = nn_outputs.shape[0]
        calculated_predictions = np.argmax(nn_outputs, axis=1)
        max_values = nn_outputs[np.arange(num_samples), calculated_predictions]

        return calculated_predictions, max_values


class SoftmaxEntropy(UncertaintyQuantifier):
    """
    The SoftmaxEntropy is a confidence metric in one-shot classification.

    Inputs/activations have to be normalized using the softmax function over all classes.
    The class with the highest activation is chosen as prediction,
    the entropy over all activations is used as uncertainty quantification.
    """

    # docstr-coverage:inherited
    @classmethod
    def aliases(cls) -> List[str]:
        return ["softmax_entropy", "SoftmaxEntropy", "se"]

    # docstr-coverage:inherited
    @classmethod
    def takes_samples(cls) -> bool:
        return False

    # docstr-coverage:inherited
    @classmethod
    def problem_type(cls) -> ProblemType:
        return ProblemType.CLASSIFICATION

    # docstr-coverage:inherited
    @classmethod
    def calculate(cls, nn_outputs: np.ndarray):
        _check_inputs_array(nn_outputs, quantifier_name="softmax-entropy")

        calculated_predictions = np.argmax(nn_outputs, axis=1)
        entropies = uwiz.quantifiers.predictive_entropy.entropy(nn_outputs, axis=1)

        return calculated_predictions, entropies


class DeepGini(UncertaintyQuantifier):
    """DeepGini - Uncertainty (1 minus sum of squared softmax outputs).


    See Feng. et. al., "Deepgini: prioritizing massive tests to enhance
    the robustness of deep neural networks" for more information. ISSTA 2020.

    The implementation is part of our paper:
    Michael Weiss and Paolo Tonella, Simple Techniques Work Surprisingly Well
    for Neural Network Test Prioritization and Active Learning (Replication Paper),
    ISSTA 2021. (forthcoming)"""

    # docstr-coverage:inherited
    @classmethod
    def aliases(cls) -> List[str]:
        return ["deep_gini", "DeepGini"]

    # docstr-coverage:inherited
    @classmethod
    def takes_samples(cls) -> bool:
        return False

    # docstr-coverage:inherited
    @classmethod
    def is_confidence(cls) -> bool:
        return False

    # docstr-coverage:inherited
    @classmethod
    def calculate(cls, nn_outputs: np.ndarray):
        predictions, _ = MaxSoftmax.calculate(nn_outputs)
        gini = 1 - np.sum(nn_outputs * nn_outputs, axis=1)
        return predictions, gini

    # docstr-coverage:inherited
    @classmethod
    def problem_type(cls) -> ProblemType:
        return ProblemType.CLASSIFICATION

"""This module contains all quantifiers used to infer prediction and confidence (or uncertainty)
from neural network outputs. It also contains the QuantifierRegistry which allows to refer to
quantifiers by alias."""

__all__ = [
    "Quantifier",
    "MutualInformation",
    "PredictiveEntropy",
    "VariationRatio",
    "MeanSoftmax",
    "PredictionConfidenceScore",
    "MaxSoftmax",
    "SoftmaxEntropy",
    "StandardDeviation",
    "QuantifierRegistry",
]


# Base class
from .mean_softmax import MeanSoftmax

# Sampling Based Classification Quantifiers
from .mutual_information import MutualInformation

# Point Predictor Classification Quantifiers
from .one_shot_classifiers import MaxSoftmax, PredictionConfidenceScore, SoftmaxEntropy
from .predictive_entropy import PredictiveEntropy
from .quantifier import Quantifier

# Registry
from .quantifier_registry import QuantifierRegistry

# Regression
from .regression_quantifiers import StandardDeviation
from .variation_ratio import VariationRatio

from .mean_softmax import MeanSoftmax
from .mutual_information import MutualInformation
from .one_shot_classifiers import (
    DeepGini,
    MaxSoftmax,
    PredictionConfidenceScore,
    SoftmaxEntropy,
)
from .predictive_entropy import PredictiveEntropy
from .quantifier import Quantifier
from .regression_quantifiers import StandardDeviation
from .variation_ratio import VariationRatio


class QuantifierRegistry:
    """
    The quantifier registry keeps track of all quantifiers and their string aliases.
    This is primarily used to allow to pass string representations of quantifiers in predict_quantified
    method calls, but may also be used for other purposes where dynamic quantifier selection is desired.
    """

    _registries = dict()

    @classmethod
    def register(cls, quantifier: Quantifier) -> None:
        """
        Use this method to add a new quantifier to the registry.
        :param quantifier: The quantifier instance to be added.
        :return: None
        """
        for alias in quantifier.aliases():
            if alias.lower() in cls._registries:
                raise ValueError(
                    f"A quantifier with alias '{alias}' is already registered."
                )
            cls._registries[alias.lower()] = quantifier

    @classmethod
    def find(cls, alias: str) -> Quantifier:
        """
        Find quantifiers by their id.
        :param alias: A string representation of the quantifier, as defined in the quantifiers aliases method
        :return: A quantifier instance
        """
        record = cls._registries.get(alias.lower())
        if record is None:
            raise ValueError(
                f"No quantifier with alias '{alias}' was found. Check if you made any typos."
                f"If you use the alias of a custom quantifier (i.e., not an uwiz default quantifier),"
                f"make sure to register it through `uwiz.QuantifierRegistry.register(...)`"
            )
        return record


# Register uwiz classification quantifiers
QuantifierRegistry.register(MaxSoftmax())
QuantifierRegistry.register(PredictionConfidenceScore())
QuantifierRegistry.register(SoftmaxEntropy())
QuantifierRegistry.register(DeepGini())
QuantifierRegistry.register(VariationRatio())
QuantifierRegistry.register(PredictiveEntropy())
QuantifierRegistry.register(MutualInformation())
QuantifierRegistry.register(MeanSoftmax())

# Register uwiz classification quantifiers
QuantifierRegistry.register(StandardDeviation())

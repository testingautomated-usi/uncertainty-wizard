import abc
import enum
from typing import List, Union

import numpy as np


class ProblemType(enum.Enum):
    """
    An enum used to distinguish between regression and classification problems.
    Might be extended later, to support also other or more specific ProblemTypes
    (e.g. one-class classification)
    """

    REGRESSION = enum.auto()
    CLASSIFICATION = enum.auto()


class Quantifier(abc.ABC):
    """
    Quantifiers are dependencies, injectable into prediction calls,
    which calculate predictions and uncertainties or confidences
    from DNN outputs.

    The quantifier class is abstract and should not be directly implemented.
    Instead, new quantifiers should extend uwiz.quantifiers.ConfidenceQuantifier
    or uwiz.quantifiers.UncertaintyQuantifier instead.
    """

    @classmethod
    @abc.abstractmethod
    def aliases(cls) -> List[str]:
        """
        Aliases are string identifiers of this quantifier.
        They are used to select quantifiers by string in predict methods (need to be registered in quantifier_registry).

        Additionally, the first identifier in the list is used for logging purpose.
        Thus, the returned list have at least length 1.

        :return: list of quantifier identifiers
        """
        pass  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def is_confidence(cls) -> bool:
        """
        Boolean flag indicating whether this quantifier quantifies uncertainty or confidence.
        They are different as follows (assuming that the quantifier actually correctly captures the chance of misprediction):

        - In `uncertainty quantification`, the higher the quantification, the higher the chance of misprediction.
        - in `confidence quantification` the lower the quantification, the higher the change of misprediction.

         :return: True iff this is a confidence quantifier, False if this is an uncertainty quantifier

        """
        pass  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def takes_samples(cls) -> bool:
        """
        A flag indicating whether this quantifier relies on monte carlo samples
        (in which case the method returns True)
        or on a single neural network output
        (in which case the method return False)

        :return: True if this quantifier expects monte carlo samples for quantification. False otherwise.
        """
        pass  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def problem_type(cls) -> ProblemType:
        """
        Specifies whether this quantifier is applicable to classification or regression problems
        :return: One of the two enum values REGRESSION or CLASSIFICATION
        """
        pass  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def calculate(cls, nn_outputs: np.ndarray):
        """
        Calculates the predictions and uncertainties.


        Note this this assumes *batches* of neural network outputs.
        When using this method for a single nn output, make sure to reshape the passed array,
        e.g. using `x = np.expand_dims(x, axis=0)`

        The method returns a tuple of

        - A prediction (int or float) or array of predictions
        - A uncertainty or confidence quantification (float) or array of uncertainties

        :param nn_outputs: The NN outputs to be considered when determining prediction and uncertainty quantification
        :return: A tuple of prediction(s) and uncertainty(-ies).
        """
        pass  # pragma: no cover

    @classmethod
    def cast_conf_or_unc(
        cls, as_confidence: Union[None, bool], superv_scores: np.ndarray
    ) -> np.ndarray:
        """
        Utility method to convert confidence metrics into uncertainty and vice versa.
        Call `is_confidence()` to find out if this is a uncertainty or a confidence metric.

        The supervisors scores are converted as follows:

         - Confidences are multiplied by (-1) iff `as_confidence` is False
         - Uncertainties are multiplied by (-1) iff `as_confidence` is True
         - Otherwise, the passed supervisor scores are returned unchanged.


        :param as_confidence: : A boolean indicating if the scores should be converted to confidences (True) or uncertainties (False)
        :param superv_scores: : The scores that are to be converted, provided a conversion is needed.
        :return: The converted scores or the unchanged `superv_scores` (if `as_confidence` is None or no conversion is needed)

        """
        if as_confidence is not None and cls.is_confidence() != as_confidence:
            return superv_scores * -1
        return superv_scores


class ConfidenceQuantifier(Quantifier, abc.ABC):
    """
    An abstract Quantifier subclass, serving as superclass for all confidence quantifying quantifiers:
    In `confidence quantification` the lower the value, the higher the chance of misprediction.
    """

    # docstr-coverage:inherited
    @classmethod
    def is_confidence(cls) -> bool:
        return True


class UncertaintyQuantifier(Quantifier, abc.ABC):
    """
    An abstract Quantifier subclass, serving as superclass for all uncertainty quantifying quantifiers:
    In `uncertainty quantification` the lower the value, the lower the chance of misprediction.
    """

    # docstr-coverage:inherited
    @classmethod
    def is_confidence(cls) -> bool:
        return False

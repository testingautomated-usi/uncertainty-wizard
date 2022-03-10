from typing import List, Tuple

import numpy as np

from .quantifier import ProblemType, UncertaintyQuantifier
from .variation_ratio import VariationRatio


def entropy(data: np.ndarray, axis: int) -> np.ndarray:
    """
    A utility method to compute the entropy. May also be used by other quantifiers which internally rely on entropy.

    Following standard convention, the logarithm used in the entropy calculation is on base 2.

    :param data: The values on which the entropy should be computed.
    :param axis: Entropy will be taken along this axis.
    :return: An array containing the entropies. It is one dimension smaller than the passed data (the specified axis was removed).
    """
    # Remove zeros from nn_outputs (to allow to take logs)
    # Note that the actual increment (1e-20) does not matter, as it is multiplied by 0 below
    increments = np.zeros(shape=data.shape, dtype=np.float32)
    indexes_of_zeros = data == 0
    increments[indexes_of_zeros] = 1e-20
    nonzero_data = data + increments

    # These arrays can be quite large and are not used anymore - we free the space for the operations below
    del increments, indexes_of_zeros

    # Calculate and return the entropy
    return -np.sum(data * np.log2(nonzero_data), axis=axis)


class PredictiveEntropy(UncertaintyQuantifier):
    """
    A predictor & uncertainty quantifier, based on multiple samples (e.g. nn outputs) in a classification problem

    The prediction is made using a plurality vote, i.e., the class with the highest value in most samples is selected.
    In the case of a tie, the class with the lowest index is selected.

    The uncertainty is quantified using the predictive entropy;
    the entropy (base 2) of the per-class means of the sampled predictions.
    """

    # docstr-coverage:inherited
    @classmethod
    def aliases(cls) -> List[str]:
        return ["predictive_entropy", "pred_entropy", "PE"]

    # docstr-coverage:inherited
    @classmethod
    def takes_samples(cls) -> bool:
        return True

    # docstr-coverage:inherited
    @classmethod
    def problem_type(cls) -> ProblemType:
        return ProblemType.CLASSIFICATION

    # docstr-coverage:inherited
    @classmethod
    def calculate(cls, nn_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # For simplicity, we let the predictions be calculated by the Variation Ratio Code
        #    accepting a slight overhead from also calculating the actual variation ratio
        predictions, _ = VariationRatio.calculate(nn_outputs)

        # Take means over the samples
        means = np.mean(nn_outputs, axis=1)

        # The predictive entropy is the entropy of the means
        predictive_entropy = entropy(means, axis=1)

        return predictions, predictive_entropy

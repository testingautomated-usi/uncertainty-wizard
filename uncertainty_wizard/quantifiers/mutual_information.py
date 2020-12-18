from typing import List, Tuple

import numpy as np

from uncertainty_wizard.quantifiers.predictive_entropy import PredictiveEntropy, entropy
from uncertainty_wizard.quantifiers.quantifier import ProblemType, UncertaintyQuantifier


class MutualInformation(UncertaintyQuantifier):
    """
    A predictor & uncertainty quantifier, based on multiple samples (e.g. nn outputs) in a classification problem

    The prediction is made using a plurality vote, i.e., the class with the highest value in most samples is selected.
    In the case of a tie, the class with the lowest index is selected.

    The uncertainty is quantified using the mutual information.
    See the docs for a precise explanation of mutual information.
    """

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
    def aliases(cls) -> List[str]:
        return ["mutu_info", "mutual_information", "mi"]

    # docstr-coverage:inherited
    @classmethod
    def calculate(cls, nn_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions, pred_entropy = PredictiveEntropy.calculate(nn_outputs)

        entropies = entropy(nn_outputs, axis=2)
        entropy_means = np.mean(entropies, axis=1)

        # The mutual information is the predictive entropy minus the mean of the entropies
        mutual_information = pred_entropy - entropy_means

        return predictions, mutual_information

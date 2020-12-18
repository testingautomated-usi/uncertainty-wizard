from typing import List, Tuple

import numpy as np

from uncertainty_wizard.quantifiers.one_shot_classifiers import MaxSoftmax
from uncertainty_wizard.quantifiers.quantifier import ConfidenceQuantifier, ProblemType


class MeanSoftmax(ConfidenceQuantifier):
    """
    A predictor & uncertainty quantifier, based on multiple samples (e.g. nn outputs) in a classification problem.

    Both the prediction and the uncertainty score are calculated using the average softmax values over all samples.
    This is sometimes also called 'ensembling', as it is often used in deep ensembles.
    """

    # docstr-coverage:inherited
    @classmethod
    def aliases(cls) -> List[str]:
        return ["mean_softmax", "ensembling", "ms"]

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
        assert len(nn_outputs.shape) == 3, (
            "nn_outputs for this quantifier must have shape "
            "(num_inputs, num_samples, num_classes)"
        )

        # Take means over the samples
        means = np.mean(nn_outputs, axis=1)

        # Calculate argmax as prediction and max as confidence
        return MaxSoftmax.calculate(means)

from typing import List

import numpy as np

from .quantifier import ProblemType, UncertaintyQuantifier


def validate_shape(nn_outputs):
    """Makes sure the nn outputs is a valid input for the regression classifiers in this file."""
    assert len(nn_outputs.shape) >= 2, (
        "nn_outputs for Average Standard Deviation must have shape "
        "(number_of_inputs, num_samples, pred_dim_1, pred_dim_2, ...)"
    )
    num_inputs = nn_outputs.shape[0]
    num_samples = nn_outputs.shape[1]
    return num_inputs, num_samples


class StandardDeviation(UncertaintyQuantifier):
    """
    Measures the standard deviation over different samples of a regression problem, i.e., an arbitrary problem,
    which is used as Uncertainty and the mean of the samples as prediction

    This implementation can handle both regression prediction consisting of a single scalar dnn output
    as well as larger-shaped dnn outputs. In the latter case, entropy is calculated and returned
    for every position in the dnn output shape.
    """

    # docstr-coverage:inherited
    @classmethod
    def takes_samples(cls) -> bool:
        return True

    # docstr-coverage:inherited
    @classmethod
    def problem_type(cls) -> ProblemType:
        return ProblemType.REGRESSION

    # docstr-coverage:inherited
    @classmethod
    def aliases(cls) -> List[str]:
        return ["standard_deviation", "std_dev", "std", "stddev", "StandardDeviation"]

    # docstr-coverage:inherited
    @classmethod
    def calculate(cls, nn_outputs: np.ndarray):
        _, _ = validate_shape(nn_outputs)
        predictions = np.mean(nn_outputs, axis=1)
        uncertainties = np.std(nn_outputs, axis=1)
        return predictions, uncertainties

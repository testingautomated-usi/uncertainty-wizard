from typing import List, Tuple

import numpy as np

from uncertainty_wizard.quantifiers.quantifier import ProblemType, UncertaintyQuantifier


class VariationRatio(UncertaintyQuantifier):
    """
    A predictor & uncertainty quantifier, based on multiple samples (e.g. nn outputs) in a classification problem

    The prediction is made using a plurality vote, i.e., the class with the highest value in most samples is selected.
    In the case of a tie, the class with the lowest index is selected.

    The uncertainty is quantified using the variation ratio `1 - w / S`,
    where w is the number of samples where the overall prediction equals the prediction of the sample
    and S is the total number of samples.
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
        return ["variation_ratio", "vr", "var_ratio"]

    # docstr-coverage:inherited
    @classmethod
    def calculate(cls, nn_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert len(nn_outputs.shape) == 3, (
            "nn_outputs for this quantifier must have shape "
            "(num_inputs, num_samples, num_classes)"
        )
        num_inputs = nn_outputs.shape[0]
        num_samples = nn_outputs.shape[1]
        num_classes = nn_outputs.shape[2]

        sofmax_table = np.reshape(nn_outputs, (num_inputs * num_samples, num_classes))
        per_sample_argmax = np.argmax(sofmax_table, axis=1)

        is_max_array = np.zeros(
            shape=(num_inputs * num_samples, num_classes), dtype=bool
        )
        is_max_array[np.arange(num_inputs * num_samples), per_sample_argmax] = True

        per_input_is_max_array = np.reshape(
            is_max_array, (num_inputs, num_samples, num_classes)
        )

        sum_array_dtype = cls._sum_array_dtype(num_samples)
        votes_counts = np.sum(per_input_is_max_array, axis=1, dtype=sum_array_dtype)

        predictions = cls._prediction_array_with_appropriate_dtype(
            num_classes, num_inputs
        )
        np.argmax(votes_counts, axis=1, out=predictions)
        max_counts = votes_counts[np.arange(num_inputs), predictions]

        vr = 1 - max_counts / num_samples
        return predictions, vr

    @classmethod
    def _sum_array_dtype(cls, num_samples):
        """Selects an appropriate dtype (np.uint16 or np.uint8) based on the number of samples"""
        # uint16 allows up to 65535 samples (which is way above reasonable for any problem)
        # If there are up to 255 samples per input, we can save half the memory using uint8
        sum_array_dtype = np.uint16
        if num_samples < 256:
            sum_array_dtype = np.uint8
        return sum_array_dtype

    @classmethod
    def _prediction_array_with_appropriate_dtype(cls, num_classes, num_inputs):
        """Creates an empty one-dimensional array with the number of inputs as length
        and an appropriate dtype (np.uint16 or np.uint8).
        This can then be used to store the predictions"""
        # uint16 allows up to 65535 samples (which is way above reasonable for any problem)
        # If there are up to 255 samples per input, we can save half the memory using uint8
        predictions = np.empty(shape=num_inputs, dtype=np.uint16)
        if num_classes < 256:
            predictions = np.empty(shape=num_inputs, dtype=np.uint8)
        return predictions

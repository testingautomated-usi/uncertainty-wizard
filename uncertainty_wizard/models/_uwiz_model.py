import warnings
from typing import Iterable, Union

from uncertainty_wizard.internal_utils import UncertaintyWizardWarning
from uncertainty_wizard.quantifiers import Quantifier, QuantifierRegistry


class _UwizModel:
    @staticmethod
    def _quantifiers_as_list(quantifier):
        is_single_quantifier = False
        if isinstance(quantifier, list):
            quantifiers = quantifier
        else:
            quantifiers = [quantifier]
            is_single_quantifier = True

        quantifier_objects = []
        for quantifier in quantifiers:
            if isinstance(quantifier, str):
                quantifier_objects.append(QuantifierRegistry.find(quantifier))
            elif isinstance(quantifier, Quantifier):
                quantifier_objects.append(quantifier)
            else:
                raise TypeError(
                    "The passed quantifier is neither a quantifier instance nor a quantifier alias (str)."
                    f"Type of the passed object {str(type(quantifier))}"
                )

        point_prediction_quantifiers = [
            q for q in quantifier_objects if q.takes_samples() is False
        ]
        samples_based_quantifiers = [
            q for q in quantifier_objects if q.takes_samples() is True
        ]
        return (
            quantifier_objects,
            point_prediction_quantifiers,
            samples_based_quantifiers,
            is_single_quantifier,
        )

    @staticmethod
    def _check_quantifier_heterogenity(
        as_confidence: Union[None, bool], quantifiers: Iterable[Quantifier]
    ) -> None:
        if as_confidence is None:
            num_uncertainties = sum(q.is_confidence() is False for q in quantifiers)
            num_confidences = sum(q.is_confidence() is True for q in quantifiers)
            if num_confidences > 0 and num_uncertainties > 0:
                warnings.warn(
                    """ 
                    You are predicting both confidences and uncertainties.
                    When comparing the two, keep in mind that confidence is expected to 
                    correlate positively with the probability of a correct prediction,
                    while uncertainty is expected to correlate negatively.
    
                    You may want to use the `as_confidence` flag when calling `predict_quantified`: 
                    If set to false, it multiplies confidences by (-1). 
                    If set to true, it multiplies uncertainties by (-1).
    
                    See uwiz.Quantifier.cast_conf_or_unc for more details.
                    """,
                    category=UncertaintyWizardWarning,
                    stacklevel=3,
                )

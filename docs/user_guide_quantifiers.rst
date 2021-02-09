User Guide: Quantifiers
##############################

Quantifiers are dependencies, injectable into prediction calls,
which calculate predictions and uncertainties or confidences
from DNN outputs:

.. code-block:: python
   :caption: Use of quantifiers on uwiz models

    # Let's use a quantifier that calculates the entropy on a regression variable as uncertainty
    predictions, entropy = model.predict_quantified(x_test, quantifier='predictive_entropy')

    # Equivalently, we can pass the quantifier as object
    quantifier = uwiz.quantifiers.PredictiveEntropy()
    predictions, entropy = model.predict_quantified(x_test, quantifier=quantifer)

    # We can also pass multiple quantifiers.
    # In that case, `predict_quantified` returns a (prediction, confidence_or_uncertainty) tuple
    # for every passed quantifier.
    results = model.predict_quantified(x_test, quantifier=['predictive_entropy', 'standard_deviation')
    # results[0] is a tuple of predictions and entropies
    # results[1] is a tuple of predictions and standard deviations

Besides the prediction, quantifiers quantify either the networks confidence or its uncertainty.
The difference between that two is as follows
(assuming that the quantifier actually correctly captures the chance of misprediction):

- In `uncertainty quantification`, the higher the value, the higher the chance of misprediction.
- In `confidence quantification` the lower the value, the higher the chance of misprediction.

For most applications where you use multiple quantifiers, you probably want to quantify
either uncertainties or confidences to allow to use the quantifiers outputs interchangeable.
Setting the param ``model.predict_quantified(..., as_confidence=True)``
convert uncertainties into confidences. ``as_confidence=False`` converts confidences into uncertainties.
The default is 'None', in which case no conversions are made.

.. note::
    Independent on how many quantifiers you pass to the `predict_quantified` method,
    the outputs of the neural networks inference are re-used wherever possible for a more efficient execution.
    Thus, it is better to call `predict_quantified` with two quantifiers than
    to call `predict_quantified` twice, with one quantifier each.



Quantifiers implemented in Uncertainty Wizard
*********************************************
This Section provides an overview of the quantifiers provided in uncertainty wizard:
For a precise discussion of the quantifiers listed here, please consult our paper
and the docstrings of the quantifiers.

Point Prediction Quantifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------------------------+------------------+------------------------------------------+
| | Class                        | | Problem Type   | | Aliases                                |
| | (uwiz.quantifiers.<...>)     | |                | | (besides class name)                   |
+================================+==================+==========================================+
| | MaxSoftmax                   | | Classification | | SM, softmax, max_softmax,              |
+--------------------------------+------------------+------------------------------------------+
| | PredictionConfidenceScore    | | Classification | | PCS, prediction_confidence_score       |
+--------------------------------+------------------+------------------------------------------+
| | SoftmaxEntropy               | | Classification | | SE, softmax_entropy                    |
+--------------------------------+------------------+------------------------------------------+


Monte Carlo Sampling Quantifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


+--------------------------------+------------------+------------------------------------------+
| | Class                        | | Problem Type   | | Aliases                                |
| | (uwiz.quantifiers.<...>)     | |                | | (besides class name)                   |
+================================+==================+==========================================+
| | VariationRatio               | | Classification | | VR, var_ratio,                         |
| |                              | |                | | variation_ratio                        |
+--------------------------------+------------------+------------------------------------------+
| | PredictiveEntropy            | | Classification | | PE, pred_entropy,                      |
| |                              | |                | | predictive_entropy                     |
+--------------------------------+------------------+------------------------------------------+
| | MutualInformation            | | Classification | | MI, mutu_info,                         |
| |                              | |                | | mutual_information                     |
+--------------------------------+------------------+------------------------------------------+
| | MeanSoftmax                  | | Classification | | MS, mean_softmax,                      |
| |                              | |                | | ensembling                             |
+--------------------------------+------------------+------------------------------------------+
| | StandardDeviation            | | Regression     | | STD, stddev, std_dev,                  |
| |                              | |                | | standard_deviation                     |
+--------------------------------+------------------+------------------------------------------+



Custom Quantifers
*****************

You can of course also use custom quantifiers with uncertainty wizard.
It's as easy as extending ``uwiz.quantifiers.Quantifier`` and implement all abstract methods according
to the description in the superclass method docstrings.

Let's for example assume you want to create an **identity function** quantifier for a sampling based DNN
(i.e., a stochastic DNN or a deep ensemble) for a classification problem,
which does not actually calculate a prediction and uncertainty, but just returns the observed DNN outputs.
This can be achieved using the following snippet:

.. code-block:: python
   :caption: Custom quantifier definition: Identity Quantifier

    class IdentityQuantifer(uwiz.quantifiers.Quantifier):
        @classmethod
        def aliases(cls) -> List[str]:
            return ["custom::identity"]

        @classmethod
        def takes_samples(cls) -> bool:
            return True

        @classmethod
        def is_confidence(cls) -> bool:
            # Does not matter for the identity function
            return False

        @classmethod
        def calculate(cls, nn_outputs: np.ndarray):
            # Return None as prediction and all DNN outputs as 'quantification'
            return None, nn_outputs

        @classmethod
        def problem_type(cls) -> uwiz.ProblemType:
            return uwiz.ProblemType.CLASSIFICATION


If you want to call your custom quantifier by its alias, you need to add it to the registry.
To prevent name clashes in future uncertainty wizard versions, where more quantifiers might be registered by default,
we recommend you to preprend "custom::" to any of your quantifiers aliases.

.. code-block:: python
   :caption: Register a quantifier in the quantifier registry

    custom_instance = IdentityQuantifier()
    uwiz.quantifiers.QuantifierRegistry().register(custom_instance)

    model = # (...) uwiz model creation, compiling and fitting
    x_test = # (...) get the data for your predictions

    # Now this call, where we calculate the variation ratio,
    # and also return the observed DNN outputs...
    results = model.predict_quantified(x_test, num_samples=20,
                                       quantifier=["var_ratio", "custom::identity"])
    # ... is equivalent to this call...
    results = model.predict_quantified(x_test, num_samples=20,
                                       quantifier=["var_ratio", IdentityQuantifier()])


.. warning::
   Quantifiers added to the registry should be stateless and all their functions should be pure functions.
   Otherwise, reproduction of results might not be possible.
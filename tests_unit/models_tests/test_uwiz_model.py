import warnings
from unittest import TestCase

import numpy as np
import tensorflow as tf

import uncertainty_wizard as uwiz
from uncertainty_wizard.internal_utils import UncertaintyWizardWarning


class EnsembleFunctionalTest(TestCase):
    @staticmethod
    def _dummy_stochastic_classifier():
        model = uwiz.models.StochasticSequential(
            layers=[
                tf.keras.layers.Input(shape=1000),
                tf.keras.layers.Dense(1000),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1000),
                tf.keras.layers.Dense(10),
                tf.keras.layers.Softmax(),
            ]
        )
        model.compile(loss="mse")
        # The labels make no sense for a softmax output layer, but this does not matter
        model.fit(x=np.ones((10, 1000)), y=np.ones((10, 10)), epochs=2)
        return model

    def test_error_if_invalid_quantifier_type(self):
        model = self._dummy_stochastic_classifier()
        # Test that only string and objects are accepted as quantifiers
        with self.assertRaises(TypeError):
            model.predict_quantified(np.ones((10, 1000)), quantifier=5)

    def test_error_if_point_predictors(self):
        model = self._dummy_stochastic_classifier()
        # Test that only string and objects are accepted as quantifiers
        with self.assertRaises(ValueError):
            model.predict_quantified(
                np.ones((10, 1000)), quantifier=["PCS", "SoftmaxEntropy"]
            )

    def test_error_if_point_predictors(self):
        model = self._dummy_stochastic_classifier()
        # Test that only string and objects are accepted as quantifiers
        with self.assertWarns(UncertaintyWizardWarning):
            model.predict_quantified(
                np.ones((10, 1000)), quantifier=["PCS", "SoftmaxEntropy"]
            )

        # Test that no warning is printed if passed individually
        with warnings.catch_warnings(record=True) as w:
            model.predict_quantified(np.ones((10, 1000)), quantifier=["SoftmaxEntropy"])
            model.predict_quantified(np.ones((10, 1000)), quantifier=["PCS"])
        self.assertEqual(len(w), 0, w)

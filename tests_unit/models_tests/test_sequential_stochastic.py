from unittest import TestCase

import numpy as np
import tensorflow as tf

import uncertainty_wizard as uwiz
from uncertainty_wizard.internal_utils import UncertaintyWizardWarning
from uncertainty_wizard.models import StochasticSequential
from uncertainty_wizard.quantifiers import StandardDeviation, VariationRatio, MaxSoftmax


class SequentialStochasticTest(TestCase):
    @staticmethod
    def _dummy_model():
        model = StochasticSequential()
        model.add(tf.keras.layers.Input(shape=1000))
        model.add(tf.keras.layers.Dropout(rate=0.5))
        return model

    @staticmethod
    def _dummy_classifier():
        model = StochasticSequential()
        model.add(tf.keras.layers.Input(shape=1000))
        model.add(tf.keras.layers.Dropout(rate=0.5))
        model.add(tf.keras.layers.Dense(10, activation="softmax"))
        # compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )
        return model

    def test_result_as_dict(self):
        model = self._dummy_classifier()
        x = np.ones((10, 1000))
        res = model.predict_quantified(x=x,
                                       quantifier=[
                                           "MaxSoftmax", VariationRatio(),
                                       ],
                                       return_alias_dict=True)

        self.assertTrue(isinstance(res, dict))
        for key, values in res.items():
            self.assertTrue(isinstance(key, str))
            self.assertTrue(isinstance(values, tuple))
            self.assertEqual(len(values), 2)
            self.assertEqual(values[0].shape, (10,))
            self.assertEqual(values[1].shape, (10,))

        for q in [MaxSoftmax(), VariationRatio()]:
            for a in q.aliases():
                self.assertTrue(a in res.keys())

    def test_return_type_default_multi_quant(self):
        model = self._dummy_classifier()
        x = np.ones((10, 1000))
        res = model.predict_quantified(x=x, quantifier=["MaxSoftmax", VariationRatio()])
        self.assertTrue(isinstance(res, list))
        self.assertTrue(len(res), 2)

    def test_return_type_default_single_quant(self):
        model = self._dummy_classifier()
        x = np.ones((10, 1000))
        res = model.predict_quantified(x=x, quantifier="MaxSoftmax")
        self.assertTrue(isinstance(res, tuple))
        self.assertTrue(len(res), 2)

    def test_predict_is_deterministic(self):
        model = self._dummy_model()
        y = model.predict(x=np.ones((10, 1000)))
        self.assertTrue(np.all(y == 1))

    def test_sampled_predict_is_not_deterministic(self):
        model = self._dummy_model()
        self._assert_random_samples(model)

    def test_sampled_turning_sampling_on_and_off_iteratively(self):
        model = self._dummy_model()
        self._test_randomized_on_off(model)

    def _test_randomized_on_off(self, model):
        for _ in range(2):
            self._assert_random_samples(model)

            y = model.predict(x=np.ones((10, 1000)))
            self.assertTrue(np.all(y == 1))

    def _assert_random_samples(self, model):
        y, std = model.predict_quantified(
            x=np.ones((10, 1000)), quantifier=StandardDeviation(), sample_size=20
        )
        self.assertFalse(np.all(y == 1), y)
        self.assertFalse(np.all(std == 0), std)

    def test_warns_on_compile_if_not_stochastic(self):
        model = StochasticSequential()
        model.add(tf.keras.layers.Input(shape=1000))
        model.add(tf.keras.layers.Dense(1000))
        with self.assertWarns(UncertaintyWizardWarning):
            model.compile()

    def test_save_and_load_model(self):
        stochastic = self._dummy_model()
        # Model can currently (as of tf 2.1) only be saved if build, fit or predict was called
        stochastic.predict(np.ones((10, 1000)))
        stochastic.save("/tmp/model")
        del stochastic
        stochastic_loaded = uwiz.models.load_model("/tmp/model")
        self._test_randomized_on_off(stochastic_loaded)

    def test_weights_and_stochasicmode_on_clone_from_keras(self):
        # Prepare a model with dropout to be used to create a StochasticModel
        keras_model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    1000, kernel_initializer="random_normal", bias_initializer="zeros"
                ),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(
                    10, kernel_initializer="random_normal", bias_initializer="zeros"
                ),
                tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
            ]
        )
        keras_model.compile(loss="mse", optimizer="adam", metrics=["mse"])
        keras_model.fit(
            np.ones((20, 1000), dtype=float), np.zeros((20, 10)), batch_size=1, epochs=1
        )

        # Call the model under test
        uwiz_model = uwiz.models.stochastic_from_keras(keras_model)

        # Demo input for tests
        input = np.ones((10, 1000), dtype=float)

        # Assert that both models make the same predictions
        keras_prediction = keras_model.predict(input)
        uwiz_prediction = uwiz_model.predict(input)
        np.testing.assert_array_equal(keras_prediction, uwiz_prediction)

        # Test that stochastic mode is working on cloned model
        self._assert_random_samples(uwiz_model)

    def test_randomness_error_on_clone_from_keras(self):
        keras_model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    10, kernel_initializer="random_normal", bias_initializer="zeros"
                ),
                tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
            ]
        )
        keras_model.compile(loss="mse", optimizer="adam", metrics=["mse"])
        keras_model.fit(
            np.ones((20, 10), dtype=float), np.zeros((20, 10)), batch_size=1, epochs=1
        )

        # make sure no validation error is thrown when determinism is expected
        _ = uwiz.models.stochastic_from_keras(keras_model, expect_determinism=True)

        self.assertRaises(
            ValueError, lambda: uwiz.models.stochastic_from_keras(keras_model)
        )

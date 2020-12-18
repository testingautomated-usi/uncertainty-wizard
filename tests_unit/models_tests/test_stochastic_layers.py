from unittest import TestCase

import numpy as np
import tensorflow as tf

import uncertainty_wizard as uwiz
from uncertainty_wizard.internal_utils import UncertaintyWizardWarning
from uncertainty_wizard.models._stochastic._stochastic_mode import StochasticMode
from uncertainty_wizard.models.stochastic_utils.layers import (
    UwizBernoulliDropout,
    UwizGaussianDropout,
    UwizGaussianNoise,
)

DUMMY_INPUT = np.ones((10, 1000), dtype=np.float32)


class StochasticLayerTest(TestCase):
    def test_warns_custom_keras_subtype(self):
        class SubDropout(tf.keras.layers.Dropout):
            def __init__(self, rate, **kwargs):
                super().__init__(rate, **kwargs)

        subclass_instance = SubDropout(rate=0.5)
        uwiz_type = uwiz.models.stochastic_utils.layers.UwizBernoulliDropout
        with self.assertWarns(UncertaintyWizardWarning) as cm:
            uwiz.models.stochastic_utils.layers._has_casting_preventing_subtype(
                subclass_instance,
                expected_type=tf.keras.layers.Dropout,
                corresponding_uw_type=uwiz_type,
            )
        the_warning = cm.warning
        self.assertTrue(
            "make sure the models stochastic mode tensor is respected"
            in the_warning.args[0]
        )

    def test_warns_custom_uwiz_subtype(self):
        class SubUwizGaussianNoise(UwizGaussianNoise):
            def __init__(self, stddev, stochastic_mode, **kwargs):
                super().__init__(stddev, stochastic_mode, **kwargs)

        subclass_instance = SubUwizGaussianNoise(
            stddev=0.5, stochastic_mode=StochasticMode()
        )
        uwiz_type = uwiz.models.stochastic_utils.layers.UwizGaussianNoise
        with self.assertWarns(UncertaintyWizardWarning) as cm:
            uwiz.models.stochastic_utils.layers._has_casting_preventing_subtype(
                subclass_instance,
                expected_type=tf.keras.layers.GaussianNoise,
                corresponding_uw_type=uwiz_type,
            )
        the_warning = cm.warning
        self.assertTrue(
            "ou know what you did and set up the stochastic mode correctly."
            in the_warning.args[0]
        )

    def assert_listens_to_forceful_enabled(self, layer_constructor):
        """
        Setup to test that the stochastic mode is read correctly in a noise layer in eager mode.
        To be called directly by the test class for the corresponding layer

        Args:
            layer_constructor (): No-Args lambda that creates a corresponding layer.

        Returns:
            None
        """
        stochastic_mode = StochasticMode()
        self.assertTrue(
            tf.executing_eagerly(), "This test is supposed to test eager execution"
        )
        layer = layer_constructor(stochastic_mode)
        stochastic_mode.as_tensor().assign(tf.ones((), dtype=bool))
        enabled_result = layer(DUMMY_INPUT)
        self.assertFalse(
            tf.reduce_all(tf.equal(enabled_result, DUMMY_INPUT).numpy()),
            "No values were dropped",
        )
        stochastic_mode.as_tensor().assign(tf.zeros((), dtype=bool))
        disabled_result = layer(DUMMY_INPUT)
        self.assertTrue(
            tf.reduce_all(tf.equal(disabled_result, DUMMY_INPUT)).numpy(),
            "Some values changed - which should not happen",
        )

    def assert_listens_to_forceful_enabled_in_sequential_predict(
        self, layer_constructor
    ):
        """
        Setup to test that the stochastic mode is read correctly in a noise layer in eager mode,
        when used in a tf.keras.Sequential model for prediction.
        To be called directly by the test class for the corresponding layer

        Args:
            layer_constructor (): No-Args lambda that creates a corresponding layer.

        Returns:
            None
        """
        stochastic_mode = StochasticMode()
        self.assertTrue(
            tf.executing_eagerly(), "This test is supposed to test eager execution"
        )
        stochastic_mode.as_tensor().assign(tf.ones((), dtype=bool))
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=1000))
        model.add(layer_constructor(stochastic_mode))
        enabled_result = model.predict(DUMMY_INPUT)
        self.assertFalse(
            tf.reduce_all(tf.equal(enabled_result, DUMMY_INPUT).numpy()),
            "No values were dropped",
        )

        stochastic_mode.as_tensor().assign(tf.zeros((), dtype=bool))
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=1000))
        model.add(layer_constructor(stochastic_mode))
        disabled_result = model.predict(DUMMY_INPUT)
        self.assertTrue(
            tf.reduce_all(tf.equal(disabled_result, DUMMY_INPUT)).numpy(),
            "Some values changed - which should not happen",
        )

    def assert_listens_to_forceful_enabled_graph_mode(self, layer_constructor):
        """
        Setup to test that the stochastic mode is read correctly in a noise layer.
        To be called directly by the test class for the corresponding layer

        Args:
            layer_constructor (): function that creates a layer. Argument: the stochastic mode instance to use

        Returns:
            None
        """
        stochastic_mode = StochasticMode()

        @tf.function
        def run_test():
            self.assertFalse(
                tf.executing_eagerly(),
                "This test is supposed to test disabled eager execution",
            )
            layer = layer_constructor(stochastic_mode)
            stochastic_mode.as_tensor().assign(tf.ones((), dtype=bool))
            enabled_result = layer(DUMMY_INPUT)
            input_as_tensor = tf.constant(DUMMY_INPUT, dtype=tf.float32)
            test_enabled = tf.debugging.Assert(
                tf.math.logical_not(
                    tf.reduce_all(tf.equal(enabled_result, input_as_tensor))
                ),
                [tf.constant("No values were dropped"), enabled_result],
            )
            stochastic_mode.as_tensor().assign(tf.zeros((), dtype=bool))
            disabled_result = layer(DUMMY_INPUT)
            test_disabled = tf.debugging.Assert(
                tf.reduce_all(tf.equal(disabled_result, input_as_tensor)),
                [
                    tf.constant("Some values changed - which should not happen"),
                    disabled_result,
                ],
            )
            stochastic_mode.as_tensor().assign(tf.zeros((), dtype=bool))

            with tf.control_dependencies([test_enabled, test_disabled]):
                pass

        run_test()


class TestBernoulliDropout(StochasticLayerTest):
    def test_listens_to_forceful_enabled(self):
        self.assert_listens_to_forceful_enabled(
            lambda sm: UwizBernoulliDropout(0.5, stochastic_mode=sm)
        )

    def test_listens_to_forceful_enabled_in_sequential(self):
        self.assert_listens_to_forceful_enabled_in_sequential_predict(
            lambda sm: UwizBernoulliDropout(0.5, stochastic_mode=sm)
        )

    def test_listens_to_forceful_enabled_non_eager(self):
        self.assert_listens_to_forceful_enabled_graph_mode(
            lambda sm: UwizBernoulliDropout(0.5, stochastic_mode=sm)
        )

    def test_cast_from_keras(self):
        plain_keras = tf.keras.layers.Dropout(rate=0.5)
        stochastic_mode = StochasticMode()
        output = UwizBernoulliDropout.from_keras_layer(
            plain_keras, stochastic_mode=stochastic_mode
        )
        self.assertIsInstance(output, UwizBernoulliDropout)
        self.assertEqual(output.rate, 0.5)
        self.assertEqual(
            stochastic_mode.as_tensor(), output.stochastic_mode.as_tensor()
        )


class TestGaussianDropout(StochasticLayerTest):
    def test_listens_to_forceful_enabled(self):
        self.assert_listens_to_forceful_enabled(
            lambda sm: UwizGaussianDropout(0.5, stochastic_mode=sm)
        )

    def test_listens_to_forceful_enabled_in_sequential(self):
        self.assert_listens_to_forceful_enabled_in_sequential_predict(
            lambda sm: UwizGaussianDropout(0.5, stochastic_mode=sm)
        )

    def test_listens_to_forceful_enabled_non_eager(self):
        self.assert_listens_to_forceful_enabled_graph_mode(
            lambda sm: UwizGaussianDropout(0.5, stochastic_mode=sm)
        )

    def test_cast_from_keras(self):
        plain_keras = tf.keras.layers.GaussianDropout(rate=0.5)
        stochastic_mode = StochasticMode()
        output = UwizGaussianDropout.from_keras_layer(
            plain_keras, stochastic_mode=stochastic_mode
        )
        self.assertIsInstance(output, UwizGaussianDropout)
        self.assertEqual(output.rate, 0.5)
        self.assertEqual(
            stochastic_mode.as_tensor(), output.stochastic_mode.as_tensor()
        )


class TestGaussianNoise(StochasticLayerTest):
    def test_listens_to_forceful_enabled(self):
        self.assert_listens_to_forceful_enabled(
            lambda sm: UwizGaussianNoise(0.5, stochastic_mode=sm)
        )

    def test_listens_to_forceful_enabled_in_sequential(self):
        self.assert_listens_to_forceful_enabled_in_sequential_predict(
            lambda sm: UwizGaussianNoise(0.5, stochastic_mode=sm)
        )

    def test_listens_to_forceful_enabled_non_eager(self):
        self.assert_listens_to_forceful_enabled_graph_mode(
            lambda sm: UwizGaussianNoise(0.5, stochastic_mode=sm)
        )

    def test_cast_from_keras(self):
        plain_keras = tf.keras.layers.GaussianNoise(stddev=0.5)
        stochastic_mode = StochasticMode()
        output = UwizGaussianNoise.from_keras_layer(
            plain_keras, stochastic_mode=stochastic_mode
        )
        self.assertIsInstance(output, UwizGaussianNoise)
        self.assertEqual(output.stddev, 0.5)
        self.assertEqual(
            stochastic_mode.as_tensor(), output.stochastic_mode.as_tensor()
        )

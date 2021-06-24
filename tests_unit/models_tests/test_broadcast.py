from unittest import TestCase

import numpy as np
import tensorflow as tf

import uncertainty_wizard as uwiz


class BroadcastingTest(TestCase):
    def test_regular_numpy_broadcasting(self):
        x = np.arange(100).T
        broadcasted = uwiz.models.StochasticSequential.broadcast(10, x)
        count = 0
        for i, e in enumerate(broadcasted):
            self.assertEqual(e, np.floor(i / 10))
            count += 1
        self.assertEqual(count, 100 * 10)

    def test_tf_data_broadcasting(self):
        x = tf.data.Dataset.from_tensor_slices(np.arange(100).T)
        broadcasted = uwiz.models.StochasticSequential.broadcast(10, x)
        count = 0
        for i, e in enumerate(broadcasted):
            self.assertEqual(e, np.floor(i / 10))
            count += 1
        self.assertEqual(count, 100 * 10)

    def test_multi_input(self):
        x1 = np.arange(100).T
        x2 = np.arange(200).reshape((100, 2))

        atomic_ds = (
            tf.data.Dataset.from_tensor_slices(x1),
            tf.data.Dataset.from_tensor_slices(x2),
        )
        dataset = tf.data.Dataset.zip(atomic_ds)
        broadcasted = uwiz.models.StochasticSequential.broadcast(10, dataset)

        count = 0
        for i, e in enumerate(broadcasted):
            self.assertEqual(len(e), 2)
            self.assertEqual(e[1].shape[0], 2)
            self.assertEqual(e[0], np.floor(i / 10))
            count += 1
        self.assertEqual(count, 100 * 10)

    def test_feature_inputs(self):
        dataset = tf.data.Dataset.zip((
            tf.data.Dataset.range(100),
            tf.data.Dataset.range(100, 200)
        )).map(lambda x, y: {"a": x, "b": y})

        broadcasted = uwiz.models.StochasticSequential.broadcast(10, dataset)
        count = 0
        for i, e in enumerate(broadcasted):
            self.assertEqual(len(e), 2)
            self.assertEqual(e['a'], np.floor(i / 10))
            self.assertEqual(e['b'], np.floor(i / 10) + 100)
            count += 1
        self.assertEqual(count, 100 * 10)

    def test_multi_input_model(self):
        # Dummy Model
        inputs = [tf.keras.Input((1,), name="a"), tf.keras.Input((1,), name="b")]
        combined = inputs[0] + inputs[1]
        dropped = tf.keras.layers.Dropout(0.5)(combined)
        output = tf.keras.layers.Dense(1)(dropped)
        model = tf.keras.Model(inputs=inputs, outputs=output)

        # Dummy Dataset
        dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(tf.zeros((10, 1))),
            tf.data.Dataset.from_tensor_slices(tf.ones((10, 1))),
        )).map(lambda x, y: {"a": x, "b": y})

        stochastic_model = uwiz.models.stochastic_from_keras(model)
        pred, unc = stochastic_model.predict_quantified(dataset,
                                                        quantifier='var_ratio',
                                                        sample_size=16)

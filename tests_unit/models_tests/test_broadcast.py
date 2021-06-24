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

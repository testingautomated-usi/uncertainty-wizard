from unittest import TestCase

import numpy as np
import tensorflow as tf


class TestDropoutSeedAssumptions(TestCase):
    def test_dropout_is_random_with_no_tf_seed(self):
        dropout = tf.keras.layers.Dropout(rate=0.1)
        noseed_a = dropout.call(np.ones((10, 100)), training=True)
        noseed_b = dropout.call(np.ones((10, 100)), training=True)
        self.assertFalse(np.all(noseed_a == noseed_b))

    def test_dropout_seed_is_not_sufficient_for_reproduction(self):
        dropout_a = tf.keras.layers.Dropout(rate=0.1, seed=1)
        noseed_a = dropout_a.call(np.ones((10, 100)), training=True)
        dropout_b = tf.keras.layers.Dropout(rate=0.1, seed=1)
        noseed_b = dropout_b.call(np.ones((10, 100)), training=True)
        self.assertFalse(np.all(noseed_a == noseed_b))

    def test_dropout_is_deterministic_with_tf_seed(self):
        dropout = tf.keras.layers.Dropout(rate=0.1)
        tf.random.set_seed(123)
        seeded_a = dropout.call(np.ones((10, 100)), training=True)
        tf.random.set_seed(123)
        seeded_b = dropout.call(np.ones((10, 100)), training=True)
        self.assertTrue(np.all(seeded_a == seeded_b))

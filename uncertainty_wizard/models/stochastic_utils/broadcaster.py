import abc
import logging
from typing import Any

import numpy as np
import tensorflow as tf


class Broadcaster(abc.ABC):
    """Abstract class to inject sampling-related logic"""

    def __init__(self, batch_size: int, verbose, steps, sample_size, **kwargs):
        self.batch_size = batch_size
        self.verbose = verbose
        self.steps = steps
        self.sample_size = sample_size

    @abc.abstractmethod
    def broadcast_inputs(self, x, **kwargs) -> Any:
        """Replicates every input in x `num_samples` times.

        Replication should happen in place. For example,
        inputs [a,b,c] with sample size 3 should lead to output
        [a,a,a,b,b,b,c,c,c] and not [a,b,c,a,b,c,a,b,c].

        The return type is arbitrary, but typically a `tf.data.Dataset`.
        It will be used as `inputs` to `self.predict`.
        """

    @abc.abstractmethod
    def predict(self, model: tf.keras.Model, inputs: Any) -> Any:
        """Returns predictions for the given inputs on the passed model"""

    @abc.abstractmethod
    def reshape_outputs(self, outputs: np.ndarray, **kwargs) -> Any:
        """Reshape predictions to be used by sampling based quantifiers.

        For the default sampling-based quantifiers shipped with uwiz, such as
        `uwiz.quantifiers.VariationRatio`, predictions are expected to have the shape
        (num_inputs, num_samples, ...).
        The outputs of `self.predict` typically have shape (num_inputs * num_samples, ...).
        It's this methods responsibility to bring the inputs to the right shape.
        """


class DefaultBroadcaster(Broadcaster):
    """Implements a Default Broadcaster, supporting the most typical usecases."""

    # docstr-coverage:inherited
    def predict(self, model: tf.keras.Model, inputs: Any) -> Any:
        if self.steps is None:
            steps = None
        else:
            steps = self.steps * self.sample_size
        return model.predict(inputs, verbose=self.verbose, steps=steps)

    # docstr-coverage:inherited
    def broadcast_inputs(self, x, **kwargs) -> tf.data.Dataset:
        if isinstance(x, tf.data.Dataset):
            logging.debug(
                "You passed a tf.data.Dataset to predict_quantified in a stochastic model"
                "using the default broadcaster."
                "tf.data.Datasets passed to this method must not be batched. We take care of the batching."
                "Please make sure that your dataset is not batched (we can not check that)."
            )
            x_as_ds = x
        elif isinstance(x, np.ndarray):
            x_as_ds = tf.data.Dataset.from_tensor_slices(x)
        else:
            raise ValueError(
                "At the moment, uwiz stochastic models support only (unbatched)"
                "numpy arrays and tf.data.Datasets as inputs. "
                "Please transform your input in one of these forms or inject a custom broadcaster."
            )

        # Repeat every input `sample_size` many times in-place
        num_samples_tensor = tf.reshape(tf.constant(self.sample_size), [1])

        @tf.function
        @tf.autograph.experimental.do_not_convert
        def _expand_to_sample_size(inp):
            shape = tf.concat((num_samples_tensor, tf.shape(inp)), axis=0)
            return tf.broadcast_to(input=inp, shape=shape)

        inputs = x_as_ds.map(_expand_to_sample_size).unbatch()

        # Batch the resulting dataset
        return inputs.batch(batch_size=self.batch_size)

    # docstr-coverage:inherited
    def reshape_outputs(self, outputs: np.ndarray, **kwargs) -> np.ndarray:
        output_shape = list(outputs.shape)
        output_shape.insert(0, -1)
        output_shape[1] = self.sample_size
        return outputs.reshape(output_shape)

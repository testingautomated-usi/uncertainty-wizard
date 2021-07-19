import abc
import logging
from typing import Any

import numpy as np
import tensorflow as tf


class Broadcaster(abc.ABC):

    def __init__(self,
                 batch_size: int,
                 verbose,
                 steps,
                 sample_size,
                 **kwargs):
        self.batch_size = batch_size
        self.verbose = verbose
        self.steps = steps
        self.sample_size = sample_size

    @abc.abstractmethod
    def broadcast_inputs(self,
                         x,
                         **kwargs) -> Any:
        pass

    @abc.abstractmethod
    def predict(self,
                model: tf.keras.Model,
                inputs: Any) -> Any:
        pass

    @abc.abstractmethod
    def reshape_outputs(self,
                        outputs: np.ndarray,
                        **kwargs) -> Any:
        pass


class DefaultBroadcaster(Broadcaster):

    def predict(self, model: tf.keras.Model, inputs: Any) -> Any:
        if self.steps is None:
            steps = None
        else:
            steps = self.steps * self.sample_size
        return model.predict(inputs,
                             verbose=self.verbose,
                             steps=steps)

    def broadcast_inputs(self, x, **kwargs) -> tf.data.Dataset:
        if isinstance(x, tf.data.Dataset):
            logging.debug(
                "You passed a tf.data.Dataset to predict_quantified in a stochastic model."
                "tf.data.Datasets passed to this method must not be batched. We take care of the batching."
                "Please make sure that your dataset is not batched (we can not check that)"
            )
            x_as_ds = x
        elif isinstance(x, np.ndarray):
            x_as_ds = tf.data.Dataset.from_tensor_slices(x)
        else:
            raise ValueError(
                "At the moment, uwiz stochastic models support only (unbatched)"
                "numpy arrays and tf.data.Datasets as inputs. "
                "Please transform your input in one of these forms."
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

    def reshape_outputs(self, outputs: np.ndarray, **kwargs) -> np.ndarray:
        output_shape = list(outputs.shape)
        output_shape.insert(0, -1)
        output_shape[1] = self.sample_size
        return outputs.reshape(output_shape)

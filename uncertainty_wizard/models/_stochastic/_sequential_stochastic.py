import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from uncertainty_wizard.models._stochastic._abstract_stochastic import Stochastic
from uncertainty_wizard.models._stochastic._stochastic_mode import StochasticMode


class StochasticSequential(Stochastic):
    """
    A stochastic wrapper of `tf.keras.models.Sequential` models, suitable for MC Dropout
    and similar sampling based approaches on randomized models.
    """

    # Include the superclass documentation
    __doc__ += Stochastic.__doc__

    # docstr-coverage:inherited
    def __init__(self, layers=None, name=None):
        super().__init__()
        # Create an empty sequential model with a stochastic mode
        self._inner_sequential = tf.keras.models.Sequential(name=name)
        self._inner_sequential._stochastic_mode_tensor = StochasticMode().as_tensor()
        # Append all the passed layers
        if layers is not None:
            for layer in layers:
                self.add(layer)

    @classmethod
    def _wrap(cls, inner, stochastic_mode_tensor=None):
        model = StochasticSequential()
        model._inner_sequential = inner
        if stochastic_mode_tensor is None:
            assert model._inner_sequential._stochastic_mode_tensor is not None, (
                "Uncertainty Wizard internal error. "
                "Trying to wrap a model that has no stochastic_mode_tensor, "
                "and no external stochastic_mode_tensor is passed to attach"
            )
        else:
            model._inner_sequential._stochastic_mode_tensor = stochastic_mode_tensor
        return model

    # docstr-coverage:inherited
    @property
    def inner(self):
        return self._inner_sequential

    # docstr-coverage:inherited
    @property
    def stochastic_mode_tensor(self):
        return self._inner_sequential._stochastic_mode_tensor

    def add(self, layer, prevent_use_for_sampling=False):
        """
        Adds the layer to the model. See docs of `tf.keras.model.Sequential.add(layer)` for details.

        In addition, layers of type
        `tf.keras.layers.Dropout`,
        `tf.keras.layers.GaussianNoise` and
        `tf.keras.layers.GaussianDropout`
        are overridden by equivalent layers which allow to be enabled during inference for randomized predictions.

        Arguments:
            layer: layer instance to be added to the model.
            prevent_use_for_sampling: Do not use the layer for randomization during inference. Has only effect on layers of type `Dropout`, `GaussianNoise` or `GaussianDropout`
        """
        # Add noise layer, if applicable
        if not prevent_use_for_sampling:
            layer = self._replace_layer_if_possible(
                layer, stochastic_mode=self._get_stochastic_mode()
            )
        # Add normal as defined by user
        self.inner.add(layer)

    def get_config(self):
        """
        Not supported
        :return: An empty config
        """
        logging.warning(
            """ 
            It looks like you are trying to serialize a StochasticSequential model.
            Please note that to save an StochasticSequential model, you have to call `model.save(...)`
            and to load it, you have to use `StochasticSequential.load_model(...)`
            """
        )
        return []

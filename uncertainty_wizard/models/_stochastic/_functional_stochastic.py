import tensorflow as tf

from uncertainty_wizard.models._stochastic._abstract_stochastic import Stochastic
from uncertainty_wizard.models._stochastic._stochastic_mode import StochasticMode


class StochasticFunctional(Stochastic):
    """
    A stochastic wrapper of a `tf.keras.Model`, allowing to build models using the functional interface.
    Note that when using the functional interface, you need to use `uwiz.models.stochastic.layers`
    or build your own Stochastic-Mode dependent stochastic layers. See the online user guide for more info.

    """

    # Include the superclass documentation
    __doc__ += Stochastic.__doc__

    def __init__(
        self, inputs, outputs, stochastic_mode: StochasticMode, name: str = None
    ):
        """
        Create a new functional model, equivalent to calling tf.keras.Model(...).

        In addition, a stochastic mode instance has to be passed.
        The same instance also has to be passed to any randomized uwiz.layers instances
        which are part of this model.
        This allows to dynamically enable and disable randomness in the predictions.
        :param inputs: See the corresponding tf.keras.Model(...) docs
        :param outputs: See the corresponding tf.keras.Model(...) docs
        :param stochastic_mode: A stochastic mode instance
        :param name: See the corresponding tf.keras.Model(...) docs
        """
        super().__init__()
        self._inner_model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
        self._inner_model._stochastic_mode_tensor = stochastic_mode.as_tensor()

    # docstr-coverage:inherited
    @property
    def inner(self):
        return self._inner_model

    # docstr-coverage:inherited
    @property
    def stochastic_mode_tensor(self):
        return self._inner_model._stochastic_mode_tensor

    @classmethod
    def _wrap(cls, inner: tf.keras.Model, stochastic_mode_tensor=None):
        if stochastic_mode_tensor is None:
            assert inner._stochastic_mode_tensor is not None, (
                "Uncertainty Wizard internal error. "
                "Trying to wrap a model that has no stochastic_mode_tensor, "
                "and no external stochastic_mode_tensor is passed to attach"
            )
            stochastic_mode_tensor = inner._stochastic_mode_tensor
        stochastic_mode = StochasticMode(stochastic_mode_tensor)
        return StochasticFunctional(
            inner.inputs, inner.outputs, stochastic_mode=stochastic_mode
        )

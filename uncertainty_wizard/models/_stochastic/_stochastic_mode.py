import tensorflow as tf

batch_size = 128
num_classes = 10


class StochasticMode:
    """
    Stochastic mode is a wrapper for a bool tensor which serves as flag during inference in an uwiz stochastic model:
    If the flag is True, the inference in randomized. Otherwise, randomization is disabled.

    When creating a StochasticFunctional model, you need to create a new StochasticMode(),
    use it for any of your (custom?) layers that should have a behavior in a stochastic environment
    than in a detererministic environment (for example your own randomization layer).
    """

    def __init__(self, tensor=None):
        """
        Create a new stochastic mode. If not provided, a new flag tensor will be created.
        :param tensor: Pass your own boolean tensorflow Variable. Use is not recommended.
        """
        if tensor is not None:
            self._enabled = tensor
        else:
            self._enabled = tf.Variable(
                initial_value=False, trainable=False, dtype=bool
            )

    def as_tensor(self):
        """
        Get the tensor wrapped by this stochastic mode
        :return: A boolean tensor
        """
        return self._enabled

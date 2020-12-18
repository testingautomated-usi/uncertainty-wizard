import shutil

import tensorflow as tf

from uncertainty_wizard.models._stochastic._stochastic_mode import StochasticMode

from ._abstract_stochastic import Stochastic
from ._functional_stochastic import StochasticFunctional
from ._sequential_stochastic import StochasticSequential


def stochastic_from_keras(
    model: tf.keras.models.Model,
    input_tensors=None,
    clone_function=None,
    expect_determinism=False,
    temp_weights_path="tmp/weights",
):
    """
    Creates a stochastic instance from a given `tf.keras.models.Sequential` model:
    The new model will have the same structure (layers) and weights as the passed model.

    All stochastic layers (e.g. tf.keras.layers.Dropout) will be used for randomization during randomized predictions.
    If no stochastic layers are present, a ValueError is thrown.
    The raising of the error can be suppressed by setting `expect_determinism` to true.

    :param model: The model to copy. Remains unchanged.
    :param input_tensors: Optional tensors to use as input_tensors for new model. See the corresponding parameter in `tf.keras.models.clone_model` for details.
    :param _clone_function: Optional function to use to clone layers. Will be applied to all layers except input layers and stochastic layers. See the corresponding parameter in `tf.keras.models.clone_model` for more details.
    :param expect_determinism: If True, deterministic models (e.g. models without stochastic layers) are accepted and no ValueError is thrown.
    :param temp_weights_path: The model weights are temporarily saved to the disk at this path. Folder is deleted after successful completion.
    :return: A newly created stochastic model
    """
    # _clone_function is some layer cloning behavior that can be specified by the user
    # If none is specified, we use keras default (see `tf.keras.models.clone_model`)
    if clone_function is None:

        def _clone_function(layer):
            return layer.__class__.from_config(layer.get_config())

    # We wrap the users (or default) clone function in a clone function
    #   that replaces stochastic layers with uncertainty wizard stochastic layers
    stochastic_mode = StochasticMode()
    is_stochastic_layer = []

    def _uncertainty_wizard_aware_clone_function(layer):
        new_layer = Stochastic._replace_layer_if_possible(
            layer, stochastic_mode=stochastic_mode
        )
        if new_layer == layer:
            # Layer was not mapped to an uncertainty wizard layer, thus the default clone function is applied
            new_layer = _clone_function(layer)
            is_stochastic_layer.append(False)
        else:
            is_stochastic_layer.append(True)
        return new_layer

    # Clone the keras model to become the new inner model
    new_inner = tf.keras.models.clone_model(
        model=model,
        input_tensors=input_tensors,
        clone_function=_uncertainty_wizard_aware_clone_function,
    )
    new_inner.stochastic_mode_tensor = stochastic_mode.as_tensor()

    if not expect_determinism and not any(is_stochastic_layer):
        raise ValueError(
            "The passed model had no stochastic layers."
            "If that is intended (and you do not plan to use any sampling based quantifiers)"
            "you can set the flag `expect_determinism = True`, i.e., "
            "calling `SequentialStochastic.clone_from_keras(keras_model,expect_determinism = True)`"
        )

    # Restore the Weights
    model.save_weights(temp_weights_path)
    new_inner.load_weights(temp_weights_path)
    # Remove temporarily stored weights
    shutil.rmtree(temp_weights_path, ignore_errors=True)

    # Put the wrapper around the new model
    if isinstance(model, tf.keras.models.Sequential):
        target_class = StochasticSequential
    else:
        target_class = StochasticFunctional

    # Consenting Adults: The _wrap method is intended to be used here
    #   but declared private as it is not intended to be used by the uwiz user
    return target_class._wrap(
        inner=new_inner, stochastic_mode_tensor=stochastic_mode.as_tensor()
    )

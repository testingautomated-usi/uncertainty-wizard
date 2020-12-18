import json
import logging
import os
import warnings

import tensorflow as tf

import uncertainty_wizard as uwiz
from uncertainty_wizard.internal_utils import (
    UncertaintyWizardWarning,
    tf_version_resolver,
)

from ._stochastic._functional_stochastic import StochasticFunctional
from ._stochastic._sequential_stochastic import StochasticSequential


def load_model(path, custom_objects: dict = None, compile=None, options=None):
    """
    Loads an uncertainty wizard model that was saved using model.save(...).
    See the documentation of `tf.keras.models.load_model` for further information about the method params.

    For lazy ensembles: As they are lazy, only the folder path and the number of models are interpreted
    by this model loading - no keras models are actually loaded yet.
    Thus, custom_objects, compile and options must not be specified.

    :param path: The path of the folder where the ensemble was saved.
    :param custom_objects: Dict containing methods for custom deserialization of objects.
    :param compile: Whether to compile the models.
    :param options: Load options, check tf.keras documentation for precise information.

    :return: An uwiz model.
    """
    # Note: a path without file extension is not necessarily an ensemble in a folder
    # It could be a user who stored a stochastic model with default (non specified) file ending
    # Thus, we have to check if the folder exists and contains an ensemble config file
    ensemble_config_path = uwiz.models.ensemble_utils._lazy_ensemble.config_file_path(
        path
    )
    if (
        os.path.isdir(path)
        and os.path.exists(path)
        and os.path.exists(ensemble_config_path)
    ):
        return _load_ensemble(
            path=path, custom_objects=custom_objects, compile=compile, options=options
        )
    else:
        return _load_stochastic(
            path=path, custom_objects=custom_objects, compile=compile, options=options
        )


def _load_stochastic(path, custom_objects: dict = None, compile=None, options=None):
    """Attempts to load the model at the provided path as a stochastic model"""

    # Note: We currently intentionally don't define stochastic layers as custom_objects
    # as they have no methods other than call that we rely on, and thus the (robust and easy to maintain)
    # tf deserialization is sufficient
    if tf_version_resolver.current_tf_version_is_older_than("2.3.0", fallback=True):
        if options is not None:
            raise ValueError(
                "Load-Options are not supported by tensorflow<2.3.0."
                "Please do not specify any options when you call 'uwiz.models.load_model'"
                "or upgrade to a tensorflow version >= 2.3.0"
            )
        inner = tf.keras.models.load_model(
            path, custom_objects=custom_objects, compile=compile
        )
    else:
        inner = tf.keras.models.load_model(
            path, custom_objects=custom_objects, compile=compile, options=options
        )
    assert hasattr(
        inner, "_stochastic_mode_tensor"
    ), "Looks like the model which is being deserialized is not an uwiz stochastic model"

    if isinstance(inner, tf.keras.models.Sequential):
        return StochasticSequential._wrap(inner)
    else:
        return StochasticFunctional._wrap(inner)


def _load_ensemble(path, custom_objects: dict = None, compile=None, options=None):
    """Creates a lazy ensemble with the provided path as root dir. No models are acutally loaded yet (as in 'lazy')."""

    if compile is not None or options is not None or custom_objects is not None:
        warnings.warn(
            "Parameters compile, custom_objects and options are still ignored in lazy ensembles."
            "Support may be added in the future.",
            UncertaintyWizardWarning,
        )

    with open(
        uwiz.models.ensemble_utils._lazy_ensemble.config_file_path(path=path), "r"
    ) as f:
        config = json.load(f)

    num_models = config["num_models"]
    ensemble = uwiz.models.LazyEnsemble(
        num_models=num_models,
        model_save_path=path,
        expect_model=True,
        delete_existing=False,
    )
    logging.info(
        "Loaded ensemble. You may want to override the default_num_processes 'model.default_num_processes'"
    )
    return ensemble

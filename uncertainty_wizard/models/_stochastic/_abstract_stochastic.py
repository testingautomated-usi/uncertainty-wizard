import abc
import logging
import warnings
from typing import Iterable, Union

import numpy as np
import tensorflow as tf

from uncertainty_wizard.internal_utils import UncertaintyWizardWarning
from uncertainty_wizard.models._stochastic._stochastic_mode import StochasticMode
from uncertainty_wizard.models._uwiz_model import _UwizModel
from uncertainty_wizard.models.stochastic_utils import layers
from uncertainty_wizard.models.stochastic_utils.layers import (
    UwizBernoulliDropout,
    UwizGaussianDropout,
    UwizGaussianNoise,
)
from uncertainty_wizard.quantifiers import Quantifier


class Stochastic(_UwizModel):
    """
    Stochastic models are models in which some randomness is added to the network during training.
    While this is typically done for network regularization, models trained in such a way can be used for
    uncertainty quantification. Simply speaking:

    Randomness (which is typically disabled during inference) can be enforced during inference,
    leading to predictions which are impacted by the random noise.
    By sampling multiple network outputs for the same input, we can infer the robustness of the network to the
    random noise. We assume that the higher the robustness, the higher the networks confidence.

    Instances of stochastic uncertainty wizard models can also be used in a non-stochastic way
    as as point prediction models (i.e., models without sampling)
    by calling the `model.predict` function
    or by passing a quantifier which does not rely on sampling to `model.predict_quantified` (such as Max-Softmax).
    Randomization during model inference is automatically enabled or disabled.
    """

    def __init__(self):
        """ABSTRACT METHOD. DO NOT CALL INIT DIRECTLY"""

    @property
    @abc.abstractmethod
    def inner(self) -> tf.keras.Model:
        """
        Direct access to the wrapped keras model.
        Use this if you want to directly work on the wrapped model.
        When using this, make sure not to modify the stochastic layers or the stochastic_mode tensor on the model.

        Returns: the tf.keras.Model wrapped in this StochasticSequential.
        """
        pass  # pragma: no cover

    @property
    @abc.abstractmethod
    def stochastic_mode_tensor(self) -> tf.Variable:
        """
        Get access to the flag used to enable and disable the stochastic behavior.

        Returns: A boolean 0-dimensions tensorflow variable.

        """
        pass  # pragma: no cover

    def call(self, inputs, training=None, mask=None):
        """
        See tf.keras.Model.call for the documentation: The call is forwarded
        :param inputs: See tf.keras docs
        :param training: See tf.keras docs
        :param mask: See tf.keras docs
        :return: See tf.keras docs
        """
        return self.inner.call(inputs, training, mask)

    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        expect_deterministic: bool = False,
    ):
        """
        This wraps the tf.keras.Model.compile method, but checks before if a stochastic layer was added to the model:
        If none was added, a warning is printed.

        This behavior can be turned of if you only intend to use the model as point predictor.
        In this case, set expect_deterministic to True.

        :param optimizer: See tf.keras.Model docs
        :param loss: See tf.keras.Model docs
        :param metrics: See tf.keras.Model docs
        :param loss_weights: See tf.keras.Model docs
        :param weighted_metrics: See tf.keras.Model docs
        :param run_eagerly: See tf.keras.Model docs
        :param expect_deterministic: Iff true, the model is not checked for randomness. Default: False
        :return: See tf.keras.Model docs
        """
        if not expect_deterministic:
            # Check is randomized
            is_stochastic = False
            for layer in self.inner.layers:
                if (
                    isinstance(layer, layers.UwizGaussianNoise)
                    or isinstance(layer, layers.UwizBernoulliDropout)
                    or isinstance(layer, layers.UwizGaussianDropout)
                ):
                    is_stochastic = True
                    break

            if not is_stochastic:
                warnings.warn(
                    "Looks like your model contains no uwiz random layers."
                    "If you added your own randomized layers, "
                    "or if you only want to use this with non-randomized point predictors, "
                    "you can savely ignore this warning."
                    "Otherwise, check the docs on uwiz stochastic models for more info.\n\n"
                    "Use compile(..., expect_deterministic=True) to disable this warning",
                    UncertaintyWizardWarning,
                )
        return self.inner.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
        )

    @property
    def fit(self):
        """
        Direct access to the fit method the wrapped keras model.
        See `tf.keras.Model.fit` for precise documentation of this method.

        Can be called as `stochastic_model.fit(...)`, equivalent to how fit would be called on a plain keras model.
        :return: The fit method of the wrapped model
        """
        return self.inner.fit

    @property
    def evaluate(self):
        """
        Direct access to the evaluate method the wrapped keras model.
        See `tf.keras.Model.evaluate` for precise documentation of this method.

        Can be called as `stochastic_model.evaluate(...)`, equivalent to how fit would be called on a plain keras model.
        This means that no stochastic sampling is done.

        :return: The evaluate method of the wrapped model
        """
        return self.inner.evaluate

    @property
    def predict(self):
        """
        Direct access to the predict method the wrapped keras model.
        See `tf.keras.Model.predict` for precise documentation of this method.

        Note that no confidences are calculated if calling this predict method, and the stochastic layers are disabled.
        To calculate confidences, call `model.predict_quantified(...)` instead of `model.predict(...)`

        Can be called as `model.predict(...)`,
        equivalent to how predict would be called on a plain keras model.
        :return: The predict method of the wrapped model
        """
        return self.inner.predict

    @property
    def summary(self):
        """
        Direct access to the summary method the wrapped keras model.
        See `tf.keras.Model.summary` for precise documentation of this method.
        """
        return self.inner.summary

    def save(
        self,
        filepath: str,
        overwrite: bool = True,
        include_optimizer: bool = True,
        save_format: str = None,
        signatures=None,
        options=None,
    ):
        """
        Save the model to file, as on plain tf models. Note that you must not use the h5 file format.

        ** Attention ** uwiz models must be loaded using `uwiz.models.load_model` AND NOT using the corresponding
        keras method.

        See below the keras documentation, with applies for this method as well
        (taking in account the limitations mentioned above)

        """
        # Append the keras documentation
        Stochastic.save.__doc__ += tf.keras.Model.save.__doc__
        assert (
            not filepath.lower().endswith("h5")
            and not filepath.lower().endswith("hdf5")
            and not filepath.lower().endswith(".keras")
            and (save_format is None or not save_format.lower() == "h5")
        ), (
            "Uncertainty Wizard does not support the deprecated h5 format to save models."
            "Change the file ending or the save_format parameter to save using the better tf `SavedModel` format."
        )

        return self.inner.save(
            filepath=filepath,
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
        )

    def _get_stochastic_mode(self):
        """
        Get access to the stochastic mode used in the model to toggle randomness during predictions.
        :return: A stochastic mode instance wrapping the stochastic mode tensor used in this model.
        """
        return StochasticMode(self.stochastic_mode_tensor)

    @classmethod
    def _replace_layer_if_possible(
        cls, layer, stochastic_mode
    ) -> tf.keras.layers.Layer:
        if isinstance(layer, tf.keras.layers.Dropout):
            return UwizBernoulliDropout.from_keras_layer(
                layer=layer, stochastic_mode=stochastic_mode
            )
        elif isinstance(layer, tf.keras.layers.GaussianNoise):
            return UwizGaussianNoise.from_keras_layer(
                layer=layer, stochastic_mode=stochastic_mode
            )
        if isinstance(layer, tf.keras.layers.GaussianDropout):
            return UwizGaussianDropout.from_keras_layer(
                layer=layer, stochastic_mode=stochastic_mode
            )
        else:
            # The passed layer is not replaceable with a stochastic layer
            return layer

    def _get_scores(
        self, x, batch_size: int, verbose, steps, sample_size
    ) -> np.ndarray:
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
        num_samples_tensor = tf.reshape(tf.constant(sample_size), [1])

        @tf.function
        @tf.autograph.experimental.do_not_convert
        def _expand_to_sample_size(inp):
            shape = tf.concat((num_samples_tensor, tf.shape(inp)), axis=0)
            return tf.broadcast_to(input=inp, shape=shape)

        inputs = x_as_ds.map(_expand_to_sample_size).unbatch()

        # Batch the resulting dataset
        inputs = inputs.batch(batch_size=batch_size)

        # Make predictions.
        if steps is not None:
            steps = steps * sample_size
        outputs = self.inner.predict(inputs, verbose=verbose, steps=steps)

        # Reshape sampled predictions, grouping by sample (such that outputs[i] contains all samples for input i
        output_shape = list(outputs.shape)
        output_shape.insert(0, -1)
        output_shape[1] = sample_size
        return outputs.reshape(output_shape)

    def predict_quantified(
        self,
        x: Union[tf.data.Dataset, np.ndarray],
        quantifier: Union[Quantifier, str, Iterable[Union[str, Quantifier]]],
        # Other Sequential.predict params (e.g. Callbacks) are not yet supported
        sample_size: int = 64,
        batch_size: int = 32,
        verbose: int = 0,
        steps=None,
        as_confidence: Union[None, bool] = None,
    ):
        """
        Calculates predictions and uncertainties (or confidences) according to the passed quantifer(s).
        Sampling is done internally.
        Both point-predictor and sampling based quantifiers can be used in the same method call.
        Uwiz automatically enables and disables the randomness of the model accordingly.
        :param x: The inputs for which the predictions should be made. tf.data.Dataset (unbatched) or numpy array.
        :param quantifier: The quantifier or quantifier alias to use (or a collection of them)
        :param sample_size: The number of samples to be used for sample-expecting quantifiers
        :param batch_size: The batch size to be used for predictions
        :param verbose: Prediction process logging, as in tf.keras.Model.fit
        :param steps: Predictions steps, as in tf.keras.Model.fit. Is adapted according to chosen sample size.
        :param as_confidence: If true, uncertainties are multiplied by (-1),
        if false, confidences are multiplied by (-1). Default: No transformations.
        :return: A tuple (predictions, uncertainties_or_confidences) if a single quantifier was
        passed as string or instance, or a collection of such tuples if the passed quantifiers was an iterable.
        """
        all_q, pp_q, sample_q, return_single_tuple = self._quantifiers_as_list(
            quantifier
        )
        self._check_quantifier_heterogenity(
            as_confidence=as_confidence, quantifiers=all_q
        )

        self._warn_if_invalid_sample_size(
            sample_size, samples_based_quantifiers=sample_q
        )

        stochastic_scores, point_prediction_scores = None, None
        if len(sample_q) > 0:
            self.stochastic_mode_tensor.assign(True)
            stochastic_scores = self._get_scores(
                x, batch_size, verbose, steps, sample_size
            )
            self.stochastic_mode_tensor.assign(False)
        if len(pp_q) > 0:
            if isinstance(x, tf.data.Dataset):
                x = x.batch(batch_size=batch_size)
            point_prediction_scores = self.predict(
                x, batch_size=batch_size, verbose=verbose, steps=steps
            )

        results = self._run_quantifiers(
            as_confidence, point_prediction_scores, all_q, stochastic_scores
        )
        if return_single_tuple:
            return results[0]
        return results

    @staticmethod
    def _run_quantifiers(
        as_confidence, point_prediction_scores, quantifiers, stochastic_scores
    ):
        results = []
        for q in quantifiers:
            if q.takes_samples():
                assert stochastic_scores is not None, (
                    "Uncertainty Wizard internal error. "
                    "Did not compute stochastic scores"
                )
                scores = stochastic_scores
            else:
                assert point_prediction_scores is not None, (
                    "Uncertainty Wizard internal error. "
                    "Did not compute point prediction scores"
                )
                scores = point_prediction_scores
            predictions, superv_scores = q.calculate(scores)
            superv_scores = q.cast_conf_or_unc(
                as_confidence=as_confidence, superv_scores=superv_scores
            )
            results.append((predictions, superv_scores))
        return results

    @staticmethod
    def _warn_if_invalid_sample_size(sample_size, samples_based_quantifiers):
        if len(samples_based_quantifiers) > 0 and sample_size < 2:
            warnings.warn(
                f"The sample_size parameter must be greater than 1, but was {sample_size}",
                category=UncertaintyWizardWarning,
                # This shows the warning in the users code where he does the wrong call
                stacklevel=3,
            )

    @classmethod
    @abc.abstractmethod
    def _wrap(cls, inner, stochastic_mode_tensor=None):
        """
        Method to wrap a passed model.
        The model must already rely on a stochastic mode tensor which either has to be passed as a param
        or has to be already attached to the model as attribute.

        Args:
            inner (): The model to wrap
            stochastic_mode_tensor (): The stochastic mode to attach on the wrapped model, if not yet done

        Returns:
            An instance of the implementing subclass (i.e., a subclass of _AbstractStochastic)
        """
        # pragma: no cover

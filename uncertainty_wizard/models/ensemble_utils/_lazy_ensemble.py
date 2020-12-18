import json
import multiprocessing
import os
import tempfile
import warnings
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import tensorflow as tf

from uncertainty_wizard.models._uwiz_model import _UwizModel
from uncertainty_wizard.models.ensemble_utils import (
    DynamicGpuGrowthContextManager,
    EnsembleContextManager,
    NoneContextManager,
    SaveConfig,
)
from uncertainty_wizard.models.ensemble_utils._callables import (
    DataLoadedPredictor,
    NumpyFitProcess,
)
from uncertainty_wizard.models.ensemble_utils._save_config import _preprocess_path
from uncertainty_wizard.quantifiers import Quantifier

T = TypeVar("T")


def _model_creation_and_saving_process(
    model_id: int,
    save_config: SaveConfig,
    inner_func: Callable[[int], Tuple[tf.keras.Model, T]],
    context: Callable[[int, dict], EnsembleContextManager],
):
    # No varargs yet. This is a placeholder for future version extensions
    varargs = dict()
    # No custom methods called on ctxt yet. But this may change in future versions.
    with context(model_id, varargs) as ctxt:
        model, ret_container = inner_func(model_id)
        ctxt.save_single_model(model_id=model_id, model=model, save_config=save_config)
    return ret_container


def _model_updating_process(
    model_id: int,
    save_config: SaveConfig,
    inner_func: Callable[[int, tf.keras.Model], Tuple[tf.keras.Model, Any]],
    context: Callable[[int, dict], EnsembleContextManager],
):
    # No varargs yet. This is a placeholder for future version extensions
    varargs = dict()
    # No custom methods called on ctxt yet. But this may change in future versions.
    with context(model_id, varargs) as ctxt:
        model = ctxt.load_single_model(model_id, save_config=save_config)
        model, ret_container = inner_func(model_id, model)
        ctxt.save_single_model(model_id=model_id, model=model, save_config=save_config)
    return ret_container


def _model_consuming_process(
    model_id: int,
    save_config: SaveConfig,
    inner_func: Callable[[int, tf.keras.Model], Any],
    context: Callable[[int, dict], EnsembleContextManager],
):
    # No varargs yet. This is a placeholder for future version extensions
    varargs = dict()
    # No custom methods called on ctxt yet. But this may change in future versions.
    with context(model_id, varargs) as ctxt:
        model = ctxt.load_single_model(model_id, save_config=save_config)
        ret_container = inner_func(model_id, model)
    return ret_container


def config_file_path(path):
    """
    Constructs the path of the config file for ensemble saving.
    Args:
        path (): The path of the folder in which the ensemble is going to be saved or was saved.
    Returns: The path of the save_config file.
    """
    # This will be a repeated preprocessing in most cases,
    #   but we do it every time nonetheless, to avoid nasty bugs due to forgotten preprocessing

    path = _preprocess_path(path)

    return f"{path}/lazy_ensemble_config.pickle"


def _store_to_temp_file(x: np.ndarray):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as file:
        file_name = file.name
    np.save(file_name, x)
    return file_name


class LazyEnsemble(_UwizModel):
    """
    LazyEnsembles are uncertainty wizards implementation of Deep Ensembles,
    where multiple atomic models are trained for the same problem;
    the output distribution (and thus uncertainty) is then inferred from predicting on all atomic models.

    ** Multi-Processing **

    This ensemble implementation is lazy as it does not keep the atomic models in memory
    (or even worse, in the tf graph).
    Instead, atomic models are persisted on the file system and only loaded when needed -
    and discarded immediately afterwards.
    To further increase performance, in particular on high performance GPU powered hardware setups,
    where a single model instance training does not use the full GPU resources,
    LazyEnsemble allows to create multiple concurrent tensorflow sessions, each running a dedicated model in parallel.
    The number of processes to be used can be specified on essentially any LazyEnsemble function.


    Models are loaded into a context, e.g. a gpu configuration which was configured before the model was loaded.
    The default context, if multiple processes are used, sets the GPU usage to dynamic memory growth.
    Pay attention: By using too many processes, it is easy to exhaust your systems resources.
    Thus, we recommend to set the number of processes conservatively, observe the system load
    and increase the number of processes if possible.
    Default contexts are uwiz.models.ensemble_utils.DynamicGpuGrowthContextManager if multiprocessing is enables,
    and uwiz.models.ensemble_utils.NoneContextManager otherwise.

    Note: Multi-Processing can be disabled by setting the number of processes to 0.
    Then, predictions will be made in the main process on the main tensorflow session.
    *Attention*: In this case, the tensorflow session will be cleared after every model execution!

    ** The LazyEnsemble Interface & Workflow **

    LazyEnsemble exposes four central functions: create, modify and consume.
    In general, every of these functions expectes a picklable function at input
    which either creates, modifies or consumes a plain keras model.
    Please refer to the specific methods documentation for details.
    Furthermore LazyEnsemble exposes utility methods wrapping the above listed methods,
    e.g. fit and predict_quantified, which expect numpy array inputs and automatically
    serialize and deserialize them to be used in parallel processes.

    ** Stability of Lazy Ensembles **

    To optimize GPU use, LazyEnsemble relies on some of tensorflows experimental features and is thus,
    by extension, also to be considered experimental.

    """

    def __init__(
        self,
        num_models: int,
        model_save_path: str,
        delete_existing: bool = True,
        expect_model: bool = False,
        default_num_processes: int = 1,
    ):
        """
        Creates a new lazy model.
        There's not much done in this method, actually.
        Essentially, this opens and prepares the folder in which the atomic models will be stored.
        :param num_models: The number of models to be used.
        :param model_save_path: The path (folder) where the models should be stored.
        :param delete_existing: If True (default) the content of the specified folder will be cleared.
        :param expect_model: If false, delete existing is false and the specified folder is not empty,
            a warning will be printed.
        :param default_num_processes: The default number of processes to use. This can be overridden in later calls.
        """
        self.default_num_processes = default_num_processes
        self.num_models = num_models
        self.save_config = SaveConfig(
            ensemble_save_base_path=model_save_path,
            expect_model=expect_model,
            delete_existing=delete_existing,
        )
        with open(config_file_path(path=model_save_path), "w") as f:
            json.dump({"num_models": num_models}, f)

    def _num_processes_or_default(self, num_processes: Optional[int]):
        return self.default_num_processes if num_processes is None else num_processes

    def create(
        self,
        create_function: Callable[[int], Tuple[tf.keras.Model, T]],
        num_processes: int = None,
        context: Callable[[int, dict], EnsembleContextManager] = None,
    ) -> List[T]:
        """
        This function takes care of the creation of new atomic models for this ensemble instance.
        At its core stands a create_function:
        This custom function takes as input the id of the model to be generated (which may be ignored),
        and is expected to return the newly created keras model and some custom,
        picklable, creation report (e.g. the fit history).
        If not required, the returned report may be None.
        You should refrain from returning extremely large report object,
        as they will be kept in memory and may occupy too many system resources.

        :param create_function: A picklable function to create new atomic models,
            as explained in the description above.
        :param num_processes: The number of processes to use.
            Default: The default or value specified when creating the lazy ensemble.
        :param context: A contextmanager which prepares a newly crated process for execution
            (e.g. by configuring the gpus). See class docstring for explanation of default values.
        :return: The reports returned by the create_function executions.
        """
        return self._run_in_processes(
            process_creator=_model_creation_and_saving_process,
            inner_function=create_function,
            num_processes=num_processes,
            context=context,
        )

    def modify(
        self,
        map_function: Callable[[int, tf.keras.Model], Tuple[tf.keras.Model, T]],
        num_processes: int = None,
        context: Callable[[int], EnsembleContextManager] = None,
    ) -> List[T]:
        """
        This function takes care of modifications of previously generated atomic models for this ensemble instance.
        At its core stands a map_function:
        This custom function takes as input the id of the model to be modified (which may be ignored)
        and the model instance.
        It is expected to return the modified (or replaced) keras model and some custom,
        picklable, modification report (e.g. the fit history).
        If not required, the returned report may be None.
        You should refrain from returning extremely large report object,
        as they will be kept in memory and may occupy too many system resources.

        *Attention* Whenever possible, try to reduce the number of calls to this function.
        For example, it is often possible to train models as part of the 'create' call.
        This will result in the creation of less processes and thus a faster overall performance.

        :param map_function: A picklable function to modify atomic models,
            as explained in the description above.
        :param num_processes: The number of processes to use.
            Default: The default or value specified when creating the lazy ensemble.
        :param context: A contextmanager which prepares a newly crated process for execution
            (e.g. by configuring the gpus). See class docstring for explanation of default values.
        :return: The reports returned by the create_function executions.
        """
        return self._run_in_processes(
            process_creator=_model_updating_process,
            inner_function=map_function,
            num_processes=num_processes,
            context=context,
        )

    def consume(
        self,
        consume_function: Callable[[int, tf.keras.Model], T],
        num_processes: int = None,
        context: Callable[[int], EnsembleContextManager] = None,
    ) -> List[T]:
        """
        This function uses the atomic models in the ensemble without changing them.
        At its core stands a consume_function:
        This custom function takes as input the id of the model to be modified (which may be ignored)
        and the model instance.
        It is expected to return a picklable consumption result.
        You should refrain from returning extremely large consumption results,
        as they will be kept in memory and may occupy too many system resources.
        In such a case, you may want to persist the results and return None as a consumption result instead.

        *Attention* While this function can be used for predictions, you'd probably prefer to use
        ensemble.quantify_predictions(...) instead, which wraps this functions and allows to apply
        quantifiers for overall prediction inference and uncertainty quantification.

        :param consume_function: A picklable function to consume atomic models,
            as explained in the description above.
        :param num_processes: The number of processes to use.
            Default: The default or value specified when creating the lazy ensemble.
        :param context: A contextmanager which prepares a newly crated process for execution
            (e.g. by configuring the gpus). See class docstring for explanation of default values.
        :return: The reports returned by the create_function executions.
        """
        return self._run_in_processes(
            process_creator=_model_consuming_process,
            inner_function=consume_function,
            num_processes=num_processes,
            context=context,
        )

    def fit(
        self,
        x: np.ndarray = None,
        y: np.ndarray = None,
        batch_size: int = None,
        epochs: int = 1,
        verbose: int = 1,
        callbacks=None,
        validation_split: float = 0.0,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        shuffle: bool = True,
        class_weight: Dict[int, float] = None,
        sample_weight: np.ndarray = None,
        initial_epoch: int = 0,
        steps_per_epoch: int = None,
        validation_steps: int = None,
        validation_freq: int = 1,
        # Note: Max_queue_size, workers and use_multiprocessing not supported as we force input to be numpy array
        pickle_arrays=True,
        num_processes=None,
        context: Callable[[int], EnsembleContextManager] = None,
    ):
        """
        An easy access to keras fit function.
        As the inputs are pickled and distributed onto processes, only numpy array are accepted for the data params
        and no callbacks can be provided.

        If this is too restrictive for your use-case, consider using model.modify to setup your fit process
        and generate the datasets / callbacks right in the map_function.

        :param x: See `tf.keras.Model.fit` documentation.
        :param y: See `tf.keras.Model.fit` documentation.
        :param batch_size: See `tf.keras.Model.fit` documentation.
        :param epochs: See `tf.keras.Model.fit` documentation.
        :param verbose: See `tf.keras.Model.fit` documentation.
        :param callbacks: See `tf.keras.Model.fit` documentation.
        :param validation_split: See `tf.keras.Model.fit` documentation.
        :param validation_data: See `tf.keras.Model.fit` documentation.
        :param shuffle: See `tf.keras.Model.fit` documentation.
        :param class_weight: See `tf.keras.Model.fit` documentation.
        :param sample_weight: See `tf.keras.Model.fit` documentation.
        :param initial_epoch: See `tf.keras.Model.fit` documentation.
        :param steps_per_epoch: See `tf.keras.Model.fit` documentation.
        :param validation_steps: See `tf.keras.Model.fit` documentation.
        :param validation_freq: See `tf.keras.Model.fit` documentation.
        :param pickle_arrays: If true, the arrays are stored to the file system
            and deserialized in every child process to save memory.
        :param num_processes: The number of processes to use.
            Default: The default or value specified when creating the lazy ensemble.
        :param context: A contextmanager which prepares a newly crated process for execution
            (e.g. by configuring the gpus). See class docstring for explanation of default values.
        :return: The fit histories of the atomic models
        """
        if callbacks is not None and callbacks != []:
            raise ValueError(
                "Callbacks are currently not supported in the _lazy_ensemble fit function."
                "Use model.modify to setup your fit process with custom callbacks."
            )
        invalid_input = None
        if not isinstance(x, np.ndarray):
            invalid_input = "x_train"
        elif not isinstance(y, np.ndarray):
            invalid_input = "y_train"
        elif validation_data is not None and (
            validation_data[0] is None or not isinstance(validation_data[0], np.ndarray)
        ):
            invalid_input = "validation_data[0]"
        elif validation_data is not None and (
            validation_data[1] is None or not isinstance(validation_data[1], np.ndarray)
        ):
            invalid_input = "validation_data[1]"
        if invalid_input is not None:
            raise TypeError(
                f"{invalid_input} is not a numpy array. "
                "Lazy Ensembles fit function supports only numpy array, as other types of inputs "
                "may not be efficiently picklable. "
                "Use model.modify to setup your fit process if you want to use other input types"
                "(such as tf.data.Datasets) and generate the dataset in the custom process "
                "to avoid serialization & deserialization."
            )

        x_path = None
        y_path = None
        val_paths = None
        if pickle_arrays:
            # Temporarily save arrays to file system
            x_path = _store_to_temp_file(x)
            y_path = _store_to_temp_file(y)
            if validation_data is not None:
                val_x_path = _store_to_temp_file(validation_data[0])
                val_y_path = _store_to_temp_file(validation_data[1])
                val_paths = (val_x_path, val_y_path)

        inner_fit_function = NumpyFitProcess(
            x=x_path if pickle_arrays else x,
            y=y_path if pickle_arrays else y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split,
            validation_data=val_paths if pickle_arrays else validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_freq=validation_freq,
        )
        res = self.modify(
            map_function=inner_fit_function,
            num_processes=num_processes,
            context=context,
        )

        if pickle_arrays:
            # Temporarily save arrays to file system
            os.remove(x_path)
            os.remove(y_path)
            if validation_data is not None:
                os.remove(val_paths[0])
                os.remove(val_paths[1])

        return res

    def predict_quantified(
        self,
        x: np.ndarray,
        quantifier: Union[Quantifier, Iterable[Union[str, Quantifier]]],
        # Other Sequential.predict params (e.g. Callbacks) are not yet supported
        batch_size: int = 32,
        verbose: int = 0,
        steps=None,
        as_confidence: Union[None, bool] = None,
        num_processes=None,
        context=None,
    ):
        """
        Utility function to make quantified predictions on numpy arrays.
        Note: The numpy arrays are replicated on every created process and will thus quickly consume a lot of memory.

        :param x: An (unbatched) numpy array, to be used in tf.keras.Model.predict
        :param quantifier: A single or a collection of (sampling expecting) uwiz.quantifiers
        :param batch_size: The batch size to use in tf.keras.Model.predict
        :param verbose: Not yet supported.
        :param steps: The number of steps to use in tf.keras.Model.predict
        :param as_confidence: If true, uncertainties are multiplied by (-1), if false,
            confidences are multiplied by (-1). Default: No transformations.
        :param num_processes: The number of processes to use. Default:
            The default or value specified when creating the lazy ensemble.
        :param context: A contextmanager which prepares a newly crated process for execution
            (e.g. by configuring the gpus). See class docstring for explanation of default values.
        :return: A tuple (predictions, uncertainties_or_confidences) if a single quantifier was passed as string
            or instance, or a collection of such tuples if the passed quantifiers was an iterable.
        """
        if verbose > 0:
            warnings.warn("Verbosity not yet supported in lazy ensemble models.")
        data_loaded_predictor = DataLoadedPredictor(
            x_test=x, batch_size=batch_size, steps=steps
        )
        return self.quantify_predictions(
            quantifier=quantifier,
            consume_function=data_loaded_predictor,
            as_confidence=as_confidence,
            num_processes=num_processes,
            context=context,
        )

    def quantify_predictions(
        self,
        quantifier: Union[Quantifier, Iterable[Quantifier]],
        consume_function: Callable[[int, tf.keras.Model], Any],
        as_confidence: bool = None,
        num_processes: int = None,
        context: Callable[[int], EnsembleContextManager] = None,
    ):
        """
        A utility function to make predictions on all atomic models and then infer overall predictions and uncertainty
        (or confidence) on those predictions.

        The test data is expected to be loaded directly in the consume function.
        This function, which gets the atomic model id and the atomic model as inputs,
        is expected to return the predictions, i.e., the results of a model.predict(..) call.
        :param quantifier: A single or a collection of (sampling expecting) uwiz.quantifiers
        :param consume_function: A picklable function to make predictions on atomic models, as explained in the description above.
        :param as_confidence: If true, uncertainties are multiplied by (-1), if false, confidences are multiplied by (-1). Default: No transformations.
        :param num_processes: The number of processes to use. Default: The default or value specified when creating the lazy ensemble.
        :param context: A contextmanager which prepares a newly crated process for execution (e.g. by configuring the gpus). See class docstring for explanation of default values.
        :return: A tuple (predictions, uncertainties_or_confidences) if a single quantifier was passed as string or instance, or a collection of such tuples if the passed quantifiers was an iterable.
        """
        all_q, pp_q, sample_q, return_single_tuple = self._quantifiers_as_list(
            quantifier
        )
        self._check_quantifier_heterogenity(
            as_confidence=as_confidence, quantifiers=all_q
        )

        nn_outputs_by_model = self.consume(
            consume_function=consume_function,
            num_processes=num_processes,
            context=context,
        )

        scores = None
        for i, predictions in enumerate(nn_outputs_by_model):
            if scores is None:
                scores_shape = list(predictions.shape)
                scores_shape.insert(1, self.num_models)
                scores = np.empty(scores_shape)
            scores[:, i] = predictions

        results = []
        for q in all_q:
            predictions, superv_scores = q.calculate(scores)
            superv_scores = q.cast_conf_or_unc(
                as_confidence=as_confidence, superv_scores=superv_scores
            )
            results.append((predictions, superv_scores))
        if return_single_tuple:
            return results[0]
        return results

    def _run_in_processes(
        self,
        process_creator,
        inner_function: Callable[[int, tf.keras.Model], Any],
        num_processes: Optional[int],
        context: Optional[Callable[[int], EnsembleContextManager]],
    ):
        num_processes = self._num_processes_or_default(num_processes=num_processes)
        if num_processes > 0:
            if context is None:
                # As default, we just grow the memory dynamically
                context = DynamicGpuGrowthContextManager
            multiprocess_spawn_ctx = multiprocessing.get_context("spawn")
            context.before_start()
            tasks_per_child = context.max_sequential_tasks_per_process()
            pool = multiprocessing.pool.Pool(
                num_processes,
                maxtasksperchild=tasks_per_child,
                context=multiprocess_spawn_ctx,
            )
            partial_process_creator = partial(
                process_creator,
                save_config=self.save_config,
                inner_func=inner_function,
                context=context,
            )
            results = pool.map(partial_process_creator, range(self.num_models))
            pool.close()
            pool.join()
            context.after_end()
            return results
        else:
            if context is None:
                context = NoneContextManager
            res = []
            context.before_start()
            for i in range(self.num_models):
                print(
                    f"Working on model {i} (of {self.num_models}) in the main process"
                )
                model_res = process_creator(
                    model_id=i,
                    save_config=self.save_config,
                    inner_func=inner_function,
                    context=context,
                )
                res.append(model_res)
            context.after_end()
            return res

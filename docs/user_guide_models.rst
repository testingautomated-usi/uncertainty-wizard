User Guide: Models
####################

Uncertainty wizard supports three types of models capable to calculate uncertainties and confidences:

* | :ref:`user_guide_stochastic_models`
  | ``Stochastic models`` use randomness during predictions (typically using dropout layers).
    Then, for a given input, multiple predictions are made (-> sampling).
    The final predictions und uncertainties are a function of the distribution of neural network outputs distributions
    over all the observed samples.
* | :ref:`user_guide_ensemble_models`
  | ``Ensemble models`` are collections of models trained to answer the same problem,
    but with slightly different weights due to different kernel initialization or slightly different training data.
    Then, for a given input, a prediction is made on each model in the collection (-> sampling).
    The final predictions und uncertainties are a function of the distribution of neural network outputs
    over all the models in the ensemble.
* | :ref:`user_guide_point_predictor_models` [for classification problems only]
  | We call models which base their prediction and uncertainty quantification based on a single
    inference on a single, traditional (i.e., non-bayesian) neural network a ``Point-Predictor models``.
    Point predictor based uncertainty quantification can typically only be applied to classification problems,
    where the uncertainty quantification is based on the distribution of values in the softmax output layer.

See our papers and the references therein for a more detailed
for a detailed information about the here described techniques.


.. _user_guide_stochastic_models:

Stochastic Models (e.g. MC-Dropout)
***************************************************
Stochastic models are models in which some randomness is added to the network during training.
While this is typically done for network regularization, models trained in such a way can be used for
uncertainty quantification. Simply speaking:

Randomness (which is typically disabled during inference) can be enforced during inference,
leading to predictions which are impacted by the random noise.
By sampling multiple network outputs for the same input, we can infer the robustness of the network to the
random noise. We assume that the higher the robustness, the higher the networks confidence.

Uwiz stochastic models wrap a keras model, and inject a mechanism to automatically control the randomization
during inference.

**TL;DR? Get started with two short snippets**

.. code-block:: python
   :caption: Stochastic API: The simplest way to stochastic models

    model = uwiz.models.StochasticSequential()

    # The following lines are equivalent to a keras Sequential model
    model.add(tf.keras.layers.Dense(100)
    # Dropout and noise layers will be used to randomize predictions
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Softmax(10))
    model.compile(..)
    model.fit(..)

    # Make predictions, and calculate the variation ratio as uncertainty metric
    # (where x_test are the inputs for which you want to predict...)
    pred, unc = model.predict_quantified(x_test, quantifier='var_ratio', num_samples=32)


.. code-block:: python
   :caption: Functional API: Full control over randomness

   model = uwiz.models.StochasticFunctional()

   # We create an object that will serve as a flag during inference
   # indicating on whether randomization should be enabled
   stochastic_mode = uwiz.models.stochastic.StochasticMode()

   # We construct input and output as for classical tensorflow models
   # Note that layers for prediction randomization have to be specified explicitly
   input_l = tf.keras.layers.Input(100)
   x = tf.keras.layers.Dense(100)(input_l)
   x = uwiz.models.stochastic.layers.UwizBernoulliDropout(0.3, stochastic_mode=stochastic_mode)(x)
   output_l = tf.keras.layers.Softmax(10)(x)
   model = uwiz.models.FunctionalStochastic(input_l, output_l, stochastic_mode=stochastic_mode)
   model.compile(..)
   model.fit(..)

   # Make predictions and calculate uncertainty (as shown in example above)
   pred, unc = model.predict_quantified(x_test, quantifier='var_ratio', num_samples=32)


.. _user_guide_ensemble_models:

Ensemble Models
***************************************************
LazyEnsembles are uncertainty wizards implementation of Deep Ensembles,
where multiple atomic models are trained for the same problem;
the output distribution (and thus uncertainty) is then inferred from predicting on all atomic models.

**Multi-Processing**

This ensemble implementation is lazy as it does not keep the atomic models in memory
(or even worse, in the tf graph).
Instead, atomic models are persisted on the file system and only loaded when needed -
and cleared from memory immediately afterwards.
To further increase performance, in particular on high performance GPU powered hardware setups,
where a single model instance training does not use the full GPU resources,
LazyEnsemble allows to create multiple concurrent tensorflow sessions, each running a dedicated model in parallel.
The number of processes to be used can be specified on essentially all of LazyEnsembles methods.

Models are loaded into a context, e.g. a gpu configuration which was configured before the model was loaded.
The default context, if multiple processes are used, sets the GPU usage to dynamic memory growth.
We recommend to set the number of processes conservatively, observe the system load
and increase the number of processes if possible.

If you use tensorflow in your main process, chances are the main thread allocates all available GPU resources.
In such case you may for example want to enabling dynamic growth on the main thread,
which can be done by calling the following utility method right after first importing tensorflow:
``uwiz.models.ensemble_utils.DynamicGpuGrowthContextManager.enable_dynamic_gpu_growth()``

.. warning::
   By using too many processes you will quickly exhaust your systems resources.
   Similarly, if you do not have a GPU: Your CPU will not be able to handle the high workload of training multiple
   models in parallel.

Multi-Processing can be disabled by setting the number of processes to 0.
Then, predictions will be made in the main process on the main tensorflow session.
*Attention*: In this case, the tensorflow session will be cleared after every model execution!

**The LazyEnsemble Interface & Workflow**

LazyEnsemble exposes five central functions:
``create``
``modify``
``consume``
``quantify_predictions``
``run_model_free``
create, modify, consume, quantify_predictions or run_model_free.
In general, every of these functions expects a picklable function as input
which either creates, modifies or consumes a plain keras model, or uses it to make predictions.
Please refer to the
`specific methods documentation <https://uncertainty-wizard.readthedocs.io/en/latest/source/uncertainty_wizard.models.html#uncertainty_wizard.models.LazyEnsemble>`_
and examples for details.

Furthermore LazyEnsemble exposes utility methods wrapping the above listed methods,
e.g. fit and predict_quantified, which expect numpy array inputs and automatically
serialize and deserialize them to be used in parallel processes.

.. note::
   The less often you call methods on your ensemble, the less often we have to deserialize and persist your models
   (which is some overhead). Thus, try reducing these calls for even faster processing:
   For example, you may want to fit your model as part of the ``ensemble.create`` call.

**Stability of Lazy Ensembles**

To optimize GPU use, LazyEnsemble relies on some of tensorflows features which are (as of August 2020) still
experimental. Thus, by extension, our ensembles are also to be considered experimental.


**TL;DR? Get started with one short snippet**

.. code-block:: python
   :caption: Stochastic API: The simplest way to ensemble models

   # Define how models should be trained. This function must be picklable.
   def model_creator(model_id: int):
      import tensorflow as tf
      model = tf.keras.models.Sequential()
      model.add(tf.keras.layers.Dense(100)
      model.add(tf.keras.layers.Dropout(0.3))
      model.add(tf.keras.layers.Softmax(10))
      model.compile(..)
      fit_history = model.fit(..)
      return model, fit_history.history

   # Define properties of the ensemble to be created
   uwiz.models.LazyEnsemble(num_models=2,
                            model_save_path="/tmp/demo_ensemble",
                            default_num_processes=5)

   # Create and train the inner models in your ensemble according to your process defined above
   ensemble.create(create_function=model_creator)

   # Now we are ready to make predictions
   pred, unc = model.predict_quantified(x_test,
                                        quantifier='var_ratio',
                                        # For the sake of this example, lets assume we want to
                                        # predict with a higher batch size and lower process number
                                        # than our default settings.
                                        batch_size=128,
                                        num_processes=2)





.. _user_guide_point_predictor_models:

Point Predictor Models
****************************************************
We call models which base their prediction and uncertainty quantification
based on a single inference on a single,
traditional (i.e., non-bayesian) neural network a Point-Predictor model.
In `uncertainty wizard`, we can use the stochastic model classes ``StochasticSequential``
and ``StochasticFunctional`` for such predictions as well.
To do so, create or re-use a stochastic model as explained above.
Of course, if we only want to do point predictions,
the stochastic model does not have to contain any stochastic layers
(i.e., it can be deterministic).
Stochastic layers (e.g. Dropout) which are included in the network
are automatically disabled when doing point predictions.

The following snippet provides three examples on how to do point predictions on a stochastic model instance `model`:

.. code-block:: python
   :caption: Using the stochastic model classes for (non-sampled) point predictions


   # Example 1: Plain Keras Prediction
   # If we just want to use the keras model output (as it there were no uncertainty_wizard)
   #   we can predict on the stochastic model as if it was a regular `tf.keras.Model`
   nn_outputs = model.predict(x_test)


   # Example 2: Point Prediction Confidence and Uncertainty Metrics
   # We can also get confidences and uncertainties using predict_quantified.
   #   For point-predictor quantifiers which don't rely on random sampling,
   #   such as the prediction confidence score (PCS), randomness is automatically disabled
   #   and the returned values are based on a one-shot prediction.
   pred, unc = model.predict_quantified(x_test, quantifier='pcs')


   # Example 2b: Doing Point-Prediction and Sampling Based Interence in one Call
   # We can even combine point-prediction based and sampling based quantifiers
   #   Randomization will only be used for the sampling based quantifiers
   res = model.predict_quantified(x_test, quantifier=['pcs', 'var_ratio'])
   #   If `quantifier` is a list, the returned res is also a list,
   #   containing a (prediction, uncertainty_or_confidence_score) tuple
   #   for every passed quantifier
   point_predictor_predictions = res[0][0]
   point_predictor_confidence_scores = res[0][1]
   sampling_based_predictions = res[1][0]
   sampling_based_var_ratio = res[1][1]



from unittest import TestCase

import numpy as np
import tensorflow as tf

import uncertainty_wizard as uwiz
from uncertainty_wizard.quantifiers import StandardDeviation

DUMMY_MODEL_PATH = "tmp/dummy_lazy_ensemble"


def create_dummy_atomic_model(model_id: int):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=1000))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model, None


def dummy_fit_process(model_id: int, model: tf.keras.Model):
    history = model.fit(
        x=np.ones((100, 1000)), y=np.ones((100, 1)), epochs=10, batch_size=100
    )
    return model, history.history


def dummy_predict_process(model_id: int, model: tf.keras.Model):
    return model.predict(np.ones((10, 1000)))


def dummy_independent_task(model_id: int):
    return model_id


# Note: So far we have mostly smoke tests.
class LazyEnsembleTest(TestCase):
    def test_dummy_in_distinct_process(self):
        ensemble = uwiz.models.LazyEnsemble(
            num_models=2, model_save_path=DUMMY_MODEL_PATH
        )
        ensemble.create(create_function=create_dummy_atomic_model)
        pred, std = ensemble.predict_quantified(
            x=np.ones((10, 1000)), quantifier="std", num_processes=1
        )
        self.assertEqual(pred.shape, (10, 1))
        self.assertEqual(std.shape, (10, 1))

    def test_dummy_in_main_process(self):
        ensemble = uwiz.models.LazyEnsemble(
            num_models=2, model_save_path=DUMMY_MODEL_PATH, default_num_processes=0
        )
        ensemble.create(create_function=create_dummy_atomic_model)
        pred, std = ensemble.predict_quantified(
            x=np.ones((10, 1000)), quantifier="std", num_processes=0
        )
        self.assertEqual(pred.shape, (10, 1))
        self.assertEqual(std.shape, (10, 1))

    def test_result_as_dict(self):
        ensemble = uwiz.models.LazyEnsemble(
            num_models=2, model_save_path=DUMMY_MODEL_PATH, default_num_processes=0
        )
        ensemble.create(create_function=create_dummy_atomic_model)
        res = ensemble.predict_quantified(
            x=np.ones((10, 1000)),
            quantifier="std",
            num_processes=0,
            return_alias_dict=True,
        )
        assert isinstance(res, dict)
        for alias in StandardDeviation().aliases():
            assert alias in res
            assert type(res[alias]) == tuple
            assert len(res[alias]) == 2
            assert res[alias][0].shape == (10, 1)
            assert res[alias][1].shape == (10, 1)

    def test_dummy_main_and_one_distinct_process_are_equivalent(self):
        ensemble = uwiz.models.LazyEnsemble(
            num_models=2, model_save_path=DUMMY_MODEL_PATH
        )
        ensemble.create(create_function=create_dummy_atomic_model)
        pred_p, std_p = ensemble.predict_quantified(
            x=np.ones((10, 1000)), quantifier="std", num_processes=1
        )

        pred_m, std_m = ensemble.predict_quantified(
            x=np.ones((10, 1000)), quantifier="std", num_processes=0
        )
        self.assertTrue(np.all(std_m == std_p))
        self.assertTrue(np.all(pred_m == pred_p))

    def test_dummy_main_and_two_distinct_processes_are_equivalent(self):
        ensemble = uwiz.models.LazyEnsemble(
            num_models=2, model_save_path=DUMMY_MODEL_PATH
        )
        ensemble.create(create_function=create_dummy_atomic_model)
        pred_p, std_p = ensemble.predict_quantified(
            x=np.ones((10, 1000)), quantifier="std", num_processes=2
        )

        pred_m, std_m = ensemble.predict_quantified(
            x=np.ones((10, 1000)), quantifier="std", num_processes=0
        )
        self.assertTrue(np.all(std_m == std_p))
        self.assertTrue(np.all(pred_m == pred_p))

    def test_save_and_load(self):
        ensemble = uwiz.models.LazyEnsemble(
            num_models=2, model_save_path=DUMMY_MODEL_PATH
        )
        ensemble.create(create_function=create_dummy_atomic_model)
        pred_p, std_p = ensemble.predict_quantified(
            x=np.ones((10, 1000)), quantifier="std", num_processes=0
        )
        # Saving not required, lazy ensembles are always saved on file
        del ensemble

        ensemble = uwiz.models.load_model(DUMMY_MODEL_PATH)
        pred_m, std_m = ensemble.predict_quantified(
            x=np.ones((10, 1000)), quantifier="std", num_processes=0
        )
        self.assertTrue(np.all(std_m == std_p))
        self.assertTrue(np.all(pred_m == pred_p))

    def smoke_full(self, num_processes):
        ensemble = uwiz.models.LazyEnsemble(
            num_models=2,
            model_save_path=DUMMY_MODEL_PATH,
            default_num_processes=num_processes,
        )
        ensemble.create(create_function=create_dummy_atomic_model)
        fit_history = ensemble.modify(map_function=dummy_fit_process)
        self.assertIsNotNone(fit_history)
        fit_history = ensemble.fit(
            x=np.ones((100, 1000)), y=np.ones((100, 1)), epochs=10, batch_size=100
        )
        self.assertIsNotNone(fit_history)

        lazy_pred_p, lazy_pred_std = ensemble.quantify_predictions(
            quantifier="std", consume_function=dummy_predict_process
        )
        np_based_pred_p, np_based_std = ensemble.predict_quantified(
            x=np.ones((10, 1000)), quantifier="std"
        )
        self.assertTrue(np.all(lazy_pred_p == np_based_pred_p))
        self.assertTrue(np.all(lazy_pred_std == np_based_std))

        returned_ids = ensemble.run_model_free(task=dummy_independent_task)
        self.assertTrue(returned_ids == [0, 1], f"returned was: {returned_ids}")

    def test_main_smoke_full(self):
        self.smoke_full(0)

    def test_main_smoke_distinct_process(self):
        self.smoke_full(1)

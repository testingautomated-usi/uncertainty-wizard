import gc
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
import tensorflow as tf


@dataclass
class DataLoadedPredictor:
    """
    The default task to be executed for predictions where the input data is a numpy array.
    Leaves the serialization and deserialization of the array to the python multiprocessing library,
    and does thus not explicitly implement it here.
    """

    x_test: np.ndarray
    batch_size: int
    steps: int = None

    def __call__(self, model_id: int, model: tf.keras.Model):
        """Simple call to keras predict, formulated as __call__ to allow for constructor params."""
        return model.predict(
            x=self.x_test, batch_size=self.batch_size, steps=self.steps, verbose=1
        )


@dataclass
class NumpyFitProcess:
    """
    This is a class used as callable for the serialization and deserialization of numpy arrays
    which are then used in the keras fit process.
    """

    x: Union[str, np.ndarray] = None
    y: Union[str, np.ndarray] = None
    batch_size: int = None
    epochs: int = 1
    verbose: int = 1
    # Callbacks not supported in this default process (as type does not guarantee picklability)
    # callbacks = None,
    validation_split: float = 0.0
    validation_data: Union[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = None
    shuffle: bool = True
    class_weight: Dict[int, float] = None
    sample_weight: np.ndarray = None
    initial_epoch: int = 0
    steps_per_epoch: int = None
    validation_steps: int = None
    validation_freq: int = 1

    # Max_queue_size, workers and use_multiprocessing not supported as we force input to be numpy array
    # max_queue_size = 10,
    # workers = 1,
    # use_multiprocessing = False

    def __call__(
        self, model_id: int, model: tf.keras.Model
    ) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """Simple call to keras fit, formulated as __call__ to allow for constructor params."""
        x = np.load(self.x, allow_pickle=True) if isinstance(self.x, str) else self.x
        y = np.load(self.y, allow_pickle=True) if isinstance(self.y, str) else self.y
        if self.validation_data is not None and isinstance(
            self.validation_data[0], str
        ):
            val_x = np.load(self.validation_data[0], allow_pickle=True)
            val_y = np.load(self.validation_data[1], allow_pickle=True)
            val_data = (val_x, val_y)
        else:
            val_data = self.validation_data
        history = model.fit(
            x=x,
            y=y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            validation_split=self.validation_split,
            validation_data=val_data,
            shuffle=self.shuffle,
            class_weight=self.class_weight,
            sample_weight=self.sample_weight,
            initial_epoch=self.initial_epoch,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_freq=self.validation_freq,
        )
        del x
        del y
        if val_data:
            del val_data
        gc.collect()
        return model, history.history

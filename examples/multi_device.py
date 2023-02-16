# ==============================
# It does not make sense to run this from colab (as only one GPU)
# Thus, this is not set up as a jupyter notebook official notebook
# ==============================
from typing import Dict

import tensorflow

import uncertainty_wizard as uwiz


class MultiGpuContext(uwiz.models.ensemble_utils.DeviceAllocatorContextManager):
    @classmethod
    def file_path(cls) -> str:
        return "temp-ensemble.txt"

    @classmethod
    def run_on_cpu(cls) -> bool:
        # Running on CPU is almost never a good idea.
        # The CPU should be available for data preprocessing.
        # Also, training on CPU is typically much slower than on a gpu.
        return False

    @classmethod
    def virtual_devices_per_gpu(cls) -> Dict[int, int]:
        # Here, we configure a setting with two gpus
        # On gpu 0, two atomic models will be executed at the same time
        # On gpu 1, three atomic models will be executed at the same time
        return {0: 2, 1: 3}

    @classmethod
    def gpu_memory_limit(cls) -> int:
        return 1500


def train_model(model_id):
    import tensorflow as tf

    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            input_shape=(32, 32, 3),
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), activation="relu", padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(
        tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), activation="relu", padding="same"
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), activation="relu", padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(
        tf.keras.layers.Conv2D(
            128, kernel_size=(3, 3), activation="relu", padding="same"
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            128, kernel_size=(3, 3), activation="relu", padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    # For the sake of this example, let's use just one epoch.
    # Of course, for higher accuracy, you should use more.
    model.fit(x_train, y_train, batch_size=32, epochs=1)

    return model, "history_not_returned"


if __name__ == "__main__":
    # Make sure the training data is cached on the fs before the multiprocessing starts
    # Otherwise, all processes will simultaneously attempt to download and cache data,
    # which will fail as they break each others caches
    tensorflow.keras.datasets.cifar10.load_data()

    # set this path to where you want to save the ensemble
    temp_dir = "/tmp/ensemble"
    ensemble = uwiz.models.LazyEnsemble(
        num_models=20,
        model_save_path=temp_dir,
        delete_existing=True,
        default_num_processes=5,
    )
    ensemble.create(train_model, context=MultiGpuContext)

    print("Ensemble was successfully trained")

import abc
import errno
import os
import pickle
import time
from typing import Dict

import tensorflow as tf

from uncertainty_wizard.models.ensemble_utils._save_config import SaveConfig

global number_of_tasks_in_this_process


def _init_num_tasks():
    global number_of_tasks_in_this_process
    number_of_tasks_in_this_process = 0


_init_num_tasks()


class EnsembleContextManager(abc.ABC):
    """
    An abstract superclass of context managers which can be used to instantiate a context
    on a newly created process.

    Note that subclasses may override the constuctor, but they must not add or remove any arguments from it.

    """

    def __init__(self, model_id: int, varargs: dict = None):
        """
        The constructor currently only receives the id of the atomic model for which
        it will have to generate a context.
        Later, to make it easier for custom child classes of EnsembleContextManager,
        a (now still empty) varargs is also passed which may be populated with more information
        in future versions of uncertainty_wizard.
        """
        self.ensemble_id = (model_id,)
        self.varargs = varargs

    def __enter__(self) -> "EnsembleContextManager":
        """
        Will be executed before session the model is executed.
        Must return 'self'
        :return: None
        """
        return self

    def __exit__(self, type, value, traceback) -> None:
        """
        Will be executed before session the model was executed. You can use this for clean up tasks.
        :return: None
        """
        global number_of_tasks_in_this_process
        number_of_tasks_in_this_process += 1

    @classmethod
    def max_sequential_tasks_per_process(cls) -> int:
        """
        This method is used to specify how often a process (i.e., for how many atomic models)
        every single process can be used before being replaced with a new spawn process.
        Extend this method and set the value to something low to prevent process pollution
        (e.g. if there's a memory leak in your process),
        but beware that process creation (which implies tf initialization) is a costly task.

        The method must be pure; every time it is called it should return the same value.

        The method is ignored if using `num_processes = 0`, i.e., when executing in the main process.

        *Default* If not overridden, this method returns `1000` which means that processes is
        infinitely reused in any reasonable ensemble.

        :return: A positive integer, specifying how often a process can be used before being replaced.
        """
        return 1000

    @classmethod
    def before_start(cls) -> None:
        """
        This method is called once whenever a context-aware call is made on an ensemble
        (e.g., create, modify, predict_quantified, ...). Hence, it is *not* called for every atomic model
        individually.

        Typically, this method would be used to setup requirements that the later created EnsembleContextManager
        instances are relying on.

        Default behavior if method not overridden: Nothing is done.
        """

    @classmethod
    def after_end(cls) -> None:
        """
        This method is called at the end of every context-aware call
        (e.g., create, modify, predict_quantified, ...). Hence, it is *not* called for every atomic model
        individually.

        Typically, this method would be used to setup requirements that the later created EnsembleContextManager
        instances are relying on.

        Default behavior if method not overridden: Nothing is done.
        """

    # Inspection disabled as overriding child classes may want to use 'self'
    # noinspection PyMethodMayBeStatic
    def save_single_model(
        self, model_id: int, model: tf.keras.Model, save_config: SaveConfig
    ) -> None:
        """
        This method will be called to store a single atomic model in the ensemble.
        :param model_id: The id of the atomic model.
        Is between 0 and the number of atomic model in the ensemble.
        :param model: The keras model to be saved.
        :param save_config: A save_config instance, providing information about the base path
        of the ensemble
        """
        path = os.path.abspath(save_config.filepath(model_id=model_id))
        model.save(filepath=path)

    # Inspection disabled as overriding child classes may want to use 'self'
    # noinspection PyMethodMayBeStatic
    def load_single_model(
        self, model_id: int, save_config: SaveConfig
    ) -> tf.keras.Model:
        """
        This method will be called to load a single atomic model in the ensemble.
        :param model_id: The id of the atomic model.
        Is between 0 and the number of atomic model in the ensemble.
        :param save_config: A save_config instance, providing information about the base path
        of the ensemble
        :return The loaded keras model.
        """
        tf.keras.backend.clear_session()
        return tf.keras.models.load_model(
            filepath=save_config.filepath(model_id=model_id)
        )


class NoneContextManager(EnsembleContextManager):
    """
    This context manager makes nothing at all,
    i.e., the model will be executed in exactly the state the process was created.

    This for example implies that the tensorflow default GPU configuration will be used.

    It is save to use this ContextManager on an existing processes where there is already a tf.session
    available.
    """


class CpuOnlyContextManager(EnsembleContextManager):
    """
    Disables all GPU use, and runs all processes on the CPU

    Note: Tensorflow will still see that cuda is installed,
    but will not find any GPU devices and thus print a warning accordingly.
    """

    # docstr-coverage:inherited
    def __enter__(self) -> "CpuOnlyContextManager":
        self.disable_all_gpus()
        return self

    @staticmethod
    def disable_all_gpus():
        """Makes sure no GPUs are visible"""
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], "GPU")
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != "GPU"
        except RuntimeError as e:
            raise ValueError(
                f"Uncertainty Wizard was unable to disable gpu use."
            ) from e


class DynamicGpuGrowthContextManager(EnsembleContextManager):
    """
    This context manager configures tensorflow such that multiple processes can use the main GPU at the same time.
    It is the default in a lazy ensemble multiprocessing environment
    """

    # docstr-coverage:inherited
    def __enter__(self) -> "DynamicGpuGrowthContextManager":
        super().__enter__()
        global number_of_tasks_in_this_process
        if number_of_tasks_in_this_process == 0:
            self.enable_dynamic_gpu_growth()
        return self

    @classmethod
    def enable_dynamic_gpu_growth(cls):
        """
        Configures tensorflow to set memory growth to ture on all GPUs
        :return: None
        """
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


global device_id


class DeviceAllocatorContextManager(EnsembleContextManager, abc.ABC):
    """
    This context manager configures tensorflow such a user-defined amount of processes for every available gpu
    are started. In addition, running a process on the CPU can be enabled.

    This is an abstract context manager. To use it, one has to subclass it and override (at least)
    the abstract methods.
    """

    # docstr-coverage: inherited
    def __enter__(self) -> "DeviceAllocatorContextManager":
        super().__enter__()
        global number_of_tasks_in_this_process
        global device_id
        if number_of_tasks_in_this_process == 0:
            device_id = self._get_availabilities_and_choose_device()
            if device_id == -1:
                CpuOnlyContextManager.disable_all_gpus()
            else:
                self._use_gpu(index=device_id)
        else:
            print(f"Process on device {device_id} is reused. Not initializing devices.")
        return self

    # docstr-coverage: inherited
    def __exit__(self, type, value, traceback) -> None:
        super().__exit__(type, value, traceback)

        global number_of_tasks_in_this_process
        global device_id
        if number_of_tasks_in_this_process == self.max_sequential_tasks_per_process():
            self._release_device(device_id)
        else:
            print(
                f"Not releasing assignment of device {device_id} as process might be reused."
            )

    # docstr-coverage: inherited
    @classmethod
    def before_start(cls):
        path = cls.file_path()
        if os.path.exists(path):
            os.remove(path)
        lock_path = cls._lock_file_path()
        if os.path.exists(lock_path):
            os.remove(lock_path)

    @classmethod
    @abc.abstractmethod
    def file_path(cls) -> str:
        """
        DeviceAllocatorContextManager uses a temporary file to corrdinate device allocation.
        This method should provide the path of that file.

        The folder where the file should be placed must exist.
        The file itself should not exist (if it does, it will be deleted).

        *Attention:* This function must be pure: Repeated calls should always return the same value.

        :return:
        """

    @classmethod
    def _lock_file_path(cls) -> str:
        """
        DeviceAllocatorContextManager uses a lockfile when coordinating device allocation.
        This method can be used to specify the path of the lock file.

        The folder where the lockfile should be placed must exist.
        The lockfile should not exist (if it does, it will be deleted).

        *Attention:* This function must be pure: Repeated calls should always return the same value.

        :return: The path of the lockfile
        """
        return cls.file_path() + ".lock"

    @classmethod
    @abc.abstractmethod
    def run_on_cpu(cls) -> bool:
        """
        Specify whether one process (in addition to the ones specified in 'virtual_devices_per_gpu'
        should run on the CPU.

        Note that even for the processes running on the GPU, some part of the workload (e.g. preprocessing)
        is typically performed on the CPU and may already consume a lot of resources.
        Do not set this to true if you have a weak CPU - your PC may freeze.

        *Attention:* This function must be pure: Repeated calls should always return the same value.

        :return: A flag indicating if the CPU should also dispatch a job to the cpu.
        """

    @classmethod
    @abc.abstractmethod
    def virtual_devices_per_gpu(cls) -> Dict[int, int]:
        """
        A dict mapping every GPU index to the number of processes which should be started concurrently
        on that GPU.

        *Attention:* This function must be pure: Repeated calls should always return the same value.

        :return: A mapping specifying how many processes of this ensemble should run concurrently per gpu.
        """

    @classmethod
    @abc.abstractmethod
    def gpu_memory_limit(cls) -> int:
        """
        Override this method to specify the amount of MB which should be used
        when creating the virtual device on the GPU. Ignored for CPUs.

        *Attention:* This function must be pure: Repeated calls should always return the same value.

        :return: The amount of MB which will be reserved on the selected gpu in the created context.
        """

    @classmethod
    def acquire_lock_timeout(cls) -> int:
        """
        DeviceAllocatorContextManager uses a lockfile when coordinating device allocation.

        This method can be used to specify a timeout. If the lock is not acquired within this time,
        the process fails.

        :return: The number of seconds to wait until timing out.
        """
        return 5

    @classmethod
    def delay(cls) -> float:
        """
        DeviceAllocatorContextManager uses a lockfile when coordinating device allocation.
        If a file is locked, DeviceAllocatorContextManager tries periodically in the
        interval specified in this method to gain the lock.

        :return: The number of seconds to wait before a retry if the lock cannot be acquired.
        """
        return 0.1

    @classmethod
    def _get_availabilities_and_choose_device(cls) -> int:
        lockfile = cls._acquire_lock()
        try:
            with open(cls.file_path(), "rb") as file:
                availabilities = pickle.load(file)
                device = cls._pick_device(availabilities)

        except OSError as e:
            if not isinstance(e, FileNotFoundError):
                # Some other error than 'file exists' occurred
                raise RuntimeError(
                    (
                        "An error occurred when trying read current allocation file "
                        "for the Uncertainty Wizard DeviceAllocatorContextManager"
                    )
                ) from e

            # All good, this process is apparently the first one and has to create the file.
            availabilities = cls.virtual_devices_per_gpu()
            if cls.run_on_cpu():
                availabilities[-1] = 1
            else:
                availabilities[-1] = 0
            device = cls._pick_device(availabilities)

        availabilities[device] = availabilities[device] - 1
        print(
            f"\nupdating availabilities file: {availabilities} (decreased availability of device {device})"
        )
        with open(cls.file_path(), "wb") as file:
            pickle.dump(availabilities, file)

        cls._release_lock(lockfile=lockfile)
        return device

    @classmethod
    def _release_device(cls, device: int) -> None:
        with open(cls.file_path(), "rb") as file:
            availabilities = pickle.load(file)
        availabilities[device] = availabilities[device] + 1
        print(
            f"\nupdating availabilities file: {availabilities} (increased availability of device {device})"
        )
        with open(cls.file_path(), "wb") as file:
            pickle.dump(availabilities, file)

    @classmethod
    def _pick_device(cls, availablilities) -> int:
        picked_device = None
        picked_device_availability = 0
        for device, availability in availablilities.items():
            if availability > picked_device_availability:
                picked_device = device
                picked_device_availability = availability
        if picked_device is None:
            raise ValueError(
                f"No available devices. Please make sure that the number processes configured "
                f"in your call to the ensemble does not exceed the number of deviced specified "
                f"in your DeviceAllocatorContextManager.virtual_devices_per_gpu extension "
                f"(+1 if run_on_cpu() is set to true)"
            )
        print(f"Availabilities: {availablilities}. Picked Device {picked_device}")
        return picked_device

    @classmethod
    def _use_gpu(cls, index: int):
        size = cls.gpu_memory_limit()
        gpus = tf.config.experimental.list_physical_devices("GPU")

        # Check if selected gpu can be found
        if gpus is None or len(gpus) <= index:
            raise ValueError(
                f"Uncertainty Wizards DeviceAllocatorContextManager was configured to use gpu {index} "
                f"but no no such gpu was found.  "
            )

        try:
            tf.config.set_visible_devices([gpus[index]], "GPU")
            tf.config.experimental.set_virtual_device_configuration(
                gpus[index],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=size)],
            )
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            raise ValueError(
                f"Uncertainty Wizard was unable to create a virtual device "
                f"on gpu {index} and memory limit {size}MB"
            ) from e

    @classmethod
    def _acquire_lock(cls) -> int:
        """
        Waits until no lockfile is present and, once possible, creates a lockfile.
        Code inspired by https://github.com/dmfrey/FileLock/blob/master/filelock/filelock.py

        :returns the file descriptor (int) of the acquired lockfile
        :raise RuntimeError if the lock acquiring times out or if an IO error prevents lock acquiring.
        """
        timeout = cls.acquire_lock_timeout()
        start_time = time.time()
        while True:
            try:
                return os.open(
                    cls._lock_file_path(),
                    (
                        os.O_CREAT  # create file if it does not exist
                        | os.O_EXCL  # error if create and file exists
                        | os.O_RDWR
                    ),  # open for reading and writing
                )
            except OSError as e:
                if e.errno != errno.EEXIST and not isinstance(e, FileNotFoundError):
                    # Some other error than 'file exists' occurred
                    raise RuntimeError(
                        "An error occurred when trying to acquire lock for device allocation "
                    ) from e
                if (time.time() - start_time) >= timeout:
                    raise RuntimeError(
                        (
                            f"Ensemble process was not capable of acquiring lock in {timeout} seconds."
                            f"Make sure that no file `{cls._lock_file_path()}` exists (delete it if it does)."
                            f"If this does not help, consider increasing the timeout by overriding `acquire_lock_timeout`"
                            f"in your DeviceAllocatorContextManager extension"
                        )
                    )
                time.sleep(cls.delay())

    @classmethod
    def _release_lock(cls, lockfile: int):
        os.close(lockfile)
        os.remove(cls._lock_file_path())

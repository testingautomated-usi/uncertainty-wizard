import abc
import os
import pathlib
import pickle
from typing import Dict
from unittest import TestCase

from uncertainty_wizard.models.ensemble_utils import DeviceAllocatorContextManager

FILE_PATH = 'temp_test_6543543168'


class TestDeviceAllocator(DeviceAllocatorContextManager, abc.ABC):

    @classmethod
    def file_path(cls) -> str:
        return FILE_PATH

    @classmethod
    def gpu_memory_limit(cls) -> int:
        return 2000

    @classmethod
    def acquire_lock_timeout(cls) -> int:
        return 1

    @classmethod
    def virtual_devices_per_gpu(cls) -> Dict[int, int]:
        return {0: 1, 1: 2}


class FunctionalStochasticTest(TestCase):

    def test_cleans_files_before_start(self):
        pathlib.Path(FILE_PATH).touch()
        TestDeviceAllocator.before_start()
        self.assertFalse(os.path.isfile(FILE_PATH), "Specified temp file (allocation or lockfile) was not deleted")

    def test_acquire_and_release_lock(self):
        TestDeviceAllocator.before_start()
        # Test lock was acquired
        lockfile = TestDeviceAllocator._acquire_lock()
        self.assertTrue(os.path.isfile(FILE_PATH + '.lock'), "Lockfile was not created")
        # Test lock file can not be acquired twice
        with self.assertRaises(RuntimeError) as e_context:
            TestDeviceAllocator._acquire_lock()
            self.assertTrue('Ensemble process was not capable of acquiring lock' in str(e_context.exception))
        # Test lock is released
        TestDeviceAllocator._release_lock(lockfile)
        self.assertFalse(os.path.isfile(FILE_PATH + '.lock'), "Lockfile was not deleted")

    def _assert_device_selection(self, chosen_device, expected_device, expected_availabilities):
        self.assertEqual(chosen_device, expected_device)
        with open(FILE_PATH, "rb") as file:
            availabilities = pickle.load(file)
            self.assertEqual(availabilities, expected_availabilities)

    def test_get_availabilities_and_choose_device(self):
        TestDeviceAllocator.before_start()
        # Make sure as a first device, the once with highest availability is selected
        chosen_device = TestDeviceAllocator._get_availabilities_and_choose_device()
        self._assert_device_selection(chosen_device, 1, {-1: 0, 0: 1, 1: 1})

        # Select most available device (with lowest id as tiebreaker)
        chosen_device = TestDeviceAllocator._get_availabilities_and_choose_device()
        self._assert_device_selection(chosen_device, 0, {-1: 0, 0: 0, 1: 1})
        chosen_device = TestDeviceAllocator._get_availabilities_and_choose_device()
        self._assert_device_selection(chosen_device, 1, {-1: 0, 0: 0, 1: 0})

        # Workaround to make CPU and GPU available. Also tests that device releasing works
        TestDeviceAllocator._release_device(-1)
        TestDeviceAllocator._release_device(0)
        with open(FILE_PATH, "rb") as file:
            availabilities = pickle.load(file)
            self.assertEqual(availabilities, {-1: 1, 0: 1, 1: 0})

        # Make sure if an alternative is available, CPU is not selected
        chosen_device = TestDeviceAllocator._get_availabilities_and_choose_device()
        self._assert_device_selection(chosen_device, 0, {-1: 1, 0: 0, 1: 0})

        # Make sure if the only thing available, CPU is selected
        chosen_device = TestDeviceAllocator._get_availabilities_and_choose_device()
        self._assert_device_selection(chosen_device, -1, {-1: 0, 0: 0, 1: 0})

        # No capacity left, make sure error is thrown
        with self.assertRaises(ValueError) as e_context:
            TestDeviceAllocator._get_availabilities_and_choose_device()
            self.assertTrue('No available devices. ' in str(e_context.exception))

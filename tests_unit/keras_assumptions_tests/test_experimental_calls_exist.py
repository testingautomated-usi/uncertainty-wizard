import inspect
import unittest
from unittest import TestCase

import tensorflow as tf

from uncertainty_wizard.internal_utils.tf_version_resolver import current_tf_version_is_older_than


class TestExperimentalAPIAreAvailable(TestCase):
    """
    These tests are 'dependency-regression tests':
    They check that the experimental functions uwiz relies on
    are present with the expected signature in the used tf version.
    This way, we ensure to observe when the experimental methods
    become stable (and thus conditional imports have to be added)
    """

    def test_list_physical_devices(self):
        self.assertTrue("list_physical_devices" in dir(tf.config.experimental))
        parameters = inspect.signature(
            tf.config.experimental.list_physical_devices
        ).parameters
        self.assertTrue("device_type" in parameters)
        self.assertEqual(1, len(parameters))

    @unittest.skipIf(not current_tf_version_is_older_than("2.10.0"),
                     "Known to fail for tf >= 2.10.0")
    def test_virtual_device_configuration(self):
        self.assertTrue("VirtualDeviceConfiguration" in dir(tf.config.experimental))
        parameters = inspect.signature(
            tf.config.experimental.VirtualDeviceConfiguration
        ).parameters
        self.assertTrue("memory_limit" in parameters)
        self.assertTrue("experimental_priority" in parameters)
        self.assertEqual(2, len(parameters))

    def test_set_visible_devices(self):
        self.assertTrue("set_visible_devices" in dir(tf.config.experimental))
        parameters = inspect.signature(
            tf.config.experimental.set_visible_devices
        ).parameters
        self.assertTrue("devices" in parameters)
        self.assertTrue("device_type" in parameters)
        self.assertEqual(2, len(parameters))

    def test_set_virtual_device_configuration(self):
        self.assertTrue(
            "set_virtual_device_configuration" in dir(tf.config.experimental)
        )
        parameters = inspect.signature(
            tf.config.experimental.set_virtual_device_configuration
        ).parameters
        self.assertTrue("device" in parameters)
        self.assertTrue("logical_devices" in parameters)
        self.assertEqual(2, len(parameters))

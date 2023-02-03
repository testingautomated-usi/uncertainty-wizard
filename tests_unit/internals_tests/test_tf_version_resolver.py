from unittest import TestCase

import tensorflow as tf

from uncertainty_wizard.internal_utils import tf_version_resolver


class EnsembleFunctionalTest(TestCase):
    def call_current_tf_is_older_than(self, version, expected_outcome):
        result = tf_version_resolver.current_tf_version_is_older_than(version=version)
        if expected_outcome:
            self.assertTrue(result)
        else:
            self.assertFalse(result)

    def test_current_tf_version_is_older_than_is_false(self):
        # Test regular case
        self.call_current_tf_is_older_than("1.2.3", False)
        self.call_current_tf_is_older_than("2.0.0", False)

        # Test release candidate
        self.call_current_tf_is_older_than("1.2.3_rc2", False)

        # Call on same version
        self.call_current_tf_is_older_than(tf.__version__, False)

    def test_current_tf_version_is_older_than_is_true(self):
        # Test regular case
        self.call_current_tf_is_older_than("3.2.1", True)
        self.call_current_tf_is_older_than("2.9.0", True)

        # Test release candidate
        self.call_current_tf_is_older_than("3.2.1_rc1", True)

    def test_tf_version_is_older_than_fallback(self):
        invalid = "1.2.invalid"

        # Test with fallback provided
        result = tf_version_resolver.current_tf_version_is_older_than(
            version=invalid, fallback=True
        )
        self.assertTrue(result)
        result = tf_version_resolver.current_tf_version_is_older_than(
            version=invalid, fallback=False
        )
        self.assertFalse(result)

        # Test without fallback provided: In this case it should return True
        self.call_current_tf_is_older_than(invalid, True)

    def test_tf_version_is_older_than_raises_warning(self):
        invalid = "1.2.invalid"
        with self.assertWarns(RuntimeWarning):
            self.call_current_tf_is_older_than(invalid, True)

    def call_current_tf_is_newer_than(self, version, expected_outcome):
        result = tf_version_resolver.current_tf_version_is_newer_than(version=version)
        if expected_outcome:
            self.assertTrue(result)
        else:
            self.assertFalse(result)

    def test_current_tf_version_is_newer_is_true(self):
        # Test regular case
        self.call_current_tf_is_newer_than("1.2.3", True)
        self.call_current_tf_is_newer_than("2.0.0", True)

        # Test release candidate
        self.call_current_tf_is_newer_than("1.2.3rc2", True)

    def test_current_tf_version_is_newer_is_false(self):
        # Call on same version
        self.call_current_tf_is_newer_than(tf.__version__, False)

    def test_tf_version_is_newer_than_fallback(self):
        invalid = "1.2.invalid"

        # Test with fallback provided
        result = tf_version_resolver.current_tf_version_is_newer_than(
            version=invalid, fallback=True
        )
        self.assertTrue(result)
        result = tf_version_resolver.current_tf_version_is_newer_than(
            version=invalid, fallback=False
        )
        self.assertFalse(result)

        # Test without fallback provided: In this case it should return False
        self.call_current_tf_is_newer_than(invalid, False)

    def test_tf_version_is_newer_than_raises_warning(self):
        invalid = "1.2.invalid"
        with self.assertWarns(RuntimeWarning):
            self.call_current_tf_is_newer_than(invalid, False)

import warnings
from typing import Union

import tensorflow as tf


def _compare_expected_to_current_tf_version(expected_version) -> Union[None, int]:
    """
    Compares the 'x.y.z' version parts of a passed expected version and the actual tensorflow version.
    The result is negative if the expected version is newer, positive if the expected version is older
    and 0 if they are the same.
    If one of the versions cannot be parsed, a warning is triggered and 'None' is returned.
    :param expected_version:
    :return: an int if a comparison was made and None if parsing was impossible
    """
    actual_version = tf.version.VERSION

    # Note: We're currently only considering versions in the format 'x.y.z'
    # (i.e., ignore RCs and multi digit versions - which is probably fine given tfs very fast major release cycles)

    # Inspection disabling reason: We really want to catch all exceptions.
    # noinspection PyBroadException
    try:
        expected_v_splits = [int(v) for v in expected_version[:5].split(".")]
        actual_v_splits = [int(v) for v in actual_version[:5].split(".")]
    except Exception:
        warnings.warn(
            f"One of the version strings '{expected_version}' (requested) "
            f"or '{actual_version}' was not parsed: "
            f"We are trying to use a suitable guess about your tf compatibility and thus,"
            f"you may not actually note any problems."
            f"However, to be safe, please report this issue (with this warning) "
            f"to the uncertainty wizard maintainers.",
            RuntimeWarning,
        )
        return None

    for act, expected in zip(actual_v_splits, expected_v_splits):
        if expected > act:
            return 1
        elif expected < act:
            return -1
    # Version equality
    return 0


def current_tf_version_is_older_than(version: str, fallback: Union[bool, None] = True):
    """
    A method to check whether the loaded tensorflow version is older than a passed version.
    :param version: A tensorflow version string, e.g. '2.3.0'
    :param fallback: If a problem occurs during parsing, the value of fallback will be returned
    :return: True if the used tensorflow version is older than the version specified in the passed string
    """
    comp = _compare_expected_to_current_tf_version(version)
    if comp is None:
        return fallback
    elif comp > 0:
        return True
    else:
        return False


def current_tf_version_is_newer_than(version: str, fallback: Union[bool, None] = False):
    """
    A method to check whether the loaded tensorflow version is younger than a passed version.
    :param version: A tensorflow version string, e.g. '2.3.0'
    :param fallback: If a problem occurs during parsing, the value of fallback will be returned
    :return: True if the used tensorflow version is newer than the version specified in the passed string
    """
    comp = _compare_expected_to_current_tf_version(version)
    if comp is None:
        return fallback
    elif comp >= 0:
        return False
    else:
        return True

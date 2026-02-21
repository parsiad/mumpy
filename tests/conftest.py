"""Shared test helpers and NumPy conversion utilities."""

import numpy as np


def to_numpy(x):
    return np.asarray(x)


def assert_array_equal(actual, expected):
    np.testing.assert_array_equal(to_numpy(actual), np.asarray(expected))


def assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=False):
    np.testing.assert_allclose(
        to_numpy(actual),
        np.asarray(expected),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )

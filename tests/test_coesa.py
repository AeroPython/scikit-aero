# coding: utf-8

import numpy as np
import scipy as sp
from numpy import testing
from scipy import constants

from skaero import atmosphere  # Just in case there are utility functions
from skaero.atmosphere import coesa


def test_sea_level():
    """Tests sea level values.

    """
    h, T, p, rho = coesa(0.0)

    np.testing.assert_equal(h, 0.0)
    np.testing.assert_equal(T, sp.constants.C2K(15))
    np.testing.assert_equal(p, sp.constants.atm)
    np.testing.assert_almost_equal(rho, 1.2250, decimal=3)


def test_sea_level_0d_array():
    """Tests sea level values using zero dimension array.

    """
    h_ = np.array(0.0)
    T_ = np.array(sp.constants.C2K(15))
    p_ = np.array(sp.constants.atm)
    rho_ = np.array(1.2250)

    h, T, p, rho = coesa(h_)

    np.testing.assert_array_almost_equal(h, h_, decimal=3)
    np.testing.assert_array_almost_equal(T, T_, decimal=3)
    np.testing.assert_array_almost_equal(p, p_, decimal=3)
    np.testing.assert_array_almost_equal(rho, rho_, decimal=3)


def test_sea_level_nd_array():
    """Tests sea level values using n dimension array.

    """

    h_ = np.array([0.0, 0.0, 0.0])

    h, T, p, rho = coesa(h_)

    np.testing.assert_array_almost_equal(
        h, h_, decimal=3)
    np.testing.assert_array_almost_equal(
        T, [288.15, 288.15, 288.15], decimal=3)
    np.testing.assert_array_almost_equal(
        p, [101325.0, 101325.0, 101325.0], decimal=3)
    np.testing.assert_array_almost_equal(
        rho, [1.2250, 1.2250, 1.2250], decimal=3)


def test_under_1000():
    """Tests altitude values under 1000.0 m"""

    h_ = np.array([50.0, 550.0, 850.0])

    h, T, p, rho = coesa(h_)

    np.testing.assert_array_almost_equal(h, h_)
    np.testing.assert_array_almost_equal(
        T, [287.825, 284.575, 282.625], decimal=3)
    # TODO: Notice the decimal=-1. Check against other validated data, or
    # wonder why is it "wrong" on the standard.
    # ANSWER: Maybe they multiplied by the pressure ratio, or something.
    # FINAL: Yes, they did.
    np.testing.assert_array_almost_equal(
        p, [100720.0, 94889.0, 91521.0], decimal=-1)  # [100725.8, ..., ...]
    np.testing.assert_array_almost_equal(
        rho, [1.2191, 1.1616, 1.1281], decimal=3)

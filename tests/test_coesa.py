# coding: utf-8

"""
Tests of the U.S. 1976 Standard Atmosphere implementation. All of them are
validated against the `standard`_.

.. _`standard`: http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770009539_1977009539.pdf

"""

from __future__ import division, absolute_import

import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_equal, assert_array_almost_equal)
import scipy as sp
import scipy.constants

from skaero.atmosphere import coesa, util


def _C2K(val):
    """Convert Celsius to Kelvins."""
    return scipy.constants.convert_temperature([val], 'C', 'K')


def test_sea_level():
    """Tests sea level values.

    """
    h = 0.0
    expected_h = 0.0
    expected_T = _C2K(15)
    expected_p = sp.constants.atm
    expected_rho = 1.2250

    h, T, p, rho = coesa.table(h)

    assert_equal(h, expected_h)
    assert_equal(T, expected_T)
    assert_equal(p, expected_p)
    assert_almost_equal(rho, expected_rho, decimal=4)


def test_sea_level_0d_array():
    """Tests sea level values using zero dimension array.

    """
    h = np.array(0.0)
    expected_h = np.array(0.0)
    expected_T = np.array(_C2K(15))
    expected_p = np.array(sp.constants.atm)
    expected_rho = np.array(1.2250)

    h, T, p, rho = coesa.table(h)

    assert_array_equal(h, expected_h)
    assert_array_almost_equal(T, expected_T)
    assert_array_almost_equal(p, expected_p)
    assert_array_almost_equal(rho, expected_rho)


def test_sea_level_nd_array():
    """Tests sea level values using n dimension array.

    """
    h = np.array([0.0, 0.0, 0.0])
    expected_h = np.array([0.0, 0.0, 0.0])
    expected_T = np.array([288.15] * 3)
    expected_p = np.array([101325.0] * 3)
    expected_rho = np.array([1.2250] * 3)

    h, T, p, rho = coesa.table(h)

    assert_array_equal(h, expected_h)
    assert_array_almost_equal(T, expected_T)
    assert_array_almost_equal(p, expected_p)
    assert_array_almost_equal(rho, expected_rho)


def test_geometric_to_geopotential():
    z = np.array([50.0, 5550.0, 10450.0])
    h = util.geometric_to_geopotential(z)
    expected_h = np.array([50.0, 5545.0, 10433.0])
    assert_array_almost_equal(h, expected_h, decimal=0)


def test_under_1000m():
    """Tests for altitude values under 1000.0 m

    """
    z = np.array([50.0, 550.0, 850.0])
    h = util.geometric_to_geopotential(z)
    expected_h = np.array([50.0, 550.0, 850.0])
    expected_T = np.array([287.825, 284.575, 282.626])
    expected_p = np.array([100720.0, 94890.0, 91523.0])
    expected_rho = np.array([1.2191, 1.1616, 1.1281])

    h, T, p, rho = coesa.table(h)

    assert_array_almost_equal(h, expected_h, decimal=0)
    assert_array_almost_equal(T, expected_T, decimal=3)
    assert_array_almost_equal(p, expected_p, decimal=-1)
    assert_array_almost_equal(rho, expected_rho, decimal=4)


def test_under_11km():
    """Tests for altitude values between 1 and 11 km

    """
    z = np.array([500.0, 2500.0, 6500.0, 9000.0, 11000.0])
    h = util.geometric_to_geopotential(z)
    expected_h = np.array([500.0, 2499.0, 6493.0, 8987.0, 10981.0])
    expected_T = np.array([284.900, 271.906, 245.943, 229.733, 216.774])
    expected_p = np.array([95461.0, 74691.0, 44075.0, 30800.0, 22699.0])
    expected_rho = np.array([1.1673, 0.95695, 0.62431, 0.46706, 0.36480])

    h, T, p, rho = coesa.table(h)
    
    assert_array_almost_equal(h, expected_h, decimal=0)
    assert_array_almost_equal(T, expected_T, decimal=3)
    assert_array_almost_equal(p, expected_p, decimal=0)
    assert_array_almost_equal(rho, expected_rho, decimal=4)


def test_under_35km():
    """Tests for altitude values between 11 and 35 km

    """
    z = np.array([15000.0, 25000.0, 35000.0])
    h = util.geometric_to_geopotential(z)
    expected_h = np.array([14965.0, 24902., 34808.0])
    expected_T = np.array([216.65, 221.552, 236.513])
    expected_p = np.array([12111.0, 2549.2, 574.59])
    expected_rho = np.array([0.19476, 0.040084, 0.0084634])

    h, T, p, rho = coesa.table(h)
    
    assert_array_almost_equal(h, expected_h, decimal=0)
    assert_array_almost_equal(T, expected_T, decimal=3)
    assert_array_almost_equal(p, expected_p, decimal=0)
    assert_array_almost_equal(rho, expected_rho, decimal=5)


def test_under_86km():
    """Tests for altitude values between 35 and 86 km

    """
    z = np.array([50000.0, 70000.0, 86000.0])
    h = util.geometric_to_geopotential(z)
    expected_h = np.array([49610.0, 69238., 84852.0])
    expected_T = np.array([270.65, 219.585, 186.87])
    expected_p = np.array([79.779, 5.2209, 0.37338])
    expected_rho = np.array([0.0010269, 0.000082829, 0.000006958])

    h, T, p, rho = coesa.table(h)
    
    assert_array_almost_equal(h, expected_h, decimal=0)
    assert_array_almost_equal(T, expected_T, decimal=2)
    assert_array_almost_equal(p, expected_p, decimal=3)
    assert_array_almost_equal(rho, expected_rho, decimal=7)

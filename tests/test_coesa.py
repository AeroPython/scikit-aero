# coding: utf-8

"""
Tests of the U.S. 1976 Standard Atmosphere implementation.

Most of the tests here are validated against the data tables provided
in the `PDAS Standard Atmosphere`_, which are included here. Only
test_under_1000 uses the values of the `standard`_, and it is retained just
for historical reasons because there seems to be some inaccuracies there.

Data generated using tables.py from PDAS.

.. _`PDAS Standard Atmosphere`: http://www.pdas.com/atmos.html
.. _`standard`: http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770009539_1977009539.pdf

TODO
----
* Too much repeated code: refactor.

"""

from __future__ import print_function

import numpy as np
import scipy as sp
import numpy.testing
import scipy.constants

from skaero.atmosphere import coesa


# Load data from PDAS atmos
data = np.loadtxt('tests/si2py.prt', skiprows=2)
print(data)


def test_sea_level():
    """Tests sea level values.

    """
    h, T, p, rho = coesa.table(0.0)

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

    h, T, p, rho = coesa.table(h_)

    np.testing.assert_array_equal(h, h_)
    np.testing.assert_array_almost_equal(T, T_, decimal=3)
    np.testing.assert_array_almost_equal(p, p_, decimal=3)
    np.testing.assert_array_almost_equal(rho, rho_, decimal=3)


def test_sea_level_nd_array():
    """Tests sea level values using n dimension array.

    """
    h_ = np.array([0.0, 0.0, 0.0])
    h, T, p, rho = coesa.table(h_)

    np.testing.assert_array_equal(h, h_)
    np.testing.assert_array_almost_equal(
        T, [288.15] * 3, decimal=3)
    np.testing.assert_array_almost_equal(
        p, [101325.0] * 3, decimal=3)
    np.testing.assert_array_almost_equal(
        rho, [1.2250] * 3, decimal=3)


def test_under_1000m():
    """Tests for altitude values under 1000.0 m

    """
    z_ = np.array([50.0, 550.0, 850.0])
    h_ = coesa.geometric_to_geopotential(z_)

    # Retrieve desired data from PDAS tables
    desired = np.array([data[data[:, 0] == x][0] for x in z_])

    T_ = desired[:, 4]
    p_ = desired[:, 5]
    rho_ = desired[:, 6]

    h, T, p, rho = coesa.table(h_)

    np.testing.assert_array_almost_equal(T, T_, decimal=1)
    np.testing.assert_array_almost_equal(p, p_, decimal=0)
    np.testing.assert_array_almost_equal(rho, rho_, decimal=3)


def test_under_11km():
    """Tests for altitude values under 11 km (first layer)

    """
    z_ = np.array([500.0, 2500.0, 6500.0, 9000.0, 11000.0])
    h_ = coesa.geometric_to_geopotential(z_)

    # Retrieve desired data from PDAS tables
    desired = np.array([data[data[:, 0] == x][0] for x in z_])

    T_ = desired[:, 4]
    p_ = desired[:, 5]
    rho_ = desired[:, 6]

    h, T, p, rho = coesa.table(h_)

    np.testing.assert_array_almost_equal(T, T_, decimal=1)
    np.testing.assert_array_almost_equal(p, p_, decimal=0)
    np.testing.assert_array_almost_equal(rho, rho_, decimal=3)

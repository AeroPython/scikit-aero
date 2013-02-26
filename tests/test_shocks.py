# coding: utf-8

"""
Tests of the shocks package.

References
----------
1. NACA-TR-1135 http://hdl.handle.net/2060/19930091059
2. NASA-TN-D-2221 http://hdl.handle.net/2060/19640007246
3. Normal shock tables on Wikipedia
   http://en.wikipedia.org/wiki/Normal_shock_tables

"""

from __future__ import division, absolute_import

import numpy as np
import numpy.testing

import pytest

from skaero.gasdynamics import shocks


def test_normal_shock_constructor():
    """Tests the constructor of the NormalShock class.

    """
    gamma = 1.4
    M_1 = 2.0
    shocks.NormalShock(M_1, gamma)


def test_normal_shock_default_adiabatic_index():
    """Tests default 1.4 value for the adiabatic index.

    """
    ns = shocks.NormalShock(2.0)
    np.testing.assert_equal(ns.gamma, 7 / 5)


def test_normal_shock_M_2():
    """Tests the computation of post-shock Mach number.

    """
    M_1_list = [1.5, 1.8, 2.1, 3.0]
    M_2_list = [
        0.7011,
        0.6165,
        0.5613,
        0.4752
    ]
    ns_list = [shocks.NormalShock(M_1, 1.4) for M_1 in M_1_list]

    for i in range(len(ns_list)):
        np.testing.assert_almost_equal(ns_list[i].M_2, M_2_list[i], decimal=4)


def test_normal_shock_fails_asubsonic_M_1():
    """Tests the constructor raises an exception if M_1 <= 1.

    """
    with pytest.raises(ValueError):
        shocks.NormalShock(0.8)


def test_normal_shock_ratios():
    """Tests normal shock ratios of thermodynamics properties.

    """
    M_1_list = [1.5, 1.8, 2.1, 3.0]
    p_ratio_list = [
        2.4583,
        3.6133,
        4.9783,
        10.3333
    ]
    rho_ratio_list = [
        1.8621,
        2.3592,
        2.8119,
        3.8571
    ]
    T_ratio_list = [
        1.3202,
        1.5316,
        1.7705,
        2.6790
    ]
    ns_list = [shocks.NormalShock(M_1, 1.4) for M_1 in M_1_list]

    for i in range(len(ns_list)):
        np.testing.assert_almost_equal(
            ns_list[i].p2_p1, p_ratio_list[i], decimal=4)
        np.testing.assert_almost_equal(
            ns_list[i].rho2_rho1, rho_ratio_list[i], decimal=4)
        np.testing.assert_almost_equal(
            ns_list[i].T2_T1, T_ratio_list[i], decimal=4)


def test_normal_shock_infinite_limit():
    gamma = 1.4
    ns = shocks.NormalShock(np.inf, gamma)
    np.testing.assert_almost_equal(
        ns.M_2, np.sqrt((gamma - 1) / 2 / gamma), decimal=3)
    np.testing.assert_almost_equal(
        ns.rho2_rho1, (gamma + 1) / (gamma - 1), decimal=3)


def test_normal_shock_zero_deflection():
    ns = shocks.NormalShock(2.0)
    assert ns.theta == 0.0


def test_error_max_deflection():
    with pytest.raises(ValueError):
        shocks.from_deflection_angle(5, np.radians(50))


def test_error_mach_angle():
    with pytest.raises(ValueError):
        shocks.ObliqueShock(5, np.radians(10))


def test_max_deflection():
    M_1_list = [1.4, 1.9, 2.2, 3.0, np.inf]
    theta_max_degrees_list = [
        9.427,
        21.17,
        26.1,
        34.07,
        45.58
    ]
    beta_theta_max_degrees_list = [
        67.72,
        64.78,
        64.62,
        65.24,
        67.79
    ]
    angles_pairs_list = [shocks.max_deflection(M_1, 1.4) for M_1 in M_1_list]

    for i in range(len(angles_pairs_list)):
        np.testing.assert_almost_equal(
            angles_pairs_list[i][0], np.radians(theta_max_degrees_list[i]),
            decimal=3)
        np.testing.assert_almost_equal(
            angles_pairs_list[i][1],
            np.radians(beta_theta_max_degrees_list[i]), decimal=3)


def test_parallel_shock_infinity_mach():
    M_1 = np.inf
    beta = 0.0
    os = shocks.ObliqueShock(M_1, beta)
    assert os.M_1n == 0.0
    assert np.isfinite(os.theta)


def test_oblique_shock_from_deflection_angle():
    # Anderson, example 4.1
    # Notice that only graphical accuracy is achieved in the original example
    M_1 = 3.0
    theta = np.radians(20.0)
    os = shocks.from_deflection_angle(M_1, theta, weak=True)

    np.testing.assert_almost_equal(os.M_1n, 1.839, decimal=2)
    np.testing.assert_almost_equal(os.M_2n, 0.6078, decimal=2)
    np.testing.assert_almost_equal(os.M_2, 1.988, decimal=1)
    np.testing.assert_almost_equal(os.p2_p1, 3.783, decimal=1)
    np.testing.assert_almost_equal(os.T2_T1, 1.562, decimal=2)

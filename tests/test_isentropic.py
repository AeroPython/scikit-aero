# coding: utf-8

"""
Tests of the isentropic package.

References
----------
1. NACA-TR-1135 http://hdl.handle.net/2060/19930091059
2. Anderson, J.D.: "Modern compressible flow", 3rd edition.

"""

from __future__ import division, absolute_import

import numpy as np
import numpy.testing

import pytest

from skaero.gasdynamics import isentropic


def test_mach_angle():
    M_list = [1.1, 1.38, 2.05, 3.0, np.inf]
    mu_list = [
        65.38,
        46.44,
        29.20,
        19.47,
        0.0
    ]
    results_list = [isentropic.mach_angle(M) for M in M_list]
    for i in range(len(M_list)):
        np.testing.assert_almost_equal(
            results_list[i], np.radians(mu_list[i]), decimal=3)


def test_mach_angle_raises_error_subsonic():
    with pytest.raises(ValueError):
        isentropic.mach_angle(0.8)


def test_pm_angle():
    gamma = 1.4
    M_list = [1.2, 1.4, 2.6, 3.2, np.inf]
    nu_list = [
        3.558,
        8.987,
        41.41,
        53.47,
        130.45
    ]
    results_list = [
        isentropic.PrandtlMeyerExpansion.nu(M, gamma) for M in M_list]
    for i in range(len(M_list)):
        np.testing.assert_almost_equal(
            results_list[i], np.radians(nu_list[i]), decimal=3)


def test_default_flow():
    pm = isentropic.PrandtlMeyerExpansion(1.2, 0.8)
    assert pm.fl.gamma == 1.4


def test_pm_expansion():
    # Example 4.13 from Anderson
    fl = isentropic.IsentropicFlow(gamma=1.4)
    pm = isentropic.PrandtlMeyerExpansion(M_1=1.5, nu=np.radians(20), fl=fl)
    np.testing.assert_almost_equal(pm.M_2, 2.207, decimal=3)
    np.testing.assert_almost_equal(pm.p2_p1, 0.340, decimal=3)
    np.testing.assert_almost_equal(pm.T2_T1, 0.735, decimal=3)
    np.testing.assert_almost_equal(pm.mu_1, np.radians(41.81), decimal=3)
    np.testing.assert_almost_equal(pm.mu_2, np.radians(26.95), decimal=3)


def test_pm_raises_error_max_deflection():
    fl = isentropic.IsentropicFlow(gamma=1.4)
    with pytest.raises(ValueError):
        isentropic.PrandtlMeyerExpansion(3.0, np.radians(125), fl)


def test_pm_function_raises_error_subsonic():
    with pytest.raises(ValueError):
        isentropic.PrandtlMeyerExpansion.nu(0.8)


def test_isentropic_flow_constructor():
    gamma = 1.4
    isentropic.IsentropicFlow(gamma)


def test_pressure_ratio():
    fl = isentropic.IsentropicFlow(1.4)
    M_list = [0.0, 0.27, 0.89, 1.0, 1.30, 2.05]
    p_ratio_list = [
        1.0,
        0.9506,
        0.5977,
        0.5283,
        0.3609,
        0.1182
    ]
    np.testing.assert_array_almost_equal(
        fl.p_p0(M_list), p_ratio_list, decimal=4
    )


def test_area_ratio():
    fl = isentropic.IsentropicFlow(1.4)
    M_list = [0.0, 0.38, 0.79, 1.0, 1.24, 2.14]
    A_Astar_list = [
        np.infty,
        1.6587,
        1.0425,
        1.0,
        1.043,
        1.902
    ]
    np.testing.assert_array_almost_equal(
        fl.A_Astar(M_list), A_Astar_list, decimal=3
    )


def test_area_ratio_no_zero_division_error():
    fl = isentropic.IsentropicFlow()
    assert np.isposinf(fl.A_Astar(0))


def test_mach_from_area_ratio_raises_error_ratio_greater_1():
    with pytest.raises(ValueError):
        isentropic.mach_from_area_ratio(0.8)


def test_mach_from_area_ratio_subsonic():
    fl = isentropic.IsentropicFlow(1.4)
    A_Astar_list = [
        np.inf,
        2.4027,
        1.7780,
        1.0382,
        1.0,
    ]
    M_sub_list = [
        0.0,
        0.25,
        0.35,
        0.8,
        1.0
    ]
    results_list = [
        isentropic.mach_from_area_ratio(A_Astar, fl)[0]  # Subsonic
        for A_Astar in A_Astar_list]
    for i in range(len(A_Astar_list)):
        np.testing.assert_almost_equal(
            results_list[i], M_sub_list[i], decimal=3)


def test_mach_from_area_ratio_supersonic():
    fl = isentropic.IsentropicFlow(1.4)
    A_Astar_list = [
        1.0,
        1.043,
        1.328,
        1.902,
        4.441
    ]
    M_sup_list = [
        1.0,
        1.24,
        1.69,
        2.14,
        3.05
    ]
    results_list = [
        isentropic.mach_from_area_ratio(A_Astar, fl)[1]  # Supersonic
        for A_Astar in A_Astar_list]
    for i in range(len(A_Astar_list)):
        np.testing.assert_almost_equal(
            results_list[i], M_sup_list[i], decimal=2)

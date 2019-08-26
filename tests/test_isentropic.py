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
    expected_mach_angles = [np.radians(val) for val in mu_list]
    mach_angles = [isentropic.mach_angle(M) for M in M_list]
    np.testing.assert_array_almost_equal(mach_angles,
                                         expected_mach_angles,
                                         decimal=3)


def test_mach_angle_raises_error_when_mach_is_subsonic():
    with pytest.raises(ValueError) as excinfo:
        isentropic.mach_angle(0.8)
    assert excinfo.exconly().startswith("ValueError: "
                                        "Mach number must be supersonic")


def test_PrandtlMeyerExpansion_angle():
    M_list = [1.2, 1.4, 2.6, 3.2, np.inf]
    nu_list = [
        3.558,
        8.987,
        41.41,
        53.47,
        130.45
    ]
    expected_angles = [np.radians(val) for val in nu_list]
    turn_angles = [
        isentropic.PrandtlMeyerExpansion.nu(M) for M in M_list]
    np.testing.assert_array_almost_equal(turn_angles,
                                         expected_angles,
                                         decimal=3)


def test_default_gamma_for_new_IsentropicFlow():
    _ = 1.0  # Unused value, equals or bigger than one
    pm = isentropic.PrandtlMeyerExpansion(_, _)
    assert pm.fl.gamma == 1.4


def test_PrandtlMeyerExpansion_example():
    # Example 4.13 from Anderson. Default gamma=1.4 used
    fl = isentropic.IsentropicFlow()
    pm = isentropic.PrandtlMeyerExpansion(M_1=1.5, theta=np.radians(20), fl=fl)
    np.testing.assert_almost_equal(pm.M_2, 2.207, decimal=3)
    np.testing.assert_almost_equal(pm.p2_p1, 0.340, decimal=3)
    np.testing.assert_almost_equal(pm.T2_T1, 0.735, decimal=3)
    np.testing.assert_almost_equal(pm.mu_1, np.radians(41.81), decimal=3)
    np.testing.assert_almost_equal(pm.mu_2, np.radians(26.95), decimal=3)


def test_PrandtlMeyerExpansion_raises_error_when_deflection_angle_is_over_the_maximum_and_mach_is_supersonic():
    mach = 3.0
    wrong_angle = np.radians(125)
    with pytest.raises(ValueError) as excinfo:
        isentropic.PrandtlMeyerExpansion(mach, wrong_angle)
    assert excinfo.exconly().startswith("ValueError: Deflection angle must "
                                        "be lower than maximum")


def test_PrandtlMeyerExpansion_raises_error_when_Mach_is_subsonic():
    wrong_mach = 0.9
    with pytest.raises(ValueError) as excinfo:
        isentropic.PrandtlMeyerExpansion.nu(wrong_mach)
    assert excinfo.exconly().startswith("ValueError: Mach number must "
                                        "be supersonic")


def test_isentropic_flow_has_the_gamma_indicated_in_constructor():
    gamma = 1.4
    flow = isentropic.IsentropicFlow(gamma)
    np.testing.assert_almost_equal(flow.gamma, gamma, decimal=3)


def test_pressure_ratio():
    fl = isentropic.IsentropicFlow(1.4)
    M_list = [0.0, 0.27, 0.89, 1.0, 1.30, 2.05]
    expected_pressure_ratios = [
        1.0,
        0.9506,
        0.5977,
        0.5283,
        0.3609,
        0.1182
    ]
    np.testing.assert_array_almost_equal(
        fl.p_p0(M_list), expected_pressure_ratios, decimal=4
    )


def test_area_ratio():
    fl = isentropic.IsentropicFlow(1.4)
    M_list = [0.0, 0.38, 0.79, 1.0, 1.24, 2.14]
    expected_area_ratios = [
        np.infty,
        1.6587,
        1.0425,
        1.0,
        1.043,
        1.902
    ]
    np.testing.assert_array_almost_equal(
        fl.A_Astar(M_list), expected_area_ratios, decimal=3
    )


def test_area_ratio_no_zero_division_error():
    fl = isentropic.IsentropicFlow()
    assert np.isposinf(fl.A_Astar(0))


def test_mach_from_area_ratio_raises_error_when_ratio_is_subsonic():
    with pytest.raises(ValueError):
        isentropic.mach_from_area_ratio(0.9)


def test_speed_of_sound_ratio():
    fl = isentropic.IsentropicFlow(1.4)
    M_list = [0.0, 0.3, 1.0, 1.3, 2.5]
    expected_sound_speed_ratios = [1.0, 0.99112, 0.91287, 0.86451, 0.6667]

    np.testing.assert_array_almost_equal(
        fl.a_a0(M_list), expected_sound_speed_ratios, decimal=3
    )


def test_mach_from_area_ratio_subsonic():
    fl = isentropic.IsentropicFlow(1.4)
    A_Astar_list = [1e4, 2.4027, 1.7780, 1.0382, 1.0]
    expected_ratios = [0.0, 0.25, 0.35, 0.8, 1.0]

    mach_from_area_ratios = [isentropic.mach_from_area_ratio(A_Astar, fl)[0]
                             for A_Astar in A_Astar_list]  # Subsonic

    np.testing.assert_array_almost_equal(mach_from_area_ratios,
                                         expected_ratios, decimal=3)


def test_mach_from_area_ratio_supersonic():
    #  https://www.engineering.com/calculators/isentropic_flow_relations.htm
    fl = isentropic.IsentropicFlow(1.4)
    A_Astar_list = [1.0, 1.043, 1.328, 1.902, 4.441, 1e5]
    expected_ratios = [1.0, 1.24, 1.69, 2.14, 3.05, 29.199]

    mach_from_area_ratios = [isentropic.mach_from_area_ratio(A_Astar, fl)[1]
                             for A_Astar in A_Astar_list]  # Supersonic

    np.testing.assert_array_almost_equal(mach_from_area_ratios,
                                         expected_ratios, decimal=2)


def test_density_ratio():
    fl = isentropic.IsentropicFlow(1.4)
    M_list = [0.0, 0.27, 0.89, 1.0, 1.30, 2.05]
    expected_density_ratios = [1.0,
                               0.96446008,
                               0.69236464,
                               0.63393815,
                               0.48290279,
                               0.21760078]
    density_ratios = fl.rho_rho0(M_list)
    np.testing.assert_array_almost_equal(
        density_ratios, expected_density_ratios, decimal=4
    )

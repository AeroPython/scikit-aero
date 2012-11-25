# coding: utf-8

"""
Tests of the shocks package.

"""

from __future__ import division

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

    Data from http://en.wikipedia.org/wiki/Normal_shock_tables.

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


def test_normal_shock_pressure_ratio():
    """Tests normal shock pressure ratio.

    Data from http://en.wikipedia.org/wiki/Normal_shock_tables.

    """
    M_1_list = [1.5, 1.8, 2.1, 3.0]
    p_ratio_list = [
        2.4583,
        3.6133,
        4.9783,
        10.3333
    ]
    ns_list = [shocks.NormalShock(M_1, 1.4) for M_1 in M_1_list]

    for i in range(len(ns_list)):
        np.testing.assert_almost_equal(
            ns_list[i].p2_p1, p_ratio_list[i], decimal=4
        )

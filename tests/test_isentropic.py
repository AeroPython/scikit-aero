# coding: utf-8

"""
Tests of the isentropic package.

Data from `NACA 1135 Report`_.

.. _`NACA 1135 Report`: http://www.grc.nasa.gov/WWW/BGH/Images/naca1135.pdf

"""

from __future__ import division

import numpy as np
import numpy.testing

import pytest

from skaero.gasdynamics import isentropic


def test_isentropic_flow_constructor():
    """Tests IsentropicFlow constructor.

    """
    gamma = 1.4
    isentropic.IsentropicFlow(gamma)


def test_negative_mach_number_pressure_ratio():
    """Tests if negative Mach number raises an error when computing pressure.

    """
    fl = isentropic.IsentropicFlow()
    with pytest.raises(ValueError):
        fl.p_p0(-1.0)


def test_pressure_ratio():
    """Test pressure ratio.

    """
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
    """Tests area ratio.

    """
    fl = isentropic.IsentropicFlow(1.4)
    M_list = [0.0, 0.38, 0.79, 1.0, 1.24, 2.14]
    A_ratio_list = [
        np.infty,
        1.6587,
        1.0425,
        1.0,
        1.043,
        1.902
    ]
    np.testing.assert_array_almost_equal(
        fl.A_Astar(M_list), A_ratio_list, decimal=3
    )

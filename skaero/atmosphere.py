# coding: utf-8

"""
Atmosphere properties.

Routines
--------
coesa(h)

Examples
--------
from skaero import atmosphere

# Properties at 1 km of altitude according to the U.S. 1976 Standard Atmosphere
h, T, p, rho = atmosphere.coesa(1000)

ToDo
----
* Check geopotential temperature
* Move to OOP

"""

from __future__ import division

import numpy as np

g_0p = 9.80665  # m / s^2
M_0 = 28.9644e-3  # kg / mol
Rs = 8.31432  # N m / (mol K)


def coesa(h):
    """Computes COESA atmosphere properties.

    Returns temperature, pressure and density COESA values at the given
    geopotential altitude.

    Parameters
    ----------
    h : array_like
       Geopotential altitude given in meters.

    Returns
    -------
    h : array_like
        Given geopotential altitude in meters.
    T : array_like
        Temperature in Kelvin.
    p : array_like
        Pressure in Pascal.
    rho : array_like
        Density in kilograms per cubic meter.

    Notes
    -----
    Based on the `U.S. 1976 Standard Atmosphere`_.

    .. _`U.S. 1976 Standard Atmosphere`: http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770009539_1977009539.pdf

    """

    # FIXME: Having these variables here feels like a total hack for me,
    # and will be worse as soon as I add more layers. I need another module,
    # or a class, or something.
    h_0 = 0.0  # m
    T_0 = 288.150  # K
    p_0 = 101325.0  # Pa
    L_0 = -6.5e-3  # K / m

    # Is this actually molecular-scale temperature?
    T = T_0 + L_0 * (h - h_0)  # Linear relation for 0 < h < 11000
    # TODO: Maybe use pressure ratio to be consistent w/ the COESA standard.
    p = p_0 * (T_0 / (T_0 + L_0 * (h - h_0))) ** (g_0p * M_0 / (Rs * L_0))
    rho = p * M_0 / (Rs * T)
    return (
        h,
        T,
        p,
        rho)

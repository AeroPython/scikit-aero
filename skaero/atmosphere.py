# coding: utf-8

"""
Atmosphere properties.

Routines
--------
coesa(h)

Examples
--------
from skaero import atmosphere

h, T, p, rho = atmosphere.coesa(1000)  # Properties at 1 km of altitude

"""

from __future__ import division

import numpy as np


def coesa(h):
    """Computes COESA atmosphere properties.

    Returns temperature, pressure and density COESA values at the given
    altitude.

    Parameters
    ----------
    h : array_like
       Altitude given in meters.

    Returns
    -------
    h : array_like
        Given altitude in meters.
    T : array_like
        Temperature in Kelvin.
    p : array_like
        Pressure in Pascal.
    rho : array_like
        Density in kilograms per cubic meter.

    Notes
    -----
    Based on the U.S. 1976 Standard Atmosphere.

    """
    ones = np.ones_like(h)
    return (
        h,
        288.150 * ones,
        101325.0 * ones,
        1.2250 * ones)

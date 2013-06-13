# coding: utf-8

"""
Utilities for atmospheric calculations.

"""

from __future__ import division, absolute_import

R_Earth = 6369.0e3  # m


def geometric_to_geopotential(z):
    """Returns geopotential altitude from geometric altitude.

    Parameters
    ----------
    z : array_like
        Geometric altitude in meters.

    Returns
    -------
    h : array_like
        Geopotential altitude in meters.

    """
    h = z * R_Earth / (z + R_Earth)
    return h

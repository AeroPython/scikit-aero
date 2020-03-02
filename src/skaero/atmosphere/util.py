# coding: utf-8

"""
Utilities for atmospheric calculations.

"""

from __future__ import absolute_import, division

import numpy as np

# effective earth's radius
R_Earth = 6356.7660e3  # m


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

    h = np.asarray(z) * R_Earth / (np.asarray(z) + R_Earth)
    return h


def geopotential_to_geometric(h):
    """Returns geometric altitude from geopotential altitude.

    Parameters
    ----------
    h : array_like
        Geopotential altitude in meters.


    Returns
    -------
    z : array_like
        Geometric altitude in meters.

    Notes
    -----
    Based on eq. 19 of the `U.S. 1976 Standard Atmosphere`_.

    .. _`U.S. 1976 Standard Atmosphere`: http://ntrs.nasa.gov/search.jsp?R=1977\0009539

    """

    z = np.asarray(h) * R_Earth / (R_Earth - np.asarray(h))
    return z

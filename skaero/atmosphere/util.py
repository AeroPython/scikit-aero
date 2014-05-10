# coding: utf-8

"""
Utilities for atmospheric calculations.

"""

from __future__ import division, absolute_import


def geometric_to_geopotential(z, R_Earth = 6369.0e3):
    """Returns geopotential altitude from geometric altitude.

    Parameters
    ----------
    z : array_like
        Geometric altitude in meters.
        
    R_Earth : array_like, optional
        Effective Earth radius in meters. Default is 6369000.
    
    Returns
    -------
    h : array_like
        Geopotential altitude in meters.

    """
    h = z * R_Earth / (z + R_Earth)
    return h
    
def geopotential_to_geometric(h, R_Earth = 6369.0e3):
    """Returns geometric altitude from geopotential altitude.

    Parameters
    ----------
    h : array_like
        Geopotential altitude in meters.
        
    R_Earth : array_like, optional
        Effective Earth radius in meters. Default is 6369000.
    
    Returns
    -------
    z : array_like
        Geometric altitude in meters.
    
    Notes
    -----
    Based on eq. 19 of the U.S. 1976 Standard Atmosphere.

    .. _`U.S. 1976 Standard Atmosphere`
        
    """
    z = h * R_Earth / (R_Earth - h)
    return z

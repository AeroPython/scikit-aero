# coding: utf-8

"""
.. _U.S. 1976 Standard Atmosphere: http://ntrs.nasa.gov/search.jsp?R=19770009539

COESA model, based on the `U.S. 1976 Standard Atmosphere`_.

"""

from __future__ import absolute_import, division

import numpy as np
from scipy import constants, interpolate

from skaero.atmosphere import util

# Constants and values : the following parameters are extracted from Notes
# reference (mainly Chap. 1.2.). Naming is consistent. WARNING : Some of these
# values are not exactly consistent with the 2010 CODATA Recommended Values of
# the Fundamental Physical constants that you can find for example in the
# scipy.constants module

# gas constant
Rs = 8.31432  # N m / (mol K), WARNING : different from the the 2010 CODATA
# Recommended Values of the Fundamental Physical Constants

# set of geopotential heights from table 2 of Notes reference
H = np.array([0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.00, 84.85205]) * 1e3  # m

# set of molecular-scale temperature gradients from table 2 of Notes reference
LM = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0]) * 1e-3  # K / m

f_LM = interpolate.interp1d(H, LM, kind="zero")

# K, standard sea-level temperature
T_0 = 288.15  # K

# mean molecular-weight at sea-level
M_0 = 28.9644e-3  # kg / mol

# set of geopotential heights from table 8 of Notes reference
H2 = np.array(
    [
        0.0,
        79005.7,
        79493.3,
        79980.8,
        80468.2,
        80955.7,
        81443.0,
        81930.2,
        82417.3,
        82904.4,
        83391.4,
        83878.4,
        84365.2,
        84852.05,
    ]
)  # m

# set of molecular weight ratios from table 8 of Notes reference
M_o_M0 = np.array(
    [
        1.0,
        1.0,
        0.999996,
        0.999989,
        0.999971,
        0.999941,
        0.999909,
        0.999870,
        0.999829,
        0.999786,
        0.999741,
        0.999694,
        0.999641,
        0.999579,
    ]
)  # -

f_M_o_M0 = interpolate.interp1d(H2, M_o_M0)

# set of pressures and temperatures (initialization)
P = np.array([constants.atm])  # Pa
TM = np.array([T_0])  # K

for k in range(1, len(H)):
    # from eq. [23] of Notes reference
    TM = np.append(TM, TM[-1] + f_LM(H[k - 1]) * (H[k] - H[k - 1]))
    if f_LM(H[k - 1]) == 0.0:
        # from eq. [33b] of Notes reference
        P = np.append(
            P, P[-1] * np.exp(-constants.g * M_0 * (H[k] - H[k - 1]) / (Rs * TM[-2]))
        )
    else:
        # from eq. [33a] of Notes reference
        P = np.append(
            P,
            P[-1] * (TM[-2] / (TM[-1])) ** (constants.g * M_0 / (Rs * f_LM(H[k - 1]))),
        )

f_TM = interpolate.interp1d(H, TM, kind="zero")
f_P = interpolate.interp1d(H, P, kind="zero")
f_H = interpolate.interp1d(H, H, kind="zero")


def table(x, kind="geopotential"):
    """Computes table of COESA atmosphere properties.

    Returns temperature, pressure and density COESA values at the given
    altitude.

    Parameters
    ----------
    x : array_like
       Geopotential or geometric altitude (depending on kind) given in meters.
    kind : str
       Specifies the kind of interpolation as altitude x ('geopotential' or 'geometric'). Default is 'geopotential'

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

    Note
    ----
    Based on the U.S. 1976 Standard Atmosphere.

    """

    # check the kind of altitude and raise an exception if necessary
    if kind == "geopotential":
        alt = x
    elif kind == "geometric":
        alt = util.geometric_to_geopotential(x)
    else:
        raise ValueError(
            "%s is unsupported: Use either geopotential or " "geometric." % kind
        )

    h = np.asarray(alt)

    # check if altitude is out of bound and raise an exception if necessary
    if (h < H[0]).any() or (h > H[-1]).any():
        raise ValueError(
            "the given altitude x is out of bound, this module is "
            "currently only valid for a geometric altitude between 0. and 86000. m"
        )

    # K, molecule-scale temperature from eq. [23] of Notes reference
    tm = f_TM(h) + f_LM(h) * (h - f_H(h))

    # K, absolute temperature from eq. [22] of Notes reference
    T = tm * f_M_o_M0(h)

    if h.shape:  # if h is not a 0-d array (like a scalar)
        # Pa, intialization of the pressure vector
        p = np.zeros(len(h))

        # points of h for which the molecular-scale temperature gradient is
        # zero
        zero_gradient = f_LM(h) == 0.0

        # points of h for which the molecular-scale temperature gradient is not
        # zero
        not_zero_gradient = f_LM(h) != 0.0

        # Pa, pressure from eq. [33b] of Notes reference
        p[zero_gradient] = f_P(h[zero_gradient]) * np.exp(
            -constants.g
            * M_0
            * (h[zero_gradient] - f_H(h[zero_gradient]))
            / (Rs * f_TM(h[zero_gradient]))
        )

        # Pa, pressure from eq. [33a] of Notes reference
        p[not_zero_gradient] = f_P(h[not_zero_gradient]) * (
            f_TM(h[not_zero_gradient])
            / (
                f_TM(h[not_zero_gradient])
                + f_LM(h[not_zero_gradient])
                * (h[not_zero_gradient] - f_H(h[not_zero_gradient]))
            )
        ) ** (constants.g * M_0 / (Rs * f_LM(h[not_zero_gradient])))

    else:
        if f_LM(h) == 0:
            # Pa, pressure from eq. [33b] of Notes reference
            p = f_P(h) * np.exp(-constants.g * M_0 * (h - f_H(h)) / (Rs * f_TM(h)))
        else:
            # Pa, pressure from eq. [33a] of Notes reference
            p = f_P(h) * (f_TM(h) / (f_TM(h) + f_LM(h) * (h - f_H(h)))) ** (
                constants.g * M_0 / (Rs * f_LM(h))
            )

    rho = p * M_0 / (Rs * tm)  # kg / m^3, mass density

    return alt, T, p, rho


def temperature(x, kind="geopotential"):
    """Computes air temperature for a given altitude using the U.S. standard atmosphere model

    Parameters
    ----------
    x : array_like
       Geopotential or geometric altitude (depending on kind) given in meters.
    kind : str
       Specifies the kind of interpolation as altitude x ('geopotential' or 'geometric'). Default is 'geopotential'

    Returns
    -------
    T : array_like
        Temperature in Kelvin.

    Note
    ----
    Based on the U.S. 1976 Standard Atmosphere.

    """

    T = table(x, kind)[1]
    return T


def pressure(x, kind="geopotential"):
    """Computes absolute pressure for a given altitude using the U.S. standard atmosphere model.

    Parameters
    ----------
    x : array_like
       Geopotential or geometric altitude (depending on kind) given in meters.
    kind : str
       Specifies the kind of interpolation as altitude x ('geopotential' or 'geometric'). Default is 'geopotential'

    Returns
    -------
    P : array_like
        Pressure in Pascal.

    Note
    ----
    Based on the U.S. 1976 Standard Atmosphere.

    """

    p = table(x, kind)[2]
    return p


def density(x, kind="geopotential"):
    """Computes air mass density for a given altitude using the U.S. standar atmosphere model.

    Parameters
    ----------
    x : array_like
       Geopotential or geometric altitude (depending on kind) given in meters.
    kind : str
       Specifies the kind of interpolation as altitude x ('geopotential' or 'geometric'). Default is 'geopotential'

    Returns
    -------
    rho : array_like
        Density in kilograms per cubic meter.

    Note
    ----
    Based on the U.S. 1976 Standard Atmosphere.

    """

    rho = table(x, kind)[3]
    return rho

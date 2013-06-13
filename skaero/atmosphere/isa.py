# coding: utf-8

"""
ISA Standard Atmosphere.

TODO: This is a simple and direct implementation of the troposphere which
needs improving regarding the addition of new layers.
TODO: Pressure

"""

from __future__ import division, absolute_import

import numpy as np

R = 287.058  # J / kg K
gamma = 1.4
T0 = 288.15  # K
rho0 = 1.225  # kg / m^3
lambda_isa = -6.5e-3  # K / m
g0 = 9.81  # m / s^2


def T(h):
    return T0 + lambda_isa * h


def rho(h):
    return rho0 / (1 + lambda_isa / T0 * h) ** (1 + g0 / R / lambda_isa)


# TODO: Does this belong to here?
def a(h):
    return np.sqrt(gamma * R * T(h))

# coding: utf-8

"""
Airspeed calculations.

Assumes ISA Standard Atmosphere.

"""

from __future__ import division, absolute_import

import numpy as np

from skaero.atmosphere import isa


def TAS2EAS(TAS, h):
    """Converts TAS to EAS.

    """
    EAS = np.sqrt(isa.rho(h) / isa.rho0) * TAS
    return EAS


def EAS2TAS(EAS, h):
    """Converts EAS to TAS.

    """
    TAS = np.sqrt(isa.rho0 / isa.rho(h)) * EAS
    return TAS

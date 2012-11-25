# coding: utf-8

"""
Shock waves.

Classes
-------
NormalShock(M_1, gamma)

Examples
--------
>>> from skaero.gasdynamics import shocks
>>> ns = shocks.NormalShock(1.4, 2.0)

"""

from __future__ import division

import numpy as np


class NormalShock(object):
    """Class representing a normal shock.

    Parameters
    ----------
    M_1 : float
        Incident Mach number.
    gamma : float, optional
        Specific heat ratio, default 7 / 5.

    Raises
    ------
    ValueError
        If the incident Mach number is less than one.

    """
    def __init__(self, M_1, gamma=1.4):
        if M_1 < 1.0:
            raise ValueError("Incident Mach number must be supersonic.")

        self.gamma = gamma
        self.M_1 = M_1
        pass

    @property
    def M_2(self):
        """Mach number behind the shock.

        """
        M_2 = np.sqrt(
            (1 + (self.gamma - 1) * self.M_1 * self.M_1 / 2) /
            (self.gamma * self.M_1 * self.M_1 - (self.gamma - 1) / 2)
        )
        return M_2

    @property
    def p2_p1(self):
        """Pressure ratio across the shock.

        Resultant pressure over incident pressure.

        """
        p2_p1 = (
            1 + 2 * self.gamma *
            (self.M_1 * self.M_1 - 1) / (self.gamma + 1)
        )
        return p2_p1

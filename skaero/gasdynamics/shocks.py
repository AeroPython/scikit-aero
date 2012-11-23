# coding: utf-8

"""
Shock waves.

"""

from __future__ import division

import numpy as np

class NormalShock(object):
    """Class representing a normal shock.

    """
    def __init__(self, M_1, gamma=1.4):
        """Constructor.

        Arguments
        ---------
        M_1 : float
            Incident Mach number.
        gamma : float, optional
            Specific heat ratio, default 7 / 5.

        Raises
        ------
        ValueError
            If the incident Mach number is less than one.

        """
        if M_1 <= 1.0:
            raise ValueError("Incident Mach number must be greater than 1.0")

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
    def p_ratio(self):
        """Pressure ratio across the shock.

        Resultant pressure over incident pressure.

        """
        p_ratio = 1 + 2 * self.gamma * (self.M_1 * self.M_1 - 1) / (self.gamma + 1)
        return p_ratio

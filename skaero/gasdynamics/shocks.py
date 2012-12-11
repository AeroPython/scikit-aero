# coding: utf-8

"""
Shock waves.

Classes
-------
NormalShock(M_1, gamma)
ObliqueShock(M_1, beta, gamma)

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


class ObliqueShock(object):
    """Class representing an oblique shock.

    Parameters
    ----------
    M_1 : float
        Incident Mach number.
    beta : float
        Shock wave angle, with respect to the incident velocity, in radians.
    strong : boolean, optional
        Parameter to specify strong shock solution, default to False
        (weak shock).
    gamma : float, optional
        Specific heat ratio, default 7 / 5.

    Raises
    ------
    ValueError
        If the incident Mach number is less than one.

    TODO
    ----
    > Although the M_1 - theta approach seems to be the "right" way to
    characterize an oblique shock (because we usually know the corner
    angle and we want to compute the shock wave angle) this can be
    computationally cumbersone, because

    * There is the double-valued thing.
    * There is the max angle thing (which involves computing it first).

    _Nevertheless_, why not characterizing an oblique shock by its angle
    and create a function that _returns_ an ObliqueShock given a corner
    angle? That would be so awesome.

    I need some tests and example cases before that.

    > On the other hand, a normal shock is a special case of an oblique
    shock (beta = pi / 2), so maybe I could do a nice subclassing here.

    """
    def __init__(self, M_1, beta, gamma=1.4):
        M_1n = M_1 * np.sin(beta)
        if M_1n < 1.0:
            raise ValueError("Normal incident Mach number must be supersonic.")
        self.M_1 = M_1
        self.M_1n = M_1n
        self.beta = beta
        self.gamma = gamma

    @property
    def theta(self):
        """Deflection angle of the shock.

        """
        tan_theta = (
            2 / np.tan(self.beta) *
            (self.M_1 * self.M_1 * np.sin(self.beta) ** 2 - 1) /
            (self.M_1 * self.M_1 * (self.gamma + np.cos(2 * self.beta)) + 2)
        )
        theta = np.arctan(tan_theta)
        return theta, tan_theta

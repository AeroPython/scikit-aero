# coding: utf-8

"""
Shock waves.

Routines
--------
from_deflection_angle(M_1, theta, weak=True, gamma=1.4)
max_deflection(M_1, gamma=1.4)

Classes
-------
NormalShock(M_1, gamma)
ObliqueShock(M_1, beta, gamma)

Examples
--------
>>> from skaero.gasdynamics import shocks
>>> ns = shocks.NormalShock(2.0, gamma=1.4) # Normal shock with M_1 = 2.0
>>> shocks.from_deflection_angle(3.0, np.radians(25), weak=True)

"""

from __future__ import division, absolute_import

import numpy as np
import scipy as sp
import scipy.optimize

from skaero.gasdynamics.isentropic import mach_angle


# Exceptions used in this module
class InvalidParametersError(Exception): pass


def max_deflection(M_1, gamma=1.4):
    """Returns maximum deflection angle and corresponding wave angle for given
    Mach number.

    Parameters
    ----------
    M_1 : float
        Upstream Mach number.
    gamma : float, optional
        Specific heat ratio, default 7 / 5.

    Returns
    -------
    theta : float
        Maximum deflection angle.
    beta : float
        Corresponding wave angle.

    """
    def eq(beta, M_1, gamma):
        os = _ShockClass(M_1, beta, gamma)
        return -os.theta

    mu = mach_angle(M_1)
    beta_theta_max = sp.optimize.fminbound(
        eq, mu, np.pi / 2, args=(M_1, gamma), disp=0)
    os = _ShockClass(M_1, beta_theta_max, gamma)
    return os.theta, os.beta


def _ShockFactory(**kwargs):
    """Returns an object representing a shock wave.

    Examples
    --------
    >>> ss1 = Shock(M_1=1.5)  # Given upstream Mach number (default beta = 90°)
    >>> ss1.M_2
    0.70108874169309943
    >>> ss1.beta
    1.5707963267948966
    >>> ss1.theta
    0.0
    >>> ss2 = Shock(M_1=3.0, theta=np.radians(20.0), weak=True)
    >>> ss2.beta  # Notice it is an oblique shock
    0.6590997534071927

    """
    methods = {
        frozenset(['M_1']): _ShockClass,
        frozenset(['M_1', 'theta']): _from_deflection_angle,
        frozenset(['M_1', 'theta', 'weak']): _from_deflection_angle
    }
    beta = kwargs.pop('beta', np.pi / 2)
    gamma = kwargs.pop('gamma', 1.4)
    params = frozenset(kwargs.keys())
    try:
        shock = methods[params](beta=beta, gamma=gamma, **kwargs)
    except KeyError:
        raise InvalidParametersError("Invalid list of parameters")
    return shock


Shock = _ShockFactory


def _from_deflection_angle(M_1, theta, weak=True, gamma=1.4, **kwargs):
    """Returns oblique shock given upstream Mach number and deflection angle.

    """
    def eq(beta, M_1, theta, gamma):
        os = _ShockClass(M_1, beta, gamma)
        return os.theta - theta

    theta_max, beta_theta_max = max_deflection(M_1)
    if theta > theta_max:
        raise ValueError("No attached solution for this deflection angle")
    else:
        if weak:
            mu = mach_angle(M_1)
            beta = sp.optimize.bisect(
                eq, mu, beta_theta_max, args=(M_1, theta, gamma))
        else:
            beta = sp.optimize.bisect(
                eq, beta_theta_max, np.pi / 2, args=(M_1, theta, gamma))

    return _ShockClass(M_1, beta, gamma)


class _ShockClass(object):
    """Class representing a shock.

    """
    def __init__(self, M_1, beta, gamma=1.4):
        mu = mach_angle(M_1)
        if beta < mu:
            raise ValueError(
                "Shock wave angle must be higher than Mach angle {:.2f}°"
                .format(np.degrees(mu)))

        self.M_1 = M_1
        self.M_1n = M_1 * np.sin(beta) if beta != 0.0 else 0.0
        self.beta = beta
        self.gamma = gamma

    def __repr__(self):
        # FIXME: What if the object is returned from different parameters?
        return ("Shock(M_1={0!r}, beta={1!r}, "
                "gamma={2!r})".format(self.M_1, self.beta, self.gamma))

    @property
    def theta(self):
        """Deflection angle of the shock.

        """
        if self.beta == 0.0 or self.beta == np.pi / 2:
            theta = 0.0
        else:
            theta = np.arctan(
                2 / np.tan(self.beta) *
                (np.sin(self.beta) ** 2 - 1 / self.M_1 / self.M_1) /
                (self.gamma + np.cos(2 * self.beta) + 2 / self.M_1 / self.M_1))
        return theta

    @property
    def M_2n(self):
        """Normal Mach number behind the shock.

        """
        M_2n = np.sqrt(
            (1 / (self.M_1n * self.M_1n) + (self.gamma - 1) / 2) /
            (self.gamma - (self.gamma - 1) / 2 / (self.M_1n * self.M_1n)))
        return M_2n

    @property
    def M_2(self):
        """Mach number behind the shock.

        """
        M_2 = self.M_2n / np.sin(self.beta - self.theta)
        return M_2

    @property
    def p2_p1(self):
        """Pressure ratio across the shock.

        """
        p2_p1 = (
            1 + 2 * self.gamma *
            (self.M_1n * self.M_1n - 1) / (self.gamma + 1))
        return p2_p1

    @property
    def rho2_rho1(self):
        """Density ratio accross the shock.

        """
        rho2_rho1 = (
            (self.gamma + 1) /
            (2 / (self.M_1n * self.M_1n) + self.gamma - 1))
        return rho2_rho1

    @property
    def T2_T1(self):
        """Temperature ratio accross the shock.

        """
        T2_T1 = self.p2_p1 / self.rho2_rho1
        return T2_T1

# coding: utf-8

"""
Shock waves.

The following notation is used:

* `X_1` property X in front of the shock.
* `X_2` property X behind the shock.
* `X2_X1` ratio between `X_2` and `X_1`.

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
import scipy as sp
import scipy.optimize


def from_deflection_angle(M_1, theta, weak=True, gamma=1.4):
    """Returns oblique shock given incident Mach number and deflection angle.

    By default weak solution is selected, unless weak=False is provided.

    """
    def eq(beta, M_1, theta, gamma):
        os = ObliqueShock(M_1, beta, gamma)
        return os.theta - theta

    theta_max, beta_theta_max = max_deflection(M_1)
    if theta > theta_max:
        raise ValueError("No solution for this deflection angle")
    else:
        if weak:
            mu = np.arcsin(1 / M_1)
            beta = sp.optimize.bisect(
                eq, mu, beta_theta_max, args=(M_1, theta, gamma))
        else:
            beta = sp.optimize.bisect(
                eq, beta_theta_max, np.pi / 2, args=(M_1, theta, gamma))

    return ObliqueShock(M_1, beta, gamma)


def max_deflection(M_1, gamma=1.4):
    """Returns maximum deflection angle and corresponding wave angle for given
    Mach number.

    """
    def eq(beta, M_1, gamma):
        os = ObliqueShock(M_1, beta, gamma)
        return -os.theta

    beta_0 = np.radians(50)
    beta_theta_max, = sp.optimize.fmin(eq, beta_0, args=(M_1, gamma))
    os = ObliqueShock(M_1, beta_theta_max, gamma)
    return os.theta, os.beta


class ObliqueShock(object):
    """Class representing an oblique shock.

    Parameters
    ----------
    M_1 : float
        Incident Mach number.
    beta : float
        Shock wave angle, with respect to the incident velocity, in radians.
    gamma : float, optional
        Specific heat ratio, default 7 / 5.

    Raises
    ------
    ValueError
        If the incident Mach number is less than one.

    """
    def __init__(self, M_1, beta, gamma=1.4):
        with np.errstate(invalid='raise'):
            try:
                mu = np.arcsin(1.0 / M_1)
                if beta < mu:
                    raise ValueError(
                        "Shock wave angle must be higher "
                        "than Mach angle {:.2f}°".format(np.degrees(mu)))
            except FloatingPointError:
                raise ValueError("Incident Mach number must be supersonic")

        self.M_1 = M_1
        self.M_1n = M_1 * np.sin(beta)
        self.beta = beta
        self.gamma = gamma

    @property
    def theta(self):
        """Deflection angle of the shock.

        """
        theta = np.arctan(
            2 / np.tan(self.beta) *
            (np.sin(self.beta) ** 2 - 1 / (self.M_1 * self.M_1)) /
            (self.gamma + np.cos(2 * self.beta) + 2 / (self.M_1 * self.M_1))
        )
        return theta

    @property
    def M_2n(self):
        """Normal Mach number behind the shock.

        """
        M_2n = np.sqrt(
            (1 / (self.M_1n * self.M_1n) + (self.gamma - 1) / 2) /
            (self.gamma - (self.gamma - 1) / 2 / (self.M_1n * self.M_1n))
        )
        return M_2n

    @property
    def M_2(self):
        """Mach number behind the shock.

        """
        M_2 = self.M_2n / np.sin(self.beta - self.theta)
        return M_2

    @property
    def rho2_rho1(self):
        """Density ratio accross the shock.

        """
        rho2_rho1 = (
            (self.gamma + 1) /
            (2 / (self.M_1n * self.M_1n) + self.gamma - 1)
        )
        return rho2_rho1

    @property
    def p2_p1(self):
        """Pressure ratio across the shock.

        """
        p2_p1 = (
            1 + 2 * self.gamma *
            (self.M_1n * self.M_1n - 1) / (self.gamma + 1)
        )
        return p2_p1

    @property
    def T2_T1(self):
        """Temperature ratio accross the shock.

        """
        T2_T1 = self.p2_p1 / self.rho2_rho1
        return T2_T1


class NormalShock(ObliqueShock):
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
        super(NormalShock, self).__init__(M_1, np.pi / 2, gamma)

    @property
    def theta(self):
        """Deflection angle of the shock.

        """
        return 0.0

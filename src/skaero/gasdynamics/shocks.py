# coding: utf-8

"""
Shock waves.
"""

from __future__ import absolute_import, division

import inspect

import numpy as np
from scipy import optimize

from skaero.gasdynamics.isentropic import IsentropicFlow, mach_angle


# Exceptions used in this module
class InvalidParametersError(Exception):
    pass


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
    beta_theta_max = optimize.fminbound(eq, mu, np.pi / 2, args=(M_1, gamma), disp=0)
    os = _ShockClass(M_1, beta_theta_max, gamma)
    return os.theta, os.beta


def _ShockFactory(**kwargs):
    """Returns an object representing a shock wave.

    Parameters
    ----------
    gamma : float, optional
        Specific heat ratio, default 7 / 5.

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

    TODO
    ----
    REMOVE MAGIC
    * This is a list of possible cases

      * M_1(, beta=np.pi / 2) -> _ShockClass(M_1, beta) (only direct case)
      * M_2(, beta=np.pi / 2)
      * p2_p1(, beta=np.pi / 2)
      * ...
      * M_1, theta(, weak=True)
      * M_2, theta(, weak=True)
      * p2_p1, theta(, weak=True)

    """
    kwargs.setdefault("gamma", 1.4)
    try:
        # We want a view of the keys, but the syntax changed in Python 3
        params = kwargs.viewkeys()
    except AttributeError:
        params = kwargs.keys()
    if "theta" not in params:
        kwargs.setdefault("beta", np.pi / 2)
        # ['X', 'beta', 'gamma']
        if len(params) != 3:
            raise InvalidParametersError("Invalid list of parameters")
    else:
        if "beta" in params:
            raise InvalidParametersError("Invalid list of parameters")
        kwargs.setdefault("weak", True)
        # ['X', 'theta', 'weak', 'gamma']
        if len(params) != 4:
            raise InvalidParametersError("Invalid list of parameters")

    # Here is the list of available resolution methods
    methods_list = [_from_deflection_angle]

    # And we generate a dictionary from it, indexed by their call arguments
    methods = {frozenset(inspect.getargspec(f)[0]): f for f in methods_list}
    # HACK, see http://stackoverflow.com/a/3999604/554319
    _k_class = frozenset(inspect.getargspec(_ShockClass.__init__)[0]) - set(["self"])
    methods[_k_class] = _ShockClass
    try:
        call_sig = frozenset(params)
        shock = methods[call_sig](**kwargs)
    except KeyError:
        raise NotImplementedError
    return shock


Shock = _ShockFactory


def _from_deflection_angle(M_1, theta, weak, gamma):
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
            beta = optimize.bisect(eq, mu, beta_theta_max, args=(M_1, theta, gamma))
        else:
            beta = optimize.bisect(
                eq, beta_theta_max, np.pi / 2, args=(M_1, theta, gamma)
            )

    return _ShockClass(M_1, beta, gamma)


class _ShockClass(object):
    """Class representing a shock.

    """

    def __init__(self, M_1, beta, gamma):
        mu = mach_angle(M_1)
        if beta < mu:
            raise ValueError(
                "Shock wave angle must be higher than Mach angle {:.2f}°".format(
                    np.degrees(mu)
                )
            )

        self.M_1 = M_1
        self.M_1n = M_1 * np.sin(beta) if beta != 0.0 else 0.0
        self.beta = beta
        self.gamma = gamma

    def __repr__(self):
        # FIXME: What if the object is returned from different parameters?
        return "Shock(M_1={0!r}, beta={1!r}, " "gamma={2!r})".format(
            self.M_1, self.beta, self.gamma
        )

    @property
    def theta(self):
        """Deflection angle of the shock.

        """
        if self.beta == mach_angle(self.M_1) or self.beta == np.pi / 2:
            theta = 0.0
        else:
            theta = np.arctan(
                2
                / np.tan(self.beta)
                * (np.sin(self.beta) ** 2 - 1 / self.M_1 / self.M_1)
                / (self.gamma + np.cos(2 * self.beta) + 2 / self.M_1 / self.M_1)
            )
        return theta

    @property
    def M_2n(self):
        """Normal Mach number behind the shock.

        FIXME: Raises ZeroDivisionError if M_1n == 0.0. Consistent?
        """
        M_2n = np.sqrt(
            (1 / (self.M_1n * self.M_1n) + (self.gamma - 1) / 2)
            / (self.gamma - (self.gamma - 1) / 2 / (self.M_1n * self.M_1n))
        )
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
        p2_p1 = 1 + 2 * self.gamma * (self.M_1n * self.M_1n - 1) / (self.gamma + 1)
        return p2_p1

    @property
    def rho2_rho1(self):
        """Density ratio accross the shock.

        """
        rho2_rho1 = (self.gamma + 1) / (2 / (self.M_1n * self.M_1n) + self.gamma - 1)
        return rho2_rho1

    @property
    def T2_T1(self):
        """Temperature ratio accross the shock.

        """
        T2_T1 = self.p2_p1 / self.rho2_rho1
        return T2_T1

    @property
    def p02_p01(self, fl=None):
        """Stagnation pressure ratio across the shock.

        Parameters
        ----------
        M_1 : float, optional
            Mach number behind the shock.
        fl : IsentropicFlow, optional
            To calculate stagnation conditions behind and after the shock.

        Returns
        -------
        p02_p01 : float
            Stagnation pressure ratio.

        """
        if fl is None:
            fl = IsentropicFlow(gamma=self.gamma)

        p02_p01 = fl.p_p0(self.M_1) / fl.p_p0(self.M_2) * self.p2_p1
        return p02_p01

    @property
    def rho02_rho01(self, fl=None):
        """Stagnation density ratio across the shock."""
        if fl is None:
            fl = IsentropicFlow(gamma=self.gamma)

        rho02_rho01 = fl.rho_rho0(self.M_1) / fl.rho_rho0(self.M_2) * self.rho2_rho1

        return rho02_rho01

    @property
    def T02_T01(self):
        """Stagnation temperature ratio across the shock."""
        T02_T01 = self.p02_p01 / self.rho02_rho01
        return T02_T01

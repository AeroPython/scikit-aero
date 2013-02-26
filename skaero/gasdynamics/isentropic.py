# coding: utf-8

"""
Isentropic relations.

Routines
--------
mach_from_area_ratio(fl, A_ratio)

Classes
-------
IsentropicFlow(gamma)

Examples
--------
>>> from skaero.gasdynamics import isentropic
>>> fl = Isentropic(gamma=1.4)
>>> _, M = isentropic.mach_from_area_ratio(fl, 2.5)

"""

from __future__ import division, absolute_import

import numpy as np
import scipy as sp
import scipy.optimize

from skaero import decorators


def mach_from_area_ratio(fl, A_ratio):
    """Computes the Mach number given an area ratio asuming isentropic flow.

    Uses the relation between Mach number and area ratio for isentropic flow,
    and returns both the subsonic and the supersonic solution.

    Parameters
    ----------
    fl : IsentropicFlow
        Isentropic flow object.
    A_ratio : float
        Cross sectional area.

    Returns
    -------
    out : tuple of floats
        Subsonic and supersonic Mach number solution of the equation.

    Raises
    ------
    ValueError
        If the area ratio is less than 1.0 (the critical area is always the
        minimum).

    """
    def eq(M, fl, A_ratio):
        return fl.A_ratio(M) - A_ratio

    if A_ratio < 1.0:
        raise ValueError("Area ratio must be greater than 1")
    elif A_ratio == 1.0:
        M_sub = M_sup = 1.0
    else:
        M_sub = sp.optimize.bisect(eq, 0.0, 1.0, args=(fl, A_ratio))
        M_sup = sp.optimize.newton(eq, 2.0, args=(fl, A_ratio))

    return M_sub, M_sup


class IsentropicFlow(object):
    """Class representing an isentropic flow.

    Parameters
    ----------
    gamma : float, optional
        Specific heat ratio, default 7 / 5.

    """
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    @decorators.arrayize
    def p_p0(self, M):
        """Pressure ratio from Mach number.

        Parameters
        ----------
        M : array_like
            Mach number.

        Returns
        -------
        p_p0 : array_like
            Pressure ratio.

        """
        p_p0 = self.T_T0(M) ** (self.gamma / (self.gamma - 1))
        return p_p0

    @decorators.arrayize
    def rho_rho0(self, M):
        """Density ratio from Mach number.

        Parameters
        ----------
        M : array_like
            Mach number.

        Returns
        -------
        rho_rho0 : array_like
            Density ratio.

        """
        rho_rho0 = self.T_T0(M) ** (1 / (self.gamma - 1))
        return rho_rho0

    @decorators.arrayize
    def T_T0(self, M):
        """Temperature ratio from Mach number.

        Parameters
        ----------
        M : array_like
            Mach number.

        Returns
        -------
        T_T0 : array_like
            Temperature ratio.

        """
        T_T0 = 1 / (1 + (self.gamma - 1) * M * M / 2)
        return T_T0

    @decorators.arrayize
    def A_ratio(self, M):
        """Area ratio from Mach number.

        Duct area divided by critial area given Mach number.

        Parameters
        ----------
        M : array_like
            Mach number.

        Returns
        -------
        A_ratio : array_like
            Area ratio.

        """
        # If there is any zero entry, NumPy array division gives infinity,
        # which is correct.
        with np.errstate(divide='ignore'):
            A_ratio = (
                (2 / self.T_T0(M) / (self.gamma + 1)) **
                ((self.gamma + 1) / (2 * (self.gamma - 1))) / M
            )
        return A_ratio


class PrandtlMeyerExpansion(object):
    """Class representing a Prandtl-Meyer expansion fan.

    Parameters
    ----------
    gamma : float, optional
        Specific heat ratio, default 7 / 5.

    """
    pass

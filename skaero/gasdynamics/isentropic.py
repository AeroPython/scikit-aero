# coding: utf-8

"""
Isentropic relations.

Routines
--------
mach_from_area_ratio(fl, A_ratio)

Classes
-------
IsentropicFlow(gamma)

"""

from __future__ import division

import numpy as np
import scipy as sp
import scipy.optimize


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
        result = fl.A_Astar(M) - A_ratio
        return result

    if A_ratio < 1.0:
        raise ValueError("Area ratio must be greater than 1.")
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

    def T_T0(self, M):
        """Temperature ratio from Mach number.

        Static tempeature divided by stagnation temperature at the point with
        given Mach number.

        Arguments
        ---------
        M : array_like
            Mach number.

        Returns
        -------
        T_T0 : array_like
            Temperature ratio.

        """
        M_ = np.asanyarray(M)
        if np.any(M_ < 0.0):
            raise ValueError("Mach number must be positive.")

        T_T0 = 1 / (1 + (self.gamma - 1) * M_ * M_ / 2)
        return T_T0

    def p_p0(self, M):
        """Pressure ratio from Mach number.

        Static pressure divided by stagnation pressure at the point with given
        Mach number.

        Arguments
        ---------
        M : array_like
            Mach number.

        Returns
        -------
        p_p0 : array_like
            Pressure ratio.

        """
        M_ = np.asanyarray(M)
        if np.any(M_ < 0.0):
            raise ValueError("Mach number must be positive.")

        p_p0 = (
            (1 + (self.gamma - 1) * M_ * M_ / 2) **
            (self.gamma / (1 - self.gamma))
        )
        return p_p0

    def A_Astar(self, M):
        """Area ratio from Mach number.

        Duct area divided by critial area given Mach number.

        Arguments
        ---------
        M : array_like
            Mach number.

        Returns
        -------
        A_Astar : array_like
            Area ratio.

        """
        M_ = np.asanyarray(M)
        if np.any(M_ < 0.0):
            raise ValueError("Mach number must be positive.")

        # If there is any zero entry, NumPy array division gives infnity,
        # which is correct.
        A_Astar = (
            (2 * (1 + (self.gamma - 1) * M_ * M_ / 2) / (self.gamma + 1)) **
            ((self.gamma + 1) / (2 * (self.gamma - 1))) / M
        )
        return A_Astar

# coding: utf-8

"""
Isentropic relations.

Routines
--------
mach_angle(M)
prandtl_meyer_function(M, fl=None)
mach_from_area_ratio(fl, A_Astar)

Classes
-------
IsentropicFlow(gamma)
PrandtlMeyerExpansion(M_1, nu, fl=None)

Examples
--------
>>> from skaero.gasdynamics import isentropic
>>> fl = IsentropicFlow(gamma=1.4)
>>> _, M = isentropic.mach_from_area_ratio(2.5, fl)

"""

from __future__ import division, absolute_import

import numpy as np
import scipy as sp
import scipy.optimize

from skaero.utils.decorators import implicit, method_decorator


def mach_angle(M):
    """Returns Mach angle given supersonic Mach number.

    Parameters
    ----------
    M : float
        Mach number.

    Returns
    -------
    mu : float
        Mach angle.

    Raises
    ------
    ValueError
        If given Mach number is subsonic.

    """
    try:
        with np.errstate(invalid="raise"):
            mu = np.arcsin(1 / M)
    except FloatingPointError:
        raise ValueError("Mach number must be supersonic")
    return mu


def mach_from_area_ratio(A_Astar, fl=None):
    """Computes the Mach number given an area ratio asuming isentropic flow.

    Uses the relation between Mach number and area ratio for isentropic flow,
    and returns both the subsonic and the supersonic solution.

    Parameters
    ----------
    A_Astar : float
        Cross sectional area.
    fl : IsentropicFlow, optional
        Isentropic flow object, default flow with gamma = 7 / 5.

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
    if not fl:
        fl = IsentropicFlow(gamma=1.4)
    eq = implicit(fl.A_Astar)
    if A_Astar < 1.0:
        raise ValueError("Area ratio must be greater than 1")
    elif A_Astar == 1.0:
        M_sub = M_sup = 1.0
    else:
        M_sub = sp.optimize.bisect(eq, 0.0, 1.0, args=(A_Astar,))
        M_sup = sp.optimize.newton(eq, 2.0, args=(A_Astar,))

    return M_sub, M_sup


class IsentropicFlow(object):
    """Class representing an isentropic flow.

    """
    def __init__(self, gamma=1.4):
        """Constructor of IsentropicFlow.

        Parameters
        ----------
        gamma : float, optional
            Specific heat ratio, default 7 / 5.

        """
        self.gamma = gamma

    @method_decorator(np.vectorize)
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

    @method_decorator(np.vectorize)
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

    @method_decorator(np.vectorize)
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

    @method_decorator(np.vectorize)
    def A_Astar(self, M):
        """Area ratio from Mach number.

        Duct area divided by critial area given Mach number.

        Parameters
        ----------
        M : array_like
            Mach number.

        Returns
        -------
        A_Astar : array_like
            Area ratio.

        """
        # If there is any zero entry, NumPy array division gives infinity,
        # which is correct.
        with np.errstate(divide='ignore'):
            A_Astar = (
                (2 / self.T_T0(M) / (self.gamma + 1)) **
                ((self.gamma + 1) / (2 * (self.gamma - 1))) / M
            )
        return A_Astar


class PrandtlMeyerExpansion(object):
    """Class representing a Prandtl-Meyer expansion fan.

    """
    @staticmethod
    def nu(M, gamma=1.4):
        """Turn angle given Mach number.

        The result is given by evaluating the Prandtl-Meyer function.

        Parameters
        ----------
        M : float
            Mach number.
        gamma : float, optional
            Specific heat ratio, default 7 / 5.

        Returns
        -------
        nu : float
            Turn angle, in radians.

        Raises
        ------
        ValueError
            If Mach number is subsonic.

        """
        try:
            with np.errstate(invalid="raise"):
                sgpgm = np.sqrt((gamma + 1) / (gamma - 1))
                nu = (
                    sgpgm * np.arctan(np.sqrt(M * M - 1) / sgpgm) -
                    np.arctan(np.sqrt(M * M - 1)))
        except FloatingPointError:
            raise ValueError("Mach number must be supersonic")
        return nu

    def __init__(self, M_1, nu, fl=None):
        """Constructor of PrandtlMeyerExpansion.

        Parameters
        ----------
        M_1 : float
            Upstream Mach number.
        nu : float
            Deflection angle, in radians.
        fl : IsentropicFlow, optional.
            Flow to be expanded, default flow with gamma = 7 / 5.

        Raises
        ------
        ValueError
            If given Mach number is subsonic.

        """
        if not fl:
            fl = IsentropicFlow(gamma=1.4)
        nu_max = (
            PrandtlMeyerExpansion.nu(np.inf, fl.gamma) -
            PrandtlMeyerExpansion.nu(M_1, fl.gamma))
        if nu > nu_max:
            raise ValueError(
                "Deflection angle must be lower than maximum {:.2f}°"
                .format(np.degrees(nu_max)))
        self.M_1 = M_1
        self.nu = nu
        self.fl = fl

    @property
    def M_2(self):
        """Downstream Mach number.

        """
        def eq(M, nu, gamma):
            return PrandtlMeyerExpansion.nu(M, gamma) - nu

        nu_2 = self.nu + PrandtlMeyerExpansion.nu(self.M_1, self.fl.gamma)
        M_2 = sp.optimize.newton(eq, self.M_1, args=(nu_2, self.fl.gamma))
        return M_2

    @property
    def mu_1(self):
        """Angle of forward Mach line.

        """
        return mach_angle(self.M_1)

    @property
    def mu_2(self):
        """Angle of rearward Mach line.

        """
        return mach_angle(self.M_2)

    @property
    def p2_p1(self):
        """Pressure ratio across the expansion fan.

        """
        p2_p1 = self.fl.p_p0(self.M_2) / self.fl.p_p0(self.M_1)
        return p2_p1

    @property
    def T2_T1(self):
        """Temperature ratio across the expansion fan.

        """
        T2_T1 = self.fl.T_T0(self.M_2) / self.fl.T_T0(self.M_1)
        return T2_T1

""" Isentropic properties. """

from __future__ import absolute_import, division

import numpy as np
import scipy as sp
from scipy.optimize import bisect, newton

from skaero.util.decorators import implicit


def mach_angle(M):
    r"""Returns Mach angle given supersonic Mach number.

    .. math::

        \mu = \arcsin{\left ( \frac{1}{M} \right )}

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
        M_sub = bisect(eq, 0.0, 1.0, args=(A_Astar,))
        M_sup = newton(eq, 2.0, args=(A_Astar,))

    return M_sub, M_sup


def mach_from_nu(nu, in_radians=True, gamma=1.4):
    r"""Computes the Mach number given a Prandtl-Meyer angle, :math:`\nu`.

    Uses the relation between Mach number and Prandtl-Meyer angle for
    isentropic flow, to iteratively compute and return the Mach number.

    Parameters
    ----------
    nu : float
        Prandtl-Meyer angle, by default in radians.
    in_radians : bool, optional
        When set as False, converts nu from degrees to radians.
    gamma : float, optional
        Specific heat ratio.

    Returns
    -------
    M : float
        Mach number corresponding to :math:`\nu`.

    Raises
    ------
    ValueError
        If :math:`\nu` is 0 or negative or above the theoretical maxima based on
        :math:`\gamma`.

    """
    if not in_radians:
        nu = np.radians(nu)

    nu_max = np.pi / 2.0 * (np.sqrt((gamma + 1.0) / (gamma - 1.0)) - 1)
    if nu <= 0.0 or nu >= nu_max:
        raise ValueError(
            "Prandtl-Meyer angle must be between (0, %f) radians." % nu_max
        )

    eq = implicit(PrandtlMeyerExpansion.nu)
    M = newton(eq, 2.0, args=(nu,))

    return M


class IsentropicFlow(object):
    """Class representing an isentropic gas flow.

    Isentropic flow is characterized by:

    * Viscous and heat conductivity effects are negligible.
    * No chemical or radioactive heat production.

    """

    def __init__(self, gamma=1.4):
        """Constructor of IsentropicFlow.

        Parameters
        ----------
        gamma : float, optional
            Specific heat ratio, default 7 / 5.

        """
        self.gamma = gamma

    def p_p0(self, M):
        r"""Pressure ratio from Mach number.

        .. math::
            \left ( \frac{P}{P_{0}} \right ) = \left ( \frac{T}{T_{0}} \right )^{\frac{\gamma}{(\gamma - 1)}}

        Parameters
        ----------
        M : array_like
            Mach number.

        Returns
        -------
        p_p0 : array_like
            Pressure ratio.

        """

        M = np.asanyarray(M)
        p_p0 = self.T_T0(M) ** (self.gamma / (self.gamma - 1))

        return p_p0

    def rho_rho0(self, M):
        r"""Density ratio from Mach number.

        .. math::
            \left ( \frac{\rho}{\rho_{0}} \right ) = \left ( \frac{T}{T_{0}} \right )^{\frac{1}{(\gamma - 1)}}

        Parameters
        ----------
        M : array_like
            Mach number.

        Returns
        -------
        rho_rho0 : array_like
            Density ratio.

        """

        M = np.asanyarray(M)
        rho_rho0 = self.T_T0(M) ** (1 / (self.gamma - 1))
        return rho_rho0

    def T_T0(self, M):
        r"""Temperature ratio from Mach number.

        .. math::
            \left ( \frac{T}{T_{0}} \right ) = \left (1 + \frac{\gamma - 1}{2}M^{2} \right )^{-1}

        Parameters
        ----------
        M : array_like
            Mach number.

        Returns
        -------
        T_T0 : array_like
            Temperature ratio.

        """
        M = np.asanyarray(M)
        T_T0 = 1 / (1 + (self.gamma - 1) * M * M / 2)
        return T_T0

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
        M = np.asanyarray(M)
        # If there is any zero entry, NumPy array division gives infinity,
        # which is correct.
        with np.errstate(divide="ignore"):
            A_Astar = (2 / self.T_T0(M) / (self.gamma + 1)) ** (
                (self.gamma + 1) / (2 * (self.gamma - 1))
            ) / M
        return A_Astar

    def a_a0(self, M):
        """ Speed of sound ratio from Mach number.

        Parameters
        ----------
        M: array_like
            Mach number.

        Returns
        -------
        a_a0: array_like
            Speed of sound ratio.
        """

        M = np.asarray(M)
        a_a0 = self.T_T0(M) ** 0.5

        return a_a0


class PrandtlMeyerExpansion(object):
    """Class representing a Prandtl-Meyer expansion fan.

    """

    @staticmethod
    def nu(M, gamma=1.4):
        r"""Prandtl-Meyer angle for a given Mach number.

        The result is given by evaluating the Prandtl-Meyer function.

        .. math::
            \nu = \sqrt{\frac{\gamma + 1}{\gamma - 1}} \tan^{-1}\left [ \sqrt{\frac{\gamma - 1}{\gamma + 1}(M^{2} - 1)} \right ] - \tan^{-1}(\sqrt{M^{2} - 1})

        Parameters
        ----------
        M : float
            Mach number.
        gamma : float, optional
            Specific heat ratio, default 7 / 5.

        Returns
        -------
        nu : float
            Prandtl-Meyer angle, in radians.

        Raises
        ------
        ValueError
            If Mach number is subsonic.

        """
        try:
            with np.errstate(invalid="raise"):
                sgpgm = np.sqrt((gamma + 1) / (gamma - 1))
                nu = sgpgm * np.arctan(np.sqrt(M * M - 1) / sgpgm) - np.arctan(
                    np.sqrt(M * M - 1)
                )
        except FloatingPointError:
            raise ValueError("Mach number must be supersonic")
        return nu

    def __init__(self, M_1, theta, fl=None, gamma=1.4):
        """Constructor of PrandtlMeyerExpansion.

        Parameters
        ----------
        M_1 : float
            Upstream Mach number.
        theta : float
            Deflection angle, in radians.
        fl : IsentropicFlow, optional.
            Flow to be expanded
        gamma : float, optional
            Specific heat ratio, default value = 7 / 5.

        Raises
        ------
        ValueError
            If given Mach number is subsonic.

        """
        if not fl:
            fl = IsentropicFlow(gamma=gamma)
        nu_max = PrandtlMeyerExpansion.nu(np.inf, fl.gamma) - PrandtlMeyerExpansion.nu(
            M_1, fl.gamma
        )
        if theta > nu_max:
            raise ValueError(
                "Deflection angle must be lower than maximum {:.2f}°".format(
                    np.degrees(nu_max)
                )
            )
        self.M_1 = M_1
        self.theta = theta
        self.fl = fl

    @property
    def nu_1(self):
        """Upstream Prandtl-Meyer angle."""
        return PrandtlMeyerExpansion.nu(self.M_1, self.fl.gamma)

    @property
    def nu_2(self):
        """Downstream Prandtl-Meyer angle."""
        return self.nu_1 + self.theta

    @property
    def M_2(self):
        """Downstream Mach number.

        """
        return mach_from_nu(nu=self.nu_2, gamma=self.fl.gamma)

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

    @property
    def rho2_rho1(self):
        """Density ratio across the expansion fan.

        """
        return self.p2_p1 / self.T2_T1

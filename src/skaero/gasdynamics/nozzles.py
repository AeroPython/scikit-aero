# coding: utf-8

"""
Utilities for working with nozzles.
"""

from __future__ import absolute_import, division

import numpy as np

from skaero.gasdynamics import isentropic, shocks


class Nozzle(object):
    """Class representing a nozzle.

    """

    def __init__(self, x, A):
        self.x = x
        self.A = A

    def plot(self, ax, where=None, interpolate=False, **kwargs):
        """Plots the nozzle geometry.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes where the nozzle geometry should be plotted.

        """
        radius = np.sqrt(self.A / np.pi)
        ax.set_xlim(self.x[0], self.x[-1])
        ax.fill_between(self.x, radius, -radius, where, interpolate, **kwargs)

    def solve_flow(self, fl, p_0, T_0, p_B):
        """Solves the flow through the nozzle.

        For the time being, it assumes isentropic flow and uses a quasi one
        dimensional model.

        Parameters
        ----------
        fl : IsentropicFlow
            Isentropic flow through the nozzle.
        p_0 : float
            Stagnation pressure at the reservoir.
        T_0 : float
            Stagnation temperature at the reservoir.
        p_B : float
            Back pressure at the exit section.

        Returns
        -------
        M : array
            Mach distribution along the nozzle.
        p : array
            Pressure distribution along the nozzle.

        Raises
        ------
        ValueError
            If the back pressure is higher than the reservoir pressure.

        Notes
        -----
        First of all, the function computes the isentropic conditions to
        discriminate if we have fully subsonic flow, chocked flow with shock
        inside nozzle or chocked fully supersonic flow.

        """
        p_ratio = p_B / p_0
        if p_ratio > 1.0:
            raise ValueError("Back pressure must be lower than reservoir pressure")

        # Exit and minimum area
        A_e = self.A[-1]
        A_min = np.min(self.A)
        i_min = np.argmin(self.A)

        # 1. Compute isentropic conditions at the exit
        # Points 1 (subsonic) and 2 (supersonic)
        M_e_1, M_e_2 = isentropic.mach_from_area_ratio(fl, A_e / A_min)

        pe_p0_1 = fl.p_p0(M_e_1)
        pe_p0_2 = fl.p_p0(M_e_2)

        # 2. Normal shock at the exit of the nozzle
        ns_e = shocks.NormalShock(M_e_2)
        pe_p0_3 = pe_p0_2 * ns_e.p2_p1

        # 3. Discriminate case
        M = np.empty_like(self.x)
        p = np.empty_like(self.x)

        if pe_p0_1 <= p_ratio <= 1.0:
            print("Fully subsonic flow")
            # 1. Compute M_e (inverse fl.p_p0(M))
            # 2. Compute area ratio from M_e (fl.A_Ac(M_e))
            # 3. The area ratio in each point is A_Ac = A * Ae_Ac / Ae
            #    so I can compute the Mach number distribution
        elif pe_p0_3 <= p_ratio < pe_p0_1:
            print("Shock inside the nozzle")
            # First I have to compute shock location and properties
        elif p_ratio < pe_p0_3:
            print("Fully isentropic subsonic-supersonic flow")
            # I already have Ac = A_min, so I can directly compute the Mach
            # number distribution
            Ac = A_min
            for i in range(i_min + 1):
                M[i] = isentropic.mach_from_area_ratio(fl, self.A[i] / Ac)[0]
            for i in range(i_min, len(M)):
                M[i] = isentropic.mach_from_area_ratio(fl, self.A[i] / Ac)[1]
            # Stagnation pressure does not change, as the flow is fully
            # isentropic
            p = p_0 * fl.p_p0(M)

        return M, p

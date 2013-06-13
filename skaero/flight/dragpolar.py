# coding: utf-8

"""
Drag polar calculations.

"""

from __future__ import division, absolute_import

import numpy as np
import numpy.ma as ma
import scipy as sp
import scipy.interpolate
from scipy.interpolate import interp1d

from skaero.atmosphere import isa  # For dragrise


class DragPolar(object):
    """Drag polar.

    """
    @classmethod
    def fromfile(cls, fname, *args, **kwargs):
        """Returns DragPolar object from file.

        """
        polar_data = np.loadtxt(fname, *args, **kwargs)
        return cls(polar_data[0], polar_data[1])

    def __init__(self, C_L, C_D):
        """Constructor.

        """
        assert C_L.shape == C_D.shape
        self.C_L = C_L
        self.C_D = C_D

    def parabolic_coeffs(self):
        """Returns coefficients of parabolic polar curve.

        """
        stall_idx = self._stall_idx
        K, C_D0 = np.polyfit(self.C_L[:stall_idx + 1] ** 2,
                             self.C_D[:stall_idx + 1], deg=1)
        return K, C_D0

    def C_D_int(self, C_L_vals):
        """Interpolated polar.

        """
        stall_idx = self._stall_idx
        C_D_int_func = interp1d(self.C_L[:stall_idx + 1],
                                self.C_D[:stall_idx + 1], 'cubic')
        return C_D_int_func(C_L_vals)

    def C_D_fit(self, C_L_vals):
        """Fitted polar.

        """
        K, C_D0 = self.parabolic_coeffs()
        return np.polyval([K, C_D0], C_L_vals ** 2)

    def plot_data(self, ax, stalled=True, *args, **kwargs):
        """Plots the data points.

        """
        if stalled:
            l = ax.plot(self.C_D, self.C_L, *args, **kwargs)
        else:
            stall_idx = self._stall_idx
            l = ax.plot(self.C_D[:stall_idx + 1], self.C_L[:stall_idx + 1],
                    *args, **kwargs)
        ax.set_xlabel("C_D")
        ax.set_ylabel("C_L")
        return l

    def plot_int(self, ax, *args, **kwargs):
        """Plots interpolated polar.

        """
        return self._plot_function(ax, self.C_D_int, *args, **kwargs)

    def plot_fit(self, ax, *args, **kwargs):
        """Plots fitted polar.

        """
        return self._plot_function(ax, self.C_D_fit, *args, **kwargs)

    @property
    def C_L_max(self):
        """Maximum lift coefficient.

        """
        return self.C_L[self._stall_idx]

    @property
    def E_max(self):
        """Maximum aerodynamic efficiency.

        """
        return np.max(self._E)

    @property
    def F_max(self):
        # TODO: Docstring
        with np.errstate(invalid='ignore'):
            F = ma.masked_invalid(self._E * np.sqrt(self.C_L))
        return np.max(F)

    @property
    def G_max(self):
        # TODO: Docstring
        with np.errstate(invalid='ignore'):
            G = ma.masked_invalid(self._E / np.sqrt(self.C_L))
        return np.max(G)

    def _plot_function(self, ax, fn, *args, **kwargs):
        C_L_dom = np.linspace(self.C_L[0], self.C_L_max)
        l = ax.plot(fn(C_L_dom), C_L_dom, *args, **kwargs)
        return l

    @property
    def _stall_idx(self):
        return np.argmax(self.C_L)

    @property
    def _E(self):
        E = self.C_L / self.C_D
        return E

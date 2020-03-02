import matplotlib.pyplot as plt
import numpy as np


def zeta_coord(radius, centre_chi=0.0, centre_eta=0.0):
    """Transform a cicle into zeta coordinates."""
    angles = np.linspace(0, 2.0 * pl.pi, 100)
    chi = radius * np.cos(angles) + centre_chi
    eta = radius * np.sin(angles) + centre_eta
    zeta = chi + 1j * eta
    return zeta


def z_coord(zeta=None, radius=1.2, centre_chi=-0.02, centre_eta=0.01, plot=False):
    """Joukowsky transformation as a complex number."""
    if zeta is None:
        zeta = zeta_coord(radius, centre_chi, centre_eta)

    z = zeta + 1.0 / zeta
    x = z.real
    y = z.imag
    if plot:
        plt.plot(x, y)
        plt.show()

    return z

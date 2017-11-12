import pylab as pl


def zeta_coord(radius, centre_chi=0., centre_eta=0.):
    angles = pl.linspace(0, 2. * pl.pi, 100)
    chi = radius * pl.cos(angles) + centre_chi
    eta = radius * pl.sin(angles) + centre_eta
    zeta = chi + 1j * eta
    return zeta


def z_coord(zeta=None, radius=1.2, centre_chi=-0.02, centre_eta=0.01):
    if zeta is None:
        zeta = zeta_coord(radius, centre_chi, centre_eta)

    z = zeta + 1. / zeta
    x = z.real
    y = z.imag
    pl.plot(x, y)
    pl.show()
    return z

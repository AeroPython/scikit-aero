"""
Mathematical and physical constants
"""

from skaero import units as u

# Earth's mean equatorial radius from IAU Working Group on Cartographic
# Coordinates and Rotational Elements: 2015
R_earth = 6.3781366e6 * u.meters

# Standard Earth's gravity acceleration value from "The International System of
# Units by Barry N.Taylor and Ambler Thompson"
g0 = 9.80665 * u.meters / u.seconds ** 2

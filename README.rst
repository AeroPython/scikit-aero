scikit-aero
===========

scikit-aero is a Python package for various aeronautical engineering
calculations. It is based on several existing Python packages on the field,
but intends to provide pythonic syntax, use of SI units and full NumPy arrays
support among other things. scikit-aero is licensed under the BSD license.

It was started by Juan Luis Cano in 2012 and it is currently developed by him.

Features
--------

* Pythonic interface.
* Use of SI units.
* Full support of NumPy arrays.
* Fully tested and documented.
* Standard atmosphere properties up to ~85 kilometers.
* Gas dynamics calculations.

Future
------

* Airspeed conversions.
* Coordinate systems.
* Most of the PDAS.

Usage
=====

Atmosphere properties::

  from skaero import atmosphere

  h, T, p, rho = atmosphere.coesa(1000)  # Altitude by default, 1 km
  a = atmosphere.speed_of_sound(T)  # Compute speed of sound from temperature

Inverse computations allowed with density and pressure, which are monotonic::

  h, T, p, rho = atmosphere.coesa(p=101325)  # Pressure of 1 atm

Isentropic flow properties::

  from skaero.flow import isentropic

  mach, T, P, rho, area = isentropic(0.85)  # Mach number by default, 0.85
  mach, T, P, rho, area = isentropic(0.65, gamma=1.3)  # Gamma of 1.3, default to 1.4

Dependencies
============

* Python >= 2.7
* NumPy >= 1.6
* SciPy >= 0.11

TODO: Test with older versions.

Install
=======

This package uses distutils. To install, execute as usual::

  python setup.py install

TODO: Create actual install script.

Testing
=======

scikit-aero recommends py.test for running the test suite. Running from the
top directory::

  py.test

License
=======

scikit-aero is released under the BSD license, hence allowing commercial use
of the library. Please refer to the COPYING file.

See also
========

* `AeroCalc`_, package written by Kevin Horton which inspired scikit-aero.
* `MATLAB Aerospace Toolbox`_,
* `PDAS`_, the Public Domain Aeronautical Software.

.. _Aerocalc: http://pypi.python.org/pypi/AeroCalc/0.11
.. _`MATLAB Aerospace Toolbox`: http://www.mathworks.com/help/aerotbx/index.html
.. _PDAS: http://www.pdas.com/index.html

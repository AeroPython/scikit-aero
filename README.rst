scikit-aero
===========

:name: scikit-aero
:website: https://github.com/Juanlu001/scikit-aero
:author: Juan Luis Cano <juanlu001@gmail.com>
:version: 0.1

scikit-aero is a Python package for various aeronautical engineering
calculations. It is based on several existing Python packages on the field,
but intends to provide pythonic syntax, use of SI units and full NumPy arrays
support among other things. scikit-aero is licensed under the BSD license.

It was started by Juan Luis Cano in 2012 and it is currently developed and
maintained by him. The source code and issue tracker are both hosted on
GitHub

https://github.com/Juanlu001/scikit-aero

**Notice**: This package is under heavy development and the API might change
at any time until a 1.0 version is reached. It is stable but not feaure
complete yet, and it might contain bugs.

Features
--------

* Pythonic interface.
* Use of SI units.
* Full support of NumPy arrays.
* Support for both Python 2 and 3.
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

Gas dynamics calculations::

  from skaero.gasdynamics import isentropic, shocks

  fl = isentropic.IsentropicFlow(gamma=1.4)
  p = 101325 * fl.p_p0(M=0.8)  # Static pressure given total pressure of 1 atm

  ns = shocks.NormalShock(M_1=2.5, gamma=1.4)
  M_2 = ns.M_2  # Mach number behind a normal shock wave

Dependencies
============

This package depends on Python, NumPy and SciPy and is usually tested on
Linux with the following versions:

* Python 2.7, NumPy 1.6, SciPy 0.11
* Python 3.3, NumPy 1.7.0b2, SciPy 0.11.0

but there is no reason it shouldn't work on Windows or Mac OS X. If you are
willing to provide testing on this platforms, please
`contact me <mailto:juanlu001@gmail.com>`_ and if you find any bugs file them
on the `issue tracker`_.

Install
=======

This package uses distutils. To install, execute as usual::

  $ python setup.py install

It is recommended that you **never ever use sudo** with distutils, pip,
setuptools and friends in Linux because you might seriously break your
system [1_][2_][3_][4_]. I recommend using `virtualenv`_, `per user directories`_
or `local installations`_.

.. _1: http://wiki.python.org/moin/CheeseShopTutorial#Distutils_Installation
.. _2: http://stackoverflow.com/questions/4314376/how-can-i-install-a-python-egg-file/4314446#comment4690673_4314446
.. _3: http://workaround.org/easy-install-debian
.. _4: http://matplotlib.1069221.n5.nabble.com/Why-is-pip-not-mentioned-in-the-Installation-Documentation-tp39779p39812.html

.. _`virtualenv`: http://pypi.python.org/pypi/virtualenv
.. _`per user directories`: http://stackoverflow.com/a/7143496/554319
.. _`local installations`: http://stackoverflow.com/a/4325047/554319

Testing
=======

scikit-aero recommends py.test for running the test suite. Running from the
top directory::

  $ py.test

Bug reporting
=============

I am pretty sure I never introduce bugs in my code, but if you want to prove
me wrong please refer to the `issue tracker`_ on GitHub.

.. _`issue tracker`: https://github.com/Juanlu001/scikit-aero/issues

Citing
======

If you use scikit-aero on your project, please
`drop me a line <mailto:juanlu001@gmail.com>`_.

License
=======

scikit-aero is released under a 2-clause BSD license, hence allowing commercial use
of the library. Please refer to the COPYING file.

See also
========

* `AeroCalc`_, package written by Kevin Horton which inspired scikit-aero.
* `MATLAB Aerospace Toolbox`_,
* `PDAS`_, the Public Domain Aeronautical Software.

.. _Aerocalc: http://pypi.python.org/pypi/AeroCalc/0.11
.. _`MATLAB Aerospace Toolbox`: http://www.mathworks.com/help/aerotbx/index.html
.. _PDAS: http://www.pdas.com/index.html

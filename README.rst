.. image:: docs/source/_static/logo.png
    :align: center

.. image:: https://img.shields.io/badge/Skaero-%E2%9C%88%EF%B8%8F-9cf
	:target: https://github.com/AeroPython/scikit-aero
	:alt: Skaero

.. image:: https://img.shields.io/badge/Built%20with-Python%20%F0%9F%92%95%20-blue
    :target: https://python.org
    :alt: Built with Python

.. image:: https://img.shields.io/pypi/l/scikit-aero.svg
    :target: https://github.com/AeroPython/scikit-aero/blob/master/COPYING
    :alt: License

.. image:: https://readthedocs.org/projects/scikit-aero/badge/?version=latest
    :target: https://scikit-aero.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://travis-ci.org/AeroPython/scikit-aero.svg?branch=master
    :target: https://travis-ci.org/AeroPython/scikit-aero
    :alt: Travis

.. image:: https://codecov.io/gh/AeroPython/scikit-aero/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/AeroPython/scikit-aero
    :alt: Coverage

.. image:: https://img.shields.io/badge/mailing%20list-groups.io-8cbcd1.svg
    :target: aeropython@groups.io
    :alt: Email


Scikit-aero
===========

Scikit-aero is a Python package for various aeronautical engineering
calculations. It is based on several existing Python packages on the field, but
intends to provide pythonic syntax, use of SI units and full NumPy arrays
support among other things. The software is licensed under the BSD license.

It was started by Juan Luis Cano in 2012 and has code from several contributors
(see AUTHORS). The source code and issue tracker are both hosted on GitHub

Installation
------------

This package fulfills the last standard PEP-517 and PEP-518 by including a
`pyproject.toml` file and making use of a tool called [Flit](https://pypi.org/project/flit/)
for the installation of the software. Follow these steps to install scikit-aero:

1. Clone the repository or just download the latest release in the [releases
section](https://github.com/AeroPython/scikit-aero/releases).

2. Run the following commands:

   .. code-block:: bash

        $ pip install pygments, flit
        $ flit install --symlink path/to/scikit-aero

Features
--------

* Pythonic interface.
* Use of SI units.
* Full support of NumPy arrays.
* Fully tested and documented.
* Standard atmosphere properties up to 86 kilometers.
* Gas dynamics calculations.

Future
------

* Airspeed conversions.
* Coordinate systems.
* Most of the PDAS.

Usage and documentation
-----------------------

Official docs are hosted [here](https://scikit-aero.readthedocs.io/en/latest/).
You can find not only installation instructions, but also a set of examples together
with the API documentation.

Testing
-------

It is possible to run the tests by making use of a tool called `tox`, which not
only tests the logic Python files but also other things such us code format and
coverage. Tox is based on what are called "environments", each one defined by its
own name in this package:

* `tox -e check`: will check if code fulfills proper format.
* `tox -e reformat`: applies proper format for passing the previous environment.
* `tox -e coverage`: runs coverage test against actual code.
* `tox -e py36`: will check if tests sucessfully run in Python3.6.

Bug reporting
-------------

If you find any bugs on the software, please refer to the `issue tracker`_ on GitHub.

.. _`issue tracker`: https://github.com/Juanlu001/scikit-aero/issues

Citing
------

If you use scikit-aero on your project, please
`drop me a line <mailto:juanlu001@gmail.com>`_.

License
-------

scikit-aero is released under a 2-clause BSD license, hence allowing commercial use
of the library. Please refer to the COPYING file.

See also
--------

* `AeroCalc`_, package written by Kevin Horton which inspired scikit-aero.
* `MATLAB Aerospace Toolbox`_,
* `PDAS`_, the Public Domain Aeronautical Software.

.. _Aerocalc: http://pypi.python.org/pypi/AeroCalc/0.11
.. _`MATLAB Aerospace Toolbox`: http://www.mathworks.com/help/aerotbx/index.html
.. _PDAS: http://www.pdas.com/index.html

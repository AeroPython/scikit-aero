# coding: utf-8
from runpy import run_path
from setuptools import setup

# Get the version from the relevant file
d = run_path('skaero/version.py')
__version__ = d['__version__']

setup(
    name="scikit-aero",
    version=__version__,
    description="Aeronautical engineering calculations in Python.",
    author="Juan Luis Cano",
    author_email="juanlu001@gmail.com",
    url="https://github.com/Juanlu001/scikit-aero",
    license="BSD",
    keywords=[
        "aero", "aeronautical", "aerospace",
        "engineering", "atmosphere", "gas"
    ],
    requires=["numpy", "scipy"],
    packages=[
        "skaero",
        "skaero.atmosphere", "skaero.gasdynamics",
        "skaero.util"
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    long_description=open('README.rst').read()
)

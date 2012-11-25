# coding: utf-8

from distutils.core import setup

from skaero import __version__

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
    packages=["skaero", "skaero.atmosphere", "skaero.gasdynamics"],
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

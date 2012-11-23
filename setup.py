# coding: utf-8

from distutils.core import setup

setup(
    name="scikit-aero",
    version="0.1",
    description="Aeronautical engineering calculations in Python.",
    author="Juan Luis Cano",
    author_email="juanlu001@gmail.com",
    license="BSD",
    keywords=[
        "aero", "aeronautical", "aerospace",
        "engineering", "atmosphere", "gas"
    ],
    requires=["numpy", "scipy"],
    packages=["skaero", "skaero.atmosphere", "skaero.gasdynamics"]
    )

setup()

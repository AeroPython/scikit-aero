# coding: utf-8

"""
COESA model.

Routines
--------

table(h, kind='geopotential')
temperature(h, kind='geopotential')
pressure(h, kind='geopotential')
density(h, kind='geopotential')

Examples
--------
>>> from skaero.atmosphere import coesa
>>> h, T, p, rho = coesa.table(1000)

Notes
-----
Based on the U.S. 1976 Standard Atmosphere.

.. _`U.S. 1976 Standard Atmosphere`

TODO
----
* Check geopotential temperature
* Move to OOP

"""

from __future__ import division, absolute_import

import numpy as np
from skaero.atmosphere import util
import scipy
from scipy import interpolate

# Constants and values : the following parameter are extracted from Notes reference (mainly Chap. 1.2.). Naming is consistent. WARNING : Some of these values are not exactly consistent with the 2010 CODATA Recommended Values of the Fundamental Physical Constants that you can find for example in the scipy.constants module

Rs = 8.31432 
# N m / (mol K), gas constant
g_0 = 9.80665 
# m / s^2, sea-level value of the acceleration of gravity
g_0p = g_0 
# m^2 / (s^2 m), dimensional constant selected to relate the standard geopotential meter to geometric height
H = np.array([0.,11.0,20.0,32.,47.,51.,71.00,84.85205])*10**3 
# m, set of geopotential heights from table 2 of Notes reference
LM = np.array([-6.5,0.,1.,2.8,0.,-2.8,-2.,0.])*10**(-3) 
# K / m, set of molecular-scale temperature gradients from table 2 of Notes reference
f_LM = interpolate.interp1d(H,LM,kind='zero')
P_0 = 1.01325*10**5 
# Pa, standard sea-level atmospheric pressure
r_e = 6356.7660*10**3  
# m, effective earth's radius
T_0 = 288.15 
# K, standard sea-level temperature
M_0 = 28.9644*10**(-3) 
# kg / mol, mean molecular-weight at sea-level
H2 = np.array([0., 79005.7, 79493.3, 79980.8, 80468.2, 80955.7, 81443.0, 81930.2, 82417.3, 82904.4, 83391.4, 83878.4, 84365.2, 84852.05]) 
# m, set of geopotential heights from table 8 of Notes reference 
M_o_M0 = np.array([1., 1., 0.999996, 0.999989, 0.999971, 0.999941, 0.999909, 0.999870, 0.999829, 0.999786, 0.999741, 0.999694, 0.999641, 0.999579]) 
# -, set of molecular weight ratios from table 8 of Notes reference 
f_M_o_M0=interpolate.interp1d(H2,M_o_M0)

P=np.array([P_0]) 
# Pa, set of pressures (initialization)
TM=np.array([T_0]) 
# K, set of mellecular-scale temperatures (initialization)

for k in range(1,len(H)):
    TM = np.append(TM, TM[-1]+f_LM(H[k-1])*(H[k]-H[k-1])) 
    # from eq. [23] of Notes reference
    if f_LM(H[k-1]) == 0.:
        P = np.append(P, P[-1]*scipy.exp(-g_0p*M_0*(H[k]-H[k-1])/(Rs*TM[-2]))) 
        # from eq. [33b] of Notes reference
    else:
        P = np.append(P, P[-1]*(TM[-2]/(TM[-1]))**(g_0p*M_0/(Rs*f_LM(H[k-1])))) 
        # from eq. [33a] of Notes reference

f_TM=interpolate.interp1d(H,TM,kind='zero')
f_P=interpolate.interp1d(H,P,kind='zero')
f_H=interpolate.interp1d(H,H,kind='zero')


def table(x, kind='geopotential'):
    """Computes table of COESA atmosphere properties.

    Returns temperature, pressure and density COESA values at the given
    altitude.

    Parameters
    ----------
    x : array_like
       Geopotential or geometric altitude (depending on kind) given in meters.
    kind : str
       Specifies the kind of interpolation as altitude x ('geopotential' or 'geometric'). Default is 'geopotential'

    Returns
    -------
    h : array_like
        Given geopotential altitude in meters.
    T : array_like
        Temperature in Kelvin.
    p : array_like
        Pressure in Pascal.
    rho : array_like
        Density in kilograms per cubic meter.

    Notes
    -----
    Based on the U.S. 1976 Standard Atmosphere.

    .. _`U.S. 1976 Standard Atmosphere`

    """
    
    # check the kind of altitude and raise an exception if necessary
    if kind=='geopotential':
        alt=x
    elif kind=='geometric':
        alt=util.geometric_to_geopotential(x,r_e)
    else:
        raise ValueError("%s is unsupported: Use either \'geopotential\' or \'geometric\'." % kind)
    
    h=np.array(alt)
    
    # check if altitude is out of bound and raise an exception if necessary
    if (h<H[0]).any() or (h>H[-1]).any():
        raise ValueError("the given altitude x is out of bound, this module is currently only valid for a geometric altitude between 0. and 86000. m")
    
    tm = f_TM(h)+f_LM(h)*(h-f_H(h)) 
    # K, molecule-scale temperature from eq. [23] of Notes reference 
    T = tm*f_M_o_M0(h) 
    # K, absolute temperature from eq. [22] of Notes reference
    
    if h.shape: # if h is not a 0-d array (like a scalar)
        p=np.zeros(len(h))
        # Pa, intialization of the pressure vector
        
        zero_gradient=(f_LM(h)==0.) 
        # points of h for which the molecular-scale temperature gradient is zero
        not_zero_gradient=(f_LM(h)!=0.) 
        # points of h for which the molecular-scale temperature gradient is not zero  
             
        p[zero_gradient] = f_P(h[zero_gradient])*scipy.exp(-g_0p*M_0*(h[zero_gradient]-f_H(h[zero_gradient]))/(Rs*f_TM(h[zero_gradient])))
        # Pa, pressure from eq. [33b] of Notes reference
        p[not_zero_gradient] = f_P(h[not_zero_gradient])*(f_TM(h[not_zero_gradient])/(f_TM(h[not_zero_gradient])+f_LM(h[not_zero_gradient])*(h[not_zero_gradient]-f_H(h[not_zero_gradient]))))**(g_0p*M_0/(Rs*f_LM(h[not_zero_gradient])))
        # Pa, pressure from eq. [33a] of Notes reference
    else: 
        if f_LM(h)==0:
            p = f_P(h)*scipy.exp(-g_0p*M_0*(h-f_H(h))/(Rs*f_TM(h)))
            # Pa, pressure from eq. [33b] of Notes reference
        else:
            p = f_P(h)*(f_TM(h)/(f_TM(h)+f_LM(h)*(h-f_H(h))))**(g_0p*M_0/(Rs*f_LM(h)))
            # Pa, pressure from eq. [33a] of Notes reference
    
    rho = p*M_0/(Rs*tm) # kg / m^3, mass density
    
    return alt, T, p, rho
    
def temperature(x,kind='geopotential'):
    """Computes air temperature for a given altitude using the U.S. standard atmosphere model

    Parameters
    ----------
    x : array_like
       Geopotential or geometric altitude (depending on kind) given in meters.
    kind : str
       Specifies the kind of interpolation as altitude x ('geopotential' or 'geometric'). Default is 'geopotential'

    Returns
    -------
    T : array_like
        Temperature in Kelvin.

    Notes
    -----
    Based on the U.S. 1976 Standard Atmosphere.

    .. _`U.S. 1976 Standard Atmosphere`

    """
    
    h, T, p, rho = table(x,kind)
    return T
    
def pressure(x,kind='geopotential'):
    """Computes absolute pressure for a given altitude using the U.S. standard atmosphere model

    Parameters
    ----------
    x : array_like
       Geopotential or geometric altitude (depending on kind) given in meters.
    kind : str
       Specifies the kind of interpolation as altitude x ('geopotential' or 'geometric'). Default is 'geopotential'

    Returns
    -------
    P : array_like
        Pressure in Pasacal.

    Notes
    -----
    Based on the U.S. 1976 Standard Atmosphere.

    .. _`U.S. 1976 Standard Atmosphere`

    """
    
    h, T, p, rho = table(x,kind)
    return p    
    
def density(x,kind='geopotential'):
    """Computes air mass density for a given altitude using the U.S. standard atmosphere model

    Parameters
    ----------
    x : array_like
       Geopotential or geometric altitude (depending on kind) given in meters.
    kind : str
       Specifies the kind of interpolation as altitude x ('geopotential' or 'geometric'). Default is 'geopotential'

    Returns
    -------
    rho : array_like
        Density in kilograms per cubic meter.

    Notes
    -----
    Based on the U.S. 1976 Standard Atmosphere.

    .. _`U.S. 1976 Standard Atmosphere`

    """
    
    h, T, p, rho = table(x,kind)
    return rho    

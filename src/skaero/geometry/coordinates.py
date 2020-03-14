# coding: utf-8

"""
    Coordinate transformations used in flight dynamics
"""

import numpy as np
from numpy import array, cos, deg2rad, rad2deg, sin


def lla2ecef(lat, lng, h):
    """
    Calculates geocentric coordinates (ECEF - Earth Centered, Earth Fixed)
    for a given set of latitude, longitude and altitude inputs, following
    the WGS84 system.

    Parameters
    ----------
    lat : float
        latitude in degrees
    lng : float
        longitude in degrees
    h : float
        geometric altitude above sea level in meters

    Returns
    -------
    array-like
        ECEF coordinates in meters
    """
    if abs(lat) > 90:
        raise ValueError("latitude should be -90º <= latitude <= 90º")

    if abs(lng) > 180:
        raise ValueError("longitude should be -180º <= longitude <= 180º")

    if not (0 <= h <= 84852.05):
        msg = "pressure model is only valid if 0 <= h <= 84852.05"
        raise ValueError(msg)

    a = 6378137  # [m] Earth equatorial axis
    b = 6356752.3142  # [m] Earth polar axis
    e = 0.081819190842622  # Earth eccentricity

    lat = deg2rad(lat)  # degrees to radians
    lng = deg2rad(lng)  # degrees to radians

    N = a / (1 - (e * sin(lat)) ** 2) ** (0.5)

    x = (N + h) * cos(lat) * cos(lng)
    y = (N + h) * cos(lat) * sin(lng)
    z = (((b / a) ** 2) * N + h) * sin(lat)

    return array([x, y, z])


def ned2ecef(v_ned, lat, lng):
    """
    Converts vector from local geodetic horizon reference frame (NED - North,
    East, Down) at a given latitude and longitude to geocentric coordinates
    (ECEF - Earth Centered, Earth Fixed).

    Parameters
    ----------
    v_ned: array-like
        vector expressed in NED coordinates
    lat : float
        latitude in degrees
    lng : float
        longitude in degrees

    Returns
    -------
    v_ecef : array-like
        vector expressed in ECEF coordinates
    """
    if abs(lat) > 90:
        raise ValueError("latitude should be -90º <= latitude <= 90º")

    if abs(lng) > 180:
        raise ValueError("longitude should be -180º <= longitude <= 180º")

    lat = deg2rad(lat)
    lng = deg2rad(lng)

    Lne = array(
        [
            [-sin(lat) * cos(lng), -sin(lat) * sin(lng), cos(lat)],
            [-sin(lng), cos(lng), 0],
            [-cos(lat) * cos(lng), -cos(lat) * sin(lng), -sin(lat)],
        ]
    )

    Len = Lne.transpose()
    v_ecef = Len.dot(v_ned)

    return v_ecef


def ecef2ned(v_ecef, lat, lng):
    """
    Converts vector from geocentric coordinates (ECEF - Earth Centered,
    Earth Fixed) at a given latitude and longitude to local geodetic horizon
    reference frame (NED - North, East, Down).

    Parameters
    ----------
    v_ecef: array-like
        vector expressed in ECEF coordinates
    lat : float
        latitude in degrees
    lng : float
        longitude in degrees

    Returns
    -------
    v_ned : array-like
        vector expressed in NED coordinates
    """
    if abs(lat) > 90:
        raise ValueError("latitude should be -90º <= latitude <= 90º")

    if abs(lng) > 180:
        raise ValueError("longitude should be -180º <= longitude <= 180º")

    lat = deg2rad(lat)
    lng = deg2rad(lng)

    Lne = array(
        [
            [-sin(lat) * cos(lng), -sin(lat) * sin(lng), cos(lat)],
            [-sin(lng), cos(lng), 0],
            [-cos(lat) * cos(lng), -cos(lat) * sin(lng), -sin(lat)],
        ]
    )

    v_ned = Lne.dot(v_ecef)

    return v_ned


def body2ned(v_body, theta, phi, psi):
    """
    Converts vector from body reference frame to local geodetic horizon
    reference frame (NED - North, East, Down)

    Parameters
    ----------
    v_body: array-like
        vector expressed in body coordinates
    theta : float
        pitch angle in radians
    phi : float
        bank angle in radians
    psi : float
        yaw angle in radians

    Returns
    -------
    v_ned : array_like
        vector expressed in local horizon (NED) coordinates
    """
    if abs(theta) > np.pi / 2:
        raise ValueError("theta should be -pi/2 <= theta <= pi/2")

    if abs(phi) > np.pi:
        raise ValueError("phi should be -pi <= phi <= pi")

    if not 0 <= psi <= 2 * np.pi:
        raise ValueError("psi should be 0 <= psi <= 2*pi")

    Lnb = array(
        [
            [
                cos(theta) * cos(psi),
                sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),
                cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi),
            ],
            [
                cos(theta) * sin(psi),
                sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi),
                cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi),
            ],
            [-sin(theta), sin(phi) * cos(theta), cos(phi) * cos(theta)],
        ]
    )

    v_ned = Lnb.dot(v_body)

    return v_ned


def ned2body(v_ned, theta, phi, psi):
    """
    Converts vector from local geodetic horizon reference frame (NED -
    North, East, Down) to body reference frame

    Parameters
    ----------
    v_ned : array_like
        vector expressed in local horizon (NED) coordinates
    theta : float
        pitch angle in radians
    phi : float
        bank angle in radians
    psi : float
        yaw angle in radians

    Returns
    -------
    v_body: array-like
        vector expressed in body coordinates
    """
    if abs(theta) > np.pi / 2:
        raise ValueError("theta should be -pi/2 <= theta <= pi/2")

    if abs(phi) > np.pi:
        raise ValueError("phi should be -pi <= phi <= pi")

    if not 0 <= psi <= 2 * np.pi:
        raise ValueError("psi should be 0 <= psi <= 2*pi")

    Lbn = array(
        [
            [cos(theta) * cos(psi), cos(theta) * sin(psi), -sin(theta)],
            [
                sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),
                sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi),
                sin(phi) * cos(theta),
            ],
            [
                cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi),
                cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi),
                cos(phi) * cos(theta),
            ],
        ]
    )

    v_body = Lbn.dot(v_ned)

    return v_body


def body2wind(v_body, alpha, beta):
    """
    Converts vector from body reference frame to wind reference frame

    Parameters
    ----------
    v_body : array_like
        vector expressed in body coordinates
    alpha : float
        angle of attack in radians
    beta : float
        sideslip angle in radians

    Returns
    -------
    v_wind : array_like
        vector expressed in wind coordinates
    """
    if abs(alpha) > np.pi / 2:
        raise ValueError("alpha should be -pi/2 <= alpha <= pi/2")

    if abs(beta) > np.pi:
        raise ValueError("beta should be -pi <= beta <= pi")

    Lwb = array(
        [
            [cos(alpha) * cos(beta), sin(beta), sin(alpha) * cos(beta)],
            [-cos(alpha) * sin(beta), cos(beta), -sin(alpha) * sin(beta)],
            [-sin(alpha), 0, cos(alpha)],
        ]
    )

    v_wind = Lwb.dot(v_body)

    return v_wind


def wind2body(v_wind, alpha, beta):
    """
    Converts vector from wind reference frame to body reference frame

    Parameters
    ----------
    v_wind : array_like
        vector expressed in wind coordinates
    alpha : float
        angle of attack in radians
    beta : float
        sideslip angle in radians

    Returns
    -------
    v_body : array_like
        vector expressed in body coordinates
    """
    if abs(alpha) > np.pi / 2:
        raise ValueError("alpha should be -pi/2 <= alpha <= pi/2")

    if abs(beta) > np.pi:
        raise ValueError("beta should be -pi <= beta <= pi")

    Lbw = array(
        [
            [cos(alpha) * cos(beta), -cos(alpha) * sin(beta), -sin(alpha)],
            [sin(beta), cos(beta), 0],
            [sin(alpha) * cos(beta), -sin(alpha) * sin(beta), cos(alpha)],
        ]
    )

    v_body = Lbw.dot(v_wind)

    return v_body


def az_elev_dist(lla, lla_ref):
    """
    Returns distance, azimuth angle and elevation angle that define the
    position in the sky of a point (defined using lla coordinates) as viewed
    from a point of reference (also defined by lla coordinates)
    Parameters
    ----------
    lla : array-like
        contains latitude and longitude (in degrees) and geometric altitude
        above sea level in meters
    lla_ref : array-like
        contains reference point latitude and longitude (in degrees) and
        geometric altitude above sea level in meters
    Returns
    -------
    out : tuple-like
        distance aircraft to reference point in m
        azimuth angle (from the reference point) in degrees
        elevation angle (from the reference point) in degrees
    """
    lat, lng, h = lla
    lat_ref, lng_ref, h_ref = lla_ref

    if abs(lat) > 90 or abs(lat_ref) > 90:
        raise ValueError("latitude should be -90º <= latitude <= 90º")

    if abs(lng) > 180 or abs(lng_ref) > 180:
        raise ValueError("longitude should be -180º <= longitude <= 180º")

    v = lla2ecef(lat, lng, h) - lla2ecef(lat_ref, lng_ref, h_ref)

    v_unit_ecef = v / np.linalg.norm(v)
    v_unit_ned = ecef2ned(v_unit_ecef, lat_ref, lng_ref)

    azimuth = np.arctan2(-v_unit_ned[1], v_unit_ned[0])

    if v_unit_ned[0] == v_unit_ned[1] == 0:
        elevation = np.pi / 2
    else:
        elevation = np.arctan(
            -v_unit_ned[2] / np.sqrt(v_unit_ned[0] ** 2 + v_unit_ned[1] ** 2)
        )

    distance = np.linalg.norm(v)

    return rad2deg(azimuth), rad2deg(elevation), distance

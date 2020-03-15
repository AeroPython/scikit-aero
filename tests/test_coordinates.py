# coding: utf-8

"""
    Tests ofc the coordinate transformations
"""

from __future__ import division, absolute_import

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy import array, deg2rad, rad2deg
import unittest as ut

from skaero.geometry.coordinates import *


class Test_lla2ecef(ut.TestCase):
    """
        Test function that returns ecef position from lat, long, altitude
    """
    def test_latitude_wrong_input(self):
        self.assertRaises(ValueError, lla2ecef, 91.0, 0, 0)
        self.assertRaises(ValueError, lla2ecef, -90.001, 0, 0)
        self.assertRaises(TypeError, lla2ecef, '91.0', 0, 0)

    def test_longitude_wrong_input(self):
        self.assertRaises(ValueError, lla2ecef, 0, -189, 0)
        self.assertRaises(ValueError, lla2ecef, 0, 181.9, 0)
        self.assertRaises(TypeError, lla2ecef, 8, '0', 0)

    def test_altitude_wrong_input(self):
        self.assertRaises(ValueError, lla2ecef, 0, 0, -1.0)
        self.assertRaises(ValueError, lla2ecef, 0, 0, 85000.0)
        self.assertRaises(TypeError, lla2ecef, 0, 0, 'e')

    def test_OX(self):
        a = 6378137  # [m] Earth equatorial axis
        b = 6356752.3142  # [m] Earth polar axis

        # OX-axis
        lat = 0
        lng = 0
        h = 0
        expected_value = array([a, 0, 0])
        self.assertTrue(np.allclose(lla2ecef(lat, lng, h),
                                    expected_value))

        lat = 0
        lng = 180
        h = 0
        expected_value = array([-a, 0, 0])
        self.assertTrue(np.allclose(lla2ecef(lat, lng, h),
                                    expected_value))

    def test_OY(self):
        a = 6378137  # [m] Earth equatorial axis
        b = 6356752.3142  # [m] Earth polar axis

        # OY-axis
        lat = 0
        lng = 90
        h = 0
        expected_value = array([0, a, 0])
        self.assertTrue(np.allclose(lla2ecef(lat, lng, h),
                                    expected_value))

        lat = 0
        lng = -90
        h = 0
        expected_value = array([0, -a, 0])
        self.assertTrue(np.allclose(lla2ecef(lat, lng, h),
                                    expected_value))

    def test_OZ(self):
        a = 6378137  # [m] Earth equatorial axis
        b = 6356752.3142  # [m] Earth polar axis

        # OZ-axis
        lat = 90
        lng = 0
        h = 0
        expected_value = array([0, 0, b])
        self.assertTrue(np.allclose(lla2ecef(lat, lng, h),
                                    expected_value))

        lat = -90
        lng = 0
        h = 0
        expected_value = array([0, 0, -b])
        self.assertTrue(np.allclose(lla2ecef(lat, lng, h),
                                    expected_value))


class Test_ned2ecef(ut.TestCase):
    """
    Test function that transforms ned-basis vectors to ecef-basis
    """
    def test_latitude_wrong_input(self):
        v_aux = array([1, 0, 0])
        self.assertRaises(ValueError, ned2ecef, v_aux, 91.0, 0)
        self.assertRaises(ValueError, ned2ecef, v_aux, -90.001, 0)
        self.assertRaises(TypeError, ned2ecef, v_aux, 't', 0)

    def test_longitude_wrong_input(self):
        v_aux = array([1, 0, 0])
        self.assertRaises(ValueError, ned2ecef, v_aux, 0, -190.1)
        self.assertRaises(ValueError, ned2ecef, v_aux, 0, 180.1)
        self.assertRaises(TypeError, ned2ecef, v_aux, 0, '0')

    def test_1(self):
        lat, lng = 0, 0

        v_ned = array([1, 0, 0])
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(ned2ecef(v_ned, lat, lng),
                                    expected_value))

        v_ned = array([0, 1, 0])
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(ned2ecef(v_ned, lat, lng),
                                    expected_value))

        v_ned = array([0, 0, 1])
        expected_value = array([-1, 0, 0])
        self.assertTrue(np.allclose(ned2ecef(v_ned, lat, lng),
                                    expected_value))

    def test_2(self):
        lat, lng = 0, 90

        v_ned = array([1, 0, 0])
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(ned2ecef(v_ned, lat, lng),
                                    expected_value))

        v_ned = array([0, 1, 0])
        expected_value = array([-1, 0, 0])
        self.assertTrue(np.allclose(ned2ecef(v_ned, lat, lng),
                                    expected_value))

        v_ned = array([0, 0, 1])
        expected_value = array([0, -1, 0])
        self.assertTrue(np.allclose(ned2ecef(v_ned, lat, lng),
                                    expected_value))

    def test_3(self):
        lat, lng = 90, 0

        v_ned = array([1, 0, 0])
        expected_value = array([-1, 0, 0])
        self.assertTrue(np.allclose(ned2ecef(v_ned, lat, lng),
                                    expected_value))

        v_ned = array([0, 1, 0])
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(ned2ecef(v_ned, lat, lng),
                                    expected_value))

        v_ned = array([0, 0, 1])
        expected_value = array([0, 0, -1])
        self.assertTrue(np.allclose(ned2ecef(v_ned, lat, lng),
                                    expected_value))

    def test_ned2ecef2ned(self):
        v = array([1, 0.43, -3])

        lat, lng = 67, 31.45
        v_ecef = ned2ecef(v, lat, lng)
        expected_value = v
        self.assertTrue(np.allclose(ecef2ned(v_ecef, lat, lng),
                                    expected_value))


class Test_ecef2ned(ut.TestCase):
    """
    Test function that transforms ecef-basis vectors to ned-basis
    """
    def test_latitude_wrong_input(self):
        v_aux = array([1, 0, 0])
        self.assertRaises(ValueError, ecef2ned, v_aux, 91.0, 0)
        self.assertRaises(ValueError, ecef2ned, v_aux, -90.001, 0)
        self.assertRaises(TypeError, ecef2ned, v_aux, 't', 0)

    def test_longitude_wrong_input(self):
        v_aux = array([1, 0, 0])
        self.assertRaises(ValueError, ecef2ned, v_aux, 0, -190.1)
        self.assertRaises(ValueError, ecef2ned, v_aux, 0, 180.1)
        self.assertRaises(TypeError, ecef2ned, v_aux, 0, '0')

    def test_1(self):
        lat, lng = 0, 0

        v_ecef = array([1, 0, 0])
        expected_value = array([0, 0, -1])
        self.assertTrue(np.allclose(ecef2ned(v_ecef, lat, lng),
                                    expected_value))

        v_ecef = array([0, 1, 0])
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(ecef2ned(v_ecef, lat, lng),
                                    expected_value))

        v_ecef = array([0, 0, 1])
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(ecef2ned(v_ecef, lat, lng),
                                    expected_value))

    def test_2(self):
        lat, lng = 0, 90

        v_ecef = array([1, 0, 0])
        expected_value = array([0, -1, 0])
        self.assertTrue(np.allclose(ecef2ned(v_ecef, lat, lng),
                                    expected_value))

        v_ecef = array([0, 1, 0])
        expected_value = array([0, 0, -1])
        self.assertTrue(np.allclose(ecef2ned(v_ecef, lat, lng),
                                    expected_value))

        v_ecef = array([0, 0, 1])
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(ecef2ned(v_ecef, lat, lng),
                                    expected_value))

    def test_3(self):
        lat, lng = 90, 0

        v_ecef = array([1, 0, 0])
        expected_value = array([-1, 0, 0])
        self.assertTrue(np.allclose(ecef2ned(v_ecef, lat, lng),
                                    expected_value))

        v_ecef = array([0, 1, 0])
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(ecef2ned(v_ecef, lat, lng),
                                    expected_value))

        v_ecef = array([0, 0, 1])
        expected_value = array([0, 0, -1])
        self.assertTrue(np.allclose(ecef2ned(v_ecef, lat, lng),
                                    expected_value))

    def test_ecef2ned2ecef(self):
        v = array([-56, 30.43, -81])

        lat, lng = -39, 178.45
        v_ned = ecef2ned(v, lat, lng)
        expected_value = v
        self.assertTrue(np.allclose(ned2ecef(v_ned, lat, lng),
                                    expected_value))


class Test_body2ned(ut.TestCase):
    """
    Test function that transforms body-basis vectors to ned-basis
    """
    def test_wrong_theta_input(self):
        v_aux = array([1, 0, 0])

        theta, phi, psi = deg2rad(90.1), deg2rad(0), deg2rad(0)
        self.assertRaises(ValueError, body2ned, v_aux, theta, phi, psi)

        theta, phi, psi = deg2rad(-90.01), deg2rad(0), deg2rad(0)
        self.assertRaises(ValueError, body2ned, v_aux, theta, phi, psi)

        theta, phi, psi = 'a', deg2rad(0), deg2rad(0)
        self.assertRaises(TypeError, body2ned, v_aux, theta, phi, psi)

    def test_wrong_phi_input(self):
        v_aux = array([1, 0, 0])

        theta, phi, psi = deg2rad(0), deg2rad(180.1), deg2rad(0)
        self.assertRaises(ValueError, body2ned, v_aux, theta, phi, psi)

        theta, phi, psi = deg2rad(0), deg2rad(-181), deg2rad(0)
        self.assertRaises(ValueError, body2ned, v_aux, theta, phi, psi)

        theta, phi, psi = deg2rad(0), 'a', deg2rad(0)
        self.assertRaises(TypeError, body2ned, v_aux, theta, phi, psi)

    def test_wrong_psi_input(self):
        v_aux = array([1, 0, 0])

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(-1)
        self.assertRaises(ValueError, body2ned, v_aux, theta, phi, psi)

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(361)
        self.assertRaises(ValueError, body2ned, v_aux, theta, phi, psi)

        theta, phi, psi = deg2rad(0), deg2rad(0), 'a'
        self.assertRaises(TypeError, body2ned, v_aux, theta, phi, psi)

    def test_OXb(self):
        v_body = array([1, 0, 0])

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(0)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(90), deg2rad(0), deg2rad(0)
        expected_value = array([0, 0, -1])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(-90), deg2rad(0), deg2rad(0)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(90), deg2rad(0)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(-90), deg2rad(0)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(90)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(270)
        expected_value = array([0, -1, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

    def test_OYb(self):
        v_body = array([0, 1, 0])

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(90), deg2rad(0), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(-90), deg2rad(0), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(90), deg2rad(0)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(-90), deg2rad(0)
        expected_value = array([0, 0, -1])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(90)
        expected_value = array([-1, 0, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(270)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

    def test_OZb(self):
        v_body = array([0, 0, 1])

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(0)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(90), deg2rad(0), deg2rad(0)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(-90), deg2rad(0), deg2rad(0)
        expected_value = array([-1, 0, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(90), deg2rad(0)
        expected_value = array([0, -1, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(-90), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(90)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(270)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))

    def test_body2ned2body(self):
        v = array([7, 12, -31])

        theta, phi, psi = deg2rad(54), deg2rad(-12), deg2rad(76)
        v_ned = body2ned(v, theta, phi, psi)
        expected_value = v
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))


class Test_ned2body(ut.TestCase):
    """
    Test function that transforms ned-basis vectors to body-basis
    """
    def test_wrong_theta_input(self):
        v_aux = array([1, 0, 0])

        theta, phi, psi = deg2rad(90.1), deg2rad(0), deg2rad(0)
        self.assertRaises(ValueError, ned2body, v_aux, theta, phi, psi)

        theta, phi, psi = deg2rad(-90.01), deg2rad(0), deg2rad(0)
        self.assertRaises(ValueError, ned2body, v_aux, theta, phi, psi)

        theta, phi, psi = 'a', deg2rad(0), deg2rad(0)
        self.assertRaises(TypeError, ned2body, v_aux, theta, phi, psi)

    def test_wrong_phi_input(self):
        v_aux = array([1, 0, 0])

        theta, phi, psi = deg2rad(0), deg2rad(180.1), deg2rad(0)
        self.assertRaises(ValueError, ned2body, v_aux, theta, phi, psi)

        theta, phi, psi = deg2rad(0), deg2rad(-181), deg2rad(0)
        self.assertRaises(ValueError, ned2body, v_aux, theta, phi, psi)

        theta, phi, psi = deg2rad(0), 'a', deg2rad(0)
        self.assertRaises(TypeError, ned2body, v_aux, theta, phi, psi)

    def test_wrong_psi_input(self):
        v_aux = array([1, 0, 0])

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(-1)
        self.assertRaises(ValueError, ned2body, v_aux, theta, phi, psi)

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(361)
        self.assertRaises(ValueError, ned2body, v_aux, theta, phi, psi)

        theta, phi, psi = deg2rad(0), deg2rad(0), 'a'
        self.assertRaises(TypeError, ned2body, v_aux, theta, phi, psi)

    def test_OXn(self):
        v_ned = array([1, 0, 0])

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(0)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(90), deg2rad(0), deg2rad(0)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(-90), deg2rad(0), deg2rad(0)
        expected_value = array([0, 0, -1])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(90), deg2rad(0)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(-90), deg2rad(0)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(90)
        expected_value = array([0, -1, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(270)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

    def test_OYn(self):
        v_ned = array([0, 1, 0])

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(90), deg2rad(0), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(-90), deg2rad(0), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(90), deg2rad(0)
        expected_value = array([0, 0, -1])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(-90), deg2rad(0)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(90)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(90), deg2rad(270)
        expected_value = array([-1, 0, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

    def test_OZn(self):
        v_ned = array([0, 0, 1])

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(0)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(90), deg2rad(0), deg2rad(0)
        expected_value = array([-1, 0, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(-90), deg2rad(0), deg2rad(0)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(90), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(-90), deg2rad(0)
        expected_value = array([0, -1, 0])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(90)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

        theta, phi, psi = deg2rad(0), deg2rad(0), deg2rad(270)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(ned2body(v_ned, theta, phi, psi),
                                    expected_value))

    def test_ned2body2ned(self):
        v = array([7, -2, 1])

        theta, phi, psi = deg2rad(5), deg2rad(21), deg2rad(6)
        v_body = ned2body(v, theta, phi, psi)
        expected_value = v
        self.assertTrue(np.allclose(body2ned(v_body, theta, phi, psi),
                                    expected_value))


class Test_body2wind(ut.TestCase):
    """
    Test function that transforms body-basis vectors to wind-basis
    """
    def test_wrong_alpha_input(self):
        v_aux = array([1, 0, 0])

        alpha, beta = deg2rad(90.1), deg2rad(0)
        self.assertRaises(ValueError, body2wind, v_aux, alpha, beta)

        alpha, beta = deg2rad(-90.01), deg2rad(0)
        self.assertRaises(ValueError, body2wind, v_aux, alpha, beta)

        alpha, beta = 'a', deg2rad(0)
        self.assertRaises(TypeError, body2wind, v_aux, alpha, beta)

    def test_wrong_beta_input(self):
        v_aux = array([1, 0, 0])

        alpha, beta = deg2rad(0), deg2rad(180.1)
        self.assertRaises(ValueError, body2wind, v_aux, alpha, beta)

        alpha, beta = deg2rad(0), deg2rad(-181)
        self.assertRaises(ValueError, body2wind, v_aux, alpha, beta)

        alpha, beta = deg2rad(0), 'a'
        self.assertRaises(TypeError, body2wind, v_aux, alpha, beta)

    def test_OXb(self):
        v_body = array([1, 0, 0])

        alpha, beta = deg2rad(0), deg2rad(0)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(90), deg2rad(0)
        expected_value = array([0, 0, -1])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(-90), deg2rad(0)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(0), deg2rad(90)
        expected_value = array([0, -1, 0])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(0), deg2rad(-90)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

    def test_OYb(self):
        v_body = array([0, 1, 0])

        alpha, beta = deg2rad(0), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(90), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(-90), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(0), deg2rad(90)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(0), deg2rad(-90)
        expected_value = array([-1, 0, 0])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

    def test_OZb(self):
        v_body = array([0, 0, 1])

        alpha, beta = deg2rad(0), deg2rad(0)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(90), deg2rad(0)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(-90), deg2rad(0)
        expected_value = array([-1, 0, 0])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(0), deg2rad(90)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(0), deg2rad(-90)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))

    def test_body2wind2body(self):
        v = array([10, -8, -1])

        alpha, beta = deg2rad(10), deg2rad(-7)
        v_wind = body2wind(v, alpha, beta)
        expected_value = v
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))


class Test_wind2body(ut.TestCase):
    """
    Test function that transforms wind-basis vectors to body-basis
    """
    def test_wrong_alpha_input(self):
        v_aux = array([1, 0, 0])

        alpha, beta = deg2rad(90.1), deg2rad(0)
        self.assertRaises(ValueError, wind2body, v_aux, alpha, beta)

        alpha, beta = deg2rad(-90.01), deg2rad(0)
        self.assertRaises(ValueError, wind2body, v_aux, alpha, beta)

        alpha, beta = 'a', deg2rad(0)
        self.assertRaises(TypeError, wind2body, v_aux, alpha, beta)

    def test_wrong_beta_input(self):
        v_aux = array([1, 0, 0])

        alpha, beta = deg2rad(0), deg2rad(180.1)
        self.assertRaises(ValueError, wind2body, v_aux, alpha, beta)

        alpha, beta = deg2rad(0), deg2rad(-181)
        self.assertRaises(ValueError, wind2body, v_aux, alpha, beta)

        alpha, beta = deg2rad(0), 'a'
        self.assertRaises(TypeError, wind2body, v_aux, alpha, beta)

    def test_OXw(self):
        v_wind = array([1, 0, 0])

        alpha, beta = deg2rad(0), deg2rad(0)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(90), deg2rad(0)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(-90), deg2rad(0)
        expected_value = array([0, 0, -1])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(0), deg2rad(90)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(0), deg2rad(-90)
        expected_value = array([0, -1, 0])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

    def test_OYw(self):
        v_wind = array([0, 1, 0])

        alpha, beta = deg2rad(0), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(90), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(-90), deg2rad(0)
        expected_value = array([0, 1, 0])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(0), deg2rad(90)
        expected_value = array([-1, 0, 0])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(0), deg2rad(-90)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

    def test_OZw(self):
        v_wind = array([0, 0, 1])

        alpha, beta = deg2rad(0), deg2rad(0)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(90), deg2rad(0)
        expected_value = array([-1, 0, 0])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(-90), deg2rad(0)
        expected_value = array([1, 0, 0])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(0), deg2rad(90)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

        alpha, beta = deg2rad(0), deg2rad(-90)
        expected_value = array([0, 0, 1])
        self.assertTrue(np.allclose(wind2body(v_wind, alpha, beta),
                                    expected_value))

    def test_wind2body2wind(self):
        v = array([0, -2, -16])

        alpha, beta = deg2rad(19), deg2rad(37)
        v_body = wind2body(v, alpha, beta)
        expected_value = v
        self.assertTrue(np.allclose(body2wind(v_body, alpha, beta),
                                    expected_value))


class Test_az_elev_dist(ut.TestCase):
    """
    Test function that calculates distance, azimuth and elevation of a point
    as seen from a reference point
    """
    def test_latitude_wrong_input(self):
        lla = array([91, 0, 0])
        lla_ref = array([0, 0, 0])
        self.assertRaises(ValueError, az_elev_dist, lla, lla_ref)

        lla = array([0, 0, 0])
        lla_ref = array([-210, 0, 0])
        self.assertRaises(ValueError, az_elev_dist, lla, lla_ref)

        lla = array(['a', 0, 0])
        lla_ref = array([0, 0, 0])
        self.assertRaises(TypeError, az_elev_dist, lla, lla_ref)

    def test_longitude_wrong_input(self):
        lla = array([0, -181, 0])
        lla_ref = array([0, 0, 0])
        self.assertRaises(ValueError, az_elev_dist, lla, lla_ref)

        lla = array([0, 0, 0])
        lla_ref = array([0, 189, 0])
        self.assertRaises(ValueError, az_elev_dist, lla, lla_ref)

        lla = array([0, 0, 0])
        lla_ref = array([0, 'a', 0])
        self.assertRaises(TypeError, az_elev_dist, lla, lla_ref)

    def test_1(self):
        lla = array([0, 0, 0])
        lla_ref = array([0, 0.00001, 0])
        expected_value = (90, 0, 1.113)
        self.assertTrue(np.allclose(az_elev_dist(lla, lla_ref),
                                    expected_value, atol=1e-3))

    def test_2(self):
        lla = array([0, 0, 0])
        lla_ref = array([0, 0, 10])
        expected_value = (0, 90, 10)
        self.assertTrue(np.allclose(az_elev_dist(lla, lla_ref),
                                    expected_value))

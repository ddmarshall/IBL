#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:09:25 2022

@author: ddmarshall
"""

import unittest
import numpy as np
import numpy.testing as npt
from scipy.misc import derivative as fd

from ibl.thwaites_method import ThwaitesMethodLinear
from ibl.thwaites_method import ThwaitesMethodNonlinear
from ibl.thwaites_method import _ThwaitesFunctionsWhite
from ibl.thwaites_method import _ThwaitesFunctionsCebeciBradshaw
from ibl.thwaites_method import _ThwaitesFunctionsSpline


class TestCurveFits(unittest.TestCase):
    """Class to test various functions and curve fits for Thwaites method"""

    f_ref = np.array([0.938, 0.953, 0.956, 0.962, 0.967, 0.969, 0.971,
                      0.970, 0.963, 0.952, 0.936, 0.919, 0.902, 0.886,
                      0.854, 0.825, 0.797, 0.770, 0.744, 0.691, 0.640,
                      0.590, 0.539, 0.490, 0.440, 0.342, 0.249, 0.156,
                      0.064,-0.028,-0.138,-0.251,-0.362,-0.702,-1.000])
    shear_ref = np.array([0.000, 0.011, 0.016, 0.024, 0.030, 0.035, 0.039,
                          0.049, 0.055, 0.067, 0.076, 0.083, 0.089, 0.094,
                          0.104, 0.113, 0.122, 0.130, 0.138, 0.153, 0.168,
                          0.182, 0.195, 0.208, 0.220, 0.244, 0.268, 0.291,
                          0.313, 0.333, 0.359, 0.382, 0.404, 0.463, 0.500])
    shape_ref = np.array([3.70,  3.69,  3.66,  3.63,  3.61,  3.59,  3.58,
                          3.52,  3.47,  3.38,  3.30,  3.23,  3.17,  3.13,
                          3.05,  2.99,  2.94,  2.90,  2.87,  2.81,  2.75,
                          2.71,  2.67,  2.64,  2.61,  2.55,  2.49,  2.44,
                          2.39,  2.34,  2.28,  2.23,  2.18,  2.07,  2.00])
    lam_ref = np.array([-0.082,-0.0818,-0.0816,-0.0812,-0.0808,-0.0804,
                        -0.080,-0.079, -0.078, -0.076, -0.074, -0.072,
                        -0.070,-0.068, -0.064, -0.060, -0.056, -0.052,
                        -0.048,-0.040, -0.032, -0.024, -0.016, -0.008,
                        +0.000, 0.016,  0.032,  0.048,  0.064,  0.080,
                        +0.10,  0.12,   0.14,   0.20,   0.25])

    def test_tabular_values(self):
        """Test the tabular value fits."""

        spline = _ThwaitesFunctionsSpline()

        # test the range of lambda
        lam_min = spline._tab_lambda[0]  # pylint: disable=protected-access
        lam_max = spline._tab_lambda[-1]  # pylint: disable=protected-access
        self.assertIsNone(npt.assert_allclose(lam_min, np.min(self.lam_ref)))
        self.assertIsNone(npt.assert_allclose(lam_max, np.max(self.lam_ref)))

        # test the tabular values
        lam = spline._tab_lambda  # pylint: disable=protected-access
        shape = spline._tab_shape  # pylint: disable=protected-access
        shear = spline._tab_shear  # pylint: disable=protected-access
        f = spline._tab_f  # pylint: disable=protected-access
        self.assertIsNone(npt.assert_allclose(self.lam_ref, lam))
        self.assertIsNone(npt.assert_allclose(self.shape_ref, shape))
        self.assertIsNone(npt.assert_allclose(self.shear_ref, shear))
        self.assertIsNone(npt.assert_allclose(self.f_ref, f))

        # compare the tabulated F with the calculated value
        f_calc = 2*(shear-lam*(2+shape))
        self.assertIsNone(npt.assert_allclose(self.f_ref, f_calc,
                                              rtol=0, atol=1e-2))

    def test_white_functions(self):
        """Test the White fits."""

        white = _ThwaitesFunctionsWhite()

        # test the range of lambda
        lam_min, lam_max = white.range()
        self.assertIsNone(npt.assert_allclose(lam_min, -0.09))
        self.assertTrue(np.isinf(lam_max))

        # create the lambdas and reference values for testing
        lam = np.linspace(lam_min, np.max(self.lam_ref), 101)

        # test S function
        def shear_fun(lam):
            return (lam + 0.09)**(0.62)

        # do loop in case hard coded functions cannot take vectors
        shear = np.zeros_like(lam)
        for i, l in enumerate(lam):
            shear[i] = shear_fun(l)
        self.assertIsNone(npt.assert_allclose(shear, white.shear(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(white.shear(lam_min),
                                              white.shear(2*lam_min)))

        # test H function
        def shape_fun(lam):
            z = 0.25-lam
            return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5

        # do loop in case hard coded functions cannot take vectors
        shape = np.zeros_like(lam)
        for i, l in enumerate(lam):
            shape[i] = shape_fun(l)
        self.assertIsNone(npt.assert_allclose(shape, white.shape(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(white.shape(lam_min),
                                              white.shape(2*lam_min)))

        # test H' function
        shape_p = np.zeros_like(lam)
        delta = 1e-5
        for i, l in enumerate(lam):
            shape_p[i] = fd(shape_fun, l, l*delta, n=1, order=3)
        self.assertIsNone(npt.assert_allclose(shape_p, white.shape_p(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(white.shape(lam_min),
                                              white.shape(2*lam_min)))

    def test_cebeci_bradshaw_functions(self):
        """Test the Cebeci & Bradshaw fits."""

        cb = _ThwaitesFunctionsCebeciBradshaw()

        # test the range of lambda
        lam_min, lam_max = cb.range()
        self.assertIsNone(npt.assert_allclose(lam_min, -0.1))
        self.assertIsNone(npt.assert_allclose(lam_max, 0.1))

        # create the lambdas and reference values for testing
        lam = np.linspace(lam_min, lam_max, 101)

        # test S function
        def shear_fun(lam):
            if lam < 0:
                return 0.22 + 1.402*lam + 0.018*lam/(0.107 + lam)
            return 0.22 + 1.57*lam - 1.8*lam**2

        # do loop in case hard coded functions cannot take vectors
        shear = np.zeros_like(lam)
        for i, l in enumerate(lam):
            shear[i] = shear_fun(l)
        self.assertIsNone(npt.assert_allclose(shear, cb.shear(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(cb.shear(lam_min),
                                              cb.shear(2*lam_min)))
        self.assertIsNone(npt.assert_allclose(cb.shear(lam_max),
                                              cb.shear(2*lam_max)))

        # test H function
        def shape_fun(lam):
            if lam < 0:
                return 2.088 + 0.0731/(0.14 + lam)
            return 2.61 - 3.75*lam + 5.24*lam**2

        # do loop in case hard coded functions cannot take vectors
        shape = np.zeros_like(lam)
        for i, l in enumerate(lam):
            shape[i] = shape_fun(l)
        self.assertIsNone(npt.assert_allclose(shape, cb.shape(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(cb.shape(lam_min),
                                              cb.shape(2*lam_min)))
        self.assertIsNone(npt.assert_allclose(cb.shape(lam_max),
                                              cb.shape(2*lam_max)))

        # test H' function
        # NOTE: Since H is discontinuous at 0, so is H' so remove that case
        lam = lam[lam != 0]
        shape_p = np.zeros_like(lam)
        delta = 1e-5
        for i, l in enumerate(lam):
            shape_p[i] = fd(shape_fun, l, np.maximum(l*delta, delta), n=1,
                            order=3)

        self.assertIsNone(npt.assert_allclose(shape_p, cb.shape_p(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(cb.shape_p(lam_min),
                                              cb.shape_p(2*lam_min)))
        self.assertIsNone(npt.assert_allclose(cb.shape_p(lam_max),
                                              cb.shape_p(2*lam_max)))

    def test_spline_functions(self):
        """Test the spline fits."""

        spline = _ThwaitesFunctionsSpline()

        # test the range of lambda
        lam_min, lam_max = spline.range()
        self.assertIsNone(npt.assert_allclose(lam_min, np.min(self.lam_ref)))
        self.assertIsNone(npt.assert_allclose(lam_max, np.max(self.lam_ref)))

        # create the lambdas and reference values for testing
        lam = self.lam_ref

        # test S function
        shear = self.shear_ref
        self.assertIsNone(npt.assert_allclose(shear, spline.shear(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(spline.shear(lam_min),
                                              spline.shear(2*lam_min)))
        self.assertIsNone(npt.assert_allclose(spline.shear(lam_max),
                                              spline.shear(2*lam_max)))

        # test H function
        shape = self.shape_ref
        self.assertIsNone(npt.assert_allclose(shape, spline.shape(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(spline.shape(lam_min),
                                              spline.shape(2*lam_min)))
        self.assertIsNone(npt.assert_allclose(spline.shape(lam_max),
                                              spline.shape(2*lam_max)))

        # test H' function
        # NOTE: cannot evaluate spline H outside of end points, so finite
        #       difference cannot be done on end points of lambda
        lam = lam[1:-2]
        shape_p = np.zeros_like(lam)
        delta = 1e-8
        for i, l in enumerate(lam):
            shape_p[i] = fd(spline.shape, l, np.maximum(l*delta, delta),
                       n=1, order=3)

        self.assertIsNone(npt.assert_allclose(shape_p, spline.shape_p(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(spline.shape_p(lam_min),
                                              spline.shape_p(2*lam_min)))
        self.assertIsNone(npt.assert_allclose(spline.shape_p(lam_max),
                                              spline.shape_p(2*lam_max)))


class ThwaitesLinearAnalytic:
    """Analytic result for power-law, linear Thwaites method."""

    def __init__(self, u_ref, m, nu, shape_fun, shear_fun):

        self.m = m
        self.u_ref = u_ref
        self.nu = nu
        self.shape_fun = shape_fun
        self.shear_fun = shear_fun

    def v_e(self, x):
        """Return the transpiration velocity."""
        ddelta_ddx = fd(self.delta_d, x, 1e-5, n=1, order=3)
        return self.u_ref*x**self.m*(self.m*self.delta_d(x)/x+ddelta_ddx)

    def delta_d(self, x):
        """Return the displacment thickness."""
        return self.delta_m(x)*self.shape_d(x)

    def delta_m(self, x):
        """Return the momentum thickness."""
        k = np.sqrt(0.45/(5*self.m+1))
        rex_sqrt = np.sqrt(self.u_ref*x**(self.m+1)/self.nu)
        return x*k/rex_sqrt

    def shape_d(self, x):
        """Return the displacement shape factor."""
        if self.m == 0:
            lam = np.zeros_like(x)
        else:
            k = np.sqrt(0.45/(5*self.m+1))
            lam = self.m*k**2*np.ones_like(x)
        return self.shape_fun(lam)

    def tau_w(self, x, rho):
        """Return the wall shear stress."""
        k = np.sqrt(0.45/(5*self.m+1))
        rex_sqrt = np.sqrt(self.u_ref*x**(self.m+1)/self.nu)

        if self.m == 0:
            lam = np.zeros_like(x)
        else:
            lam = self.m*k**2*np.ones_like(x)
        return rho*(self.u_ref*x**self.m)**2*self.shear_fun(lam)/(k*rex_sqrt)


class TestLinearThwaites(unittest.TestCase):
    """Class to test the implementation of the linear Thwaites method"""

    def test_blaisus_case(self):
        """Test the flat plate case."""
        # set parameters
        u_ref = 10
        m = 0
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def u_e_fun(x):
            return u_ref*x**m

        def du_e_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*u_ref*x**(m-1)

        def d2u_e_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*u_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun,
                                  data_fits="Spline")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x),
                                              tm_ref.shape_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              rtol=1e-4))

        # test with White fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun, data_fits="White")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x),
                                              tm_ref.shape_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              rtol=1e-4))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun,
                                  data_fits="Cebeci-Bradshaw")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x),
                                              tm_ref.shape_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              rtol=1e-4))

        # test creating with own functions for S, H, H'
        def shear_fun(lam):
            return (lam + 0.09)**(0.62)

        def shape_fun(lam):
            z = 0.25-lam
            return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5

        def shape_p_fun(lam):
            z = 0.25-lam
            return -(4.14 + z*(-2*83.5 + z*(3*854 + z*(-4*3337 + z*5*4576))))

        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun,
                                  data_fits=(shear_fun, shape_fun,
                                             shape_p_fun))
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, shape_fun, shear_fun)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x),
                                              tm_ref.shape_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              rtol=1e-4))

        # test creating with own functions for S, H
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun,
                                  data_fits=(shear_fun, shape_fun))
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, shape_fun, shear_fun)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x),
                                              tm_ref.shape_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              rtol=1e-4))

        # test creating with invalid name
        with self.assertRaises(ValueError):
            ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                 d2U_edx2=d2u_e_fun, data_fits="My Own")

    def test_wedge_050_case(self):
        """Test the m=0.50 wedge case."""
        # set parameters
        u_ref = 10
        m = 0.5
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def u_e_fun(x):
            return u_ref*x**m

        def du_e_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*u_ref*x**(m-1)

        def d2u_e_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*u_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun,
                                  data_fits="Spline")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              rtol=3e-6))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=1e-5))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              rtol=1e-4))

        # test with White fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun, data_fits="White")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              rtol=3e-6))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=1e-5))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              rtol=1e-4))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun,
                                  data_fits="Cebeci-Bradshaw")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              rtol=3e-6))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=1e-5))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              rtol=1e-4))

    def test_wedge_072_case(self):
        """Test the m=0.72 wedge case."""
        # set parameters
        u_ref = 10
        m = 0.72
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def u_e_fun(x):
            return u_ref*x**m

        def du_e_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*u_ref*x**(m-1)

        def d2u_e_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*u_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun,
                                  data_fits="Spline")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=1e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              rtol=3e-6))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=6e-6))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              rtol=7e-5))

        # test with White fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun, data_fits="White")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=1e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              rtol=3e-6))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=6e-6))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              rtol=7e-5))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun,
                                  data_fits="Cebeci-Bradshaw")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=1e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              rtol=3e-6))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=6e-6))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              rtol=8e-5))

    def test_wedge_100_case(self):
        """Test the m=1.00 wedge case."""
        # set parameters
        u_ref = 10
        m = 1
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def u_e_fun(x):
            return u_ref*x**m

        def du_e_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*u_ref*x**(m-1)

        def d2u_e_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*u_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun,
                                  data_fits="Spline")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x)))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x),
                                              tm_ref.shape_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho)))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x)))

        # test with White fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun, data_fits="White")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x)))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x),
                                              tm_ref.shape_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho)))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x)))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun,
                                  data_fits="Cebeci-Bradshaw")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x)))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x),
                                              tm_ref.shape_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho)))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x)))


class TestNonlinearThwaites(unittest.TestCase):
    """Class to test the implementation of the nonlinear Thwaites method"""

    def test_blaisus_case(self):
        """Test the flat plate case."""
        # set parameters
        u_ref = 10
        m = 0
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def u_e_fun(x):
            return u_ref*x**m

        def du_e_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*u_ref*x**(m-1)

        def d2u_e_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*u_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Spline")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m =tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-2))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=2e-2))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x),
                                              tm_ref.shape_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=2e-2))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=3e-2))

        # test with White fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="White")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x),
                                              tm_ref.shape_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=2e-3))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Cebeci-Bradshaw")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-2))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=2e-2))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x),
                                              tm_ref.shape_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=2e-2))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=3e-2))

        # test creating with own functions for S, H, H'
        def shear_fun(lam):
            return (lam + 0.09)**(0.62)

        def shape_fun(lam):
            z = 0.25-lam
            return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5

        def shape_p_fun(lam):
            z = 0.25-lam
            return -(4.14 + z*(-2*83.5 + z*(3*854 + z*(-4*3337 + z*5*4576))))

        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits=(shear_fun, shape_fun,
                                                shape_p_fun))
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, shape_fun, shear_fun)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x),
                                              tm_ref.shape_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=2e-3))

        # test creating with own functions for S, H
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits=(shear_fun, shape_fun))
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, shape_fun, shear_fun)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x),
                                              tm_ref.shape_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=2e-3))

        # test creating with invalid name
        with self.assertRaises(ValueError):
            ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                    d2U_edx2=d2u_e_fun,
                                    data_fits="My Own")

    def test_wedge_050_case(self):
        """Test the m=0.50 wedge case."""
        # set parameters
        u_ref = 10
        m = 0.5
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def u_e_fun(x):
            return u_ref*x**m

        def du_e_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*u_ref*x**(m-1)

        def d2u_e_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*u_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Spline")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=3e-3))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=1e-2))

        # test with White fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="White")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=3e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=3e-3))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=1e-2))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Cebeci-Bradshaw")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=3e-3))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=1e-2))

    def test_wedge_072_case(self):
        """Test the m=0.72 wedge case."""
        # set parameters
        u_ref = 10
        m = 0.72
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def u_e_fun(x):
            return u_ref*x**m

        def du_e_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*u_ref*x**(m-1)

        def d2u_e_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*u_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Spline")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=1e-2))

        # test with White fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="White")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=1e-2))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Cebeci-Bradshaw")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=1e-4))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=1e-4))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              atol=0, rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=3e-5))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=3e-4))

    def test_wedge_100_case(self):
        """Test the m=1.00 wedge case."""
        # set parameters
        u_ref = 10
        m = 1
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def u_e_fun(x):
            return u_ref*x**m

        def du_e_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*u_ref*x**(m-1)

        def d2u_e_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*u_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Spline")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-4))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=2e-4))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              atol=0, rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-4))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=1e-3))

        # test with White fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="White")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=4e-4))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=5e-4))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              atol=0, rtol=1e-4))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=2e-4))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=3e-3))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Cebeci-Bradshaw")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = tm_ref.delta_m(x[0])
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.shape_d(x), tm_ref.shape_d(x),
                                              atol=0, rtol=4e-4))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                              atol=0, rtol=1e-2))


if __name__ == "__main__":
    unittest.main(verbosity=1)

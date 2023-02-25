#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:09:25 2022

@author: ddmarshall
"""
# pylint: disable=protected-access

import unittest
import numpy as np
import numpy.testing as npt
from scipy.misc import derivative as fd

from pyBL.thwaites_method import ThwaitesMethodLinear
from pyBL.thwaites_method import ThwaitesMethodNonlinear
from pyBL.thwaites_method import _ThwaitesFunctionsWhite
from pyBL.thwaites_method import _ThwaitesFunctionsCebeciBradshaw
from pyBL.thwaites_method import _ThwaitesFunctionsSpline


class TestCurveFits(unittest.TestCase):
    """Class to test various functions and curve fits for Thwaites method"""

    F_ref = np.array([0.938, 0.953, 0.956, 0.962, 0.967, 0.969, 0.971,
                      0.970, 0.963, 0.952, 0.936, 0.919, 0.902, 0.886,
                      0.854, 0.825, 0.797, 0.770, 0.744, 0.691, 0.640,
                      0.590, 0.539, 0.490, 0.440, 0.342, 0.249, 0.156,
                      0.064,-0.028,-0.138,-0.251,-0.362,-0.702,-1.000])
    S_ref = np.array([0.000, 0.011, 0.016, 0.024, 0.030, 0.035, 0.039,
                      0.049, 0.055, 0.067, 0.076, 0.083, 0.089, 0.094,
                      0.104, 0.113, 0.122, 0.130, 0.138, 0.153, 0.168,
                      0.182, 0.195, 0.208, 0.220, 0.244, 0.268, 0.291,
                      0.313, 0.333, 0.359, 0.382, 0.404, 0.463, 0.500])
    H_ref = np.array([3.70,  3.69,  3.66,  3.63,  3.61,  3.59,  3.58,
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
        lam_min = spline._tab_lambda[0]
        lam_max = spline._tab_lambda[-1]
        self.assertIsNone(npt.assert_allclose(lam_min, np.min(self.lam_ref)))
        self.assertIsNone(npt.assert_allclose(lam_max, np.max(self.lam_ref)))

        # test the tabular values
        lam = spline._tab_lambda
        H = spline._tab_H
        S = spline._tab_S
        F = spline._tab_F
        self.assertIsNone(npt.assert_allclose(self.lam_ref, lam))
        self.assertIsNone(npt.assert_allclose(self.H_ref, H))
        self.assertIsNone(npt.assert_allclose(self.S_ref, S))
        self.assertIsNone(npt.assert_allclose(self.F_ref, F))

        # compare the tabulated F with the calculated value
        F_calc = 2*(S-lam*(2+H))
        self.assertIsNone(npt.assert_allclose(self.F_ref, F_calc,
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
        def S_fun(lam):
            return (lam + 0.09)**(0.62)

        # do loop in case hard coded functions cannot take vectors
        S = np.zeros_like(lam)
        for i, l in enumerate(lam):
            S[i] = S_fun(l)
        self.assertIsNone(npt.assert_allclose(S, white.S(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(white.S(lam_min),
                                              white.S(2*lam_min)))

        # test H function
        def H_fun(lam):
            z = 0.25-lam
            return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5

        # do loop in case hard coded functions cannot take vectors
        H = np.zeros_like(lam)
        for i, l in enumerate(lam):
            H[i] = H_fun(l)
        self.assertIsNone(npt.assert_allclose(H, white.H(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(white.H(lam_min),
                                              white.H(2*lam_min)))

        # test H' function
        Hp = np.zeros_like(lam)
        delta = 1e-5
        for i, l in enumerate(lam):
            Hp[i] = fd(H_fun, l, l*delta, n=1, order=3)
        self.assertIsNone(npt.assert_allclose(Hp, white.Hp(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(white.H(lam_min),
                                              white.H(2*lam_min)))

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
        def S_fun(lam):
            if lam < 0:
                return 0.22 + 1.402*lam + 0.018*lam/(0.107 + lam)
            return 0.22 + 1.57*lam - 1.8*lam**2

        # do loop in case hard coded functions cannot take vectors
        S = np.zeros_like(lam)
        for i, l in enumerate(lam):
            S[i] = S_fun(l)
        self.assertIsNone(npt.assert_allclose(S, cb.S(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(cb.S(lam_min), cb.S(2*lam_min)))
        self.assertIsNone(npt.assert_allclose(cb.S(lam_max), cb.S(2*lam_max)))

        # test H function
        def H_fun(lam):
            if lam < 0:
                return 2.088 + 0.0731/(0.14 + lam)
            return 2.61 - 3.75*lam + 5.24*lam**2

        # do loop in case hard coded functions cannot take vectors
        H = np.zeros_like(lam)
        for i, l in enumerate(lam):
            H[i] = H_fun(l)
        self.assertIsNone(npt.assert_allclose(H, cb.H(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(cb.H(lam_min), cb.H(2*lam_min)))
        self.assertIsNone(npt.assert_allclose(cb.H(lam_max), cb.H(2*lam_max)))

        # test H' function
        # NOTE: Since H is discontinuous at 0, so is H' so remove that case
        lam = lam[lam != 0]
        Hp = np.zeros_like(lam)
        delta = 1e-5
        for i, l in enumerate(lam):
            Hp[i] = fd(H_fun, l, np.maximum(l*delta, delta), n=1, order=3)

        self.assertIsNone(npt.assert_allclose(Hp, cb.Hp(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(cb.Hp(lam_min),
                                              cb.Hp(2*lam_min)))
        self.assertIsNone(npt.assert_allclose(cb.Hp(lam_max),
                                              cb.Hp(2*lam_max)))

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
        S = self.S_ref
        self.assertIsNone(npt.assert_allclose(S, spline.S(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(spline.S(lam_min),
                                              spline.S(2*lam_min)))
        self.assertIsNone(npt.assert_allclose(spline.S(lam_max),
                                              spline.S(2*lam_max)))

        # test H function
        H = self.H_ref
        self.assertIsNone(npt.assert_allclose(H, spline.H(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(spline.H(lam_min),
                                              spline.H(2*lam_min)))
        self.assertIsNone(npt.assert_allclose(spline.H(lam_max),
                                              spline.H(2*lam_max)))

        # test H' function
        # NOTE: cannot evaluate spline H outside of end points, so finite
        #       difference cannot be done on end points of lambda
        lam = lam[1:-2]
        Hp = np.zeros_like(lam)
        delta = 1e-8
        for i, l in enumerate(lam):
            Hp[i] = fd(spline.H, l, np.maximum(l*delta, delta),
                       n=1, order=3)

        self.assertIsNone(npt.assert_allclose(Hp, spline.Hp(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(npt.assert_allclose(spline.Hp(lam_min),
                                              spline.Hp(2*lam_min)))
        self.assertIsNone(npt.assert_allclose(spline.Hp(lam_max),
                                              spline.Hp(2*lam_max)))


class ThwaitesLinearAnalytic:
    """Analytic result for power-law, linear Thwaites method."""

    def __init__(self, U_ref, m, nu, H_fun, S_fun):

        self.m = m
        self.U_ref = U_ref
        self.nu = nu
        self.H_fun = H_fun
        self.S_fun = S_fun

    def V_e(self, x):
        """Return the transpiration velocity."""
        ddelta_ddx = fd(self.delta_d, x, 1e-5, n=1, order=3)
        return self.U_ref*x**self.m*(self.m*self.delta_d(x)/x+ddelta_ddx)

    def delta_d(self, x):
        """Return the displacment thickness."""
        return self.delta_m(x)*self.H_d(x)

    def delta_m(self, x):
        """Return the momentum thickness."""
        K = np.sqrt(0.45/(5*self.m+1))
        Rex_sqrt = np.sqrt(self.U_ref*x**(self.m+1)/self.nu)
        return x*K/Rex_sqrt

    def H_d(self, x):
        """Return the displacement shape factor."""
        if self.m == 0:
            lam = np.zeros_like(x)
        else:
            K = np.sqrt(0.45/(5*self.m+1))
            lam = self.m*K**2*np.ones_like(x)
        return self.H_fun(lam)

    def tau_w(self, x, rho):
        """Return the wall shear stress."""
        K = np.sqrt(0.45/(5*self.m+1))
        Rex_sqrt = np.sqrt(self.U_ref*x**(self.m+1)/self.nu)

        if self.m == 0:
            lam = np.zeros_like(x)
        else:
            lam = self.m*K**2*np.ones_like(x)
        return rho*(self.U_ref*x**self.m)**2*self.S_fun(lam)/(K*Rex_sqrt)


class TestLinearThwaites(unittest.TestCase):
    """Class to test the implementation of the linear Thwaites method"""

    def testBlaisusCase(self):
        """Test the flat plate case."""
        # set parameters
        U_ref = 10
        m = 0
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def U_e_fun(x):
            return U_ref*x**m

        def dU_edx_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*U_ref*x**(m-1)

        def d2U_edx2_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*U_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun,
                                  data_fits="Spline")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              rtol=1e-4))

        # test with White fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun, data_fits="White")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              rtol=1e-4))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun,
                                  data_fits="Cebeci-Bradshaw")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              rtol=1e-4))

        # test creating with own functions for S, H, H'
        def S_fun(lam):
            return (lam + 0.09)**(0.62)

        def H_fun(lam):
            z = 0.25-lam
            return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5

        def Hp_fun(lam):
            z = 0.25-lam
            return -(4.14 + z*(-2*83.5 + z*(3*854 + z*(-4*3337 + z*5*4576))))

        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun,
                                  data_fits=(S_fun, H_fun, Hp_fun))
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, H_fun, S_fun)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              rtol=1e-4))

        # test creating with own functions for S, H
        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun,
                                  data_fits=(S_fun, H_fun))
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, H_fun, S_fun)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              rtol=1e-4))

        # test creating with invalid name
        with self.assertRaises(ValueError):
            ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                 d2U_edx2=d2U_edx2_fun, data_fits="My Own")

    def testWedge050Case(self):
        """Test the m=0.50 wedge case."""
        # set parameters
        U_ref = 10
        m = 0.5
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def U_e_fun(x):
            return U_ref*x**m

        def dU_edx_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*U_ref*x**(m-1)

        def d2U_edx2_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*U_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun,
                                  data_fits="Spline")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              rtol=3e-6))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=1e-5))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              rtol=1e-4))

        # test with White fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun, data_fits="White")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              rtol=3e-6))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=1e-5))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              rtol=1e-4))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun,
                                  data_fits="Cebeci-Bradshaw")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              rtol=3e-6))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=1e-5))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              rtol=1e-4))

    def testWedge072Case(self):
        """Test the m=0.72 wedge case."""
        # set parameters
        U_ref = 10
        m = 0.72
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def U_e_fun(x):
            return U_ref*x**m

        def dU_edx_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*U_ref*x**(m-1)

        def d2U_edx2_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*U_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun,
                                  data_fits="Spline")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=1e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              rtol=3e-6))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=6e-6))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              rtol=7e-5))

        # test with White fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun, data_fits="White")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=1e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              rtol=3e-6))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=6e-6))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              rtol=7e-5))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun,
                                  data_fits="Cebeci-Bradshaw")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x), rtol=1e-5))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x), rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              rtol=3e-6))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho), rtol=6e-6))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              rtol=8e-5))

    def testWedge100Case(self):
        """Test the m=1.00 wedge case."""
        # set parameters
        U_ref = 10
        m = 1
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def U_e_fun(x):
            return U_ref*x**m

        def dU_edx_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*U_ref*x**(m-1)

        def d2U_edx2_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*U_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun,
                                  data_fits="Spline")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x)))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho)))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x)))

        # test with White fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun, data_fits="White")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x)))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho)))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x)))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                  d2U_edx2=d2U_edx2_fun,
                                  data_fits="Cebeci-Bradshaw")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x),
                                              tm_ref.delta_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x),
                                              tm_ref.delta_m(x)))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho)))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x)))


class TestNonlinearThwaites(unittest.TestCase):
    """Class to test the implementation of the nonlinear Thwaites method"""

    def testBlaisusCase(self):
        """Test the flat plate case."""
        # set parameters
        U_ref = 10
        m = 0
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def U_e_fun(x):
            return U_ref*x**m

        def dU_edx_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*U_ref*x**(m-1)

        def d2U_edx2_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*U_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits="Spline")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-2))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=2e-2))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=2e-2))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=3e-2))

        # test with White fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits="White")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=2e-3))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits="Cebeci-Bradshaw")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-2))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=2e-2))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=2e-2))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=3e-2))

        # test creating with own functions for S, H, H'
        def S_fun(lam):
            return (lam + 0.09)**(0.62)

        def H_fun(lam):
            z = 0.25-lam
            return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5

        def Hp_fun(lam):
            z = 0.25-lam
            return -(4.14 + z*(-2*83.5 + z*(3*854 + z*(-4*3337 + z*5*4576))))

        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits=(S_fun, H_fun, Hp_fun))
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, H_fun, S_fun)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=2e-3))

        # test creating with own functions for S, H
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits=(S_fun, H_fun))
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, H_fun, S_fun)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x)))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=2e-3))

        # test creating with invalid name
        with self.assertRaises(ValueError):
            ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                    d2U_edx2=d2U_edx2_fun,
                                    data_fits="My Own")

    def testWedge050Case(self):
        """Test the m=0.50 wedge case."""
        # set parameters
        U_ref = 10
        m = 0.5
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def U_e_fun(x):
            return U_ref*x**m

        def dU_edx_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*U_ref*x**(m-1)

        def d2U_edx2_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*U_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits="Spline")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=3e-3))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=1e-2))

        # test with White fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits="White")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=3e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=3e-3))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=1e-2))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits="Cebeci-Bradshaw")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=3e-3))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=1e-2))

    def testWedge072Case(self):
        """Test the m=0.72 wedge case."""
        # set parameters
        U_ref = 10
        m = 0.72
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def U_e_fun(x):
            return U_ref*x**m

        def dU_edx_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*U_ref*x**(m-1)

        def d2U_edx2_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*U_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits="Spline")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=1e-2))

        # test with White fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits="White")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=1e-2))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits="Cebeci-Bradshaw")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=1e-4))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=1e-4))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              atol=0, rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=3e-5))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=3e-4))

    def testWedge100Case(self):
        """Test the m=1.00 wedge case."""
        # set parameters
        U_ref = 10
        m = 1
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def U_e_fun(x):
            return U_ref*x**m

        def dU_edx_fun(x):
            if m == 0:
                return np.zeros_like(x)
            return m*U_ref*x**(m-1)

        def d2U_edx2_fun(x):
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*U_ref*x**(m-2)

        # test with spline of tabular data
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits="Spline")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-4))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=2e-4))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              atol=0, rtol=5e-5))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-4))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=1e-3))

        # test with White fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits="White")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=4e-4))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=5e-4))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              atol=0, rtol=1e-4))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=2e-4))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=3e-3))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=U_e_fun, dU_edx=dU_edx_fun,
                                     d2U_edx2=d2U_edx2_fun,
                                     data_fits="Cebeci-Bradshaw")
        tm_ref = ThwaitesLinearAnalytic(U_ref, m, nu, tm._model.H, tm._model.S)
        tm.set_initial_parameters(delta_m0=tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(npt.assert_allclose(tm.delta_d(x), tm_ref.delta_d(x),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.delta_m(x), tm_ref.delta_m(x),
                                              atol=0, rtol=2e-3))
        self.assertIsNone(npt.assert_allclose(tm.H_d(x), tm_ref.H_d(x),
                                              atol=0, rtol=4e-4))
        self.assertIsNone(npt.assert_allclose(tm.tau_w(x, rho),
                                              tm_ref.tau_w(x, rho),
                                              atol=0, rtol=1e-3))
        self.assertIsNone(npt.assert_allclose(tm.V_e(x), tm_ref.V_e(x),
                                              atol=0, rtol=1e-2))


if __name__ == "__main__":
    unittest.main(verbosity=1)

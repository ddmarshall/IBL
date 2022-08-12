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

from pyBL.thwaites_method import ThwaitesMethod as tm


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
                         0.000, 0.016,  0.032,  0.048,  0.064,  0.080,
                         0.10,  0.12,   0.14,   0.20,   0.25])
    
    def test_tabular_values(self):
        
        # test the range of lambda
        lam_min, lam_max = tm._tabular_range()
        self.assertIsNone(npt.assert_allclose(lam_min, np.min(self.lam_ref)))
        self.assertIsNone(npt.assert_allclose(lam_max, np.max(self.lam_ref)))
        
        # test the tabular values
        lam, H, S, F = tm._tabular_data()
        self.assertIsNone(npt.assert_allclose(self.lam_ref, lam))
        self.assertIsNone(npt.assert_allclose(self.H_ref, H))
        self.assertIsNone(npt.assert_allclose(self.S_ref, S))
        self.assertIsNone(npt.assert_allclose(self.F_ref, F))
        
        # compare the tabulated F with the calculated value
        F_calc = 2*(S-lam*(2+H))
        self.assertIsNone(npt.assert_allclose(self.F_ref, F_calc, rtol=0, atol=1e-2))
        
    def test_white_functions(self):
        
        # test the range of lambda
        lam_min, lam_max = tm._white_range()
        self.assertIsNone(npt.assert_allclose(lam_min, -0.09))
        self.assertTrue(np.isinf(lam_max))
        
        # create the lambdas and reference values for testing
        lam = np.linspace(lam_min, np.minimum(np.amax(self.lam_ref), lam_max), 101)
        
        # test S function
        def S_fun(lam):
            return (lam + 0.09)**(0.62)
        
        S=np.zeros_like(lam)
        for i, l in enumerate(lam):
            S[i] = S_fun(l)
        
        self.assertIsNone(npt.assert_allclose(S, tm._white_S(lam)))
        
        # check to make sure raises error when asked for out of range data
        with self.assertRaises(ValueError):
            tm._white_S(2*lam_min)
        
        # test H function
        def H_fun(lam):
            z = 0.25-lam
            return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5
        
        H=np.zeros_like(lam)
        for i, l in enumerate(lam):
            H[i] = H_fun(l)
        
        self.assertIsNone(npt.assert_allclose(H, tm._white_H(lam)))
        
        # check to make sure raises error when asked for out of range data
        with self.assertRaises(ValueError):
            tm._white_H(2*lam_min)
        
        # test H' function
        Hp=np.zeros_like(lam)
        delta = 1e-5
        for i, l in enumerate(lam):
            Hp[i] = fd(H_fun, l, l*delta, n=1, order=3)
        
        self.assertIsNone(npt.assert_allclose(Hp, tm._white_Hp(lam)))
        
        # check to make sure raises error when asked for out of range data
        with self.assertRaises(ValueError):
            tm._white_Hp(2*lam_min)
    
    def test_cebeci_and_bradshaw_functions(self):
        
        # test the range of lambda
        lam_min, lam_max = tm._cb_range()
        self.assertIsNone(npt.assert_allclose(lam_min, -0.1))
        self.assertIsNone(npt.assert_allclose(lam_max, 0.1))
        
        # create the lambdas and reference values for testing
        lam = np.linspace(lam_min, lam_max, 101)
        
        # test S function
        def S_fun(lam):
            if (lam < 0):
                return 0.22 + 1.402*lam + 0.018*lam/(0.107 + lam)
            else:
                return 0.22 + 1.57*lam - 1.8*lam**2
        
        S=np.zeros_like(lam)
        for i, l in enumerate(lam):
            S[i] = S_fun(l)
        
        self.assertIsNone(npt.assert_allclose(S, tm._cb_S(lam)))
        
        # check to make sure raises error when asked for out of range data
        with self.assertRaises(ValueError):
            tm._cb_S(2*lam_min)
        with self.assertRaises(ValueError):
            tm._cb_S(2*lam_max)
        
        # test H function
        def H_fun(lam):
            if (lam < 0):
                return 2.088 + 0.0731/(0.14 + lam)
            else:
                return 2.61 - 3.75*lam + 5.24*lam**2
        
        H=np.zeros_like(lam)
        for i, l in enumerate(lam):
            H[i] = H_fun(l)
        
        self.assertIsNone(npt.assert_allclose(H, tm._cb_H(lam)))
        
        # check to make sure raises error when asked for out of range data
        with self.assertRaises(ValueError):
            tm._cb_H(2*lam_min)
        with self.assertRaises(ValueError):
            tm._cb_H(2*lam_max)
        
        # test H' function
        # NOTE: Since H is discontinuous at 0, so is H' so remove that case
        lam=lam[lam!=0]
        Hp=np.zeros_like(lam)
        delta = 1e-5
        for i, l in enumerate(lam):
            Hp[i] = fd(H_fun, l, max(l*delta, delta), n=1, order=3)
        
        self.assertIsNone(npt.assert_allclose(Hp, tm._cb_Hp(lam)))
        
        # check to make sure raises error when asked for out of range data
        with self.assertRaises(ValueError):
            tm._cb_Hp(2*lam_min)
        with self.assertRaises(ValueError):
            tm._cb_Hp(2*lam_max)
    
    def test_spline_functions(self):
        
        # test the range of lambda
        lam_min, lam_max = tm._spline_range()
        self.assertIsNone(npt.assert_allclose(lam_min, np.amin(self.lam_ref)))
        self.assertIsNone(npt.assert_allclose(lam_max, np.amax(self.lam_ref)))
        
        # create the lambdas and reference values for testing
        lam = self.lam_ref
        
        # test S function
        S=self.S_ref
        self.assertIsNone(npt.assert_allclose(S, tm._spline_S(lam)))
        
        # check to make sure raises error when asked for out of range data
        with self.assertRaises(ValueError):
            tm._spline_S(2*lam_min)
        with self.assertRaises(ValueError):
            tm._spline_S(2*lam_max)
        
        # test H function
        H=self.H_ref
        self.assertIsNone(npt.assert_allclose(H, tm._spline_H(lam)))
        
        # check to make sure raises error when asked for out of range data
        with self.assertRaises(ValueError):
            tm._spline_H(2*lam_min)
        with self.assertRaises(ValueError):
            tm._spline_H(2*lam_max)
        
        # test H' function
        # NOTE: cannot evaluate spline H outside of end points, so finite 
        #       difference cannot be done on end points of lambda
        lam = lam[1:-2]
        Hp=np.zeros_like(lam)
        delta = 1e-8
        for i, l in enumerate(lam):
            Hp[i] = fd(tm._spline_H, l, max(l*delta, delta), n=1, order=3)
        
        self.assertIsNone(npt.assert_allclose(Hp, tm._spline_Hp(lam)))
        
        # check to make sure raises error when asked for out of range data
        with self.assertRaises(ValueError):
            tm._spline_Hp(2*lam_min)
        with self.assertRaises(ValueError):
            tm._spline_Hp(2*lam_max)


if (__name__ == "__main__"):
    unittest.main(verbosity=1)

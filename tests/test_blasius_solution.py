#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 17:23:25 2022

@author: ddmarshall
"""

import unittest
import numpy as np
import numpy.testing as npt
from scipy.integrate import quadrature

from pyBL.blasius_solution import BlasiusSolution


class TestCurveFits(unittest.TestCase):
    """Class to test various functions and curve fits for Thwaites method"""
    
    # Tabluated data from White (2011)
    # Note that there are errors in the LSD in:
    #    * f at eta = (4.0, 4.2, 4.6, 5.2)
    #    * f' at eta = (1.2, 5.2)
    eta_ref = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2,
                        2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4,
                        4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0])
    f_ref = np.array([0.00000, 0.00235, 0.00939, 0.02113, 0.03755, 0.05864,
                      0.08439, 0.11474, 0.14967, 0.18911, 0.23299, 0.28121,
                      0.33366, 0.39021, 0.45072, 0.51503, 0.58296, 0.65430,
                      0.72887, 0.80644, 0.88680, 1.05495, 1.23153, 1.41482,
                      1.60328, 1.79557, 1.99058, 2.18747, 2.38559, 2.58450,
                      2.78388, 2.98355, 3.18338, 3.38329, 3.58325, 3.78323,
                      3.98322, 4.18322, 4.38322, 4.58322, 4.78322])
    fp_ref = np.array([0.00000,  0.04696,  0.09391,  0.14081,  0.18761,
                       0.23423,  0.28058,  0.32653,  0.37196,  0.41672,
                       0.46063,  0.50354,  0.54525,  0.58559,  0.62439,
                       0.66147,  0.69670,  0.72993,  0.76106,  0.79000,
                       0.81669,  0.86330,  0.90107,  0.93060,  0.95288,
                       0.96905,  0.98037,  0.98797,  0.99289,  0.99594,
                       0.99777,  0.99882,  0.99940,  0.99970,  0.99986,
                       0.99994,  0.999971, 0.999988, 0.999995, 0.999998,
                       0.999999])
    fpp_ref = np.array([0.46960,  0.46956,  0.46931,  0.46861,  0.46725,
                        0.46503,  0.46173,  0.45718,  0.45119,  0.44363,
                        0.43438,  0.42337,  0.41057,  0.39598,  0.37969,
                        0.36180,  0.34249,  0.32195,  0.30045,  0.27825,
                        0.25567,  0.21058,  0.16756,  0.12861,  0.09511,
                        0.06771,  0.04637,  0.03054,  0.01933,  0.01176,
                        0.00687,  0.00386,  0.00208,  0.00108,  0.00054,
                        0.00026,  0.000119, 0.000052, 0.000022, 0.000009,
                        0.000004])
    eta_d_ref = 1.21678
    eta_m_ref = 0.46960
    eta_s_ref = 3.5
    eta_k_ref = 0
    V_e_term = 0.8604
    
    def testBasicSolution(self):
        U_inf = 10
        nu = 1e-5
        bs = BlasiusSolution(U_ref = U_inf, fpp0 = self.fpp_ref[0], nu = nu)
        
        ## Test the solution for f
        f = bs.f(self.eta_ref)
        self.assertIsNone(npt.assert_allclose(f, self.f_ref, rtol=0, atol=1e-5))
        
        ## Test the solution for f'
        fp = bs.fp(self.eta_ref)
        self.assertIsNone(npt.assert_allclose(fp, self.fp_ref, rtol=0,
                                              atol=1e-5))
        
        ## Test the solution for f''
        fpp = bs.fpp(self.eta_ref)
        self.assertIsNone(npt.assert_allclose(fpp, self.fpp_ref, rtol=0,
                                              atol=1e-5))
    
    def testBoundaryLayerParameters(self):
        U_inf = 10
        nu = 1e-5
        rho = 1
        bs = BlasiusSolution(U_ref = U_inf, fpp0 = self.fpp_ref[0], nu = nu)
        
        ## Test the values in terms of eta
        # displacement thickness
        eta_d = bs.eta_d()
        self.assertIsNone(npt.assert_allclose(eta_d, self.eta_d_ref, rtol = 0,
                                              atol = 1e-4))
        # momentum thickness
        eta_m = bs.eta_m()
        self.assertIsNone(npt.assert_allclose(eta_m, self.eta_m_ref, rtol = 0,
                                              atol = 1e-4))
        
        # shear thickness
        eta_s = bs.eta_s()
        self.assertIsNone(npt.assert_allclose(eta_s, self.eta_s_ref, rtol = 0,
                                              atol = 1e-1))
        
        # kinetic energy thickness
        def fun(eta):
            fp = bs.fp(eta)
            return fp*(1-fp**2)
        eta_k_ref = quadrature(fun, 0, 10)[0]
        
        eta_k = bs.eta_k()
        self.assertIsNone(npt.assert_allclose(eta_k, eta_k_ref, rtol = 1e-6,
                          atol = 0))
        
        ## Test the values in terms of x
        x = np.linspace(0.2, 2, 101)
        g = np.sqrt(0.5*U_inf/(nu*x))
        
        # test the transpiration velocity
        V_e_ref = nu*g*np.sqrt(2)*self.V_e_term
        self.assertIsNone(npt.assert_allclose(bs.V_e(x), V_e_ref, rtol = 1e-5,
                                              atol = 0))

        # test displacement thickness
        delta_d_ref = eta_d/g
        self.assertIsNone(npt.assert_allclose(bs.delta_d(x), delta_d_ref))
        
        # test momentum thickness
        delta_m_ref = eta_m/g
        self.assertIsNone(npt.assert_allclose(bs.delta_m(x), delta_m_ref))
        
        # test kinetic energy thickness
        delta_k_ref = eta_k/g
        self.assertIsNone(npt.assert_allclose(bs.delta_k(x), delta_k_ref))
        
        # test shear thickness
        delta_s_ref = eta_s/g
        self.assertIsNone(npt.assert_allclose(bs.delta_s(x), delta_s_ref))
        
        # test shape factors
        H_d_ref = delta_d_ref/delta_m_ref
        H_k_ref = delta_k_ref/delta_m_ref
        self.assertIsNone(npt.assert_allclose(bs.H_d(x), H_d_ref))
        self.assertIsNone(npt.assert_allclose(bs.H_k(x), H_k_ref))
        
        # test wall shear stress
        tau_w_ref = rho*nu*U_inf*g*bs.fpp(0)
        self.assertIsNone(npt.assert_allclose(bs.tau_w(x, rho), tau_w_ref))
        
        # test dissipation
        def fun(eta):
            return bs.fpp(eta)**2
        D_ref = rho*nu*U_inf**2*g*quadrature(fun, 0, 10)[0]
        self.assertIsNone(npt.assert_allclose(bs.D(x, rho), D_ref))
    
    def testLocalProperties(self):
        U_inf = 10
        nu = 1e-5
        bs = BlasiusSolution(U_ref = U_inf, fpp0 = self.fpp_ref[0], nu = nu)
        
        ## Test the values in terms of x,y
        x0 = 0.4
        y0 = 2e-3
        
        # test eta transformation
        g = np.sqrt(0.5*U_inf/(nu*x0))
        eta_ref = y0*g
        self.assertIsNone(npt.assert_allclose(bs.eta(x0, y0), eta_ref))
        
        # test sample in y
        x = x0
        y = np.linspace(1e-4, 5e-3, 11)
        g = np.sqrt(0.5*U_inf/(nu*x))
        eta_ref = y*g
        u_ref = U_inf*bs.fp(eta_ref)
        v_ref = np.sqrt(0.5*nu*U_inf/x)*(eta_ref*bs.fp(eta_ref)-bs.f(eta_ref))
        self.assertIsNone(npt.assert_allclose(bs.u(x, y), u_ref))
        self.assertIsNone(npt.assert_allclose(bs.v(x, y), v_ref))
        
        
        # test sample in x
        x = np.linspace(0.2, 0.5, 11)
        y = y0
        g = np.sqrt(0.5*U_inf/(nu*x))
        eta_ref = y*g
        u_ref = U_inf*bs.fp(eta_ref)
        v_ref = np.sqrt(0.5*nu*U_inf/x)*(eta_ref*bs.fp(eta_ref)-bs.f(eta_ref))
        self.assertIsNone(npt.assert_allclose(bs.u(x, y), u_ref))
        self.assertIsNone(npt.assert_allclose(bs.v(x, y), v_ref))


if (__name__ == "__main__"):
    unittest.main(verbosity=1)

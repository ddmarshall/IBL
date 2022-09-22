#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 01:17:56 2022

@author: ddmarshall
"""

import unittest
import numpy as np
import numpy.testing as npt
from scipy.integrate import quadrature

from pyBL.falkner_skan import FalknerSkanSolution


class TestCurveFits(unittest.TestCase):
    """Class to test various functions and curve fits for Thwaites method"""

    # Tabluated data from White (2011)
    # Note that there are errors in the data:
    #    * beta = -0.18 @ eta = 5.0
    #    * CASE 2
    eta_ref = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2,
                        3.4, 3.6, 3.8, 4.0, 4.5, 5.0])
    fp_ref = np.array([[0.00000, 0.00099, 0.00398, 0.00895, 0.01591, 0.02485,
                        0.03578, 0.04868, 0.06355, 0.08038, 0.09913, 0.14232,
                        0.19274, 0.24982, 0.31271, 0.38026, 0.45097, 0.52308,
                        0.59460, 0.66348, 0.72776, 0.78578, 0.83635, 0.87882,
                        0.91315, 0.93982, 0.97940, 0.99439],
                       [0.00000, 0.01376, 0.02933, 0.04668, 0.06582, 0.08673,
                        0.10937, 0.13373, 0.15975, 0.18737, 0.21651, 0.27899,
                        0.34622, 0.41691, 0.48946, 0.56205, 0.63269, 0.69942,
                        0.76048, 0.81449, 0.86061, 0.89853, 0.92854, 0.95138,
                        0.96805, 0.97975, 0.99449, 0.99881],
                       [0.00000, 0.04696, 0.09391, 0.14081, 0.18761, 0.23423,
                        0.28058, 0.32653, 0.37196, 0.41672, 0.46063, 0.54525,
                        0.62439, 0.69670, 0.76106, 0.81669, 0.86330, 0.90107,
                        0.93060, 0.95288, 0.96905, 0.98037, 0.98797, 0.99289,
                        0.99594, 0.99777, 0.99957, 0.99994],
                       [0.00000, 0.07597, 0.14894, 0.21886, 0.28569, 0.34938,
                        0.40988, 0.46713, 0.52107, 0.57167, 0.61890, 0.70322,
                        0.77425, 0.83254, 0.87906, 0.91509, 0.94211, 0.96173,
                        0.97548, 0.98480, 0.99088, 0.99471, 0.99704, 0.99840,
                        0.99916, 0.99958, 0.99994, 0.99999],
                       [0.00000, 0.11826, 0.22661, 0.32524, 0.41446, 0.49465,
                        0.56628, 0.62986, 0.68594, 0.73508, 0.77787, 0.84667,
                        0.89681, 0.93235, 0.95683, 0.97322, 0.98385, 0.99055,
                        0.99463, 0.99705, 0.99842, 0.99919, 0.99959, 0.99980,
                        0.99991, 0.99996, 0.99999, 1.00000],
                       [0.00000, 0.15876, 0.29794, 0.41854, 0.52190, 0.60964,
                        0.68343, 0.74496, 0.79587, 0.83767, 0.87172, 0.92142,
                        0.95308, 0.97269, 0.98452, 0.99146, 0.99542, 0.99761,
                        0.99879, 0.99940, 0.99972, 0.99987, 0.99995, 0.99998,
                        0.99999, 1.00000, 1.00000, 1.00000]])
    beta_ref = np.array([-0.19884, -0.18, 0, 0.3, 1.0, 2.0])
    fpp0_ref = np.array([0, 0.12864, 0.46960, 0.77476, 1.23259, 1.68722])
    eta_d_ref = np.array([2.35885, 1.87157, 1.21678, 0.91099, 0.64790,
                          0.49743])
    eta_m_ref = np.array([0.58544, 0.56771, 0.46960, 0.38574, 0.29235,
                          0.23079])
    eta_s_ref = np.array([4.79, 4.28, 3.47, 2.965, 2.379, 1.9])

    def testBeta0Solution(self):
        """Test the first case from White table."""
        # pylint: disable=too-many-locals
        idx = 0
        U_inf = 10
        nu = 1e-5
        rho = 1
        beta = self.beta_ref[idx] + 1e-6  # hack to get this case to work
        m = beta/(2-beta)
        fs = FalknerSkanSolution(U_ref=U_inf, m=beta/(2-beta), nu=nu)

        # Test the solution for f'
        fp = fs.fp(self.eta_ref)
        self.assertIsNone(npt.assert_allclose(fp, self.fp_ref[idx,:], rtol=0,
                                              atol=1e-5))

        # Test the solved boundary condition
        fpp0 = fs.fpp(0)
        self.assertIsNone(npt.assert_allclose(fpp0, self.fpp0_ref[idx],
                                              rtol=0, atol=1e-5))

        # Test the boundary layer values
        #
        # similarity terms
        eta_d = fs.eta_d()
        self.assertIsNone(npt.assert_allclose(eta_d, self.eta_d_ref[idx],
                                              rtol=0, atol=1e-4))
        eta_m = fs.eta_m()
        self.assertIsNone(npt.assert_allclose(eta_m, self.eta_m_ref[idx],
                                              rtol=0, atol=1e-5))
        eta_s = fs.eta_s()
        self.assertIsNone(npt.assert_allclose(eta_s, self.eta_s_ref[idx],
                                              rtol=0, atol=1e-2))

        def ke_fun(eta):
            fp = fs.fp(eta)
            return fp*(1-fp**2)
        eta_k_ref = quadrature(ke_fun, 0, 10)[0]
        eta_k = fs.eta_k()
        self.assertIsNone(npt.assert_allclose(eta_k, eta_k_ref, rtol=1e-4,
                          atol=0))

        # dimensional terms
        x = np.linspace(0.2, 2, 101)
        g = np.sqrt(0.5*(m+1)*U_inf*x**m/(nu*x))
        U_e = U_inf*x**m
        delta_d_ref = eta_d/g
        self.assertIsNone(npt.assert_allclose(fs.delta_d(x), delta_d_ref))
        delta_m_ref = eta_m/g
        self.assertIsNone(npt.assert_allclose(fs.delta_m(x), delta_m_ref))
        delta_k_ref = eta_k/g
        self.assertIsNone(npt.assert_allclose(fs.delta_k(x), delta_k_ref))
        delta_s_ref = eta_s/g
        self.assertIsNone(npt.assert_allclose(fs.delta_s(x), delta_s_ref))
        H_d_ref = delta_d_ref/delta_m_ref
        H_k_ref = delta_k_ref/delta_m_ref
        self.assertIsNone(npt.assert_allclose(fs.H_d(x), H_d_ref))
        self.assertIsNone(npt.assert_allclose(fs.H_k(x), H_k_ref))

        # test wall shear stress
        tau_w_ref = rho*nu*U_e*g*fs.fpp(0)
        self.assertIsNone(npt.assert_allclose(fs.tau_w(x, rho), tau_w_ref))

        # test dissipation
        def D_fun(eta):
            return fs.fpp(eta)**2
        D_ref = rho*nu*U_e**2*g*quadrature(D_fun, 0, 10)[0]
        self.assertIsNone(npt.assert_allclose(fs.D(x, rho), D_ref, rtol=2e-5,
                          atol=0))

    def testBeta1Solution(self):
        """Test the second case from White table."""
        # pylint: disable=too-many-locals
        idx = 1
        U_inf = 10
        nu = 1e-5
        rho = 1
        beta = self.beta_ref[idx]
        m = beta/(2-beta)
        fs = FalknerSkanSolution(U_ref=U_inf, m=beta/(2-beta), nu=nu)

        # Test the solution for f'
        fp = fs.fp(self.eta_ref)
        self.assertIsNone(npt.assert_allclose(fp, self.fp_ref[idx,:], rtol=0,
                                              atol=1e-5))

        # Test the solved boundary condition
        fpp0 = fs.fpp(0)
        self.assertIsNone(npt.assert_allclose(fpp0, self.fpp0_ref[idx],
                                              rtol=0, atol=1e-5))

        # Test the boundary layer values
        #
        # similarity terms
        eta_d = fs.eta_d()
        self.assertIsNone(npt.assert_allclose(eta_d, self.eta_d_ref[idx],
                                              rtol=0, atol=1e-5))
        eta_m = fs.eta_m()
        self.assertIsNone(npt.assert_allclose(eta_m, self.eta_m_ref[idx],
                                              rtol=0, atol=1e-5))
        eta_s = fs.eta_s()
        self.assertIsNone(npt.assert_allclose(eta_s, self.eta_s_ref[idx],
                                              rtol=0, atol=1e-2))

        def ke_fun(eta):
            fp = fs.fp(eta)
            return fp*(1-fp**2)
        eta_k_ref = quadrature(ke_fun, 0, 10)[0]
        eta_k = fs.eta_k()
        self.assertIsNone(npt.assert_allclose(eta_k, eta_k_ref, rtol=1e-5,
                          atol=0))

        # dimensional terms
        x = np.linspace(0.2, 2, 101)
        g = np.sqrt(0.5*(m+1)*U_inf*x**m/(nu*x))
        U_e = U_inf*x**m
        delta_d_ref = eta_d/g
        self.assertIsNone(npt.assert_allclose(fs.delta_d(x), delta_d_ref))
        delta_m_ref = eta_m/g
        self.assertIsNone(npt.assert_allclose(fs.delta_m(x), delta_m_ref))
        delta_k_ref = eta_k/g
        self.assertIsNone(npt.assert_allclose(fs.delta_k(x), delta_k_ref))
        delta_s_ref = eta_s/g
        self.assertIsNone(npt.assert_allclose(fs.delta_s(x), delta_s_ref))
        H_d_ref = delta_d_ref/delta_m_ref
        H_k_ref = delta_k_ref/delta_m_ref
        self.assertIsNone(npt.assert_allclose(fs.H_d(x), H_d_ref))
        self.assertIsNone(npt.assert_allclose(fs.H_k(x), H_k_ref))

        # test wall shear stress
        tau_w_ref = rho*nu*U_e*g*fs.fpp(0)
        self.assertIsNone(npt.assert_allclose(fs.tau_w(x, rho), tau_w_ref))

        # test dissipation
        def D_fun(eta):
            return fs.fpp(eta)**2
        D_ref = rho*nu*U_e**2*g*quadrature(D_fun, 0, 10)[0]
        self.assertIsNone(npt.assert_allclose(fs.D(x, rho), D_ref, atol=1e-5,
                          rtol=0))

    def testBeta2Solution(self):
        """Test the third case from White table."""
        # pylint: disable=too-many-locals
        idx = 2
        U_inf = 10
        nu = 1e-5
        rho = 1
        beta = self.beta_ref[idx]
        m = beta/(2-beta)
        fs = FalknerSkanSolution(U_ref=U_inf, m=beta/(2-beta), nu=nu)

        # Test the solution for f'
        fp = fs.fp(self.eta_ref)
        self.assertIsNone(npt.assert_allclose(fp, self.fp_ref[idx,:], rtol=0,
                                              atol=1e-5))

        # Test the solved boundary condition
        fpp0 = fs.fpp(0)
        self.assertIsNone(npt.assert_allclose(fpp0, self.fpp0_ref[idx],
                                              rtol=0, atol=1e-5))

        # Test the boundary layer values
        #
        # similarity terms
        eta_d = fs.eta_d()
        self.assertIsNone(npt.assert_allclose(eta_d, self.eta_d_ref[idx],
                                              rtol=0, atol=1e-5))
        eta_m = fs.eta_m()
        self.assertIsNone(npt.assert_allclose(eta_m, self.eta_m_ref[idx],
                                              rtol=0, atol=1e-5))
        eta_s = fs.eta_s()
        self.assertIsNone(npt.assert_allclose(eta_s, self.eta_s_ref[idx],
                                              rtol=0, atol=1e-2))

        def ke_fun(eta):
            fp = fs.fp(eta)
            return fp*(1-fp**2)
        eta_k_ref = quadrature(ke_fun, 0, 10)[0]
        eta_k = fs.eta_k()
        self.assertIsNone(npt.assert_allclose(eta_k, eta_k_ref, rtol=1e-5,
                          atol=0))

        # dimensional terms
        x = np.linspace(0.2, 2, 101)
        g = np.sqrt(0.5*(m+1)*U_inf*x**m/(nu*x))
        U_e = U_inf*x**m
        delta_d_ref = eta_d/g
        self.assertIsNone(npt.assert_allclose(fs.delta_d(x), delta_d_ref))
        delta_m_ref = eta_m/g
        self.assertIsNone(npt.assert_allclose(fs.delta_m(x), delta_m_ref))
        delta_k_ref = eta_k/g
        self.assertIsNone(npt.assert_allclose(fs.delta_k(x), delta_k_ref))
        delta_s_ref = eta_s/g
        self.assertIsNone(npt.assert_allclose(fs.delta_s(x), delta_s_ref))
        H_d_ref = delta_d_ref/delta_m_ref
        H_k_ref = delta_k_ref/delta_m_ref
        self.assertIsNone(npt.assert_allclose(fs.H_d(x), H_d_ref))
        self.assertIsNone(npt.assert_allclose(fs.H_k(x), H_k_ref))

        # test wall shear stress
        tau_w_ref = rho*nu*U_e*g*fs.fpp(0)
        self.assertIsNone(npt.assert_allclose(fs.tau_w(x, rho), tau_w_ref))

        # test dissipation
        def D_fun(eta):
            return fs.fpp(eta)**2
        D_ref = rho*nu*U_e**2*g*quadrature(D_fun, 0, 10)[0]
        self.assertIsNone(npt.assert_allclose(fs.D(x, rho), D_ref))

    def testBeta3Solution(self):
        """Test the fourth case from White table."""
        # pylint: disable=too-many-locals
        idx = 3
        U_inf = 10
        nu = 1e-5
        rho = 1
        beta = self.beta_ref[idx]
        m = beta/(2-beta)
        fs = FalknerSkanSolution(U_ref=U_inf, m=beta/(2-beta), nu=nu)

        # Test the solution for f'
        fp = fs.fp(self.eta_ref)
        self.assertIsNone(npt.assert_allclose(fp, self.fp_ref[idx,:], rtol=0,
                                              atol=1e-5))

        # Test the solved boundary condition
        fpp0 = fs.fpp(0)
        self.assertIsNone(npt.assert_allclose(fpp0, self.fpp0_ref[idx],
                                              rtol=0, atol=1e-5))

        # Test the boundary layer values
        #
        # similarity terms
        eta_d = fs.eta_d()
        self.assertIsNone(npt.assert_allclose(eta_d, self.eta_d_ref[idx],
                                              rtol=0, atol=1e-5))
        eta_m = fs.eta_m()
        self.assertIsNone(npt.assert_allclose(eta_m, self.eta_m_ref[idx],
                                              rtol=0, atol=1e-5))
        eta_s = fs.eta_s()
        self.assertIsNone(npt.assert_allclose(eta_s, self.eta_s_ref[idx],
                                              rtol=0, atol=1e-4))

        def ke_fun(eta):
            fp = fs.fp(eta)
            return fp*(1-fp**2)
        eta_k_ref = quadrature(ke_fun, 0, 10)[0]
        eta_k = fs.eta_k()
        self.assertIsNone(npt.assert_allclose(eta_k, eta_k_ref, rtol=1e-5,
                          atol=0))

        # dimensional terms
        x = np.linspace(0.2, 2, 101)
        g = np.sqrt(0.5*(m+1)*U_inf*x**m/(nu*x))
        U_e = U_inf*x**m
        delta_d_ref = eta_d/g
        self.assertIsNone(npt.assert_allclose(fs.delta_d(x), delta_d_ref))
        delta_m_ref = eta_m/g
        self.assertIsNone(npt.assert_allclose(fs.delta_m(x), delta_m_ref))
        delta_k_ref = eta_k/g
        self.assertIsNone(npt.assert_allclose(fs.delta_k(x), delta_k_ref))
        delta_s_ref = eta_s/g
        self.assertIsNone(npt.assert_allclose(fs.delta_s(x), delta_s_ref))
        H_d_ref = delta_d_ref/delta_m_ref
        H_k_ref = delta_k_ref/delta_m_ref
        self.assertIsNone(npt.assert_allclose(fs.H_d(x), H_d_ref))
        self.assertIsNone(npt.assert_allclose(fs.H_k(x), H_k_ref))

        # test wall shear stress
        tau_w_ref = rho*nu*U_e*g*fs.fpp(0)
        self.assertIsNone(npt.assert_allclose(fs.tau_w(x, rho), tau_w_ref))

        # test dissipation
        def D_fun(eta):
            return fs.fpp(eta)**2
        D_ref = rho*nu*U_e**2*g*quadrature(D_fun, 0, 10)[0]
        self.assertIsNone(npt.assert_allclose(fs.D(x, rho), D_ref))

    def testBeta4Solution(self):
        """Test the fifth case from White table."""
        # pylint: disable=too-many-locals
        idx = 4
        U_inf = 10
        nu = 1e-5
        rho = 1
        beta = self.beta_ref[idx]
        m = beta/(2-beta)
        fs = FalknerSkanSolution(U_ref=U_inf, m=beta/(2-beta), nu=nu)

        # Test the solution for f'
        fp = fs.fp(self.eta_ref)
        self.assertIsNone(npt.assert_allclose(fp, self.fp_ref[idx,:], rtol=0,
                                              atol=1e-5))

        # Test the solved boundary condition
        fpp0 = fs.fpp(0)
        self.assertIsNone(npt.assert_allclose(fpp0, self.fpp0_ref[idx],
                                              rtol=0, atol=1e-5))

        # Test the boundary layer values
        #
        # similarity terms
        eta_d = fs.eta_d()
        self.assertIsNone(npt.assert_allclose(eta_d, self.eta_d_ref[idx],
                                              rtol=0, atol=1e-5))
        eta_m = fs.eta_m()
        self.assertIsNone(npt.assert_allclose(eta_m, self.eta_m_ref[idx],
                                              rtol=0, atol=1e-5))
        eta_s = fs.eta_s()
        self.assertIsNone(npt.assert_allclose(eta_s, self.eta_s_ref[idx],
                                              rtol=0, atol=1e-3))

        def ke_fun(eta):
            fp = fs.fp(eta)
            return fp*(1-fp**2)
        eta_k_ref = quadrature(ke_fun, 0, 10)[0]
        eta_k = fs.eta_k()
        self.assertIsNone(npt.assert_allclose(eta_k, eta_k_ref, rtol=1e-5,
                          atol=0))

        # dimensional terms
        x = np.linspace(0.2, 2, 101)
        g = np.sqrt(0.5*(m+1)*U_inf*x**m/(nu*x))
        U_e = U_inf*x**m
        delta_d_ref = eta_d/g
        self.assertIsNone(npt.assert_allclose(fs.delta_d(x), delta_d_ref))
        delta_m_ref = eta_m/g
        self.assertIsNone(npt.assert_allclose(fs.delta_m(x), delta_m_ref))
        delta_k_ref = eta_k/g
        self.assertIsNone(npt.assert_allclose(fs.delta_k(x), delta_k_ref))
        delta_s_ref = eta_s/g
        self.assertIsNone(npt.assert_allclose(fs.delta_s(x), delta_s_ref))
        H_d_ref = delta_d_ref/delta_m_ref
        H_k_ref = delta_k_ref/delta_m_ref
        self.assertIsNone(npt.assert_allclose(fs.H_d(x), H_d_ref))
        self.assertIsNone(npt.assert_allclose(fs.H_k(x), H_k_ref))

        # test wall shear stress
        tau_w_ref = rho*nu*U_e*g*fs.fpp(0)
        self.assertIsNone(npt.assert_allclose(fs.tau_w(x, rho), tau_w_ref))

        # test dissipation
        def D_fun(eta):
            return fs.fpp(eta)**2
        D_ref = rho*nu*U_e**2*g*quadrature(D_fun, 0, 10)[0]
        self.assertIsNone(npt.assert_allclose(fs.D(x, rho), D_ref))


if __name__ == "__main__":
    unittest.main(verbosity=1)

"""Module to test the Falkner-Skan solution functionality."""

import platform

from typing import Union

import unittest
import numpy as np
import numpy.typing as npt
import numpy.testing as np_test
from scipy.integrate import quad

from ibl.analytic import FalknerSkan


class TestFalknerSkan(unittest.TestCase):
    """Class to test the Falkner-Skan class."""

    # Tabluated data from White (2011)
    # Note that there are errors in the data:
    #    * beta = -0.1988 beta = -0.19883785
    #    * beta = -0.18 @ eta = 5.0
    #    * CASE 2
    eta_ref = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2,
                        3.4, 3.6, 3.8, 4.0, 4.5, 5.0])
    f_p_ref = np.array([[0.00000, 0.00099, 0.00398, 0.00895, 0.01591, 0.02485,
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
    # Note: eta_inf_ref[0] is significantly different on MacOS and Windows
    if platform.system() == 'Darwin':
        eta_inf_ref = np.array([9.9113972, 7.9263214, 7.1451323, 6.7517934,
                                6.0672747, 5.6670098])
    elif platform.system() == 'Windows':
        eta_inf_ref = np.array([8.1305755, 7.9263214, 7.1451323, 6.7517934,
                                6.0672747, 5.6670098])
    else:
        eta_inf_ref = np.array([8.2571417, 7.9263214, 7.1451323, 6.7517934,
                                6.0672747, 5.6670098])
    # Note: beta_ref[0] is different than White because of rounding
    beta_ref = np.array([-0.19883785, -0.18, 0, 0.3, 1.0, 2.0])
    fw_pp_ref = np.array([0, 0.12864, 0.46960, 0.77476, 1.23259, 1.68722])
    eta_d_ref = np.array([2.35885, 1.87157, 1.21678, 0.91099, 0.64790,
                          0.49743])
    eta_m_ref = np.array([0.58544, 0.56771, 0.46960, 0.38574, 0.29235,
                          0.23079])
    eta_k_ref = np.array([0.88698, 0.86649, 0.73848, 0.61780, 0.47528,
                          0.37790])  # from calculation
    eta_s_ref = np.array([4.79, 4.28, 3.47, 2.965, 2.379, 1.949])

    def test_setters(self) -> None:
        """Test the setters."""
        test_idx = 2
        sol = FalknerSkan(beta=self.beta_ref[test_idx], u_ref=100.0,
                          nu_ref=1e-5)

        # test the default values
        self.assertEqual(sol.u_ref, 100.0)
        self.assertAlmostEqual(sol.nu_ref, 1e-5)
        self.assertAlmostEqual(sol.fw_pp, self.fw_pp_ref[test_idx])
        self.assertAlmostEqual(sol.eta_inf, self.eta_inf_ref[test_idx])
        self.assertEqual(sol.beta, self.beta_ref[test_idx])
        self.assertEqual(sol.m,
                         self.beta_ref[test_idx]/(2-self.beta_ref[test_idx]))

        # test manually setting values
        sol.reset_beta(beta=self.beta_ref[test_idx])
        self.assertEqual(sol.beta, self.beta_ref[test_idx])
        sol.reset_beta(beta=0.2)
        self.assertEqual(sol.beta, 0.2)
        self.assertAlmostEqual(sol.m, 1.0/9.0)
        sol.reset_m(m=1.0)
        self.assertEqual(sol.beta, 1.0)
        self.assertEqual(sol.m, 1.0)
        sol.reset_beta(beta=2.0)
        self.assertEqual(sol.beta, 2.0)
        self.assertEqual(sol.m, np.inf)
        sol.reset_m(m=np.inf)
        self.assertEqual(sol.beta, 2.0)
        self.assertEqual(sol.m, np.inf)
        test_idx = 3
        sol2 = FalknerSkan(beta=self.beta_ref[test_idx], u_ref=10.0,
                           nu_ref=1e-5, eta_inf=8.0)
        self.assertEqual(sol2.eta_inf, 8.0)
        self.assertAlmostEqual(sol2.fw_pp, self.fw_pp_ref[test_idx],
                               delta=1e-5)

        # test setting bad values
        with self.assertRaises(ValueError):
            _ = FalknerSkan(beta=3.0, u_ref=10.0, nu_ref=1e-5)
        with self.assertRaises(ValueError):
            sol.reset_beta(beta=-1.0)

    def test_beta_solutions(self) -> None:
        """Test the cases from White table."""
        u_inf = 10
        nu = 1e-5
        rho = 1

        FunType = Union[float, np.floating, npt.NDArray]

        for idx in range(0, 6):
            with self.subTest(i=idx):
                sol = FalknerSkan(beta=self.beta_ref[idx], u_ref=u_inf,
                                  nu_ref=nu)

                # Test the solved boundary condition
                self.assertIsNone(np_test.assert_allclose(sol.fw_pp,
                                                          self.fw_pp_ref[idx],
                                                          atol=1e-3))

                # Test the solution for f'
                self.assertIsNone(
                    np_test.assert_allclose(sol.f_p(self.eta_ref),
                                            self.f_p_ref[idx,:],
                                            atol=2e-3))

                # test the eta_inf
                self.assertAlmostEqual(sol.eta_inf, self.eta_inf_ref[idx],
                                       delta=5e-7)

                # Test the boundary layer values
                #
                # def ke_fun(eta: FunType) -> FunType:
                #     f_p = sol.f_p(eta)
                #     return f_p*(1-f_p**2)
                # eta_k_ref = quadrature(ke_fun, 0, 10)[0]
                # print(f"eta_k = {eta_k_ref:.10f}")

                # similarity terms
                self.assertIsNone(np_test.assert_allclose(sol.eta_d,
                                                          self.eta_d_ref[idx],
                                                          rtol=5e-3))
                self.assertIsNone(np_test.assert_allclose(sol.eta_m,
                                                          self.eta_m_ref[idx],
                                                          rtol=3e-5))
                self.assertIsNone(np_test.assert_allclose(sol.eta_s,
                                                          self.eta_s_ref[idx],
                                                          atol=0.007))
                self.assertIsNone(np_test.assert_allclose(sol.eta_k,
                                                          self.eta_k_ref[idx],
                                                          rtol=8e-6))

                # dimensional terms
                if idx < 5:
                    x = np.linspace(0.2, 2, 101)
                    u_e = u_inf*x**sol.m
                    g = np.sqrt(0.5*(sol.m+1)*u_e/(nu*x))
                    delta_d_ref = self.eta_d_ref[idx]/g
                    self.assertIsNone(np_test.assert_allclose(sol.delta_d(x),
                                                              delta_d_ref,
                                                              rtol=3e-3))
                    delta_m_ref = self.eta_m_ref[idx]/g
                    self.assertIsNone(np_test.assert_allclose(sol.delta_m(x),
                                                              delta_m_ref,
                                                              rtol=3e-5))
                    delta_k_ref = self.eta_k_ref[idx]/g
                    self.assertIsNone(np_test.assert_allclose(sol.delta_k(x),
                                                              delta_k_ref,
                                                              rtol=8e-6))
                    delta_s_ref = self.eta_s_ref[idx]/g
                    self.assertIsNone(np_test.assert_allclose(sol.delta_s(x),
                                                              delta_s_ref,
                                                              rtol=2e-3))
                    shape_d_ref = delta_d_ref/delta_m_ref
                    shape_k_ref = delta_k_ref/delta_m_ref
                    self.assertIsNone(np_test.assert_allclose(sol.shape_d(x),
                                                              shape_d_ref,
                                                              rtol=2e-3))
                    self.assertIsNone(np_test.assert_allclose(sol.shape_k(x),
                                                              shape_k_ref,
                                                              rtol=2e-5))

                    # test wall shear stress
                    tau_w_ref = rho*nu*u_e*g*self.fw_pp_ref[idx]
                    self.assertIsNone(
                        np_test.assert_allclose(sol.tau_w(x, rho), tau_w_ref,
                                                atol=2e-4))

                    # test dissipation
                    def diss_fun(eta: FunType) -> FunType:
                        # pylint: disable-next=cell-var-from-loop
                        return sol.f_pp(eta)**2
                    diss_ref = rho*nu*u_e**2*g*quad(diss_fun, 0, 10)[0]
                    diss = sol.dissipation(x, rho)
                    self.assertIsNone(np_test.assert_allclose(diss, diss_ref,
                                                              atol=2e-5))

    def test_problem_values(self) -> None:
        """Test values against Asaithambi (1997) results."""
        beta_test = [-0.1988,  -0.18,    -0.15,    -0.12,    -0.1,
                     0.0,        0.5,      1.0,      2.0]
        fw_pp_test = [0.005218, 0.128636, 0.216362, 0.281761, 0.319270,
                      0.469600, 0.927680, 1.232588, 1.687218]
        for beta, fw_pp in zip(beta_test, fw_pp_test):
            fs = FalknerSkan(beta=beta, u_ref=1, nu_ref=1)
            self.assertAlmostEqual(fs.fw_pp, fw_pp, delta=1e-6)

    def test_drela_comparison(self) -> None:
        """Test values against what Drela (2014) lists in Table 4.1."""
        m_test = [2.0, 1.0, 0.6, 0.3, 0.1, 0.0, -0.05, -0.08, -0.09042860905]
        dis_thick_ref = [0.47648, 0.64790, 0.79760, 1.01961, 1.34787, 1.72080,
                         2.11777, 2.67173, 3.49786]
        mom_thick_ref = [0.21775, 0.29235, 0.35483, 0.44196, 0.55660, 0.66412,
                         0.75148, 0.82973, 0.86814]
        dis_shape_ref = [2.18820, 2.21622, 2.24783, 2.30702, 2.42161, 2.59109,
                         2.81815, 3.22000, 4.02916]
        kin_shape_ref = [1.63101, 1.62575, 1.62006, 1.61009, 1.59308, 1.57259,
                         1.55196, 1.52916, 1.51509]
        shear_term_ref = [1.71507, 1.23259, 0.97532, 0.72574, 0.49657, 0.33206,
                          0.21348, 0.10155, 0.00000]
        diss_term_ref = [0.97666, 0.71292, 0.57484, 0.44474, 0.33251, 0.26109,
                         0.21867, 0.19031, 0.18014]
        lam_term_ref = [0.09483, 0.08546, 0.07554, 0.05860, 0.03098, 0.00000,
                        -0.02824, -0.05508, -0.06815]
        s_term_ref = [0.37345, 0.36034, 0.34608, 0.32075, 0.27639, 0.22053,
                      0.16043, 0.08426, 0.00022]
        f_term_ref = [-0.04742, 0.00000, 0.05036, 0.13672, 0.27882, 0.44105,
                      0.59294, 0.74351, 0.82179]

        tol = [8e-6, 1e-5, 8e-6, 8e-6, 2e-5, 2e-5, 3e-5, 7e-5, 4e-3]
        fs = FalknerSkan(beta=0.0, u_ref=1, nu_ref=1)
        for idx, m in enumerate(m_test):
            with self.subTest(i=idx):
                fs.reset_m(m=m)
                beta = fs.beta
                eta_m = fs.eta_m
                eta_d = fs.eta_d
                eta_k = fs.eta_k
                fw_pp = fs.fw_pp

                # s-free calculations
                self.assertAlmostEqual(dis_thick_ref[idx],
                                       eta_d*np.sqrt(2/(1+m)), delta=tol[idx])
                self.assertAlmostEqual(mom_thick_ref[idx],
                                       eta_m*np.sqrt(2/(1+m)), delta=tol[idx])
                self.assertAlmostEqual(dis_shape_ref[idx], eta_d/eta_m,
                                       delta=tol[idx])
                self.assertAlmostEqual(kin_shape_ref[idx], eta_k/eta_m,
                                       delta=tol[idx])
                self.assertAlmostEqual(shear_term_ref[idx],
                                       fw_pp*np.sqrt((m+1)/2), delta=tol[idx])
                self.assertAlmostEqual(diss_term_ref[idx],
                                       np.sqrt(0.5/(1+m))*0.5*(5*m+1)*eta_k,
                                       delta=tol[idx])
                self.assertAlmostEqual(lam_term_ref[idx], beta*eta_m**2,
                                       delta=tol[idx])
                self.assertAlmostEqual(s_term_ref[idx], eta_m*fw_pp,
                                       delta=tol[idx])
                self.assertAlmostEqual(f_term_ref[idx],
                                       2*(eta_m*fw_pp
                                          - (eta_d/eta_m + 2)*beta*eta_m**2),
                                       delta=tol[idx])


if __name__ == "__main__":
    _ = unittest.main(verbosity=1)

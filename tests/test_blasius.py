"""Module to test the Blasius solution functionality."""

# pyright: reportPrivateUsage=false
# pylint: disable=protected-access

from typing import Union

import unittest
import numpy as np
import numpy.typing as npt
import numpy.testing as np_test
from scipy.integrate import quadrature

from ibl.analytic import Blasius


class TestBlasius(unittest.TestCase):
    """Class to test the Blasius class."""

    # Tabluated data from White (2011) using eta_inf=10 and fw_pp=0.46960
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
    f_p_ref = np.array([0.00000,  0.04696,  0.09391,  0.14081,  0.18761,
                        0.23423,  0.28058,  0.32653,  0.37196,  0.41672,
                        0.46063,  0.50354,  0.54525,  0.58559,  0.62439,
                        0.66147,  0.69670,  0.72993,  0.76106,  0.79000,
                        0.81669,  0.86330,  0.90107,  0.93060,  0.95288,
                        0.96905,  0.98037,  0.98797,  0.99289,  0.99594,
                        0.99777,  0.99882,  0.99940,  0.99970,  0.99986,
                        0.99994,  0.999971, 0.999988, 0.999995, 0.999998,
                        0.999999])
    f_pp_ref = np.array([0.46960,  0.46956,  0.46931,  0.46861,  0.46725,
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
    eta_k_ref = 0.7384846902  # from calculation
    v_e_term = 0.8604
    eta_inf_ref = 10.0

    def test_setters(self) -> None:
        """Test the setters."""
        sol = Blasius(u_ref=100.0, nu_ref=1e-5, eta_inf=self.eta_inf_ref,
                      fw_pp=self.f_pp_ref[0])

        # test the default values
        self.assertEqual(sol.u_ref, 100.0)
        self.assertAlmostEqual(sol.nu_ref, 1e-5)
        self.assertAlmostEqual(sol.fw_pp, self.f_pp_ref[0])
        self.assertEqual(sol.eta_inf, self.eta_inf_ref)

        # test manually setting values
        sol.u_ref = 12.0
        sol.nu_ref = 2.e-5
        self.assertEqual(sol.u_ref, 12.0)
        self.assertAlmostEqual(sol.nu_ref, 2e-5)
        sol.set_solution_parameters(eta_inf=8.0, fw_pp=0.469)
        self.assertAlmostEqual(sol.fw_pp, 0.469)
        self.assertAlmostEqual(sol.eta_inf, 8.0)

        # test setting bad values
        with self.assertRaises(ValueError):
            sol.u_ref = 0.0
        with self.assertRaises(ValueError):
            sol.nu_ref = 0.0
        with self.assertRaises(ValueError):
            sol.set_solution_parameters(eta_inf=-1.0)
        with self.assertRaises(ValueError):
            sol.set_solution_parameters(fw_pp=-1.0)

        # simulate could not find solution but continuing
        sol._f = None
        with self.assertRaises(ValueError):
            _ = sol.f(0.0)
        with self.assertRaises(ValueError):
            _ = sol.f_p(0.0)
        with self.assertRaises(ValueError):
            _ = sol.f_pp(0.0)
        self.assertEqual(sol.fw_pp, np.inf)

    def test_basic_solution(self) -> None:
        """Test the calculation of the basic solution."""
        u_inf = 10
        nu = 1e-5

        # using default values
        sol = Blasius(u_ref=u_inf, nu_ref=nu)
        eta_inf_ref = sol.eta_inf_default
        fw_pp_ref = sol.fw_pp_default

        # test the solution for f, f', and f''
        self.assertIsNone(np_test.assert_allclose(sol.f(self.eta_ref),
                                                  self.f_ref, atol=1e-5))
        self.assertIsNone(np_test.assert_allclose(sol.f_p(self.eta_ref),
                                                  self.f_p_ref, atol=1e-5))
        self.assertIsNone(np_test.assert_allclose(sol.f_pp(self.eta_ref),
                                                  self.f_pp_ref, atol=1e-5))
        self.assertAlmostEqual(sol.fw_pp, fw_pp_ref)
        self.assertAlmostEqual(sol.eta_inf, eta_inf_ref)

        # setting fw_pp and solving for eta_inf
        sol = Blasius(u_ref=u_inf, nu_ref=nu, fw_pp=fw_pp_ref)

        # test the solution for f, f', and f''
        # Note: eta_inf is very sensitive, so only get limited convergence
        self.assertIsNone(np_test.assert_allclose(sol.f(self.eta_ref),
                                                  self.f_ref, atol=1e-5))
        self.assertIsNone(np_test.assert_allclose(sol.f_p(self.eta_ref),
                                                  self.f_p_ref, atol=1e-5))
        self.assertIsNone(np_test.assert_allclose(sol.f_pp(self.eta_ref),
                                                  self.f_pp_ref, atol=1e-5))
        self.assertAlmostEqual(sol.fw_pp, fw_pp_ref)
        self.assertAlmostEqual(sol.eta_inf, eta_inf_ref, delta=3e-3)

        # setting eta_inf and solving for fw_pp
        sol = Blasius(u_ref=u_inf, nu_ref=nu, eta_inf=eta_inf_ref)

        # test the solution for f, f', and f''
        self.assertIsNone(np_test.assert_allclose(sol.f(self.eta_ref),
                                                  self.f_ref, atol=1e-5))
        self.assertIsNone(np_test.assert_allclose(sol.f_p(self.eta_ref),
                                                  self.f_p_ref, atol=1e-5))
        self.assertIsNone(np_test.assert_allclose(sol.f_pp(self.eta_ref),
                                                  self.f_pp_ref, atol=1e-5))
        self.assertAlmostEqual(sol.fw_pp, fw_pp_ref)
        self.assertAlmostEqual(sol.eta_inf, eta_inf_ref)

        # solving for eta_inf and fw_pp
        sol = Blasius(u_ref=u_inf, nu_ref=nu)
        sol.set_solution_parameters(eta_inf=None, fw_pp=None)

        # test the solution for f, f', and f''
        self.assertIsNone(np_test.assert_allclose(sol.f(self.eta_ref),
                                                  self.f_ref, atol=1e-5))
        self.assertIsNone(np_test.assert_allclose(sol.f_p(self.eta_ref),
                                                  self.f_p_ref, atol=1e-5))
        self.assertIsNone(np_test.assert_allclose(sol.f_pp(self.eta_ref),
                                                  self.f_pp_ref, atol=1e-5))
        self.assertAlmostEqual(sol.fw_pp, fw_pp_ref)
        self.assertAlmostEqual(sol.eta_inf, eta_inf_ref)

    def test_eta_boundary_layer_parameters(self) -> None:
        """Test the reporting of the boundary layer parameters in eta."""
        u_inf = 10
        nu = 1e-5
        sol = Blasius(u_ref=u_inf, nu_ref=nu, eta_inf=self.eta_inf_ref,
                      fw_pp=self.f_pp_ref[0])

        # Test the values in terms of eta
        #
        # This code is used to report what the kinetic energy thickness is:
        #
        # FunType = Union[float, npt.NDArray]

        # def ke_fun(eta: FunType) -> FunType:
        #     f_p = sol.f_p(eta)
        #     return f_p*(1-f_p**2)
        # self.eta_k_ref = quadrature(ke_fun, 0, 10)[0]
        # print(f"eta_k = {eta_k_ref:.10f}")

        # displacement thickness
        self.assertAlmostEqual(sol.eta_d, self.eta_d_ref, places=5)
        # momentum thickness
        self.assertAlmostEqual(sol.eta_m, self.eta_m_ref)
        # # shear thickness
        self.assertAlmostEqual(sol.eta_s, self.eta_s_ref, places=1)
        # kinetic energy thickness
        self.assertAlmostEqual(sol.eta_k, self.eta_k_ref, delta=5e-7)

    def test_x_boundary_layer_parameters(self) -> None:
        """Test the reporting of the boundary layer parameters in x."""
        u_inf = 10
        nu = 1e-5
        rho = 1
        sol = Blasius(u_ref=u_inf, nu_ref=nu, eta_inf=self.eta_inf_ref,
                      fw_pp=self.f_pp_ref[0])
        # Test the values in terms of x
        #
        x = np.linspace(0.2, 2, 101)

        # test the edge velocity
        u_e_ref = u_inf*np.ones_like(x)
        self.assertIsNone(np_test.assert_allclose(sol.u_e(x), u_e_ref))

        # test the transformation function
        g_ref = np.sqrt(0.5*u_inf/(nu*x))
        self.assertIsNone(np_test.assert_allclose(sol._g(x),g_ref))

        # test the transpiration velocity
        v_e_ref = nu*g_ref*np.sqrt(2)*self.v_e_term
        self.assertIsNone(np_test.assert_allclose(sol.v_e(x), v_e_ref,
                                                  atol=2e-7))

        # test displacement thickness
        delta_d_ref = self.eta_d_ref/g_ref
        self.assertIsNone(np_test.assert_allclose(sol.delta_d(x), delta_d_ref,
                                                  rtol=4e-7))

        # test momentum thickness
        delta_m_ref = self.eta_m_ref/g_ref
        self.assertIsNone(np_test.assert_allclose(sol.delta_m(x), delta_m_ref))

        # test kinetic energy thickness
        delta_k_ref = self.eta_k_ref/g_ref
        self.assertIsNone(np_test.assert_allclose(sol.delta_k(x), delta_k_ref,
                                                  rtol=5e-7))

        # test shear thickness
        delta_s_ref = self.eta_s_ref/g_ref
        self.assertIsNone(np_test.assert_allclose(sol.delta_s(x), delta_s_ref,
                                                  rtol=9e-3))

        # test shape factors
        shape_d_ref = delta_d_ref/delta_m_ref
        shape_k_ref = delta_k_ref/delta_m_ref
        self.assertIsNone(np_test.assert_allclose(sol.shape_d(x), shape_d_ref,
                                                  rtol=4e-7))
        self.assertIsNone(np_test.assert_allclose(sol.shape_k(x), shape_k_ref,
                                                  rtol=5e-7))

        # test wall shear stress
        tau_w_ref = rho*nu*u_inf*g_ref*self.f_pp_ref[0]
        self.assertIsNone(np_test.assert_allclose(sol.tau_w(x, rho),
                                                  tau_w_ref))

        # test dissipation
        FunType = Union[float, np.floating, npt.NDArray]

        def diss_fun(eta: FunType) -> FunType:
            return sol.f_pp(eta)**2
        diss_ref = rho*nu*u_inf**2*g_ref*quadrature(diss_fun, 0, 10)[0]
        self.assertIsNone(np_test.assert_allclose(sol.dissipation(x, rho),
                                                  diss_ref))

    def test_local_properties(self) -> None:
        """Test the local property calculations."""
        u_inf = 10
        nu = 1e-5
        sol = Blasius(u_ref=u_inf, nu_ref=nu, eta_inf=self.eta_inf_ref,
                      fw_pp=self.f_pp_ref[0])

        # Test the values in terms of x,y
        x0 = 0.4
        y0 = 2e-3

        # test eta transformation
        g = np.sqrt(0.5*u_inf/(nu*x0))
        eta_ref = y0*g
        self.assertIsNone(np_test.assert_allclose(sol.eta(x0, y0), eta_ref))

        # test sample in y
        x: Union[float, npt.NDArray] = x0
        y: Union[float, npt.NDArray] = np.linspace(1e-4, 5e-3, 11)
        g = np.sqrt(0.5*u_inf/(nu*x))
        eta_ref = y*g
        u_ref = u_inf*sol.f_p(eta_ref)
        v_ref = np.sqrt(0.5*nu*u_inf/x)*(eta_ref*sol.f_p(eta_ref)
                                         - sol.f(eta_ref))
        self.assertIsNone(np_test.assert_allclose(sol.u(x, y), u_ref))
        self.assertIsNone(np_test.assert_allclose(sol.v(x, y), v_ref))

        # test sample in x
        x = np.linspace(0.2, 0.5, 11)
        y = y0
        g = np.sqrt(0.5*u_inf/(nu*x))
        eta_ref = y*g
        u_ref = u_inf*sol.f_p(eta_ref)
        v_ref = np.sqrt(0.5*nu*u_inf/x)*(eta_ref*sol.f_p(eta_ref)
                                         - sol.f(eta_ref))
        self.assertIsNone(np_test.assert_allclose(sol.u(x, y), u_ref))
        self.assertIsNone(np_test.assert_allclose(sol.v(x, y), v_ref))


if __name__ == "__main__":
    _ = unittest.main(verbosity=1)

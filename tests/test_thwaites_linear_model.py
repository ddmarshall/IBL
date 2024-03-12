"""Test Thwaites linear models."""
# pylint: disable=duplicate-code

import unittest

import numpy as np
import numpy.typing as npt
import numpy.testing as np_test

from _thwaites_linear_analytic import ThwaitesLinearAnalytic

from ibl.thwaites_method import ThwaitesMethodLinear
from ibl.typing import InputParam


class TestLinearThwaites(unittest.TestCase):
    """Class to test the implementation of the linear Thwaites method"""

    def test_custom_functions(self) -> None:
        """Test the setting of custom functions."""
        # set parameters
        u_ref = 10
        m = 0
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def u_e_fun(x: InputParam) -> InputParam:
            return u_ref*x**m

        def du_e_fun(x: InputParam) -> InputParam:
            if m == 0:
                return np.zeros_like(x)
            return m*u_ref*x**(m-1)

        def d2u_e_fun(x: InputParam) -> InputParam:
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*u_ref*x**(m-2)

        # test creating with own functions for S, H, H'
        def shear_fun(lam: InputParam) -> InputParam:
            return (lam + 0.09)**(0.62)

        def shape_fun(lam: InputParam) -> InputParam:
            z = 0.25-lam
            return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5

        def shape_p_fun(lam: InputParam) -> InputParam:
            z = 0.25-lam
            return -(4.14 + z*(-2*83.5 + z*(3*854 + z*(-4*3337 + z*5*4576))))

        fits = [(shear_fun, shape_fun, shape_p_fun), (shear_fun, shape_fun)]
        for idx, fit in enumerate(fits):
            with self.subTest(i=idx):
                tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                          d2U_edx2=d2u_e_fun, data_fits=fit)

                tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, shape_fun,
                                                shear_fun)
                tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
                rtn = tm.solve(x0=x[0], x_end=x[-1])
                self.assertTrue(rtn.success)
                self.assertEqual(rtn.status, 0)
                self.assertEqual(rtn.message, "Completed")
                self.assertEqual(rtn.x_end, x[-1])
                self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                          tm_ref.delta_d(x),
                                                          rtol=5e-5))
                self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                          tm_ref.delta_m(x),
                                                          rtol=5e-5))
                self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                          tm_ref.shape_d(x)))
                self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                          tm_ref.tau_w(x, rho),
                                                          rtol=5e-5))
                self.assertIsNone(np_test.assert_allclose(tm.v_e(x),
                                                          tm_ref.v_e(x),
                                                          rtol=1e-4))

        # test creating with invalid name
        with self.assertRaises(ValueError):
            ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                 d2U_edx2=d2u_e_fun, data_fits="My Own")

    def test_blaisus_case(self) -> None:
        """Test the flat plate case."""
        # set parameters
        u_ref = 10
        m = 0
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def u_e_fun(x: InputParam) -> InputParam:
            return u_ref*x**m

        def du_e_fun(x: InputParam) -> InputParam:
            if m == 0:
                return np.zeros_like(x)
            return m*u_ref*x**(m-1)

        def d2u_e_fun(x: InputParam) -> InputParam:
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*u_ref*x**(m-2)

        # test models
        models = ["Spline", "White", "Cebeci-Bradshaw", "Drela"]
        for idx, model in enumerate(models):
            with self.subTest(i=idx):
                tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                          d2U_edx2=d2u_e_fun, data_fits=model)

                tm_shape = tm._model.shape  # pylint: disable=protected-access
                tm_shear = tm._model.shear  # pylint: disable=protected-access
                tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape,
                                                tm_shear)
                tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
                rtn = tm.solve(x0=x[0], x_end=x[-1])
                self.assertTrue(rtn.success)
                self.assertEqual(rtn.status, 0)
                self.assertEqual(rtn.message, "Completed")
                self.assertEqual(rtn.x_end, x[-1])
                self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                          tm_ref.delta_d(x),
                                                          rtol=5e-5))
                self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                          tm_ref.delta_m(x),
                                                          rtol=5e-5))
                self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                          tm_ref.shape_d(x)))
                self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                          tm_ref.tau_w(x, rho),
                                                          rtol=5e-5))
                self.assertIsNone(np_test.assert_allclose(tm.v_e(x),
                                                          tm_ref.v_e(x),
                                                          rtol=1e-4))

    def test_wedge_050_case(self) -> None:
        """Test the m=0.50 wedge case."""
        # set parameters
        u_ref = 10
        m = 0.5
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def u_e_fun(x: InputParam) -> InputParam:
            return u_ref*x**m

        def du_e_fun(x: InputParam) -> InputParam:
            if m == 0:
                return np.zeros_like(x)
            return m*u_ref*x**(m-1)

        def d2u_e_fun(x: InputParam) -> InputParam:
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*u_ref*x**(m-2)

        # test models
        models = ["Spline", "White", "Cebeci-Bradshaw", "Drela"]
        for idx, model in enumerate(models):
            with self.subTest(i=idx):
                tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                          d2U_edx2=d2u_e_fun, data_fits=model)
                tm_shape = tm._model.shape  # pylint: disable=protected-access
                tm_shear = tm._model.shear  # pylint: disable=protected-access
                tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape,
                                                tm_shear)
                tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
                rtn = tm.solve(x0=x[0], x_end=x[-1])
                self.assertTrue(rtn.success)
                self.assertEqual(rtn.status, 0)
                self.assertEqual(rtn.message, "Completed")
                self.assertEqual(rtn.x_end, x[-1])
                self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                          tm_ref.delta_d(x),
                                                          rtol=2e-5))
                self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                          tm_ref.delta_m(x),
                                                          rtol=2e-5))
                self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                          tm_ref.shape_d(x),
                                                          rtol=3e-6))
                self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                          tm_ref.tau_w(x, rho),
                                                          rtol=1e-5))
                self.assertIsNone(np_test.assert_allclose(tm.v_e(x),
                                                          tm_ref.v_e(x),
                                                          rtol=1e-4))

    def test_wedge_072_case(self) -> None:
        """Test the m=0.72 wedge case."""
        # set parameters
        u_ref = 10
        m = 0.72
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def u_e_fun(x: InputParam) -> InputParam:
            return u_ref*x**m

        def du_e_fun(x: InputParam) -> InputParam:
            if m == 0:
                return np.zeros_like(x)
            return m*u_ref*x**(m-1)

        def d2u_e_fun(x: InputParam) -> InputParam:
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*u_ref*x**(m-2)

        # test models
        models = ["Spline", "White", "Cebeci-Bradshaw", "Drela"]
        for idx, model in enumerate(models):
            with self.subTest(i=idx):
                tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                          d2U_edx2=d2u_e_fun, data_fits=model)
                tm_shape = tm._model.shape  # pylint: disable=protected-access
                tm_shear = tm._model.shear  # pylint: disable=protected-access
                tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape,
                                                tm_shear)
                tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
                rtn = tm.solve(x0=x[0], x_end=x[-1])
                self.assertTrue(rtn.success)
                self.assertEqual(rtn.status, 0)
                self.assertEqual(rtn.message, "Completed")
                self.assertEqual(rtn.x_end, x[-1])
                self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                          tm_ref.delta_d(x),
                                                          rtol=1e-5))
                self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                          tm_ref.delta_m(x),
                                                          rtol=2e-5))
                self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                          tm_ref.shape_d(x),
                                                          rtol=3e-6))
                self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                          tm_ref.tau_w(x, rho),
                                                          rtol=6e-6))
                self.assertIsNone(np_test.assert_allclose(tm.v_e(x),
                                                          tm_ref.v_e(x),
                                                          rtol=8e-5))

    def test_wedge_100_case(self) -> None:
        """Test the m=1.00 wedge case."""
        # set parameters
        u_ref = 10
        m = 1
        nu = 1e-5
        rho = 1
        x = np.linspace(0.1, 2, 101)

        # create edge velocity functions
        def u_e_fun(x: InputParam) -> InputParam:
            return u_ref*x**m

        def du_e_fun(x: InputParam) -> InputParam:
            if m == 0:
                return np.zeros_like(x)
            return m*u_ref*x**(m-1)

        def d2u_e_fun(x: InputParam) -> InputParam:
            if m in (0, 1):
                return np.zeros_like(x)
            return m*(m-1)*u_ref*x**(m-2)

        # test models
        models = ["Spline", "White", "Cebeci-Bradshaw", "Drela"]
        for idx, model in enumerate(models):
            with self.subTest(i=idx):
                tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                          d2U_edx2=d2u_e_fun, data_fits=model)
                tm_shape = tm._model.shape  # pylint: disable=protected-access
                tm_shear = tm._model.shear  # pylint: disable=protected-access
                tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape,
                                                tm_shear)
                tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
                rtn = tm.solve(x0=x[0], x_end=x[-1])
                self.assertTrue(rtn.success)
                self.assertEqual(rtn.status, 0)
                self.assertEqual(rtn.message, "Completed")
                self.assertEqual(rtn.x_end, x[-1])
                self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                          tm_ref.delta_d(x)))
                self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                          tm_ref.delta_m(x)))
                self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                          tm_ref.shape_d(x)))
                self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                          tm_ref.tau_w(x,
                                                                       rho)))
                self.assertIsNone(np_test.assert_allclose(tm.v_e(x),
                                                          tm_ref.v_e(x)))

    def test_linear_sample_calculations(self) -> None:
        """Test sample calculations for linear model."""
        x = np.linspace(1e-3, 2, 20)
        u_inf = 10
        m = 0.8
        nu = 1e-5
        rho = 1.0

        # Set up the velocity functions
        def u_e_fun(x: InputParam) -> npt.NDArray:
            x = np.asarray(x)
            return u_inf*x**m

        def du_e_fun(x: InputParam) -> npt.NDArray:
            x = np.asarray(x)
            return m*u_inf*x**(m-1)

        def d2u_e_fun(x: InputParam) -> npt.NDArray:
            x = np.asarray(x)
            return m*(m-1)*u_inf*x**(m-2)

        # Get the solutions for comparisons
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun, data_fits="Spline")
        tm.initial_delta_m = 0.00035202829985968135
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)

        # # print out reference values
        # print("u_e_ref =", tm.u_e(x))
        # print("v_e_ref =", tm.v_e(x))
        # print("delta_d_ref =", tm.delta_d(x))
        # print("delta_m_ref =", tm.delta_m(x))
        # print("delta_k_ref =", tm.delta_k(x))
        # print("shape_d_ref =", tm.shape_d(x))
        # print("shape_k_ref =", tm.shape_k(x))
        # print("tau_w_ref =", tm.tau_w(x, rho))
        # print("dissipation_ref =", tm.dissipation(x, rho))

        # reference data
        u_e_ref = [0.03981072,   1.66316007,  2.88481910,  3.98513815,
                   5.01325667,   5.99077748,  6.92976171,  7.83785495,
                   8.72030922,   9.58093932, 10.42263254, 11.24764450,
                   12.05778215, 12.85452300, 13.63909583, 14.41253731,
                   15.17573284, 15.92944665, 16.67434453, 17.41101127]
        v_e_ref = [0.00177710, 0.00799042, 0.00745885, 0.00716360, 0.00696100,
                   0.00680771, 0.00668493, 0.00658282, 0.00649561, 0.00641964,
                   0.00635242, 0.00629222, 0.00623775, 0.00618806, 0.00614241,
                   0.00610020, 0.00606098, 0.00602437, 0.00599005, 0.00595777]
        delta_d_ref = [0.00070406, 0.00056696, 0.00060737, 0.00063241,
                       0.00065082, 0.00066547, 0.00067769, 0.00068821,
                       0.00069744, 0.00070570, 0.00071317, 0.00071999,
                       0.00072628, 0.00073211, 0.00073755, 0.00074265,
                       0.00074746, 0.00075200, 0.00075631, 0.00076041]
        delta_m_ref = [0.00035203, 0.00023974, 0.00025682, 0.00026741,
                       0.00027519, 0.00028139, 0.00028656, 0.00029100,
                       0.00029491, 0.00029840, 0.00030156, 0.00030444,
                       0.00030710, 0.00030957, 0.00031187, 0.00031402,
                       0.00031606, 0.00031798, 0.00031980, 0.00032153]
        delta_k_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        shape_d_ref = [2.0,        2.36496408, 2.36496136, 2.36496034,
                       2.36495979, 2.36495944, 2.36495920, 2.36495902,
                       2.36495888, 2.36495877, 2.36495868, 2.36495860,
                       2.36495854, 2.36495848, 2.36495844, 2.36495839,
                       2.36495836, 2.36495832, 2.36495829, 2.36495827]
        shape_k_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tau_w_ref = [0.00056545, 0.02241132, 0.03628715, 0.04814338,
                     0.05885093, 0.06877744, 0.07812259, 0.08701029,
                     0.09552419, 0.10372418, 0.11165499, 0.11935118,
                     0.12684017, 0.13414416, 0.14128146, 0.14826739,
                     0.15511497, 0.16183533, 0.16843812, 0.17493177]
        dissipation_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.assertIsNone(np_test.assert_allclose(u_e_ref, tm.u_e(x)))
        self.assertIsNone(np_test.assert_allclose(v_e_ref, tm.v_e(x),
                                                  atol=1e-7))
        self.assertIsNone(np_test.assert_allclose(delta_d_ref, tm.delta_d(x),
                                                  atol=1e-7))
        self.assertIsNone(np_test.assert_allclose(delta_m_ref, tm.delta_m(x),
                                                  atol=1e-7))
        self.assertIsNone(np_test.assert_allclose(delta_k_ref, tm.delta_k(x)))
        self.assertIsNone(np_test.assert_allclose(shape_d_ref, tm.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(shape_k_ref, tm.shape_k(x)))
        self.assertIsNone(np_test.assert_allclose(tau_w_ref, tm.tau_w(x, rho),
                                                  atol=1e-7))
        self.assertIsNone(np_test.assert_allclose(dissipation_ref,
                                                  tm.dissipation(x, rho)))


if __name__ == "__main__":
    unittest.main(verbosity=1)

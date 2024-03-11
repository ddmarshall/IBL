"""Test Thwaites nonlinear models."""
# pylint: disable=duplicate-code

import unittest

import numpy as np
import numpy.typing as npt
import numpy.testing as np_test

from _thwaites_linear_analytic import ThwaitesLinearAnalytic

from ibl.thwaites_method import ThwaitesMethodLinear
from ibl.thwaites_method import ThwaitesMethodNonlinear
from ibl.typing import InputParam


class TestNonlinearThwaites(unittest.TestCase):
    """Class to test the implementation of the nonlinear Thwaites method"""

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

        # test with spline of tabular data
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Spline")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=2e-2))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=2e-2))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=2e-2))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=3e-2))

        # test with White fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="White")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=2e-3))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Cebeci-Bradshaw")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=2e-2))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=2e-2))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=2e-2))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=3e-2))

        # test creating with own functions for S, H, H'
        def shear_fun(lam: InputParam) -> InputParam:
            return (lam + 0.09)**(0.62)

        def shape_fun(lam: InputParam) -> InputParam:
            z = 0.25-lam
            return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5

        def shape_p_fun(lam: InputParam) -> InputParam:
            z = 0.25-lam
            return -(4.14 + z*(-2*83.5 + z*(3*854 + z*(-4*3337 + z*5*4576))))

        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits=(shear_fun, shape_fun,
                                                shape_p_fun))
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, shape_fun, shear_fun)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=2e-3))

        # test creating with own functions for S, H
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits=(shear_fun, shape_fun))
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, shape_fun, shear_fun)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=2e-3))

        # test creating with invalid name
        with self.assertRaises(ValueError):
            ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                    d2U_edx2=d2u_e_fun,
                                    data_fits="My Own")

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

        # test with spline of tabular data
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Spline")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=2e-3))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=3e-3))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=2e-3))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=1e-2))

        # test with White fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="White")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=3e-3))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=3e-3))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=2e-3))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=1e-2))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Cebeci-Bradshaw")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=2e-3))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=3e-3))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=1e-2))

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

        # test with spline of tabular data
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Spline")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=1e-2))

        # test with White fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="White")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=2e-3))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=2e-3))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=1e-2))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Cebeci-Bradshaw")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=1e-4))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=1e-4))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x),
                                                  atol=0, rtol=2e-5))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=3e-5))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=3e-4))

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

        # test with spline of tabular data
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Spline")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=2e-4))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=2e-4))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x),
                                                  atol=0, rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=1e-4))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=1e-3))

        # test with White fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="White")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=4e-4))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=5e-4))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x),
                                                  atol=0, rtol=1e-4))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=2e-4))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=3e-3))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                     d2U_edx2=d2u_e_fun,
                                     data_fits="Cebeci-Bradshaw")
        tm_shape = tm._model.shape  # pylint: disable=protected-access
        tm_shear = tm._model.shear  # pylint: disable=protected-access
        tm_ref = ThwaitesLinearAnalytic(u_ref, m, nu, tm_shape, tm_shear)
        tm.initial_delta_m = float(tm_ref.delta_m(x[0]))
        rtn = tm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x[-1])
        self.assertIsNone(np_test.assert_allclose(tm.delta_d(x),
                                                  tm_ref.delta_d(x),
                                                  atol=0, rtol=2e-3))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  atol=0, rtol=2e-3))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x),
                                                  atol=0, rtol=4e-4))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  atol=0, rtol=1e-3))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  atol=0, rtol=1e-2))

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

    def test_nonlinear_sample_calculations(self) -> None:
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
        tm = ThwaitesMethodNonlinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
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
        v_e_ref = [-0.00100255, 0.00798703, 0.00745569, 0.00716056,
                   +0.00695805, 0.00680483, 0.00668210, 0.00658003,
                   +0.00649286, 0.00641692, 0.00634973, 0.00628956,
                   +0.00623511, 0.00618544, 0.00613980, 0.00609762,
                   +0.00605842, 0.00602182, 0.00598752, 0.00595525]
        delta_d_ref = [0.00070406, 0.00056672, 0.00060712, 0.00063214,
                       0.00065054, 0.00066519, 0.00067741, 0.00068791,
                       0.00069715, 0.00070540, 0.00071286, 0.00071968,
                       0.00072597, 0.0007318,  0.00073724, 0.00074234,
                       0.00074714, 0.00075168, 0.00075599, 0.00076009]
        delta_m_ref = [0.00035203, 0.00023961, 0.00025669, 0.00026727,
                       0.00027505, 0.00028124, 0.00028641, 0.00029085,
                       0.00029475, 0.00029824, 0.00030140, 0.00030428,
                       0.00030694, 0.00030940, 0.00031170, 0.00031386,
                       0.00031589, 0.00031781, 0.00031963, 0.00032136]
        delta_k_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        shape_d_ref = [2.0,        2.36519861, 2.36519579, 2.36519473,
                       2.36519416, 2.36519380, 2.36519355, 2.36519336,
                       2.36519322, 2.36519311, 2.36519301, 2.36519293,
                       2.36519287, 2.36519281, 2.36519276, 2.36519272,
                       2.36519268, 2.36519264, 2.36519261, 2.36519259]
        shape_k_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tau_w_ref = [0.00056545, 0.02241660, 0.03629568, 0.04815470,
                     0.05886477, 0.06879362, 0.07814096, 0.08703075,
                     0.09554666, 0.10374857, 0.11168125, 0.11937925,
                     0.12687000, 0.13417571, 0.14131468, 0.14830226,
                     0.15515145, 0.16187339, 0.16847773, 0.17497291]
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
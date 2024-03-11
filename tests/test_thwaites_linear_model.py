"""Test Thwaites linear models."""
# pylint: disable=duplicate-code

import unittest

import numpy as np
import numpy.testing as np_test

from _thwaites_linear_analytic import ThwaitesLinearAnalytic

from ibl.thwaites_method import ThwaitesMethodLinear
from ibl.typing import InputParam


class TestLinearThwaites(unittest.TestCase):
    """Class to test the implementation of the linear Thwaites method"""

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
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
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
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  rtol=1e-4))

        # test with White fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun, data_fits="White")
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
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  rtol=1e-4))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
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
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  rtol=1e-4))

        # test creating with own functions for S, H, H'
        def shear_fun(lam: InputParam) -> InputParam:
            return (lam + 0.09)**(0.62)

        def shape_fun(lam: InputParam) -> InputParam:
            z = 0.25-lam
            return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5

        def shape_p_fun(lam: InputParam) -> InputParam:
            z = 0.25-lam
            return -(4.14 + z*(-2*83.5 + z*(3*854 + z*(-4*3337 + z*5*4576))))

        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
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
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  rtol=1e-4))

        # test creating with own functions for S, H
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
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
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x),
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho),
                                                  rtol=5e-5))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  rtol=1e-4))

        # test creating with invalid name
        with self.assertRaises(ValueError):
            ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                 d2U_edx2=d2u_e_fun, data_fits="My Own")

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
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
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
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  rtol=1e-4))

        # test with White fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun, data_fits="White")
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
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  rtol=1e-4))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
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
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
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

        # test with spline of tabular data
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
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
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  rtol=7e-5))

        # test with White fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun, data_fits="White")
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
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
                                                  rtol=7e-5))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
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
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x),
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

        # test with spline of tabular data
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
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
                                                  tm_ref.delta_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x)))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho)))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x)))

        # test with White fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                                  d2U_edx2=d2u_e_fun, data_fits="White")
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
                                                  tm_ref.delta_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x)))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho)))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x)))

        # test with Cebeci and Bradshaw fits
        tm = ThwaitesMethodLinear(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
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
                                                  tm_ref.delta_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.delta_m(x),
                                                  tm_ref.delta_m(x)))
        self.assertIsNone(np_test.assert_allclose(tm.shape_d(x),
                                                  tm_ref.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(tm.tau_w(x, rho),
                                                  tm_ref.tau_w(x, rho)))
        self.assertIsNone(np_test.assert_allclose(tm.v_e(x), tm_ref.v_e(x)))


if __name__ == "__main__":
    unittest.main(verbosity=1)

"""Tests Thwaites curve fits."""

import unittest
from typing import Callable

import numpy as np
import numpy.testing as np_test

# from scipy.misc import derivative as fd

from ibl.thwaites_method import _ThwaitesFunctionsWhite
from ibl.thwaites_method import _ThwaitesFunctionsCebeciBradshaw
from ibl.thwaites_method import _ThwaitesFunctionsDrela
from ibl.thwaites_method import _ThwaitesFunctionsSpline
from ibl.typing import InputParam


def fd_1f(fun: Callable, xo: InputParam, dx: float) -> InputParam:
    """Use finite differences to approximate the derivative of function."""
    return ((fun(xo-2*dx) - fun(xo+2*dx))/12
            - 2*(fun(xo-dx) - fun(xo+dx))/3)/dx


class TestCurveFits(unittest.TestCase):
    """Class to test various functions and curve fits for Thwaites method"""

    f_ref = np.array([0.999,   0.996,  0.990,  0.982,  0.960,  0.937,
                      0.912,   0.8822, 0.853,  0.825,  0.797,  0.770,
                      0.744,   0.691,  0.640,  0.539,  0.440,  0.342,
                      0.249,   0.156,  0.064, -0.028, -0.138, -0.251,
                      -0.362, -0.702, -1.000])
    shear_ref = np.array([0.000, 0.015, 0.027, 0.038, 0.056, 0.072,
                          0.085, 0.095, 0.104, 0.113, 0.122, 0.130,
                          0.138, 0.153, 0.168, 0.195, 0.220, 0.244,
                          0.268, 0.291, 0.313, 0.333, 0.359, 0.382,
                          0.404, 0.463, 0.500])
    shape_ref = np.array([3.55, 3.49, 3.44, 3.39, 3.30, 3.22,
                          3.15, 3.09, 3.04, 2.99, 2.94, 2.90,
                          2.87, 2.81, 2.75, 2.67, 2.61, 2.55,
                          2.49, 2.44, 2.39, 2.34, 2.28, 2.23,
                          2.18, 2.07, 2.00])
    lam_ref = np.array([-0.090, -0.088, -0.086, -0.084, -0.080, -0.076,
                        -0.072, -0.068, -0.064, -0.060, -0.056, -0.052,
                        -0.048, -0.040, -0.032, -0.016,  0.000,  0.016,
                        +0.032,  0.048,  0.064,  0.080,  0.10,   0.12,
                        +0.14,   0.20,   0.25])

    f_ref_orig = np.array([0.938, 0.953, 0.956, 0.962, 0.967, 0.969, 0.971,
                           0.970, 0.963, 0.952, 0.936, 0.919, 0.902, 0.886,
                           0.854, 0.825, 0.797, 0.770, 0.744, 0.691, 0.640,
                           0.590, 0.539, 0.490, 0.440, 0.342, 0.249, 0.156,
                           0.064,-0.028,-0.138,-0.251,-0.362,-0.702,-1.000])
    shear_ref_orig = np.array([0.000, 0.011, 0.016, 0.024, 0.030, 0.035, 0.039,
                               0.049, 0.055, 0.067, 0.076, 0.083, 0.089, 0.094,
                               0.104, 0.113, 0.122, 0.130, 0.138, 0.153, 0.168,
                               0.182, 0.195, 0.208, 0.220, 0.244, 0.268, 0.291,
                               0.313, 0.333, 0.359, 0.382, 0.404, 0.463, 0.50])
    shape_ref_orig = np.array([3.70,  3.69,  3.66,  3.63,  3.61,  3.59,  3.58,
                               3.52,  3.47,  3.38,  3.30,  3.23,  3.17,  3.13,
                               3.05,  2.99,  2.94,  2.90,  2.87,  2.81,  2.75,
                               2.71,  2.67,  2.64,  2.61,  2.55,  2.49,  2.44,
                               2.39,  2.34,  2.28,  2.23,  2.18,  2.07,  2.00])
    lam_ref_orig = np.array([-0.082,-0.0818,-0.0816,-0.0812,-0.0808,-0.0804,
                             -0.080,-0.079, -0.078, -0.076, -0.074, -0.072,
                             -0.070,-0.068, -0.064, -0.060, -0.056, -0.052,
                             -0.048,-0.040, -0.032, -0.024, -0.016, -0.008,
                             +0.000, 0.016,  0.032,  0.048,  0.064,  0.080,
                             +0.10,  0.12,   0.14,   0.20,   0.25])

    def test_tabular_values(self) -> None:
        """Test the tabular value fits."""

        spline = _ThwaitesFunctionsSpline()

        # test the range of lambda
        lam_min = spline._tab_lambda[0]  # pylint: disable=protected-access
        lam_max = spline._tab_lambda[-1]  # pylint: disable=protected-access
        self.assertIsNone(np_test.assert_allclose(lam_min,
                                                  np.min(self.lam_ref)))
        self.assertIsNone(np_test.assert_allclose(lam_max,
                                                  np.max(self.lam_ref)))

        # test the tabular values
        lam = spline._tab_lambda  # pylint: disable=protected-access
        shape = spline._tab_shape  # pylint: disable=protected-access
        shear = spline._tab_shear  # pylint: disable=protected-access
        f = spline._tab_f  # pylint: disable=protected-access
        self.assertIsNone(np_test.assert_allclose(self.lam_ref, lam))
        self.assertIsNone(np_test.assert_allclose(self.shape_ref, shape))
        self.assertIsNone(np_test.assert_allclose(self.shear_ref, shear))
        self.assertIsNone(np_test.assert_allclose(self.f_ref, f))

        # compare the tabulated F with the calculated value
        f_calc = 2*(shear-lam*(2+shape))
        self.assertIsNone(np_test.assert_allclose(self.f_ref, f_calc,
                                                  rtol=0, atol=1e-2))

    def test_white_functions(self) -> None:
        """Test the White fits."""

        white = _ThwaitesFunctionsWhite()

        # test the range of lambda
        lam_min, lam_max = white.range()
        self.assertIsNone(np_test.assert_allclose(lam_min, -0.09))
        self.assertTrue(np.isinf(lam_max))

        # create the lambdas and reference values for testing
        lam = np.linspace(lam_min, np.max(self.lam_ref), 101)

        # test S function
        def shear_fun(lam: InputParam) -> InputParam:
            return (lam + 0.09)**(0.62)

        # do loop in case hard coded functions cannot take vectors
        shear = np.zeros_like(lam)
        for i, l in enumerate(lam):
            shear[i] = shear_fun(l)
        self.assertIsNone(np_test.assert_allclose(shear, white.shear(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(np_test.assert_allclose(white.shear(lam_min),
                                                  white.shear(2*lam_min)))

        # test H function
        def shape_fun(lam: InputParam) -> InputParam:
            z = 0.25-lam
            return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5

        # do loop in case hard coded functions cannot take vectors
        shape = np.zeros_like(lam)
        for i, l in enumerate(lam):
            shape[i] = shape_fun(l)
        self.assertIsNone(np_test.assert_allclose(shape, white.shape(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(np_test.assert_allclose(white.shape(lam_min),
                                                  white.shape(2*lam_min)))

        # test H' function
        shape_p = np.zeros_like(lam)
        delta = 1e-5
        for i, l in enumerate(lam):
            shape_p[i] = fd_1f(shape_fun, l, l*delta)
        self.assertIsNone(np_test.assert_allclose(shape_p, white.shape_p(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(np_test.assert_allclose(white.shape(lam_min),
                                                  white.shape(2*lam_min)))

    def test_cebeci_bradshaw_functions(self) -> None:
        """Test the Cebeci & Bradshaw fits."""

        cb = _ThwaitesFunctionsCebeciBradshaw()

        # test the range of lambda
        lam_min, lam_max = cb.range()
        self.assertIsNone(np_test.assert_allclose(lam_min, -0.1))
        self.assertIsNone(np_test.assert_allclose(lam_max, 0.1))

        # create the lambdas and reference values for testing
        lam = np.linspace(lam_min, lam_max, 101)

        # test S function
        def shear_fun(lam: InputParam) -> InputParam:
            if lam < 0:
                return 0.22 + 1.402*lam + 0.018*lam/(0.107 + lam)
            return 0.22 + 1.57*lam - 1.8*lam**2

        # do loop in case hard coded functions cannot take vectors
        shear = np.zeros_like(lam)
        for i, l in enumerate(lam):
            shear[i] = shear_fun(l)
        self.assertIsNone(np_test.assert_allclose(shear, cb.shear(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(np_test.assert_allclose(cb.shear(lam_min),
                                                  cb.shear(2*lam_min)))
        self.assertIsNone(np_test.assert_allclose(cb.shear(lam_max),
                                                  cb.shear(2*lam_max)))

        # test H function
        def shape_fun(lam: InputParam) -> InputParam:
            if lam < 0:
                return 2.088 + 0.0731/(0.14 + lam)
            return 2.61 - 3.75*lam + 5.24*lam**2

        # do loop in case hard coded functions cannot take vectors
        shape = np.zeros_like(lam)
        for i, l in enumerate(lam):
            shape[i] = shape_fun(l)
        self.assertIsNone(np_test.assert_allclose(shape, cb.shape(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(np_test.assert_allclose(cb.shape(lam_min),
                                                  cb.shape(2*lam_min)))
        self.assertIsNone(np_test.assert_allclose(cb.shape(lam_max),
                                                  cb.shape(2*lam_max)))

        # test H' function
        # NOTE: Since H is discontinuous at 0, so is H' so remove that case
        lam = lam[lam != 0]
        shape_p = np.zeros_like(lam)
        delta = 1e-5
        for i, l in enumerate(lam):
            shape_p[i] = fd_1f(shape_fun, l, np.maximum(l*delta, delta))

        self.assertIsNone(np_test.assert_allclose(shape_p, cb.shape_p(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(np_test.assert_allclose(cb.shape_p(lam_min),
                                                  cb.shape_p(2*lam_min)))
        self.assertIsNone(np_test.assert_allclose(cb.shape_p(lam_max),
                                                  cb.shape_p(2*lam_max)))

    def test_drela_functions(self) -> None:
        """Test the Drela fits."""

        drela = _ThwaitesFunctionsDrela()

        # test the range of lambda
        lam_min, lam_max = drela.range()
        self.assertIsNone(np_test.assert_allclose(lam_min, -0.09))
        self.assertTrue(np.isinf(lam_max))

        # create the lambdas and reference values for testing
        lam = np.linspace(lam_min, np.max(self.lam_ref), 101)

        # test S function
        def shear_fun(lam: InputParam) -> InputParam:
            return 0.220 + 1.52*lam - 5*lam**3 - 0.072*lam**2/(lam + 0.18)**2

        # do loop in case hard coded functions cannot take vectors
        shear = np.zeros_like(lam)
        for i, l in enumerate(lam):
            shear[i] = shear_fun(l)
        self.assertIsNone(np_test.assert_allclose(shear, drela.shear(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(np_test.assert_allclose(drela.shear(lam_min),
                                                  drela.shear(2*lam_min)))

        # test H function
        def shape_fun(lam: InputParam) -> InputParam:
            return 2.61 - 4.1*lam + 14*lam**3 + 0.56*lam**2/(lam + 0.18)**2

        # do loop in case hard coded functions cannot take vectors
        shape = np.zeros_like(lam)
        for i, l in enumerate(lam):
            shape[i] = shape_fun(l)
        self.assertIsNone(np_test.assert_allclose(shape, drela.shape(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(np_test.assert_allclose(drela.shape(lam_min),
                                                  drela.shape(2*lam_min)))

        # test H' function
        shape_p = np.zeros_like(lam)
        delta = 1e-5
        for i, l in enumerate(lam):
            shape_p[i] = fd_1f(shape_fun, l, l*delta)
        self.assertIsNone(np_test.assert_allclose(shape_p, drela.shape_p(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(np_test.assert_allclose(drela.shape(lam_min),
                                                  drela.shape(2*lam_min)))

    def test_spline_functions(self) -> None:
        """Test the spline fits."""

        spline = _ThwaitesFunctionsSpline()

        # test the range of lambda
        lam_min, lam_max = spline.range()
        self.assertIsNone(np_test.assert_allclose(lam_min,
                                                  np.min(self.lam_ref)))
        self.assertIsNone(np_test.assert_allclose(lam_max,
                                                  np.max(self.lam_ref)))

        # create the lambdas and reference values for testing
        lam = self.lam_ref

        # test S function
        shear = self.shear_ref
        self.assertIsNone(np_test.assert_allclose(shear, spline.shear(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(np_test.assert_allclose(spline.shear(lam_min),
                                                  spline.shear(2*lam_min)))
        self.assertIsNone(np_test.assert_allclose(spline.shear(lam_max),
                                                  spline.shear(2*lam_max)))

        # test H function
        shape = self.shape_ref
        self.assertIsNone(np_test.assert_allclose(shape, spline.shape(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(np_test.assert_allclose(spline.shape(lam_min),
                                                  spline.shape(2*lam_min)))
        self.assertIsNone(np_test.assert_allclose(spline.shape(lam_max),
                                                  spline.shape(2*lam_max)))

        # test H' function
        # NOTE: cannot evaluate spline H outside of end points, so finite
        #       difference cannot be done on end points of lambda
        lam = lam[1:-2]
        shape_p = np.zeros_like(lam)
        delta = 1e-8
        for i, l in enumerate(lam):
            shape_p[i] = fd_1f(spline.shape, l, np.maximum(l*delta, delta))

        self.assertIsNone(np_test.assert_allclose(shape_p,
                                                  spline.shape_p(lam)))

        # check to make sure does not calculate outside of range
        self.assertIsNone(np_test.assert_allclose(spline.shape_p(lam_min),
                                                  spline.shape_p(2*lam_min)))
        self.assertIsNone(np_test.assert_allclose(spline.shape_p(lam_max),
                                                  spline.shape_p(2*lam_max)))


if __name__ == "__main__":
    unittest.main(verbosity=1)

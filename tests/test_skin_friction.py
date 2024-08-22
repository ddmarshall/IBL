"""Module to test the Head method functionality."""

import unittest
import numpy as np
import numpy.testing as npt

from ibl.skin_friction import ludwieg_tillman
from ibl.skin_friction import felsch
from ibl.skin_friction import white
from ibl.typing import InputParam


class TestSkinFrictionCalculations(unittest.TestCase):
    """Class to test various functions and curve fits for Thwaites method"""

    def test_ludwieg_tillman(self) -> None:
        """Test the Ludwieg-Tillman function."""
        shape_d_range = np.array([1.2, 3.4])
        re_delta_m_range = np.array([1e2, 1e5])

        def fun(re_delta_m: InputParam, shape_d: InputParam) -> InputParam:
            return 0.246/(re_delta_m**0.268*10**(0.678*shape_d))

        # calculate range of displacement shape parameter
        re_delta_m = np.average(re_delta_m_range)
        shape_d = np.linspace(shape_d_range[0], shape_d_range[-1])
        c_f_ref = fun(re_delta_m, shape_d)
        c_f = ludwieg_tillman(re_delta_m, shape_d)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))

        # calculate range of Reynolds number
        re_delta_m = np.logspace(np.log10(re_delta_m_range[0]),
                                 np.log10(re_delta_m_range[-1]))
        shape_d = np.average(shape_d_range)
        c_f_ref = fun(re_delta_m, shape_d)
        c_f = ludwieg_tillman(re_delta_m, shape_d)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))

    def test_felsch(self) -> None:
        """Test the Felsch function."""
        shape_d_range = np.array([1.2, 2.998])
        re_delta_m_range = np.array([1e2, 1e5])

        def fun(re_delta_m: InputParam, shape_d: InputParam) -> InputParam:
            return 0.058*(0.93
                          - 1.95*np.log10(shape_d))**1.705/(re_delta_m**0.268)

        # calculate range of displacement shape parameter
        re_delta_m = np.average(re_delta_m_range)
        shape_d = np.linspace(shape_d_range[0], shape_d_range[-1])
        c_f_ref = fun(re_delta_m, shape_d)
        c_f = felsch(re_delta_m, shape_d)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))

        # calculate range of Reynolds number
        re_delta_m = np.logspace(np.log10(re_delta_m_range[0]),
                                 np.log10(re_delta_m_range[-1]))
        shape_d = np.average(shape_d_range)
        c_f_ref = fun(re_delta_m, shape_d)
        c_f = felsch(re_delta_m, shape_d)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))

        # check case when separated
        c_f_ref = 0
        c_f = felsch(1e4, 3)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f, rtol=0, atol=1e-7))

    def test_white(self) -> None:
        """Test the White function."""
        shape_d_range = np.array([1.2, 3.4])
        re_delta_m_range = np.array([1e2, 1e5])

        def fun(re_delta_m: InputParam, shape_d: InputParam) -> InputParam:
            return 0.3/(np.exp(1.33*shape_d)
                        * np.log10(re_delta_m)**(1.74+0.31*shape_d))

        # calculate range of displacement shape parameter
        re_delta_m = np.average(re_delta_m_range)
        shape_d = np.linspace(shape_d_range[0], shape_d_range[-1])
        c_f_ref = fun(re_delta_m, shape_d)
        c_f = white(re_delta_m, shape_d)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))

        # calculate range of Reynolds number
        re_delta_m = np.logspace(np.log10(re_delta_m_range[0]),
                                 np.log10(re_delta_m_range[-1]))
        shape_d = np.average(shape_d_range)
        c_f_ref = fun(re_delta_m, shape_d)
        c_f = white(re_delta_m, shape_d)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))


if __name__ == "__main__":
    _ = unittest.main(verbosity=1)

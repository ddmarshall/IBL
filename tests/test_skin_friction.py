#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:51:34 2022

@author: ddmarshall
"""

import unittest
import numpy as np
import numpy.testing as npt

from pyBL.skin_friction import c_f_LudwiegTillman
from pyBL.skin_friction import c_f_Felsch
from pyBL.skin_friction import c_f_White


class TestSkinFrictionCalculations(unittest.TestCase):
    """Class to test various functions and curve fits for Thwaites method"""

    def test_ludwieg_tillman(self):
        """Test the Ludwieg-Tillman function."""
        H_d_range = [1.2, 3.4]
        Re_delta_m_range = [1e2, 1e5]

        def fun(Re_delta_m, H_d):
            return 0.246/(Re_delta_m**0.268*10**(0.678*H_d))

        # calculate range of displacement shape parameter
        Re_delta_m = np.average(Re_delta_m_range)
        H_d = np.linspace(H_d_range[0], H_d_range[-1])
        c_f_ref = fun(Re_delta_m, H_d)
        c_f = c_f_LudwiegTillman(Re_delta_m, H_d)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))

        # calculate range of Reynolds number
        Re_delta_m = np.logspace(np.log10(Re_delta_m_range[0]),
                                 np.log10(Re_delta_m_range[-1]))
        H_d = np.average(H_d_range)
        c_f_ref = fun(Re_delta_m, H_d)
        c_f = c_f_LudwiegTillman(Re_delta_m, H_d)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))

    def test_felsch(self):
        """Test the Felsch function."""
        H_d_range = [1.2, 2.998]
        Re_delta_m_range = [1e2, 1e5]

        def fun(Re_delta_m, H_d):
            return 0.058*(0.93 - 1.95*np.log10(H_d))**1.705/(Re_delta_m**0.268)

        # calculate range of displacement shape parameter
        Re_delta_m = np.average(Re_delta_m_range)
        H_d = np.linspace(H_d_range[0], H_d_range[-1])
        c_f_ref = fun(Re_delta_m, H_d)
        c_f = c_f_Felsch(Re_delta_m, H_d)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))

        # calculate range of Reynolds number
        Re_delta_m = np.logspace(np.log10(Re_delta_m_range[0]),
                                 np.log10(Re_delta_m_range[-1]))
        H_d = np.average(H_d_range)
        c_f_ref = fun(Re_delta_m, H_d)
        c_f = c_f_Felsch(Re_delta_m, H_d)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))

        # check case when separated
        c_f_ref = 0
        c_f = c_f_Felsch(1e4, 3)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f, rtol=0, atol=1e-7))

    def test_white(self):
        """Test the White function."""
        H_d_range = [1.2, 3.4]
        Re_delta_m_range = [1e2, 1e5]

        def fun(Re_delta_m, H_d):
            return 0.3/(np.exp(1.33*H_d)*np.log10(Re_delta_m)**(1.74+0.31*H_d))

        # calculate range of displacement shape parameter
        Re_delta_m = np.average(Re_delta_m_range)
        H_d = np.linspace(H_d_range[0], H_d_range[-1])
        c_f_ref = fun(Re_delta_m, H_d)
        c_f = c_f_White(Re_delta_m, H_d)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))

        # calculate range of Reynolds number
        Re_delta_m = np.logspace(np.log10(Re_delta_m_range[0]),
                                 np.log10(Re_delta_m_range[-1]))
        H_d = np.average(H_d_range)
        c_f_ref = fun(Re_delta_m, H_d)
        c_f = c_f_White(Re_delta_m, H_d)
        self.assertIsNone(npt.assert_allclose(c_f_ref, c_f))


if __name__ == "__main__":
    unittest.main(verbosity=1)

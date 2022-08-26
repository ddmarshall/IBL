#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:38:44 2022

@author: ddmarshall
"""

import unittest
import numpy.testing as npt

from pyBL.stanford_olympics import StanfordOlympics1968


class TestStanfordOlympics1968(unittest.TestCase):
    """Class to test importing data from the 1968 Stanford Olympics"""
    
    def test_case_1100(self):
        so1100 = StanfordOlympics1968("1100")
        
        ## test the case info
        self.assertEqual(so1100.case, "1100")
        self.assertIsNone(npt.assert_allclose(so1100.nu, 1.55e-5))
        
        ## test some station data
        station = so1100.get_station(0)
        self.assertIsNone(npt.assert_allclose(station.x, 0.782))
        self.assertIsNone(npt.assert_allclose(station.U_e, 33.90))
        self.assertIsNone(npt.assert_allclose(station.dU_edx, -2.300))
        self.assertIsNone(npt.assert_allclose(station.delta_m, 0.00276))
        self.assertIsNone(npt.assert_allclose(station.H_d, 1.381))
        self.assertIsNone(npt.assert_allclose(station.H_k, 1.778))
        self.assertIsNone(npt.assert_allclose(station.G, 7.307))
        self.assertIsNone(npt.assert_allclose(station.c_f, 0.00285))
        self.assertIsNone(npt.assert_allclose(station.c_f_LT, 0.00276))
        self.assertIsNone(npt.assert_allclose(station.c_f_E, 0.00271))
        self.assertIsNone(npt.assert_allclose(station.beta, 0.181))
        station = so1100.get_station(10)
        self.assertIsNone(npt.assert_allclose(station.x, 4.132))
        self.assertIsNone(npt.assert_allclose(station.U_e, 23.60))
        self.assertIsNone(npt.assert_allclose(station.dU_edx, -2.250))
        self.assertIsNone(npt.assert_allclose(station.delta_m, 0.02246))
        self.assertIsNone(npt.assert_allclose(station.H_d, 1.594))
        self.assertIsNone(npt.assert_allclose(station.H_k, 1.664))
        self.assertIsNone(npt.assert_allclose(station.G, 14.960))
        self.assertIsNone(npt.assert_allclose(station.c_f, 0.00124))
        self.assertIsNone(npt.assert_allclose(station.c_f_LT, 0.00124))
        self.assertIsNone(npt.assert_allclose(station.c_f_E, 0.00126))
        self.assertIsNone(npt.assert_allclose(station.beta, 5.499))
        
        ## test some smooth data
        x, U_e, dU_edx = so1100.get_smooth_velocity()
        self.assertIsNone(npt.assert_allclose(x[0], 0.50))
        self.assertIsNone(npt.assert_allclose(U_e[0], 34.41))
        self.assertIsNone(npt.assert_allclose(dU_edx[0], -1.69))
        self.assertIsNone(npt.assert_allclose(x[7], 2.25))
        self.assertIsNone(npt.assert_allclose(U_e[7], 28.78))
        self.assertIsNone(npt.assert_allclose(dU_edx[7], -3.62))
        self.assertIsNone(npt.assert_allclose(x[14], 4.00))
        self.assertIsNone(npt.assert_allclose(U_e[14], 23.85))
        self.assertIsNone(npt.assert_allclose(dU_edx[14], -2.29))


if (__name__ == "__main__"):
    unittest.main(verbosity=1)

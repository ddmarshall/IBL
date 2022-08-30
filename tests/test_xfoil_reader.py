#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 06:21:08 2022

@author: ddmarshall
"""


import unittest
import numpy.testing as npt

from pyBL.xfoil_reader import XFoilReader


class TestXFoilDumpReader(unittest.TestCase):
    """Class to test importing data from XFoil dump file"""
    
    def test_case_1100(self):
        
        ## Common XFoil case settings
        airfoil_name = "NACA 0003"
        alpha = 0
        c = 1 # (m)
        U_inf = 20 # (m/s)
        Re = 1000
        rho = 1.2
        nu_inf = U_inf*c/Re
        x_trans = 1
        n_trans = 9
        
        ## Read a dump file from inviscid solution
        inv_filename = "data/xfoil_inviscid_dump.txt"
        xfoil_inv = XFoilReader(inv_filename, airfoil = airfoil_name,
                                alpha = alpha, c = c)
        
        # test case info
        self.assertEqual(xfoil_inv.aifoil, airfoil_name)
        self.assertEqual(xfoil_inv.alpha, alpha)
        self.assertEqual(xfoil_inv.c, c)
        self.assertIsNone(xfoil_inv.Re)
        self.assertIsNone(xfoil_inv.x_trans)
        self.assertIsNone(xfoil_inv.n_trans)
        
        ## Read a dump file from viscous solution
        visc_filename = "data/xfoil_laminar_dump.txt"
        xfoil_visc = XFoilReader(visc_filename, airfoil = airfoil_name,
                                 alpha = alpha, c = c, Re = Re, x_trans = x_trans,
                                 n_trans = n_trans)
        
        # test case info
        self.assertEqual(xfoil_visc.aifoil, airfoil_name)
        self.assertEqual(xfoil_visc.alpha, alpha)
        self.assertEqual(xfoil_visc.c, c)
        self.assertEqual(xfoil_visc.Re, Re)
        self.assertEqual(xfoil_visc.x_trans[0], x_trans)
        self.assertEqual(xfoil_visc.x_trans[1], x_trans)
        self.assertEqual(xfoil_visc.n_trans, n_trans)
        
#        ## test some station data
#        station = so1100.get_station(0)
#        self.assertIsNone(npt.assert_allclose(station.x, 0.782))
#        self.assertIsNone(npt.assert_allclose(station.U_e, 33.90))
#        self.assertIsNone(npt.assert_allclose(station.dU_edx, -2.300))
#        self.assertIsNone(npt.assert_allclose(station.delta_m, 0.00276))
#        self.assertIsNone(npt.assert_allclose(station.H_d, 1.381))
#        self.assertIsNone(npt.assert_allclose(station.H_k, 1.778))
#        self.assertIsNone(npt.assert_allclose(station.G, 7.307))
#        self.assertIsNone(npt.assert_allclose(station.c_f, 0.00285))
#        self.assertIsNone(npt.assert_allclose(station.c_f_LT, 0.00276))
#        self.assertIsNone(npt.assert_allclose(station.c_f_E, 0.00271))
#        self.assertIsNone(npt.assert_allclose(station.beta, 0.181))
#        station = so1100.get_station(10)
#        self.assertIsNone(npt.assert_allclose(station.x, 4.132))
#        self.assertIsNone(npt.assert_allclose(station.U_e, 23.60))
#        self.assertIsNone(npt.assert_allclose(station.dU_edx, -2.250))
#        self.assertIsNone(npt.assert_allclose(station.delta_m, 0.02246))
#        self.assertIsNone(npt.assert_allclose(station.H_d, 1.594))
#        self.assertIsNone(npt.assert_allclose(station.H_k, 1.664))
#        self.assertIsNone(npt.assert_allclose(station.G, 14.960))
#        self.assertIsNone(npt.assert_allclose(station.c_f, 0.00124))
#        self.assertIsNone(npt.assert_allclose(station.c_f_LT, 0.00124))
#        self.assertIsNone(npt.assert_allclose(station.c_f_E, 0.00126))
#        self.assertIsNone(npt.assert_allclose(station.beta, 5.499))
#        
#        ## test some smooth data
#        x, U_e, dU_edx = so1100.get_smooth_velocity()
#        self.assertIsNone(npt.assert_allclose(x[0], 0.50))
#        self.assertIsNone(npt.assert_allclose(U_e[0], 34.41))
#        self.assertIsNone(npt.assert_allclose(dU_edx[0], -1.69))
#        self.assertIsNone(npt.assert_allclose(x[7], 2.25))
#        self.assertIsNone(npt.assert_allclose(U_e[7], 28.78))
#        self.assertIsNone(npt.assert_allclose(dU_edx[7], -3.62))
#        self.assertIsNone(npt.assert_allclose(x[14], 4.00))
#        self.assertIsNone(npt.assert_allclose(U_e[14], 23.85))
#        self.assertIsNone(npt.assert_allclose(dU_edx[14], -2.29))


if (__name__ == "__main__"):
    unittest.main(verbosity=1)

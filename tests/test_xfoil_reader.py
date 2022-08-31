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
    
    def test_case_inviscid(self):
        
        ## Common XFoil case settings
        airfoil_name = "NACA 0003"
        alpha = 0
        c = 1 # (m)
        
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
        self.assertEqual(xfoil_inv.num_points_upper(), 12)
        self.assertEqual(xfoil_inv.num_points_lower(), 12)
        self.assertEqual(xfoil_inv.num_points_wake(), 0)
        
        # test point info
        s_upper_ref = [0.0, 0.000695, 0.092695, 0.196925, 0.301285, 0.392635,
                       0.497055, 0.601495, 0.692885, 0.797335, 0.901785,
                       1.001785]
        s_lower_ref = [0, 0.000695, 0.105705, 0.209965, 0.301285, 0.405685,
                       0.510105, 0.601485, 0.705935, 0.810385, 0.901775,
                       1.001775]
        s_wake_ref = []
        x_upper_ref = [0.0, 0.00024, 0.09114, 0.19533, 0.29968, 0.39103,
                       0.49545, 0.59987, 0.69123, 0.79565, 0.90005, 1.0]
        x_lower_ref = [0.0, 0.00024, 0.10414, 0.20837, 0.29968, 0.40408,
                       0.5085, 0.59987, 0.70429, 0.8087, 0.90005, 1.0]
        x_wake_ref = []
        y_upper_ref = [0.0, 0.00069, 0.01132, 0.01427, 0.015, 0.01459, 0.01331,
                       0.01141, 0.00937, 0.00668, 0.00362, 0.00031]
        y_lower_ref = [0.0, -0.00069, -0.01188, -0.01446, -0.015, -0.01447,
                       -0.0131, -0.01141, -0.00906, -0.00632, -0.00362,
                       -0.00031]
        y_wake_ref = []
        U_e_upper_ref = [0.0, 0.72574, 1.05098, 1.04538, 1.03909, 1.03339,
                         1.02688, 1.02041, 1.01457, 1.00702, 0.9962, 0.92497]
        U_e_lower_ref = [0.0, 0.72574, 1.05034, 1.04462, 1.03909, 1.03258,
                         1.02607, 1.02041, 1.0137, 1.00594, 0.9962, 0.92497]
        U_e_wake_ref = []
        delta_d_upper_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0]
        delta_d_lower_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0]
        delta_d_wake_ref = []
        delta_m_upper_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0]
        delta_m_lower_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0]
        delta_m_wake_ref = []
        delta_k_upper_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0]
        delta_k_lower_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0]
        H_d_upper_ref = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         1.0]
        H_d_lower_ref = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         1.0]
        H_d_wake_ref = []
        H_k_upper_ref = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                         2.0]
        H_k_lower_ref = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                         2.0]
        c_f_upper_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0]
        c_f_lower_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0]
        m_upper_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0]
        m_lower_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0]
        P_upper_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0]
        P_lower_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0]
        K_upper_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0]
        K_lower_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0]
        self.assertIsNone(npt.assert_allclose(s_upper_ref,
                                              xfoil_inv.s_upper()))
        self.assertIsNone(npt.assert_allclose(s_lower_ref,
                                              xfoil_inv.s_lower()))
        self.assertEqual(s_wake_ref, xfoil_inv.s_wake())
        self.assertIsNone(npt.assert_allclose(x_upper_ref,
                                              xfoil_inv.x_upper()))
        self.assertIsNone(npt.assert_allclose(x_lower_ref,
                                              xfoil_inv.x_lower()))
        self.assertEqual(x_wake_ref, xfoil_inv.x_wake())
        self.assertIsNone(npt.assert_allclose(y_upper_ref,
                                              xfoil_inv.y_upper()))
        self.assertIsNone(npt.assert_allclose(y_lower_ref,
                                              xfoil_inv.y_lower()))
        self.assertEqual(y_wake_ref, xfoil_inv.y_wake())
        self.assertIsNone(npt.assert_allclose(U_e_upper_ref,
                                              xfoil_inv.U_e_upper()))
        self.assertIsNone(npt.assert_allclose(U_e_lower_ref,
                                              xfoil_inv.U_e_lower()))
        self.assertEqual(U_e_wake_ref, xfoil_inv.U_e_wake())
        self.assertIsNone(npt.assert_allclose(delta_d_upper_ref,
                                              xfoil_inv.delta_d_upper()))
        self.assertIsNone(npt.assert_allclose(delta_d_lower_ref,
                                              xfoil_inv.delta_d_lower()))
        self.assertEqual(delta_d_wake_ref, xfoil_inv.delta_d_wake())
        self.assertIsNone(npt.assert_allclose(delta_m_upper_ref,
                                              xfoil_inv.delta_m_upper()))
        self.assertIsNone(npt.assert_allclose(delta_m_lower_ref,
                                              xfoil_inv.delta_m_lower()))
        self.assertEqual(delta_m_wake_ref, xfoil_inv.delta_m_wake())
        self.assertIsNone(npt.assert_allclose(delta_k_upper_ref,
                                              xfoil_inv.delta_k_upper()))
        self.assertIsNone(npt.assert_allclose(delta_k_lower_ref,
                                              xfoil_inv.delta_k_lower()))
        self.assertIsNone(npt.assert_allclose(H_d_upper_ref,
                                              xfoil_inv.H_d_upper()))
        self.assertIsNone(npt.assert_allclose(H_d_lower_ref,
                                              xfoil_inv.H_d_lower()))
        self.assertEqual(H_d_wake_ref, xfoil_inv.H_d_wake())
        self.assertIsNone(npt.assert_allclose(H_k_upper_ref,
                                              xfoil_inv.H_k_upper()))
        self.assertIsNone(npt.assert_allclose(H_k_lower_ref,
                                              xfoil_inv.H_k_lower()))
        self.assertIsNone(npt.assert_allclose(c_f_upper_ref,
                                              xfoil_inv.c_f_upper()))
        self.assertIsNone(npt.assert_allclose(c_f_lower_ref,
                                              xfoil_inv.c_f_lower()))
        self.assertIsNone(npt.assert_allclose(m_upper_ref,
                                              xfoil_inv.m_upper()))
        self.assertIsNone(npt.assert_allclose(m_lower_ref,
                                              xfoil_inv.m_lower()))
        self.assertIsNone(npt.assert_allclose(P_upper_ref,
                                              xfoil_inv.P_upper()))
        self.assertIsNone(npt.assert_allclose(P_lower_ref,
                                              xfoil_inv.P_lower()))
        self.assertIsNone(npt.assert_allclose(K_upper_ref,
                                              xfoil_inv.K_upper()))
        self.assertIsNone(npt.assert_allclose(K_lower_ref,
                                              xfoil_inv.K_lower()))
    
    def test_case_viscous(self):
        
        ## Common XFoil case settings
        airfoil_name = "NACA 0003"
        alpha = 0
        c = 1 # (m)
        Re = 1000
        x_trans = 1
        n_trans = 9
        
        ## Read a dump file from viscous solution
        visc_filename = "data/xfoil_viscous_dump.txt"
        xfoil_visc = XFoilReader(visc_filename, airfoil = airfoil_name,
                                 alpha = alpha, c = c, Re = Re,
                                 x_trans = x_trans, n_trans = n_trans)
        
        # test case info
        self.assertEqual(xfoil_visc.aifoil, airfoil_name)
        self.assertEqual(xfoil_visc.alpha, alpha)
        self.assertEqual(xfoil_visc.c, c)
        self.assertEqual(xfoil_visc.Re, Re)
        self.assertEqual(xfoil_visc.x_trans[0], x_trans)
        self.assertEqual(xfoil_visc.x_trans[1], x_trans)
        self.assertEqual(xfoil_visc.n_trans, n_trans)
        self.assertEqual(xfoil_visc.num_points_upper(), 12)
        self.assertEqual(xfoil_visc.num_points_lower(), 12)
        self.assertEqual(xfoil_visc.num_points_wake(), 8)
        
        # test point info
        s_upper_ref = [0.0, 0.000695, 0.092695, 0.196925, 0.301285, 0.392635,
                       0.497055, 0.601495, 0.692885, 0.797335, 0.901785,
                       1.001785]
        s_lower_ref = [0, 0.000695, 0.105705, 0.209965, 0.301285, 0.405685,
                       0.510105, 0.601485, 0.705935, 0.810385, 0.901775,
                       1.001775]
        s_wake_ref = [0.0, 0.11471, 0.19813, 0.32026, 0.43176, 0.49908,
                      0.76089, 0.9999]
        x_upper_ref = [0.0, 0.00024, 0.09114, 0.19533, 0.29968, 0.39103,
                       0.49545, 0.59987, 0.69123, 0.79565, 0.90005, 1.0]
        x_lower_ref = [0.0, 0.00024, 0.10414, 0.20837, 0.29968, 0.40408,
                       0.5085, 0.59987, 0.70429, 0.8087, 0.90005, 1.0]
        x_wake_ref = [1.0001, 1.11481, 1.19822, 1.32036, 1.43186, 1.49918,
                      1.76099, 2.0]
        y_upper_ref = [0.0, 0.00069, 0.01132, 0.01427, 0.015, 0.01459, 0.01331,
                       0.01141, 0.00937, 0.00668, 0.00362, 0.00031]
        y_lower_ref = [0.0, -0.00069, -0.01188, -0.01446, -0.015, -0.01447,
                       -0.0131, -0.01141, -0.00906, -0.00632, -0.00362,
                       -0.00031]
        y_wake_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        U_e_upper_ref = [0.0, 0.4447, 1.06059, 1.05732, 1.05267, 1.0485,
                         1.04388, 1.03951, 1.03593, 1.03207, 1.02842, 1.02513]
        U_e_lower_ref = [0.0, 0.4447, 1.06044, 1.05676, 1.05267, 1.04791,
                         1.04332, 1.03951, 1.03543, 1.03161, 1.02842, 1.02513]
        U_e_wake_ref = [1.02513, 1.0147, 1.0081, 0.99912, 0.99122, 0.98638,
                        0.96382, 0.92209]
        delta_d_upper_ref = [0.000811, 0.000811, 0.015435, 0.023266, 0.029346,
                             0.033966, 0.038734, 0.043102, 0.04667, 0.050523,
                             0.054191, 0.05753]
        delta_d_lower_ref = [0.000811, 0.000811, 0.016586, 0.024091, 0.029346,
                             0.034589, 0.0393, 0.043102, 0.047164, 0.05099,
                             0.054191, 0.05753]
        delta_d_wake_ref = [0.11569, 0.115166, 0.114388, 0.113283, 0.112767,
                            0.11289, 0.120079, 0.173472]
        delta_m_upper_ref = [0.000364, 0.000364, 0.006013, 0.009, 0.011294,
                             0.013019, 0.014787, 0.016399, 0.017711, 0.019121,
                             0.020455, 0.021671]
        delta_m_lower_ref = [0.000364, 0.000364, 0.006454, 0.009313, 0.011294,
                             0.013251, 0.014996, 0.016399, 0.017892, 0.019292,
                             0.020455, 0.021671]
        delta_m_wake_ref = [0.043341, 0.04543, 0.046781, 0.048653, 0.050336,
                            0.051386, 0.056558, 0.068368]
        delta_k_upper_ref = [0.0005900804, 0.0005900804, 0.0094879127,
                             0.0141849, 0.0177857912, 0.0204906041,
                             0.0232584723, 0.0257808679, 0.0278328365,
                             0.0300352668, 0.032118441, 0.0339346189]
        delta_k_lower_ref = [0.0005900804, 0.0005900804, 0.0101818304,
                             0.014677288, 0.0177857912, 0.0208530987,
                             0.0235857088, 0.0257808679, 0.0281154888,
                             0.0303038736, 0.032118441, 0.0339346189]
        H_d_upper_ref = [2.2295, 2.2295, 2.5668, 2.585, 2.5984, 2.6089, 2.6194,
                         2.6284, 2.6352, 2.6423, 2.6493, 2.6548]
        H_d_lower_ref = [2.2295, 2.2295, 2.5699, 2.5868, 2.5984, 2.6103, 2.6206,
                         2.6284, 2.6361, 2.6431, 2.6493, 2.6548]
        H_d_wake_ref = [2.6693, 2.535, 2.4452, 2.3284, 2.2403, 2.1969, 2.1231,
                        2.5373]
        H_k_upper_ref = [1.6211, 1.6211, 1.5779, 1.5761, 1.5748, 1.5739, 1.5729,
                         1.5721, 1.5715, 1.5708, 1.5702, 1.5659]
        H_k_lower_ref = [1.6211, 1.6211, 1.5776, 1.576, 1.5748, 1.5737, 1.5728,
                         1.5721, 1.5714, 1.5708, 1.5702, 1.5659]
        c_f_upper_ref = [0.8771445, 0.877145, 0.078378, 0.050783, 0.039477,
                         0.033566, 0.02895, 0.025638, 0.023409, 0.021365,
                         0.019684, 0.018362]
        c_f_lower_ref = [0.8771445, 0.877144, 0.072669, 0.048918, 0.039477,
                         0.03289, 0.028478, 0.025638, 0.023129, 0.021137,
                         0.019684, 0.018362]
        m_upper_ref = [0.00036, 0.00036, 0.01637, 0.0246, 0.03089, 0.03561,
                       0.04043, 0.04481, 0.04835, 0.05214, 0.05573, 0.05898]
        m_lower_ref = [0.00036, 0.00036, 0.01759, 0.02546, 0.03089, 0.03625,
                       0.041, 0.04481, 0.04884, 0.0526, 0.05573, 0.05898]
        P_upper_ref = [7e-05, 7e-05, 0.00676, 0.01006, 0.01251, 0.01431,
                       0.01611, 0.01772, 0.01901, 0.02037, 0.02163, 0.02277]
        P_lower_ref = [7e-05, 7e-05, 0.00726, 0.0104, 0.01251, 0.01455,
                       0.01632, 0.01772, 0.01918, 0.02053, 0.02163, 0.02277]
        K_upper_ref = [5e-05, 5e-05, 0.01132, 0.01677, 0.02075, 0.02362,
                       0.02646, 0.02896, 0.03094, 0.03302, 0.03494, 0.03656]
        K_lower_ref = [5e-05, 5e-05, 0.01214, 0.01732, 0.02075, 0.024, 0.02679,
                       0.02896, 0.03121, 0.03327, 0.03494, 0.03656]
        self.assertIsNone(npt.assert_allclose(s_upper_ref,
                                              xfoil_visc.s_upper()))
        self.assertIsNone(npt.assert_allclose(s_lower_ref,
                                              xfoil_visc.s_lower()))
        self.assertIsNone(npt.assert_allclose(s_wake_ref,
                                              xfoil_visc.s_wake()))
        self.assertIsNone(npt.assert_allclose(x_upper_ref,
                                              xfoil_visc.x_upper()))
        self.assertIsNone(npt.assert_allclose(x_lower_ref,
                                              xfoil_visc.x_lower()))
        self.assertIsNone(npt.assert_allclose(x_wake_ref,
                                              xfoil_visc.x_wake()))
        self.assertIsNone(npt.assert_allclose(y_upper_ref,
                                              xfoil_visc.y_upper()))
        self.assertIsNone(npt.assert_allclose(y_lower_ref,
                                              xfoil_visc.y_lower()))
        self.assertIsNone(npt.assert_allclose(y_wake_ref,
                                              xfoil_visc.y_wake()))
        self.assertIsNone(npt.assert_allclose(U_e_upper_ref,
                                              xfoil_visc.U_e_upper()))
        self.assertIsNone(npt.assert_allclose(U_e_lower_ref,
                                              xfoil_visc.U_e_lower()))
        self.assertIsNone(npt.assert_allclose(U_e_wake_ref,
                                              xfoil_visc.U_e_wake()))
        self.assertIsNone(npt.assert_allclose(delta_d_upper_ref,
                                              xfoil_visc.delta_d_upper()))
        self.assertIsNone(npt.assert_allclose(delta_d_lower_ref,
                                              xfoil_visc.delta_d_lower()))
        self.assertIsNone(npt.assert_allclose(delta_d_wake_ref,
                                              xfoil_visc.delta_d_wake()))
        self.assertIsNone(npt.assert_allclose(delta_m_upper_ref,
                                              xfoil_visc.delta_m_upper()))
        self.assertIsNone(npt.assert_allclose(delta_m_lower_ref,
                                              xfoil_visc.delta_m_lower()))
        self.assertIsNone(npt.assert_allclose(delta_m_wake_ref,
                                              xfoil_visc.delta_m_wake()))
        self.assertIsNone(npt.assert_allclose(delta_k_upper_ref,
                                              xfoil_visc.delta_k_upper()))
        self.assertIsNone(npt.assert_allclose(delta_k_lower_ref,
                                              xfoil_visc.delta_k_lower()))
        self.assertIsNone(npt.assert_allclose(H_d_upper_ref,
                                              xfoil_visc.H_d_upper()))
        self.assertIsNone(npt.assert_allclose(H_d_lower_ref,
                                              xfoil_visc.H_d_lower()))
        self.assertIsNone(npt.assert_allclose(H_d_wake_ref,
                                              xfoil_visc.H_d_wake()))
        self.assertIsNone(npt.assert_allclose(H_k_upper_ref,
                                              xfoil_visc.H_k_upper()))
        self.assertIsNone(npt.assert_allclose(H_k_lower_ref,
                                              xfoil_visc.H_k_lower()))
        self.assertIsNone(npt.assert_allclose(c_f_upper_ref,
                                              xfoil_visc.c_f_upper()))
        self.assertIsNone(npt.assert_allclose(c_f_lower_ref,
                                              xfoil_visc.c_f_lower()))
        self.assertIsNone(npt.assert_allclose(m_upper_ref,
                                              xfoil_visc.m_upper()))
        self.assertIsNone(npt.assert_allclose(m_lower_ref,
                                              xfoil_visc.m_lower()))
        self.assertIsNone(npt.assert_allclose(P_upper_ref,
                                              xfoil_visc.P_upper()))
        self.assertIsNone(npt.assert_allclose(P_lower_ref,
                                              xfoil_visc.P_lower()))
        self.assertIsNone(npt.assert_allclose(K_upper_ref,
                                              xfoil_visc.K_upper()))
        self.assertIsNone(npt.assert_allclose(K_lower_ref,
                                              xfoil_visc.K_lower()))


if (__name__ == "__main__"):
    unittest.main(verbosity=1)

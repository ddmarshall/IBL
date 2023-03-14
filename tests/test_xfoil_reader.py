#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 06:21:08 2022

@author: ddmarshall
"""


from os.path import abspath, dirname
import unittest

import numpy as np
import numpy.testing as npt

from ibl.reference import XFoilReader
from ibl.reference import XFoilAirfoilData
from ibl.reference import XFoilWakeData


class TestXFoilDumpReader(unittest.TestCase):
    """Class to test importing data from XFoil dump file"""

    def test_aifoil_data(self) -> None:
        """Test the aifoil data."""
        xfa = XFoilAirfoilData(data="")

        # test setting values manually
        xfa.s = 2.0
        self.assertEqual(xfa.s, 2.0)
        with self.assertRaises(ValueError):
            xfa.s = -1
        xfa.x = 2.0
        self.assertEqual(xfa.x, 2.0)
        xfa.y = 2.0
        self.assertEqual(xfa.y, 2.0)
        xfa.u_e_rel = 1.0
        self.assertEqual(xfa.u_e_rel, 1.0)
        xfa.delta_d = 2.0
        self.assertEqual(xfa.delta_d, 2.0)
        with self.assertRaises(ValueError):
            xfa.delta_d = -1
        xfa.delta_m = 2.0
        self.assertEqual(xfa.delta_m, 2.0)
        with self.assertRaises(ValueError):
            xfa.delta_m = -1
        xfa.c_f = 2.0
        self.assertEqual(xfa.c_f, 2.0)
        with self.assertRaises(ValueError):
            xfa.c_f = -1
        xfa.shape_d = 2.0
        self.assertEqual(xfa.shape_d, 2.0)
        with self.assertRaises(ValueError):
            xfa.shape_d = 0
        xfa.shape_k = 2.0
        self.assertEqual(xfa.shape_k, 2.0)
        with self.assertRaises(ValueError):
            xfa.shape_k = 0
        xfa.mass_defect = 2.0
        self.assertEqual(xfa.mass_defect, 2.0)
        xfa.mom_defect = 2.0
        self.assertEqual(xfa.mom_defect, 2.0)
        xfa.ke_defect = 2.0
        self.assertEqual(xfa.ke_defect, 2.0)

        # test setting values from string
        data = ("   0.40029  0.59987  0.01141  1.03951  0.043102  0.016399  "
                "0.025638    2.6284    1.5721  0.01772  0.04481  0.02896")
        xfa.reset(data=data)

    def test_wake_data(self) -> None:
        """Test the wake data."""
        xfw = XFoilWakeData(data="")

        # test setting values manually
        xfw.s = 2.0
        self.assertEqual(xfw.s, 2.0)
        with self.assertRaises(ValueError):
            xfw.s = -1
        xfw.x = 2.0
        self.assertEqual(xfw.x, 2.0)
        xfw.y = 2.0
        self.assertEqual(xfw.y, 2.0)
        xfw.u_e_rel = 1.0
        self.assertEqual(xfw.u_e_rel, 1.0)
        xfw.delta_d = 2.0
        self.assertEqual(xfw.delta_d, 2.0)
        with self.assertRaises(ValueError):
            xfw.delta_d = -1
        xfw.delta_m = 2.0
        self.assertEqual(xfw.delta_m, 2.0)
        with self.assertRaises(ValueError):
            xfw.delta_m = -1
        xfw.shape_d = 2.0
        self.assertEqual(xfw.shape_d, 2.0)
        with self.assertRaises(ValueError):
            xfw.shape_d = 0

        # test setting values from string
        data = ("   2.11827  1.11481 -0.00000  1.01470  0.115166  0.045430  "
                "0.000000    2.5350")
        xfw = XFoilWakeData(data=data)

    def test_setters(self) -> None:
        """Test manually setting parameters."""
        xf = XFoilReader(filename="")

        self.assertEqual(xf.filename, "")
        xf.name = "no name"
        self.assertEqual(xf.name, "no name")
        xf.alpha = 0.5
        self.assertEqual(xf.alpha, 0.5)
        xf.c = 2.0
        self.assertEqual(xf.c, 2.0)
        with self.assertRaises(ValueError):
            xf.c = 0
        xf.reynolds = 0.0
        self.assertEqual(xf.reynolds, 0.0)
        with self.assertRaises(ValueError):
            xf.reynolds = -1
        xf.n_trans = 9.0
        self.assertEqual(xf.n_trans, 9.0)
        with self.assertRaises(ValueError):
            xf.n_trans = 0
        xf.x_trans_upper = 0.5
        self.assertEqual(xf.x_trans_upper, 0.5)
        xf.x_trans_lower = 0.5
        self.assertEqual(xf.x_trans_lower, 0.5)
        with self.assertRaises(FileNotFoundError):
            xf.filename = "badname"

        # test accessing empty data
        self.assertEqual(xf.upper_count(), 0)
        self.assertEqual(xf.lower_count(), 0)
        self.assertEqual(xf.wake_count(), 0)
        with self.assertRaises(IndexError):
            xf.upper(0)
        with self.assertRaises(IndexError):
            xf.lower(0)
        with self.assertRaises(IndexError):
            xf.wake(0)

    def test_case_inviscid(self) -> None:
        """Test importing an inviscid case."""
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        # Common XFoil case settings
        airfoil_name = "NACA 0003"
        alpha = np.deg2rad(0)  # (rad)
        c = 1  # (m)

        # Read a dump file from inviscid solution
        directory = dirname(abspath(__file__))
        inv_filename = directory + "/data/xfoil_inviscid_dump.txt"
        xfoil_inv = XFoilReader(inv_filename)
        xfoil_inv.name = airfoil_name
        xfoil_inv.alpha = alpha
        xfoil_inv.c = c

        # test case info
        self.assertEqual(xfoil_inv.name, airfoil_name)
        self.assertEqual(xfoil_inv.alpha, alpha)
        self.assertEqual(xfoil_inv.c, c)
        self.assertEqual(xfoil_inv.reynolds, 0)
        self.assertEqual(xfoil_inv.upper_count(), 12)
        self.assertEqual(xfoil_inv.lower_count(), 12)
        self.assertEqual(xfoil_inv.wake_count(), 0)

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
        u_e_rel_upper_ref = [0.0, 0.72574, 1.05098, 1.04538, 1.03909, 1.03339,
                             1.02688, 1.02041, 1.01457, 1.00702, 0.9962,
                             0.92497]
        u_e_rel_lower_ref = [0.0, 0.72574, 1.05034, 1.04462, 1.03909, 1.03258,
                             1.02607, 1.02041, 1.0137, 1.00594, 0.9962,
                             0.92497]
        u_e_rel_wake_ref = []
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
        shape_d_upper_ref = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0]
        shape_d_lower_ref = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0]
        shape_d_wake_ref = []
        shape_k_upper_ref = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                             2.0, 2.0]
        shape_k_lower_ref = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                             2.0, 2.0]
        c_f_upper_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0]
        c_f_lower_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0]
        mass_defect_upper_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0]
        mass_defect_lower_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0]
        mom_defect_upper_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0]
        mom_defect_lower_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0]
        ke_defect_upper_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0]
        ke_defect_lower_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0]
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
        self.assertIsNone(npt.assert_allclose(u_e_rel_upper_ref,
                                              xfoil_inv.u_e_rel_upper()))
        self.assertIsNone(npt.assert_allclose(u_e_rel_lower_ref,
                                              xfoil_inv.u_e_rel_lower()))
        self.assertEqual(u_e_rel_wake_ref, xfoil_inv.u_e_rel_wake())
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
        self.assertIsNone(npt.assert_allclose(shape_d_upper_ref,
                                              xfoil_inv.shape_d_upper()))
        self.assertIsNone(npt.assert_allclose(shape_d_lower_ref,
                                              xfoil_inv.shape_d_lower()))
        self.assertEqual(shape_d_wake_ref, xfoil_inv.shape_d_wake())
        self.assertIsNone(npt.assert_allclose(shape_k_upper_ref,
                                              xfoil_inv.shape_k_upper()))
        self.assertIsNone(npt.assert_allclose(shape_k_lower_ref,
                                              xfoil_inv.shape_k_lower()))
        self.assertIsNone(npt.assert_allclose(c_f_upper_ref,
                                              xfoil_inv.c_f_upper()))
        self.assertIsNone(npt.assert_allclose(c_f_lower_ref,
                                              xfoil_inv.c_f_lower()))
        self.assertIsNone(npt.assert_allclose(mass_defect_upper_ref,
                                              xfoil_inv.mass_defect_upper()))
        self.assertIsNone(npt.assert_allclose(mass_defect_lower_ref,
                                              xfoil_inv.mass_defect_lower()))
        self.assertIsNone(npt.assert_allclose(mom_defect_upper_ref,
                                              xfoil_inv.mom_defect_upper()))
        self.assertIsNone(npt.assert_allclose(mom_defect_lower_ref,
                                              xfoil_inv.mom_defect_lower()))
        self.assertIsNone(npt.assert_allclose(ke_defect_upper_ref,
                                              xfoil_inv.ke_defect_upper()))
        self.assertIsNone(npt.assert_allclose(ke_defect_lower_ref,
                                              xfoil_inv.ke_defect_lower()))

        # test getting upper, lower, and wake data
        idx = 2
        upper = xfoil_inv.upper(idx)
        self.assertAlmostEqual(upper.s, s_upper_ref[idx])
        self.assertAlmostEqual(upper.x, x_upper_ref[idx])
        self.assertAlmostEqual(upper.y, y_upper_ref[idx])
        self.assertAlmostEqual(upper.u_e_rel, u_e_rel_upper_ref[idx])
        self.assertAlmostEqual(upper.delta_d, delta_d_upper_ref[idx])
        self.assertAlmostEqual(upper.delta_m, delta_m_upper_ref[idx])
        self.assertAlmostEqual(upper.c_f, c_f_upper_ref[idx])
        self.assertAlmostEqual(upper.shape_d, shape_d_upper_ref[idx])
        self.assertAlmostEqual(upper.shape_k, shape_k_upper_ref[idx])
        self.assertAlmostEqual(upper.mass_defect, mass_defect_upper_ref[idx])
        self.assertAlmostEqual(upper.mom_defect, mom_defect_upper_ref[idx])
        self.assertAlmostEqual(upper.ke_defect, ke_defect_upper_ref[idx])
        idx = 3
        lower = xfoil_inv.lower(idx)
        self.assertAlmostEqual(lower.s, s_lower_ref[idx])
        self.assertAlmostEqual(lower.x, x_lower_ref[idx])
        self.assertAlmostEqual(lower.y, y_lower_ref[idx])
        self.assertAlmostEqual(lower.u_e_rel, u_e_rel_lower_ref[idx])
        self.assertAlmostEqual(lower.delta_d, delta_d_lower_ref[idx])
        self.assertAlmostEqual(lower.delta_m, delta_m_lower_ref[idx])
        self.assertAlmostEqual(lower.c_f, c_f_lower_ref[idx])
        self.assertAlmostEqual(lower.shape_d, shape_d_lower_ref[idx])
        self.assertAlmostEqual(lower.shape_k, shape_k_lower_ref[idx])
        self.assertAlmostEqual(lower.mass_defect, mass_defect_lower_ref[idx])
        self.assertAlmostEqual(lower.mom_defect, mom_defect_lower_ref[idx])
        self.assertAlmostEqual(lower.ke_defect, ke_defect_lower_ref[idx])

    def test_case_viscous(self) -> None:
        """Test importing a viscous case."""
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        # Common XFoil case settings
        airfoil_name = "NACA 0003"
        alpha = np.deg2rad(0)  # (rad)
        c = 1  # (m)
        re = 1000
        x_trans = 1  # (m)
        n_trans = 9

        # Read a dump file from viscous solution
        directory = dirname(abspath(__file__))
        visc_filename = directory + "/data/xfoil_viscous_dump.txt"
        xfoil_visc = XFoilReader(visc_filename)
        xfoil_visc.name = airfoil_name
        xfoil_visc.alpha = alpha
        xfoil_visc.c = c
        xfoil_visc.reynolds = re
        xfoil_visc.x_trans_lower = x_trans
        xfoil_visc.x_trans_upper = x_trans
        xfoil_visc.n_trans = n_trans

        # test case info
        self.assertEqual(xfoil_visc.name, airfoil_name)
        self.assertEqual(xfoil_visc.alpha, alpha)
        self.assertEqual(xfoil_visc.c, c)
        self.assertEqual(xfoil_visc.reynolds, re)
        self.assertEqual(xfoil_visc.x_trans_upper, x_trans)
        self.assertEqual(xfoil_visc.x_trans_lower, x_trans)
        self.assertEqual(xfoil_visc.n_trans, n_trans)
        self.assertEqual(xfoil_visc.upper_count(), 12)
        self.assertEqual(xfoil_visc.lower_count(), 12)
        self.assertEqual(xfoil_visc.wake_count(), 8)

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
        u_e_rel_upper_ref = [0.0, 0.4447, 1.06059, 1.05732, 1.05267, 1.0485,
                             1.04388, 1.03951, 1.03593, 1.03207, 1.02842,
                             1.02513]
        u_e_rel_lower_ref = [0.0, 0.4447, 1.06044, 1.05676, 1.05267, 1.04791,
                             1.04332, 1.03951, 1.03543, 1.03161, 1.02842,
                             1.02513]
        u_e_rel_wake_ref = [1.02513, 1.0147, 1.0081, 0.99912, 0.99122, 0.98638,
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
        shape_d_upper_ref = [2.2295, 2.2295, 2.5668, 2.585, 2.5984, 2.6089,
                             2.6194, 2.6284, 2.6352, 2.6423, 2.6493, 2.6548]
        shape_d_lower_ref = [2.2295, 2.2295, 2.5699, 2.5868, 2.5984, 2.6103,
                             2.6206, 2.6284, 2.6361, 2.6431, 2.6493, 2.6548]
        shape_d_wake_ref = [2.6693, 2.535, 2.4452, 2.3284, 2.2403, 2.1969,
                            2.1231, 2.5373]
        shape_k_upper_ref = [1.6211, 1.6211, 1.5779, 1.5761, 1.5748, 1.5739,
                             1.5729, 1.5721, 1.5715, 1.5708, 1.5702, 1.5659]
        shape_k_lower_ref = [1.6211, 1.6211, 1.5776, 1.576, 1.5748, 1.5737,
                             1.5728, 1.5721, 1.5714, 1.5708, 1.5702, 1.5659]
        c_f_upper_ref = [0.8771445, 0.877145, 0.078378, 0.050783, 0.039477,
                         0.033566, 0.02895, 0.025638, 0.023409, 0.021365,
                         0.019684, 0.018362]
        c_f_lower_ref = [0.8771445, 0.877144, 0.072669, 0.048918, 0.039477,
                         0.03289, 0.028478, 0.025638, 0.023129, 0.021137,
                         0.019684, 0.018362]
        mass_defect_upper_ref = [0.00036, 0.00036, 0.01637, 0.0246, 0.03089,
                                 0.03561, 0.04043, 0.04481, 0.04835, 0.05214,
                                 0.05573, 0.05898]
        mass_defect_lower_ref = [0.00036, 0.00036, 0.01759, 0.02546, 0.03089,
                                 0.03625, 0.041, 0.04481, 0.04884, 0.0526,
                                 0.05573, 0.05898]
        mom_defect_upper_ref = [7e-05, 7e-05, 0.00676, 0.01006, 0.01251,
                                0.01431, 0.01611, 0.01772, 0.01901, 0.02037,
                                0.02163, 0.02277]
        mom_defect_lower_ref = [7e-05, 7e-05, 0.00726, 0.0104, 0.01251,
                                0.01455, 0.01632, 0.01772, 0.01918, 0.02053,
                                0.02163, 0.02277]
        ke_defect_upper_ref = [5e-05, 5e-05, 0.01132, 0.01677, 0.02075,
                               0.02362, 0.02646, 0.02896, 0.03094, 0.03302,
                               0.03494, 0.03656]
        ke_defect_lower_ref = [5e-05, 5e-05, 0.01214, 0.01732, 0.02075, 0.024,
                               0.02679, 0.02896, 0.03121, 0.03327, 0.03494,
                               0.03656]
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
        self.assertIsNone(npt.assert_allclose(u_e_rel_upper_ref,
                                              xfoil_visc.u_e_rel_upper()))
        self.assertIsNone(npt.assert_allclose(u_e_rel_lower_ref,
                                              xfoil_visc.u_e_rel_lower()))
        self.assertIsNone(npt.assert_allclose(u_e_rel_wake_ref,
                                              xfoil_visc.u_e_rel_wake()))
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
        self.assertIsNone(npt.assert_allclose(shape_d_upper_ref,
                                              xfoil_visc.shape_d_upper()))
        self.assertIsNone(npt.assert_allclose(shape_d_lower_ref,
                                              xfoil_visc.shape_d_lower()))
        self.assertIsNone(npt.assert_allclose(shape_d_wake_ref,
                                              xfoil_visc.shape_d_wake()))
        self.assertIsNone(npt.assert_allclose(shape_k_upper_ref,
                                              xfoil_visc.shape_k_upper()))
        self.assertIsNone(npt.assert_allclose(shape_k_lower_ref,
                                              xfoil_visc.shape_k_lower()))
        self.assertIsNone(npt.assert_allclose(c_f_upper_ref,
                                              xfoil_visc.c_f_upper()))
        self.assertIsNone(npt.assert_allclose(c_f_lower_ref,
                                              xfoil_visc.c_f_lower()))
        self.assertIsNone(npt.assert_allclose(mass_defect_upper_ref,
                                              xfoil_visc.mass_defect_upper()))
        self.assertIsNone(npt.assert_allclose(mass_defect_lower_ref,
                                              xfoil_visc.mass_defect_lower()))
        self.assertIsNone(npt.assert_allclose(mom_defect_upper_ref,
                                              xfoil_visc.mom_defect_upper()))
        self.assertIsNone(npt.assert_allclose(mom_defect_lower_ref,
                                              xfoil_visc.mom_defect_lower()))
        self.assertIsNone(npt.assert_allclose(ke_defect_upper_ref,
                                              xfoil_visc.ke_defect_upper()))
        self.assertIsNone(npt.assert_allclose(ke_defect_lower_ref,
                                              xfoil_visc.ke_defect_lower()))

        # test getting upper, lower, and wake data
        idx = 2
        upper = xfoil_visc.upper(idx)
        self.assertAlmostEqual(upper.s, s_upper_ref[idx])
        self.assertAlmostEqual(upper.x, x_upper_ref[idx])
        self.assertAlmostEqual(upper.y, y_upper_ref[idx])
        self.assertAlmostEqual(upper.u_e_rel, u_e_rel_upper_ref[idx])
        self.assertAlmostEqual(upper.delta_d, delta_d_upper_ref[idx])
        self.assertAlmostEqual(upper.delta_m, delta_m_upper_ref[idx])
        self.assertAlmostEqual(upper.c_f, c_f_upper_ref[idx])
        self.assertAlmostEqual(upper.shape_d, shape_d_upper_ref[idx])
        self.assertAlmostEqual(upper.shape_k, shape_k_upper_ref[idx])
        self.assertAlmostEqual(upper.mass_defect, mass_defect_upper_ref[idx])
        self.assertAlmostEqual(upper.mom_defect, mom_defect_upper_ref[idx])
        self.assertAlmostEqual(upper.ke_defect, ke_defect_upper_ref[idx])
        idx = 3
        lower = xfoil_visc.lower(idx)
        self.assertAlmostEqual(lower.s, s_lower_ref[idx])
        self.assertAlmostEqual(lower.x, x_lower_ref[idx])
        self.assertAlmostEqual(lower.y, y_lower_ref[idx])
        self.assertAlmostEqual(lower.u_e_rel, u_e_rel_lower_ref[idx])
        self.assertAlmostEqual(lower.delta_d, delta_d_lower_ref[idx])
        self.assertAlmostEqual(lower.delta_m, delta_m_lower_ref[idx])
        self.assertAlmostEqual(lower.c_f, c_f_lower_ref[idx])
        self.assertAlmostEqual(lower.shape_d, shape_d_lower_ref[idx])
        self.assertAlmostEqual(lower.shape_k, shape_k_lower_ref[idx])
        self.assertAlmostEqual(lower.mass_defect, mass_defect_lower_ref[idx])
        self.assertAlmostEqual(lower.mom_defect, mom_defect_lower_ref[idx])
        self.assertAlmostEqual(lower.ke_defect, ke_defect_lower_ref[idx])
        idx = 5
        wake = xfoil_visc.wake(idx)
        self.assertAlmostEqual(wake.s, s_wake_ref[idx])
        self.assertAlmostEqual(wake.x, x_wake_ref[idx])
        self.assertAlmostEqual(wake.y, y_wake_ref[idx])
        self.assertAlmostEqual(wake.u_e_rel, u_e_rel_wake_ref[idx])
        self.assertAlmostEqual(wake.delta_d, delta_d_wake_ref[idx])
        self.assertAlmostEqual(wake.delta_m, delta_m_wake_ref[idx])
        self.assertAlmostEqual(wake.shape_d, shape_d_wake_ref[idx])

    def test_stagnation_point(self) -> None:
        """Test the stagnation point."""
        directory = dirname(abspath(__file__))
        filename = directory + "/data/xfoil_viscous_dump.txt"
        xf = XFoilReader(filename)

        up = xf.upper(0)
        lo = xf.lower(0)

        self.assertAlmostEqual(lo.s, 0)
        self.assertAlmostEqual(lo.x, 0)
        self.assertAlmostEqual(lo.y, 0)
        self.assertAlmostEqual(lo.u_e_rel, 0)
        self.assertAlmostEqual(lo.delta_d, 0.000811)
        self.assertAlmostEqual(lo.delta_m, 0.000364)
        self.assertAlmostEqual(lo.c_f, 0.8771445)
        self.assertAlmostEqual(lo.shape_d, 2.2295)
        self.assertAlmostEqual(lo.shape_k, 1.6211)
        self.assertAlmostEqual(lo.mass_defect, 0.00036)
        self.assertAlmostEqual(lo.mom_defect, 7e-5)
        self.assertAlmostEqual(lo.ke_defect, 5e-5)
        self.assertAlmostEqual(lo.s, up.s)
        self.assertAlmostEqual(lo.x, up.x)
        self.assertAlmostEqual(lo.y, up.y)
        self.assertAlmostEqual(lo.u_e_rel, up.u_e_rel)
        self.assertAlmostEqual(lo.delta_d, up.delta_d)
        self.assertAlmostEqual(lo.delta_m, up.delta_m)
        self.assertAlmostEqual(lo.c_f, up.c_f)
        self.assertAlmostEqual(lo.shape_d, up.shape_d)
        self.assertAlmostEqual(lo.shape_k, up.shape_k)
        self.assertAlmostEqual(lo.mass_defect, up.mass_defect)
        self.assertAlmostEqual(lo.mom_defect, up.mom_defect)
        self.assertAlmostEqual(lo.ke_defect, up.ke_defect)


if __name__ == "__main__":
    unittest.main(verbosity=1)

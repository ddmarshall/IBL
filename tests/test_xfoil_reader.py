"""Test XFoil reader class."""

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
        xf.u_ref = 2.0
        self.assertEqual(xf.u_ref, 2.0)
        with self.assertRaises(ValueError):
            xf.u_ref = 0
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
            _ = xf.upper(0)
        with self.assertRaises(IndexError):
            _ = xf.lower(0)
        with self.assertRaises(IndexError):
            _ = xf.wake(0)

    def test_case_inviscid(self) -> None:
        """Test importing an inviscid case."""
        airfoil_name = "NACA 0003"
        alpha = np.deg2rad(0)  # (rad)
        c = 1  # (m)

        # Read a dump file from inviscid solution
        directory = dirname(abspath(__file__))
        filename = directory + "/data/xfoil_inviscid_dump.txt"
        xf = XFoilReader(filename)
        xf.name = airfoil_name
        xf.alpha = alpha
        xf.c = c

        # test case info
        self.assertEqual(xf.name, airfoil_name)
        self.assertEqual(xf.alpha, alpha)
        self.assertEqual(xf.c, c)
        self.assertEqual(xf.reynolds, 0)
        self.assertEqual(xf.upper_count(), 12)
        self.assertEqual(xf.lower_count(), 12)
        self.assertEqual(xf.wake_count(), 0)

    def test_inviscid_upper(self) -> None:
        """Test getting upper airfoil info from inviscid case."""
        directory = dirname(abspath(__file__))
        filename = directory + "/data/xfoil_inviscid_dump.txt"
        xf = XFoilReader(filename)

        # test point info
        buff = [0.0, 0.000695, 0.092695, 0.196925, 0.301285, 0.392635,
                0.497055, 0.601495, 0.692885, 0.797335, 0.901785, 1.001785]
        self.assertIsNone(npt.assert_allclose(buff, xf.s_upper()))
        buff = [0.0, 0.00024, 0.09114, 0.19533, 0.29968, 0.39103, 0.49545,
                0.59987, 0.69123, 0.79565, 0.90005, 1.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.x_upper()))
        buff = [0.0, 0.00069, 0.01132, 0.01427, 0.015, 0.01459, 0.01331,
                0.01141, 0.00937, 0.00668, 0.00362, 0.00031]
        self.assertIsNone(npt.assert_allclose(buff, xf.y_upper()))
        buff = [0.0, 0.72574, 1.05098, 1.04538, 1.03909, 1.03339, 1.02688,
                1.02041, 1.01457, 1.00702, 0.9962, 0.92497]
        self.assertIsNone(npt.assert_allclose(buff, xf.u_e_upper()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_d_upper()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_m_upper()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_k_upper()))
        buff = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.shape_d_upper()))
        buff = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.shape_k_upper()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.c_f_upper()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.mass_defect_upper()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.mom_defect_upper()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.ke_defect_upper()))

    def test_inviscid_lower(self) -> None:
        """Test accessing airfoil wake from inviscid case."""
        # Read a dump file from inviscid solution
        directory = dirname(abspath(__file__))
        filename = directory + "/data/xfoil_inviscid_dump.txt"
        xf = XFoilReader(filename)

        buff = [0, 0.000695, 0.105705, 0.209965, 0.301285, 0.405685, 0.510105,
                0.601485, 0.705935, 0.810385, 0.901775, 1.001775]
        self.assertIsNone(npt.assert_allclose(buff, xf.s_lower()))
        buff = [0.0, 0.00024, 0.10414, 0.20837, 0.29968, 0.40408, 0.5085,
                0.59987, 0.70429, 0.8087, 0.90005, 1.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.x_lower()))
        buff = [0.0, -0.00069, -0.01188, -0.01446, -0.015, -0.01447, -0.0131,
                -0.01141, -0.00906, -0.00632, -0.00362, -0.00031]
        self.assertIsNone(npt.assert_allclose(buff, xf.y_lower()))
        buff = [0.0, 0.72574, 1.05034, 1.04462, 1.03909, 1.03258, 1.02607,
                1.02041, 1.0137, 1.00594, 0.9962, 0.92497]
        self.assertIsNone(npt.assert_allclose(buff, xf.u_e_lower()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_d_lower()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_m_lower()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_k_lower()))
        buff = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.shape_d_lower()))
        buff = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.shape_k_lower()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.c_f_lower()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.mass_defect_lower()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.mom_defect_lower()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.ke_defect_lower()))

    def test_inviscid_wake(self) -> None:
        """Test accessing airfoil wake from inviscid case."""
        # Read a dump file from inviscid solution
        directory = dirname(abspath(__file__))
        filename = directory + "/data/xfoil_inviscid_dump.txt"
        xf = XFoilReader(filename)

        self.assertTrue(xf.s_wake().size == 0)
        self.assertTrue(xf.x_wake().size == 0)
        self.assertTrue(xf.y_wake().size == 0)
        self.assertTrue(xf.u_e_wake().size == 0)
        self.assertTrue(xf.delta_d_wake().size == 0)
        self.assertTrue(xf.delta_m_wake().size == 0)
        self.assertTrue(xf.shape_d_wake().size == 0)

    def test_inviscid_element_access(self) -> None:
        """Test accessing inviscid properties at specific point."""
        # Read a dump file from inviscid solution
        directory = dirname(abspath(__file__))
        filename = directory + "/data/xfoil_inviscid_dump.txt"
        xf = XFoilReader(filename)

        # test getting upper, lower, and wake data
        upper = xf.upper(2)
        self.assertAlmostEqual(upper.s, 0.092695)
        self.assertAlmostEqual(upper.x, 0.09114)
        self.assertAlmostEqual(upper.y, 0.01132)
        self.assertAlmostEqual(upper.u_e_rel, 1.05098)
        self.assertAlmostEqual(upper.delta_d, 0.000000)
        self.assertAlmostEqual(upper.delta_m, 0.000000)
        self.assertAlmostEqual(upper.c_f, 0.000000)
        self.assertAlmostEqual(upper.shape_d, 1.0000)
        self.assertAlmostEqual(upper.shape_k, 2.0000)
        self.assertAlmostEqual(upper.mass_defect, 0.00000)
        self.assertAlmostEqual(upper.mom_defect, 0.00000)
        self.assertAlmostEqual(upper.ke_defect, 0.00000)
        idx = 3
        lower = xf.lower(idx)
        self.assertAlmostEqual(lower.s, 0.209965)
        self.assertAlmostEqual(lower.x, 0.20837)
        self.assertAlmostEqual(lower.y, -0.01446)
        self.assertAlmostEqual(lower.u_e_rel, 1.04462)
        self.assertAlmostEqual(lower.delta_d, 0.000000)
        self.assertAlmostEqual(lower.delta_m, 0.000000)
        self.assertAlmostEqual(lower.c_f, 0.000000)
        self.assertAlmostEqual(lower.shape_d, 1.0000)
        self.assertAlmostEqual(lower.shape_k, 2.0000)
        self.assertAlmostEqual(lower.mass_defect, 0.00000)
        self.assertAlmostEqual(lower.mom_defect, 0.00000)
        self.assertAlmostEqual(lower.ke_defect, 0.00000)

    def test_case_viscous(self) -> None:
        """Test importing a viscous case."""
        airfoil_name = "NACA 0003"
        alpha = np.deg2rad(0)  # (rad)
        c = 1  # (m)
        re = 1000
        x_trans = 1  # (m)
        n_trans = 9

        # Read a dump file from viscous solution
        directory = dirname(abspath(__file__))
        filename = directory + "/data/xfoil_viscous_dump.txt"
        xf = XFoilReader(filename)
        xf.name = airfoil_name
        xf.alpha = alpha
        xf.c = c
        xf.reynolds = re
        xf.x_trans_lower = x_trans
        xf.x_trans_upper = x_trans
        xf.n_trans = n_trans

        # test case info
        self.assertEqual(xf.name, airfoil_name)
        self.assertEqual(xf.alpha, alpha)
        self.assertEqual(xf.c, c)
        self.assertEqual(xf.reynolds, re)
        self.assertEqual(xf.x_trans_upper, x_trans)
        self.assertEqual(xf.x_trans_lower, x_trans)
        self.assertEqual(xf.n_trans, n_trans)
        self.assertEqual(xf.upper_count(), 12)
        self.assertEqual(xf.lower_count(), 12)
        self.assertEqual(xf.wake_count(), 8)

    def test_viscous_upper(self) -> None:
        """Test getting upper airfoil info from viscous case."""
        directory = dirname(abspath(__file__))
        filename = directory + "/data/xfoil_viscous_dump.txt"
        xf = XFoilReader(filename)

        buff = [0.0, 0.000695, 0.092695, 0.196925, 0.301285, 0.392635,
                0.497055, 0.601495, 0.692885, 0.797335, 0.901785, 1.001785]
        self.assertIsNone(npt.assert_allclose(buff, xf.s_upper()))
        buff = [0.0, 0.00024, 0.09114, 0.19533, 0.29968, 0.39103, 0.49545,
                0.59987, 0.69123, 0.79565, 0.90005, 1.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.x_upper()))
        buff = [0.0, 0.00069, 0.01132, 0.01427, 0.015, 0.01459, 0.01331,
                0.01141, 0.00937, 0.00668, 0.00362, 0.00031]
        self.assertIsNone(npt.assert_allclose(buff, xf.y_upper()))
        buff = [0.0, 0.4447, 1.06059, 1.05732, 1.05267, 1.0485, 1.04388,
                1.03951, 1.03593, 1.03207, 1.02842, 1.02513]
        self.assertIsNone(npt.assert_allclose(buff,xf.u_e_upper()))
        buff = [0.000811, 0.000811, 0.015435, 0.023266, 0.029346, 0.033966,
                0.038734, 0.043102, 0.04667, 0.050523, 0.054191, 0.05753]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_d_upper()))
        buff = [0.000364, 0.000364, 0.006013, 0.009, 0.011294, 0.013019,
                0.014787, 0.016399, 0.017711, 0.019121, 0.020455, 0.021671]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_m_upper()))
        buff = [0.0005900804, 0.0005900804, 0.0094879127, 0.0141849,
                0.0177857912, 0.0204906041, 0.0232584723, 0.0257808679,
                0.0278328365, 0.0300352668, 0.032118441, 0.0339346189]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_k_upper()))
        buff = [2.2295, 2.2295, 2.5668, 2.585, 2.5984, 2.6089, 2.6194, 2.6284,
                2.6352, 2.6423, 2.6493, 2.6548]
        self.assertIsNone(npt.assert_allclose(buff, xf.shape_d_upper()))
        buff = [1.6211, 1.6211, 1.5779, 1.5761, 1.5748, 1.5739, 1.5729,
                1.5721, 1.5715, 1.5708, 1.5702, 1.5659]
        self.assertIsNone(npt.assert_allclose(buff, xf.shape_k_upper()))
        buff = [0.8771445, 0.877145, 0.078378, 0.050783, 0.039477, 0.033566,
                0.02895, 0.025638, 0.023409, 0.021365, 0.019684, 0.018362]
        self.assertIsNone(npt.assert_allclose(buff, xf.c_f_upper()))
        buff = [0.00036, 0.00036, 0.01637, 0.0246, 0.03089, 0.03561, 0.04043,
                0.04481, 0.04835, 0.05214, 0.05573, 0.05898]
        self.assertIsNone(npt.assert_allclose(buff, xf.mass_defect_upper()))
        buff = [7e-05, 7e-05, 0.00676, 0.01006, 0.01251, 0.01431, 0.01611,
                0.01772, 0.01901, 0.02037, 0.02163, 0.02277]
        self.assertIsNone(npt.assert_allclose(buff, xf.mom_defect_upper()))
        buff = [5e-05, 5e-05, 0.01132, 0.01677, 0.02075, 0.02362, 0.02646,
                0.02896, 0.03094, 0.03302, 0.03494, 0.03656]
        self.assertIsNone(npt.assert_allclose(buff, xf.ke_defect_upper()))

    def test_viscous_lower(self) -> None:
        """Test getting lower airfoil info from viscous case."""
        directory = dirname(abspath(__file__))
        filename = directory + "/data/xfoil_viscous_dump.txt"
        xf = XFoilReader(filename)

        buff = [0, 0.000695, 0.105705, 0.209965, 0.301285, 0.405685, 0.510105,
                0.601485, 0.705935, 0.810385, 0.901775, 1.001775]
        self.assertIsNone(npt.assert_allclose(buff, xf.s_lower()))
        buff = [0.0, 0.00024, 0.10414, 0.20837, 0.29968, 0.40408, 0.5085,
                0.59987, 0.70429, 0.8087, 0.90005, 1.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.x_lower()))
        buff = [0.0, -0.00069, -0.01188, -0.01446, -0.015, -0.01447, -0.0131,
                -0.01141, -0.00906, -0.00632, -0.00362, -0.00031]
        self.assertIsNone(npt.assert_allclose(buff, xf.y_lower()))
        buff = [0.0, 0.4447, 1.06044, 1.05676, 1.05267, 1.04791, 1.04332,
                1.03951, 1.03543, 1.03161, 1.02842, 1.02513]
        self.assertIsNone(npt.assert_allclose(buff, xf.u_e_lower()))
        buff = [0.000811, 0.000811, 0.016586, 0.024091, 0.029346, 0.034589,
                0.0393, 0.043102, 0.047164, 0.05099, 0.054191, 0.05753]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_d_lower()))
        buff = [0.000364, 0.000364, 0.006454, 0.009313, 0.011294, 0.013251,
                0.014996, 0.016399, 0.017892, 0.019292, 0.020455, 0.021671]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_m_lower()))
        buff = [0.0005900804, 0.0005900804, 0.0101818304, 0.014677288,
                0.0177857912, 0.0208530987, 0.0235857088, 0.0257808679,
                0.0281154888, 0.0303038736, 0.032118441, 0.0339346189]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_k_lower()))
        buff = [2.2295, 2.2295, 2.5699, 2.5868, 2.5984, 2.6103, 2.6206,
                2.6284, 2.6361, 2.6431, 2.6493, 2.6548]
        self.assertIsNone(npt.assert_allclose(buff, xf.shape_d_lower()))
        buff = [1.6211, 1.6211, 1.5776, 1.576, 1.5748, 1.5737, 1.5728, 1.5721,
                1.5714, 1.5708, 1.5702, 1.5659]
        self.assertIsNone(npt.assert_allclose(buff, xf.shape_k_lower()))
        buff = [0.8771445, 0.877144, 0.072669, 0.048918, 0.039477, 0.03289,
                0.028478, 0.025638, 0.023129, 0.021137, 0.019684, 0.018362]
        self.assertIsNone(npt.assert_allclose(buff, xf.c_f_lower()))
        buff = [0.00036, 0.00036, 0.01759, 0.02546, 0.03089, 0.03625, 0.041,
                0.04481, 0.04884, 0.0526, 0.05573, 0.05898]
        self.assertIsNone(npt.assert_allclose(buff, xf.mass_defect_lower()))
        buff = [7e-05, 7e-05, 0.00726, 0.0104, 0.01251, 0.01455, 0.01632,
                0.01772, 0.01918, 0.02053, 0.02163, 0.02277]
        self.assertIsNone(npt.assert_allclose(buff, xf.mom_defect_lower()))
        buff = [5e-05, 5e-05, 0.01214, 0.01732, 0.02075, 0.024, 0.02679,
                0.02896, 0.03121, 0.03327, 0.03494, 0.03656]
        self.assertIsNone(npt.assert_allclose(buff, xf.ke_defect_lower()))

    def test_viscous_wake(self) -> None:
        """Test getting airfoil wake info from viscous case."""
        directory = dirname(abspath(__file__))
        filename = directory + "/data/xfoil_viscous_dump.txt"
        xf = XFoilReader(filename)

        buff = [0.0, 0.11471, 0.19813, 0.32026, 0.43176, 0.49908, 0.76089,
                0.9999]
        self.assertIsNone(npt.assert_allclose(buff, xf.s_wake()))
        buff = [1.0001, 1.11481, 1.19822, 1.32036, 1.43186, 1.49918, 1.76099,
                2.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.x_wake()))
        buff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(npt.assert_allclose(buff, xf.y_wake()))
        buff = [1.02513, 1.0147, 1.0081, 0.99912, 0.99122, 0.98638, 0.96382,
                0.92209]
        self.assertIsNone(npt.assert_allclose(buff, xf.u_e_wake()))
        buff = [0.11569, 0.115166, 0.114388, 0.113283, 0.112767, 0.11289,
                0.120079, 0.173472]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_d_wake()))
        buff = [0.043341, 0.04543, 0.046781, 0.048653, 0.050336, 0.051386,
                0.056558, 0.068368]
        self.assertIsNone(npt.assert_allclose(buff, xf.delta_m_wake()))
        buff = [2.6693, 2.535, 2.4452, 2.3284, 2.2403, 2.1969, 2.1231, 2.5373]
        self.assertIsNone(npt.assert_allclose(buff, xf.shape_d_wake()))

    def test_viscous_element_access(self) -> None:
        """Test accessing viscous properties at specific point."""
        # Read a dump file from viscous solution
        directory = dirname(abspath(__file__))
        filename = directory + "/data/xfoil_viscous_dump.txt"
        xf = XFoilReader(filename)

        # test getting upper, lower, and wake data
        upper = xf.upper(2)
        self.assertAlmostEqual(upper.s, 0.092695)
        self.assertAlmostEqual(upper.x, 0.09114)
        self.assertAlmostEqual(upper.y, 0.01132)
        self.assertAlmostEqual(upper.u_e_rel, 1.06059)
        self.assertAlmostEqual(upper.delta_d, 0.015435)
        self.assertAlmostEqual(upper.delta_m, 0.006013)
        self.assertAlmostEqual(upper.c_f, 0.078378)
        self.assertAlmostEqual(upper.shape_d, 2.5668)
        self.assertAlmostEqual(upper.shape_k, 1.5779)
        self.assertAlmostEqual(upper.mass_defect, 0.01637)
        self.assertAlmostEqual(upper.mom_defect, 0.00676)
        self.assertAlmostEqual(upper.ke_defect, 0.01132)

        lower = xf.lower(3)
        self.assertAlmostEqual(lower.s, 0.209965)
        self.assertAlmostEqual(lower.x, 0.20837)
        self.assertAlmostEqual(lower.y, -0.01446)
        self.assertAlmostEqual(lower.u_e_rel, 1.05676)
        self.assertAlmostEqual(lower.delta_d, 0.024091)
        self.assertAlmostEqual(lower.delta_m, 0.009313)
        self.assertAlmostEqual(lower.c_f, 0.048918)
        self.assertAlmostEqual(lower.shape_d, 2.5868)
        self.assertAlmostEqual(lower.shape_k, 1.5760)
        self.assertAlmostEqual(lower.mass_defect, 0.02546)
        self.assertAlmostEqual(lower.mom_defect, 0.01040)
        self.assertAlmostEqual(lower.ke_defect, 0.01732)

        wake = xf.wake(5)
        self.assertAlmostEqual(wake.s, 0.49908)
        self.assertAlmostEqual(wake.x, 1.49918)
        self.assertAlmostEqual(wake.y, 0.00000)
        self.assertAlmostEqual(wake.u_e_rel, 0.98638)
        self.assertAlmostEqual(wake.delta_d, 0.112890)
        self.assertAlmostEqual(wake.delta_m, 0.051386)
        self.assertAlmostEqual(wake.shape_d, 2.1969)

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
    _ = unittest.main(verbosity=1)

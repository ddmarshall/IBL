"""Module to test the Stanford Olympics functionality."""

# pyright: reportPrivateUsage=false
# pylint: disable=protected-access

import unittest
import numpy as np
import numpy.testing as np_test

from ibl.reference import StanfordOlympics1968
from ibl.reference import StanfordOlympics1968StationData
from ibl.reference import StanfordOlympics1968SmoothVel


class TestStanfordOlympics1968(unittest.TestCase):
    """Class to test importing data from the 1968 Stanford Olympics"""

    def test_for_data(self) -> None:
        """Test accessing data files."""
        stol = StanfordOlympics1968()

        self.assertTrue(not stol._get_case_data("xxxx"))
        self.assertFalse(not stol._get_case_data("1100"))

    # pylint: disable=too-many-statements
    def test_station_setters(self) -> None:
        """Test the stations data."""
        summary_data = ("0.782  33.90  -2.300  0.276  1.381  1.778   7.307  "
                        "0.00285  0.00276  0.00271  0.181")
        station_summary = ("0.7820  1.280  33.900  0.155000  0.2760  0.3812  "
                           "0.4908   10.0941  1.3811   7.3070  1.7783   "
                           "6036.6   8337.1")
        station_data = ["   0.0   0.0   0.0    0.0     0.0     26.48",
                        "  41.3  14.43  0.050  0.5450  0.0050  12.05"]
        sd = StanfordOlympics1968StationData(si_unit=True,
                                             summ_data=summary_data,
                                             stat_summ=station_summary,
                                             stat_data=station_data)

        sd.u_e = 30.0
        self.assertAlmostEqual(sd.u_e, 30.0)
        with self.assertRaises(ValueError):
            sd.u_e = 0
        sd.delta_d = 1e-4
        self.assertAlmostEqual(sd.delta_d, 1e-4)
        with self.assertRaises(ValueError):
            sd.delta_d = 0.0
        sd.delta_m = 1e-4
        self.assertAlmostEqual(sd.delta_m, 1e-4)
        with self.assertRaises(ValueError):
            sd.delta_m = 0.0
        sd.delta_k = 1e-4
        self.assertAlmostEqual(sd.delta_k, 1e-4)
        with self.assertRaises(ValueError):
            sd.delta_k = 0.0
        sd.delta_c = 1e-4
        self.assertAlmostEqual(sd.delta_c, 1e-4)
        with self.assertRaises(ValueError):
            sd.delta_c = 0.0
        sd.shape_d = 2.0
        self.assertAlmostEqual(sd.shape_d, 2.0)
        with self.assertRaises(ValueError):
            sd.shape_d = 0.0
        sd.shape_k = 2.0
        self.assertAlmostEqual(sd.shape_k, 2.0)
        with self.assertRaises(ValueError):
            sd.shape_k = 0.0
        sd.shape_eq = 2.0
        self.assertAlmostEqual(sd.shape_eq, 2.0)
        with self.assertRaises(ValueError):
            sd.shape_eq = 0.0
        sd.u_star = 30.0
        self.assertAlmostEqual(sd.u_star, 30.0)
        with self.assertRaises(ValueError):
            sd.u_star = 0
        sd.nu = 1e-5
        self.assertAlmostEqual(sd.nu, 1e-5)
        with self.assertRaises(ValueError):
            sd.nu = 0
        sd.re_delta_d = 300.0
        self.assertAlmostEqual(sd.re_delta_d, 300.0)
        with self.assertRaises(ValueError):
            sd.re_delta_d = 0
        sd.re_delta_m = 300.0
        self.assertAlmostEqual(sd.re_delta_m, 300.0)
        with self.assertRaises(ValueError):
            sd.re_delta_m = 0
        sd.y = np.ones_like(sd.y)
        with self.assertRaises(ValueError):
            sd.y = -np.ones_like(sd.y)
        sd.y_plus = np.ones_like(sd.y_plus)
        with self.assertRaises(ValueError):
            sd.y_plus = -np.ones_like(sd.y_plus)
        sd.y_c = np.ones_like(sd.y_c)
        with self.assertRaises(ValueError):
            sd.y_c = -np.ones_like(sd.y_c)
        sd.u = np.ones_like(sd.u)
        with self.assertRaises(ValueError):
            sd.u = -np.ones_like(sd.u)
        sd.u_plus = np.ones_like(sd.u_plus)
        with self.assertRaises(ValueError):
            sd.u_plus = -np.ones_like(sd.u_plus)
        sd.u_defect = np.ones_like(sd.u_defect)
        with self.assertRaises(ValueError):
            sd.u_defect = -np.ones_like(sd.u_defect)
    # pylint: enable=too-many-statements

    def test_station_data(self) -> None:
        """Test the stations data."""
        summary_data = ("0.782  33.90  -2.300  0.276  1.381  1.778   7.307  "
                        "0.00285  0.00276  0.00271  0.181")
        station_summary = ("0.7820  1.280  33.900  0.155000  0.2760  0.3812  "
                           "0.4908   10.0941  1.3811   7.3070  1.7783   "
                           "6036.6   8337.1")
        station_data = ["   0.0   0.0   0.0    0.0     0.0     26.48",
                        "  41.3  14.43  0.050  0.5450  0.0050  12.05"]
        sd = StanfordOlympics1968StationData(si_unit=True,
                                             summ_data=summary_data,
                                             stat_summ="", stat_data=[])

        # test setting summary data
        row = np.array([float(x) for x in summary_data.split()])
        self.assertAlmostEqual(sd.x, row[0])
        self.assertAlmostEqual(sd.u_e, row[1])
        self.assertAlmostEqual(sd.du_e, row[2])
        self.assertAlmostEqual(sd.delta_m, row[3]*1e-2)
        self.assertAlmostEqual(sd.shape_d, row[4])
        self.assertAlmostEqual(sd.shape_k, row[5])
        self.assertAlmostEqual(sd.shape_eq, row[6])
        self.assertAlmostEqual(sd.c_f, row[7])
        self.assertAlmostEqual(sd.c_f_lt, row[8])
        self.assertAlmostEqual(sd.c_f_exp, row[9])
        self.assertAlmostEqual(sd.beta_eq, row[10])

        # test setting station summary data
        sd.reset(si_unit=True, summ_data=summary_data,
                 stat_summ=station_summary, stat_data=[])
        row = np.array([float(x) for x in station_summary.split()])
        self.assertAlmostEqual(sd.x, row[0])
        self.assertAlmostEqual(sd.u_star, row[1])
        self.assertAlmostEqual(sd.u_e, row[2])
        self.assertAlmostEqual(sd.nu, row[3]*1e-4)
        self.assertAlmostEqual(sd.delta_m, row[4]*1e-2)
        self.assertAlmostEqual(sd.delta_d, row[5]*1e-2)
        self.assertAlmostEqual(sd.delta_k, row[6]*1e-2)
        self.assertAlmostEqual(sd.delta_c, row[7]*1e-2)
        self.assertAlmostEqual(sd.shape_d, row[8])
        self.assertAlmostEqual(sd.shape_eq, row[9])
        self.assertAlmostEqual(sd.shape_k, row[10])
        self.assertAlmostEqual(sd.re_delta_m, row[11])
        self.assertAlmostEqual(sd.re_delta_d, row[12])

        # test setting station data
        sd.reset(si_unit=True, summ_data=summary_data,
                 stat_summ=station_summary, stat_data=station_data)
        rows = np.array([[float(x) for x in station_data[0].split()],
                         [float(x) for x in station_data[1].split()]])
        self.assertIsNone(np_test.assert_allclose(sd.y_plus, rows[:,0]))
        self.assertIsNone(np_test.assert_allclose(sd.u_plus, rows[:,1]))
        self.assertIsNone(np_test.assert_allclose(sd.y, rows[:,2]*1e-2))
        self.assertIsNone(np_test.assert_allclose(sd.u, rows[:,3]*sd.u_e))
        self.assertIsNone(np_test.assert_allclose(sd.y_c, rows[:,4]))
        self.assertIsNone(np_test.assert_allclose(sd.u_defect, rows[:,5]))

    def test_smoothed_velocity(self) -> None:
        """Test the smoothed velocity data."""
        data = ["0.75  33.98  -2.25",
                "1.00  33.38  -2.75",
                "1.25  32.63  -3.29"]
        sv = StanfordOlympics1968SmoothVel(si_unit=True, data=data)

        # test setting data
        rows = np.array([[float(x) for x in data[0].split()],
                         [float(x) for x in data[1].split()],
                         [float(x) for x in data[2].split()]])
        self.assertIsNone(np_test.assert_allclose(sv.x, rows[:, 0]))
        self.assertIsNone(np_test.assert_allclose(sv.u_e, rows[:, 1]))
        self.assertIsNone(np_test.assert_allclose(sv.du_e, rows[:, 2]))

    def test_case_station_data(self) -> None:
        """Test getting station data."""
        so = StanfordOlympics1968("1100")

        # test the case info
        self.assertEqual(so.case, "1100")
        self.assertAlmostEqual(so.nu_ref, 1.55e-5)
        self.assertEqual(so.station_count(), 12)

        # test some station data
        station = so.station(0)
        self.assertAlmostEqual(station.x, 0.782)
        self.assertAlmostEqual(station.u_e, 33.90)
        self.assertAlmostEqual(station.du_e, -2.300)
        self.assertAlmostEqual(station.c_f, 0.00285)
        self.assertAlmostEqual(station.c_f_lt, 0.00276)
        self.assertAlmostEqual(station.c_f_exp, 0.00271)
        self.assertAlmostEqual(station.beta_eq, 0.181)
        self.assertAlmostEqual(station.u_star, 1.280)
        self.assertAlmostEqual(station.nu, 1.55e-5)
        self.assertAlmostEqual(station.delta_m, 0.002760)
        self.assertAlmostEqual(station.delta_d, 3.812e-3)
        self.assertAlmostEqual(station.delta_k, 4.908e-3)
        self.assertAlmostEqual(station.delta_c, 1.00941e-1)
        self.assertAlmostEqual(station.shape_d, 1.3811)
        self.assertAlmostEqual(station.shape_eq, 7.3070)
        self.assertAlmostEqual(station.shape_k, 1.7783)
        self.assertAlmostEqual(station.re_delta_m, 6036.6)
        self.assertAlmostEqual(station.re_delta_d, 8337.1)

        self.assertEqual(station.sample_count(), 14)

        idx = 0
        self.assertAlmostEqual(station.y_plus[idx], 0.0)
        self.assertAlmostEqual(station.u_plus[idx], 0.0)
        self.assertAlmostEqual(station.y[idx], 0.0)
        self.assertAlmostEqual(station.u[idx], 0.0)
        self.assertAlmostEqual(station.y_c[idx], 0.0)
        self.assertAlmostEqual(station.u_defect[idx], 26.48)

        idx = 9
        self.assertAlmostEqual(station.y_plus[idx], 1238.9)
        self.assertAlmostEqual(station.u_plus[idx], 24.11)
        self.assertAlmostEqual(station.y[idx], 0.01500)
        self.assertAlmostEqual(station.u[idx], 0.9105*station.u_e)
        self.assertAlmostEqual(station.y_c[idx], 0.1486)
        self.assertAlmostEqual(station.u_defect[idx], 2.37)

    def test_vector_data(self) -> None:
        """Test getting a vector of data."""
        so = StanfordOlympics1968("1100")

        # test getting summary data
        ref = np.array([0.782, 1.282, 1.782, 2.282, 2.782, 3.132, 3.332,
                        3.532, 3.732, 3.932, 4.132, 4.332])
        self.assertIsNone(np_test.assert_allclose(so.x(), ref))
        ref = np.array([33.9, 32.6, 30.7, 28.6, 27.1, 26.05, 25.75, 24.85,
                        24.5, 24.05, 23.6, 23.1])
        self.assertIsNone(np_test.assert_allclose(so.u_e(), ref))
        ref = np.array([-2.3, -3.35, -4.32, -3.58, -3.0, -2.74, -2.6, -2.5,
                        -2.4, -2.3, -2.25, -2.18])
        self.assertIsNone(np_test.assert_allclose(so.du_e(), ref))
        ref = np.array([0.00285, 0.00249, 0.00221, 0.00205, 0.0018, 0.00168,
                        0.00162, 0.0015, 0.00141, 0.00133, 0.00124, 0.00117])
        self.assertIsNone(np_test.assert_allclose(so.c_f(), ref))
        ref = np.array([0.00276, 0.00246, 0.00222, 0.00202, 0.00181, 0.00167,
                        0.00161, 0.00151, 0.00142, 0.00133, 0.00124, 0.00117])
        self.assertIsNone(np_test.assert_allclose(so.c_f_lt(), ref))
        ref = np.array([0.00271, 0.00249, 0.00219, 0.00202, 0.00186, 0.00171,
                        0.0016, 0.00153, 0.00152, 0.00134, 0.00126, 0.00117])
        self.assertIsNone(np_test.assert_allclose(so.c_f_exp(), ref))
        ref = np.array([0.181, 0.475, 1.083, 1.415, 1.919, 2.379, 2.672,
                        3.282, 3.806, 4.53, 5.499, 6.615])
        self.assertIsNone(np_test.assert_allclose(so.beta_eq(), ref))

        # test getting station summary
        ref = np.array([1.28, 1.15, 1.02, 0.915, 0.814, 0.754, 0.732, 0.681,
                        0.65, 0.619, 0.588, 0.558])
        self.assertIsNone(np_test.assert_allclose(so.u_star(), ref))
        ref = np.array([0.155, 0.155, 0.155, 0.155, 0.155, 0.155, 0.155,
                        0.155, 0.155, 0.155, 0.155, 0.155])*1e-4
        self.assertIsNone(np_test.assert_allclose(so.nu(), ref))
        ref = np.array([0.276, 0.4128, 0.6059, 0.8108, 1.0736, 1.2761, 1.4325,
                        1.6139, 1.7733, 2.0046, 2.2458, 2.5276])*1e-2
        self.assertIsNone(np_test.assert_allclose(so.delta_m(), ref))
        ref = np.array([0.3812, 0.5755, 0.8496, 1.157, 1.5645, 1.8947, 2.1379,
                        2.4504, 2.7349, 3.1386, 3.58, 4.0894])*1e-2
        self.assertIsNone(np_test.assert_allclose(so.delta_d(), ref))
        ref = np.array([0.4908, 0.7265, 1.0619, 1.4064, 1.8464, 2.1774,
                        2.4397, 2.7316, 2.9855, 3.3585, 3.738, 4.1868])*1e-2
        self.assertIsNone(np_test.assert_allclose(so.delta_k(), ref))
        ref = np.array([10.0941, 16.314, 25.5722, 36.1677, 52.0837, 65.4626,
                        75.2101, 89.4145, 103.0786, 121.9346, 143.7026,
                        169.303])*1e-2
        self.assertIsNone(np_test.assert_allclose(so.delta_c(), ref))
        ref = np.array([1.3811, 1.3941, 1.4022, 1.4269, 1.4572, 1.4848,
                        1.4924, 1.5186, 1.5423, 1.5657, 1.5941, 1.6179])
        self.assertIsNone(np_test.assert_allclose(so.shape_d(), ref))
        ref = np.array([1.7783, 1.76, 1.7525, 1.7345, 1.7198, 1.7063, 1.7031,
                        1.6929, 1.6837, 1.6754, 1.6644, 1.6564])
        self.assertIsNone(np_test.assert_allclose(so.shape_k(), ref))
        ref = np.array([7.307, 8.0144, 8.633, 9.353, 10.4452, 11.2802,
                        11.6073, 12.4609, 13.2526, 14.0373, 14.9599, 15.811])
        self.assertIsNone(np_test.assert_allclose(so.shape_eq(), ref))
        ref = np.array([6036.6, 8681.5, 12000.9, 14961, 18771.5, 21447,
                        23797.9, 25869.7, 28028.9, 31102.9, 34193.8, 37670.1])
        self.assertIsNone(np_test.assert_allclose(so.re_delta_m(), ref))
        ref = np.array([8337.1, 12103, 16827.1, 21348.4, 27354.3, 31843.5,
                        35516.1, 39285.2, 43229.1, 48698.9, 54508.9, 60945.9])
        self.assertIsNone(np_test.assert_allclose(so.re_delta_d(), ref))

        # test getting station data
        ref = np.array([0, 32.9, 65.8, 131.6, 263.2, 394.8, 526.4, 658, 789.6,
                        987, 1184.4, 1316, 1645.1, 1974.1, 2632.1, 3948.1])
        self.assertIsNone(np_test.assert_allclose(so.y_plus(2), ref))
        ref = np.array([0, 13.91, 15.35, 16.92, 18.6, 19.67, 20.59, 21.28,
                        22.02, 22.97, 23.9, 24.38, 25.81, 27.04, 29.05, 30.1])
        self.assertIsNone(np_test.assert_allclose(so.u_plus(2), ref))
        ref = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2,
                        2.5, 3, 4, 6])*1e-2
        self.assertIsNone(np_test.assert_allclose(so.y(2), ref))
        ref = np.array([0, 0.462, 0.51, 0.562, 0.618, 0.6535, 0.684, 0.707,
                        0.7315, 0.763, 0.794, 0.81, 0.8575, 0.8985, 0.965,
                        1])*so.u_e()[2]
        self.assertIsNone(np_test.assert_allclose(so.u(2), ref))
        ref = np.array([0, 0.002, 0.0039, 0.0078, 0.0156, 0.0235, 0.0313,
                        0.0391, 0.0469, 0.0587, 0.0704, 0.0782, 0.0978,
                        0.1173, 0.1564, 0.2346])
        self.assertIsNone(np_test.assert_allclose(so.y_c(2), ref))
        ref = np.array([30.1, 16.19, 14.75, 13.18, 11.5, 10.43, 9.51, 8.82,
                        8.08, 7.13, 6.2, 5.72, 4.29, 3.06, 1.05, 0])
        self.assertIsNone(np_test.assert_allclose(so.u_defect(2), ref))

    def test_smooth_velocity_data(self) -> None:
        """Test getting smooth velocity data."""
        so = StanfordOlympics1968()

        # test case when no smooth data exists
        self.assertEqual(so.x_smooth().size, 0)
        self.assertEqual(so.u_e_smooth().size, 0)
        self.assertEqual(so.du_e_smooth().size, 0)

        # test some smooth data
        so.case = "1100"
        self.assertIsNone(np_test.assert_allclose(so.x_smooth()[0], 0.75))
        self.assertIsNone(np_test.assert_allclose(so.u_e_smooth()[0], 33.98))
        self.assertIsNone(np_test.assert_allclose(so.du_e_smooth()[0], -2.25))
        self.assertIsNone(np_test.assert_allclose(so.x_smooth()[6], 2.25))
        self.assertIsNone(np_test.assert_allclose(so.u_e_smooth()[6], 28.78))
        self.assertIsNone(np_test.assert_allclose(so.du_e_smooth()[6], -3.62))
        self.assertIsNone(np_test.assert_allclose(so.x_smooth()[13], 4.00))
        self.assertIsNone(np_test.assert_allclose(so.u_e_smooth()[13], 23.85))
        self.assertIsNone(np_test.assert_allclose(so.du_e_smooth()[13], -2.29))

    # def test_changing_cases(self):
    #     """Test switching from valid case."""
    #     # switch from valid case to valid case
    #     # switch from valid case to invalid case

    def test_data_consistency(self) -> None:
        """Test that the supported files pass consistency checks."""
        raised = False

        # loading case will raise an exception if data doesn't match
        try:
            _ = StanfordOlympics1968("1100")
        except ValueError as e:
            print(f"{e} for case 1100")
            raised = True
        self.assertFalse(raised)
        try:
            _ = StanfordOlympics1968("1200")
        except ValueError as e:
            print(f"{e} for case 1200")
            raised = True
        self.assertFalse(raised)
        try:
            _ = StanfordOlympics1968("1300")
        except ValueError as e:
            print(f"{e} for case 1300")
            raised = True
        self.assertFalse(raised)
        try:
            _ = StanfordOlympics1968("2200")
        except ValueError as e:
            print(f"{e} for case 2200")
            raised = True
        self.assertFalse(raised)
        try:
            _ = StanfordOlympics1968("2300")
        except ValueError as e:
            print(f"{e} for case 2300")
            raised = True
        self.assertFalse(raised)


if __name__ == "__main__":
    _ = unittest.main(verbosity=1)

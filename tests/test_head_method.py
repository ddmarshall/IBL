"""Module to test the Head method functionality."""

# pyright: reportPrivateUsage=false
# pylint: disable=protected-access

import unittest
import numpy as np
import numpy.testing as np_test

from ibl.head_method import HeadMethod
from ibl.typing import InputParam


class TestHeadMethod(unittest.TestCase):
    """Class to test the implementation of the Head method"""

    def test_setters(self) -> None:
        """Test setting parameters."""
        hm = HeadMethod()
        hm.initial_delta_m = 0.5
        self.assertEqual(hm.initial_delta_m, 0.5)
        hm.initial_shape_d = 2.0
        self.assertEqual(hm.initial_shape_d, 2.0)

        with self.assertRaises(ValueError):
            hm.initial_delta_m = 0
        with self.assertRaises(ValueError):
            hm.initial_shape_d = 0
        with self.assertRaises(ValueError):
            _ = hm.v_e(1.0)
        with self.assertRaises(ValueError):
            _ = hm.delta_m(1.0)
        with self.assertRaises(ValueError):
            _ = hm.shape_d(1.0)
        with self.assertRaises(ValueError):
            _ = hm.tau_w(1.0, 1.0)

    def test_shape_entrainment_calculation(self) -> None:
        """Test the entrainment shape calculation."""
        eps = 1e-9

        # test that H1 is continuous over H_d=1.6
        shape_d_break = 1.6
        shape_d_low = shape_d_break-eps
        shape_d_high = shape_d_break+eps
        shape_entrainment_low = HeadMethod._shape_entrainment(shape_d_low)
        shape_entrainment_high = HeadMethod._shape_entrainment(shape_d_high)
        self.assertIsNone(np_test.assert_allclose(shape_entrainment_low,
                                                  shape_entrainment_high))

        # test that H_d is continuous of H1~3.6374
        shape_entrainment_break = HeadMethod._shape_entrainment(shape_d_break)
        shape_entrainment_low = shape_entrainment_break-eps
        shape_entrainment_high = shape_entrainment_break+eps
        shape_d_low = float(HeadMethod._shape_d(shape_entrainment_low))
        shape_d_high = float(HeadMethod._shape_d(shape_entrainment_high))
        self.assertIsNone(np_test.assert_allclose(shape_d_low, shape_d_high))

        # test H1 for a range of H_d
        def shape_entrainment_fun(shape_d: float) -> float:
            if shape_d <= 1.6:
                return 3.3 + 0.8234/(shape_d - 1.1)**1.287
            return 3.32254659218600974 + 1.5501/(shape_d - 0.6778)**3.064
        shape_d = np.linspace(1.11, 2.4, 101)
        it = np.nditer([shape_d,
                        None  # type: ignore [arg-type]
                        ])
        with it:
            for sd, se in it:
                se[...] = shape_entrainment_fun(float(sd))
            shape_entrainment_ref = it.operands[1]
        shape_entrainment = HeadMethod._shape_entrainment(shape_d)
        self.assertIsNone(np_test.assert_allclose(shape_entrainment,
                                                  shape_entrainment_ref))

        # test H_d can be recoverd from H1 function
        shape_d_ref = shape_d
        shape_d = HeadMethod._shape_d(shape_entrainment)
        self.assertIsNone(np_test.assert_allclose(shape_d, shape_d_ref))

        # test for invalid values
        shape_entrainment = HeadMethod._shape_entrainment(1.1)
        shape_entrainment_ref = HeadMethod._shape_entrainment(1.1001)
        self.assertIsNone(np_test.assert_allclose(shape_entrainment,
                                                  shape_entrainment_ref))
        ref = HeadMethod._shape_d(3.3)
        self.assertIsNone(np_test.assert_allclose(ref,
                                                  HeadMethod._shape_d(3.323)))

    def test_entrainment_velocity_calculation(self) -> None:
        """Test the entrainment velocity calculations."""
        # test calculation of term
        def fun(shape_entrainment: InputParam) -> InputParam:
            return 0.0306/(shape_entrainment-3)**0.6169

        shape_entrainment = np.linspace(3.01, 5, 101)
        e_term_ref = fun(shape_entrainment)
        e_term = HeadMethod._entrainment_velocity(shape_entrainment)
        self.assertIsNone(np_test.assert_allclose(e_term, e_term_ref))

        # test invalid values
        e_term = HeadMethod._entrainment_velocity(3)
        e_term_ref = HeadMethod._entrainment_velocity(3.001)
        self.assertIsNone(np_test.assert_allclose(e_term, e_term_ref))

    def test_sample_calculations(self) -> None:
        """Test sample calculations."""
        x = np.linspace(0.8, 3.0, 20)
        nu = 1e-5
        rho = 1.0

        def u_e_fun(x: InputParam) -> InputParam:
            return 0.5*(x-0.8)**2 + 3.5*(x-0.8) + 11.5  # accelerating flow

        def du_e_fun(x: InputParam) -> InputParam:
            return 0.5*(x-0.8) + 3.5  # accelerating flow

        def d2u_e_fun(x: InputParam) -> InputParam:
            _ = x  # avoid unused variable warning
            return 0.5  # accelerating flow

        hm = HeadMethod(nu=nu, U_e=u_e_fun, dU_edx=du_e_fun,
                        d2U_edx2=d2u_e_fun)
        hm.initial_delta_m = 0.0014
        hm.initial_shape_d = 1.42
        rtn = hm.solve(x0=x[0], x_end=x[-1])
        self.assertTrue(rtn.success)

        # # print out reference values
        # print("u_e_ref =", hm.u_e(x))
        # print("v_e_ref =", hm.v_e(x))
        # print("delta_d_ref =", hm.delta_d(x))
        # print("delta_m_ref =", hm.delta_m(x))
        # print("delta_k_ref =", hm.delta_k(x))
        # print("shape_d_ref =", hm.shape_d(x))
        # print("shape_k_ref =", hm.shape_k(x))
        # print("tau_w_ref =", hm.tau_w(x, rho))
        # print("dissipation_ref =", hm.dissipation(x, rho))

        # reference data
        u_e_ref = [11.5,        11.91196676, 12.33734072, 12.77612188,
                   13.22831025, 13.69390582, 14.17290859, 14.66531856,
                   15.17113573, 15.69036011, 16.22299169, 16.76903047,
                   17.32847645, 17.90132964, 18.48759003, 19.08725762,
                   19.70033241, 20.32681440, 20.96670360, 21.62]
        v_e_ref = [0.00728674, 0.01083954, 0.01249412, 0.01339932,
                   0.01396050, 0.01435180, 0.01465730, 0.01492076,
                   0.01516601, 0.01540642, 0.01564952, 0.01589951,
                   0.01615867, 0.01642807, 0.01670814, 0.01699885,
                   0.01729995, 0.01761106, 0.01793173, 0.01826148]
        delta_d_ref = [0.00198800, 0.00201035, 0.00205305, 0.00210297,
                       0.00215477, 0.00220609, 0.00225588, 0.00230370,
                       0.00234942, 0.00239310, 0.00243483, 0.00247477,
                       0.00251307, 0.00254989, 0.00258536, 0.00261962,
                       0.00265280, 0.00268499, 0.00271630, 0.00274680]
        delta_m_ref = [0.0014,     0.00144945, 0.00150204, 0.00155435,
                       0.00160495, 0.00165328, 0.00169917, 0.00174267,
                       0.00178392, 0.00182309, 0.00186037, 0.00189593,
                       0.00192996, 0.00196260, 0.00199400, 0.00202428,
                       0.00205355, 0.00208192, 0.00210948, 0.00213629]
        delta_k_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        shape_d_ref = [1.42,       1.38698027, 1.36684448, 1.35295507,
                       1.34257558, 1.33437679, 1.32763781, 1.32193439,
                       1.31700027, 1.31265931, 1.30878965, 1.30530373,
                       1.30213659, 1.29923860, 1.29657098, 1.29410276,
                       1.29180876, 1.28966821, 1.28766376, 1.28578071]
        shape_k_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tau_w_ref = [0.24496919, 0.27160252, 0.29500543, 0.31735812,
                     0.33964136, 0.36236513, 0.38582495, 0.41020865,
                     0.43564600, 0.46223385, 0.49004963, 0.51915895,
                     0.54962008, 0.58148666, 0.61480943, 0.64963728,
                     0.68601793, 0.7239985, 0.76362573, 0.80494622]
        dissipation_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.assertIsNone(np_test.assert_allclose(u_e_ref, hm.u_e(x),
                                                  atol=1e-7))
        self.assertIsNone(np_test.assert_allclose(v_e_ref, hm.v_e(x),
                                                  atol=1e-7))
        self.assertIsNone(np_test.assert_allclose(delta_d_ref, hm.delta_d(x),
                                                  atol=1e-7))
        self.assertIsNone(np_test.assert_allclose(delta_m_ref, hm.delta_m(x),
                                                  atol=1e-7))
        self.assertIsNone(np_test.assert_allclose(delta_k_ref, hm.delta_k(x)))
        self.assertIsNone(np_test.assert_allclose(shape_d_ref, hm.shape_d(x)))
        self.assertIsNone(np_test.assert_allclose(shape_k_ref, hm.shape_k(x)))
        self.assertIsNone(np_test.assert_allclose(tau_w_ref, hm.tau_w(x, rho)))
        self.assertIsNone(np_test.assert_allclose(dissipation_ref,
                                                  hm.dissipation(x, rho)))


if __name__ == "__main__":
    _ = unittest.main(verbosity=1)

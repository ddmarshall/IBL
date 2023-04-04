#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:51:51 2022

@author: ddmarshall
"""

import unittest
import numpy as np
import numpy.testing as npt

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

    def test_shape_entrainment_calculation(self) -> None:
        """Test the entrainment shape calculation."""
        eps = 1e-9

        # test that H1 is continuous over H_d=1.6
        shape_d_break = 1.6
        shape_d_low = shape_d_break-eps
        shape_d_high = shape_d_break+eps
        shape_entrainment_low = HeadMethod.shape_entrainment(shape_d_low)
        shape_entrainment_high = HeadMethod.shape_entrainment(shape_d_high)
        self.assertIsNone(npt.assert_allclose(shape_entrainment_low,
                                              shape_entrainment_high))

        # test that H_d is continuous of H1~3.6374
        shape_entrainment_break = HeadMethod.shape_entrainment(shape_d_break)
        shape_entrainment_low = shape_entrainment_break-eps
        shape_entrainment_high = shape_entrainment_break+eps
        # pylint: disable-next=protected-access
        shape_d_low = float(HeadMethod._shape_d(shape_entrainment_low))
        # pylint: disable-next=protected-access
        shape_d_high = float(HeadMethod._shape_d(shape_entrainment_high))
        self.assertIsNone(npt.assert_allclose(shape_d_low, shape_d_high))

        # test H1 for a range of H_d
        def shape_entrainment_fun(shape_d: float) -> float:
            if shape_d <= 1.6:
                return 3.3 + 0.8234/(shape_d - 1.1)**1.287
            return 3.32254659218600974 + 1.5501/(shape_d - 0.6778)**3.064
        shape_d = np.linspace(1.11, 2.4, 101)
        it = np.nditer([shape_d, None])  # type: ignore [arg-type]
        with it:
            for sd, se in it:
                se[...] = shape_entrainment_fun(float(sd))
            shape_entrainment_ref = it.operands[1]
        shape_entrainment = HeadMethod.shape_entrainment(shape_d)
        self.assertIsNone(npt.assert_allclose(shape_entrainment,
                                              shape_entrainment_ref))

        # test H_d can be recoverd from H1 function
        shape_d_ref = shape_d
        # pylint: disable-next=protected-access
        shape_d = HeadMethod._shape_d(shape_entrainment)
        self.assertIsNone(npt.assert_allclose(shape_d, shape_d_ref))

        # test for invalid values
        shape_entrainment = HeadMethod.shape_entrainment(1.1)
        shape_entrainment_ref = HeadMethod.shape_entrainment(1.1001)
        self.assertIsNone(npt.assert_allclose(shape_entrainment,
                                              shape_entrainment_ref))
        # pylint: disable-next=protected-access
        ref = HeadMethod._shape_d(3.3)
        # pylint: disable-next=protected-access
        self.assertIsNone(npt.assert_allclose(ref, HeadMethod._shape_d(3.323)))

    def test_entrainment_velocity_calculation(self) -> None:
        """Test the entrainment velocity calculations."""
        # test calculation of term
        def fun(shape_entrainment: InputParam) -> InputParam:
            return 0.0306/(shape_entrainment-3)**0.6169

        # pylint: disable-next=protected-access
        shape_entrainment = np.linspace(3.01, 5, 101)
        e_term_ref = fun(shape_entrainment)
        # pylint: disable-next=protected-access
        e_term = HeadMethod._entrainment_velocity(shape_entrainment)
        self.assertIsNone(npt.assert_allclose(e_term, e_term_ref))

        # test invalid values
        # pylint: disable-next=protected-access
        e_term = HeadMethod._entrainment_velocity(3)
        # pylint: disable-next=protected-access
        e_term_ref = HeadMethod._entrainment_velocity(3.001)
        self.assertIsNone(npt.assert_allclose(e_term, e_term_ref))


if __name__ == "__main__":
    unittest.main(verbosity=1)

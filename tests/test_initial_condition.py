#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 22:22:24 2022

@author: ddmarshall
"""


import unittest
import numpy as np

from ibl.initial_condition import FalknerSkanStagCondition
from ibl.initial_condition import ManualCondition


class TestInitialConditions(unittest.TestCase):
    """Class to test setting initial conditions for IBL solvers"""

    def test_setters(self) -> None:
        """Test initial condition setters."""
        ic = FalknerSkanStagCondition(du_e=0.0, nu=1e-5)

        self.assertEqual(ic.du_e, 0.0)
        self.assertEqual(ic.nu, 1e-5)
        with self.assertRaises(ValueError):
            ic.nu = 0

    def test_falkner_skan_stagnation_condition(self) -> None:
        """Test the Falkner-Skan stagnation point condition."""
        du_e = 10
        nu = 1.2e-5
        eta_m = 0.29235
        shape_d = 2.2162
        shape_k = 1.6257
        fpp0 = 1.23259
        delta_m = np.sqrt(nu*eta_m*fpp0/((shape_d+2)*du_e))
        delta_d = shape_d*delta_m
        delta_k = shape_k*delta_m

        sc = FalknerSkanStagCondition(du_e=du_e, nu=nu)

        self.assertEqual(sc.shape_d(), shape_d)
        self.assertEqual(sc.shape_k(), shape_k)
        self.assertEqual(sc.delta_d(), delta_d)
        self.assertEqual(sc.delta_m(), delta_m)
        self.assertEqual(sc.delta_k(), delta_k)

    def test_manual_condition(self) -> None:
        """Test the manual setting of conditions."""
        delta_d = 2
        delta_m = 4
        delta_k = 3
        shape_d = delta_d/delta_m
        shape_k = delta_k/delta_m

        sc = ManualCondition(delta_d=delta_d, delta_m=delta_m, delta_k=delta_k)

        self.assertEqual(sc.shape_d(), shape_d)
        self.assertEqual(sc.shape_k(), shape_k)
        self.assertEqual(sc.delta_d(), delta_d)
        self.assertEqual(sc.delta_m(), delta_m)
        self.assertEqual(sc.delta_k(), delta_k)


if __name__ == "__main__":
    unittest.main(verbosity=1)

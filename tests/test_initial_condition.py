#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 22:22:24 2022

@author: ddmarshall
"""


import unittest
import numpy as np

from pyBL.initial_condition import FalknerSkanStagnationCondition
from pyBL.initial_condition import ManualCondition


class TestInitialConditions(unittest.TestCase):
    """Class to test setting initial conditions for IBL solvers"""

    def test_falkner_skan_stagnation_condition(self):
        """Test the Falkner-Skan stagnation point condition."""
        dU_edx = 10
        nu = 1.2e-5
        eta_m = 0.29235
        H_d = 2.2162
        H_k = 1.6257
        fpp0 = 1.23259
        delta_m = np.sqrt(nu*eta_m*fpp0/((H_d+2)*dU_edx))
        delta_d = H_d*delta_m
        delta_k = H_k*delta_m

        sc = FalknerSkanStagnationCondition(dU_edx=dU_edx, nu=nu)

        self.assertEqual(sc.H_d(), H_d)
        self.assertEqual(sc.H_k(), H_k)
        self.assertEqual(sc.delta_d(), delta_d)
        self.assertEqual(sc.delta_m(), delta_m)
        self.assertEqual(sc.delta_k(), delta_k)

    def test_manual_condition(self):
        """Test the manual setting of conditions."""
        delta_d = 2
        delta_m = 4
        delta_k = 3
        H_d = delta_d/delta_m
        H_k = delta_k/delta_m

        sc = ManualCondition(delta_d=delta_d, delta_m=delta_m, delta_k=delta_k)

        self.assertEqual(sc.H_d(), H_d)
        self.assertEqual(sc.H_k(), H_k)
        self.assertEqual(sc.delta_d(), delta_d)
        self.assertEqual(sc.delta_m(), delta_m)
        self.assertEqual(sc.delta_k(), delta_k)


if __name__ == "__main__":
    unittest.main(verbosity=1)

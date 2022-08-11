#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 23:34:35 2022

@author: ddmarshall
"""

import unittest
import numpy as np
import numpy.testing as npt

from scipy.integrate import RK45

from pyBL.dense_output_ppoly import RKDenseOutputPPoly

class RKDenseOutputPPolyTest(unittest.TestCase):
    """Class to test the RKDenseOutputPPoly class"""
    def test_single_ode_solution(self):
        def simple_fun(t, y):
            return t
        
        ## propagate first step
        ode = RK45(fun=simple_fun, t0=0, y0=[1], t_bound=2, )
        ode.step()
        status = ode.status
        self.assertNotEqual(status, "failed")

        dense_out = ode.dense_output()
        dopp = RKDenseOutputPPoly(dense_out)
        t_ref = np.array(5e-5)
        do_ref = np.array(dense_out(t_ref))
        self.assertIsNone(npt.assert_allclose(dopp(t_ref), do_ref))
        
        ## take another step
        ode.step()
        status = ode.status
        self.assertNotEqual(status, "failed")

        dense_out = ode.dense_output()
        dopp.extend(dense_out)
        t_ref = np.append(t_ref, 0.001)
        do_ref = np.append(do_ref, dense_out(t_ref[-1]))
        self.assertIsNone(npt.assert_allclose(dopp(t_ref), do_ref))
        
        ## take another step
        ode.step()
        status = ode.status
        self.assertNotEqual(status, "failed")

        dense_out = ode.dense_output()
        dopp.extend(dense_out)
        t_ref = np.append(t_ref, 0.01)
        do_ref = np.append(do_ref, dense_out(t_ref[-1]))
        self.assertIsNone(npt.assert_allclose(dopp(t_ref), do_ref))
        
        ## loop until end
        while ode.status=="running":
            ode.step()
            status = ode.status
            self.assertNotEqual(status, "failed")
            
            dense_out = ode.dense_output()
            dopp.extend(dense_out)
            t_ref = np.append(t_ref, 0.5*(dense_out.t+dense_out.t_old))
            do_ref = np.append(do_ref, dense_out(t_ref[-1]))
        self.assertIsNone(npt.assert_allclose(dopp(t_ref), do_ref))
    
    def test_multiple_ode_solutions(self):
        def simple_fun(t, y):
            return np.array([t, t**2])
        
        ## propagate first step
        ode = RK45(fun=simple_fun, t0=0, y0=[1, 2], t_bound=2, )
        ode.step()
        status = ode.status
        self.assertNotEqual(status, "failed")

        dense_out = ode.dense_output()
        dopp = RKDenseOutputPPoly(dense_out)
        t_ref = np.array(5e-5)
        do_ref = np.array(dense_out(t_ref), ndmin=2)
        self.assertIsNone(npt.assert_allclose(dopp(t_ref), do_ref))
        
        ## take another step
        ode.step()
        status = ode.status
        self.assertNotEqual(status, "failed")

        dense_out = ode.dense_output()
        dopp.extend(dense_out)
        t_ref = np.append(t_ref, 0.001)
        do_ref = np.append(do_ref, [dense_out(t_ref[-1]).T], axis=0)
        self.assertIsNone(npt.assert_allclose(dopp(t_ref), do_ref))
        
        ## take another step
        ode.step()
        status = ode.status
        self.assertNotEqual(status, "failed")

        dense_out = ode.dense_output()
        dopp.extend(dense_out)
        t_ref = np.append(t_ref, 0.01)
        do_ref = np.append(do_ref, [dense_out(t_ref[-1]).T], axis=0)
        self.assertIsNone(npt.assert_allclose(dopp(t_ref), do_ref))
        
        ## loop until end
        while ode.status=="running":
            ode.step()
            status = ode.status
            self.assertNotEqual(status, "failed")
            
            dense_out = ode.dense_output()
            dopp.extend(dense_out)
            t_ref = np.append(t_ref, 0.5*(dense_out.t+dense_out.t_old))
            do_ref = np.append(do_ref, [dense_out(t_ref[-1]).T], axis=0)
        self.assertIsNone(npt.assert_allclose(dopp(t_ref), do_ref))


if (__name__ == "__main__"):
    unittest.main(verbosity=1)

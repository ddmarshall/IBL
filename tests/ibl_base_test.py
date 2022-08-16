#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:26:49 2022

@author: ddmarshall
"""


import unittest
import numpy as np
import numpy.testing as npt
from scipy.interpolate import CubicSpline

from pyBL.ibl_base import IBLBase
from pyBL.ibl_base import IBLTermEventBase


class _IBLBaseTestTermEvent(IBLTermEventBase):
    """
    Sample class to test the termination capabilities of the base class.
    
    This is a callable object that the ODE integrator will use to determine if
    the integration should terminate before the end location.
    
    Attributes
    ----------
        _x_kill: x-location that the integrator should stop.
    """
    def __init__(self, x_kill):
        self._x_kill = x_kill
        super().__init__()
    
    def _call_impl(self, x, F):
        """
        Information used to determine if IBL test integrator should terminate.
        
        This will terminate once x passed specified value (x_kill) and will be
        negative before then (positive afterwards).
        
        Args
        ----
            x: Current x-location of the integration
            F: Current state value(s)
        
        Returns
        -------
            Negative value when the integration should continue, positive when
            the integration has passed the termination condition, and zero at
            the state when the integrator should terminate.
        """
        return x - self._x_kill
    
    def event_info(self):
        return -1, ""


class IBLBaseTest(IBLBase):
    """Generic class to test the concrete methods in IBLBase"""
    def __init__(self, U_e = None, dU_edx = None, d2U_edx2 = None, 
                 x_kill = None):
        # setup base class
        super().__init__(U_e=U_e, dU_edx=dU_edx, d2U_edx2=d2U_edx2)
        
        # set up this class
        if x_kill is not None:
            self._set_kill_event(_IBLBaseTestTermEvent(x_kill))
    
    def solve(self, x_start, x_end, y0, term_event = None):
        self._set_x_range(x_start, x_end)
        return self._solve_impl(y0, term_event = term_event)
    
    def _ode_impl(self, x, F):
        """
        This is the derivatives of the ODEs that are to be solved
        
        Args
        ----
            x: x-location of current step
            F: current step state value(s)
        """
        return x
    
    def U_n(self, x):
        return np.zeros_like(x)
    
    def delta_d(self, x):
        return np.zeros_like(x)
    
    def delta_m(self, x):
        return np.zeros_like(x)
    
    def H(self, x):
        return np.zeros_like(x)
    
    def tau_w(self, x, rho):
        return np.zeros_like(x)


class IBLBaseTestTransition(IBLTermEventBase):
    """
    Generic class to test the passing of termination events during the solve 
    method.
    """
    def __init__(self, F_kill):
        self._F_kill = F_kill
        super().__init__()
    
    def _call_impl(self, x, F):
        return F - self._F_kill
    
    def event_info(self):
        return 1, ""


class TestEdgeVelocity(unittest.TestCase):
    """Class to test various functions and curve fits for Thwaites method"""
    ## define the edge velocity functions
    @classmethod
    def U_e_fun(cls, x, C, m):
        x = np.asarray(x)
        if m==0:
            return C*np.ones_like(x)
        else:
            return C*x**m
    
    @classmethod
    def dU_edx_fun(cls, x, C, m):
        x = np.asarray(x)
        if m==0:
            return np.zeros_like(x)
        elif m==1:
            return C*np.ones_like(x)
        else:
            return m*C*x**(m-1)
    
    @classmethod
    def d2U_edx2_fun(cls, x, C, m):
        x = np.asarray(x)
        if (m==0) or (m==1):
            return np.zeros_like(x)
        elif (m==2):
            return m*C*np.ones_like(x)
        else:
            return m*(m-1)*C*x**(m-2)
    
    @classmethod
    def d3U_edx3_fun(cls, x, C, m):
        x = np.asarray(x)
        if (m==0) or (m==1) or (m==2):
            return np.zeros_like(x)
        elif (m==3):
            return m*(m-1)*C*np.ones_like(x)
        else:
            return m*(m-1)*(m-2)*C*x**(m-3)
    
    def test_setting_velocity_functions(self):
        ## create test class with all three functions
        U_inf = 10
        m = 0.75
        iblb = IBLBaseTest(U_e = lambda x: self.U_e_fun(x, U_inf, m),
                           dU_edx = lambda x: self.dU_edx_fun(x, U_inf, m),
                           d2U_edx2 = lambda x: self.d2U_edx2_fun(x, U_inf, m))
        
        x=np.linspace(0.1, 5, 21)
        U_e_ref = self.U_e_fun(x, U_inf, m)
        dU_edx_ref = self.dU_edx_fun(x, U_inf, m)
        d2U_edx2_ref = self.d2U_edx2_fun(x, U_inf, m)
        self.assertIsNone(npt.assert_allclose(iblb.U_e(x), U_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.dU_edx(x), dU_edx_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2U_edx2(x), d2U_edx2_ref))
        
        ## create test class with two functions
        U_inf = 10
        m = 0.75
        iblb = IBLBaseTest(U_e = lambda x: self.U_e_fun(x, U_inf, m),
                           dU_edx = lambda x: self.dU_edx_fun(x, U_inf, m))
        
        x=np.linspace(0.1, 5, 21)
        U_e_ref = self.U_e_fun(x, U_inf, m)
        dU_edx_ref = self.dU_edx_fun(x, U_inf, m)
        d2U_edx2_ref = self.d2U_edx2_fun(x, U_inf, m)
        self.assertIsNone(npt.assert_allclose(iblb.U_e(x), U_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.dU_edx(x), dU_edx_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2U_edx2(x), d2U_edx2_ref))
        
        ## create test class with one function
        U_inf = 10
        m = 0.75
        iblb = IBLBaseTest(U_e = lambda x: self.U_e_fun(x, U_inf, m))
        
        x=np.linspace(0.1, 5, 21)
        U_e_ref = self.U_e_fun(x, U_inf, m)
        dU_edx_ref = self.dU_edx_fun(x, U_inf, m)
        d2U_edx2_ref = self.d2U_edx2_fun(x, U_inf, m)
        self.assertIsNone(npt.assert_allclose(iblb.U_e(x), U_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.dU_edx(x), dU_edx_ref))
        # NOTE: second derivative has slightly larger errors
        self.assertIsNone(npt.assert_allclose(iblb.d2U_edx2(x), d2U_edx2_ref,
                                              rtol=1e-5, atol=0))
    
    def test_setting_velocity_splines(self):
        ## set the edge velocity spline
        x_sample = np.linspace(0.1, 5, 8)
        U_inf = 10
        m = 1.25
        U_e = CubicSpline(x_sample, self.U_e_fun(x_sample, U_inf, m))
        dU_edx = U_e.derivative()
        d2U_edx2 = dU_edx.derivative()
        iblb = IBLBaseTest(U_e = U_e)
        
        x=np.linspace(0.1, 5, 21)
        U_e_ref = U_e(x)
        dU_edx_ref = dU_edx(x)
        d2U_edx2_ref = d2U_edx2(x)
        self.assertIsNone(npt.assert_allclose(iblb.U_e(x), U_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.dU_edx(x), dU_edx_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2U_edx2(x), d2U_edx2_ref))
        
        ## set the edge velocity derivative spline
        x_sample = np.linspace(0.1, 5, 8)
        U_inf = 10
        m = 1.25
        dU_edx = CubicSpline(x_sample, self.dU_edx_fun(x_sample, U_inf, m),
                             bc_type=((1, self.d2U_edx2_fun(x_sample[0],
                                                            U_inf, m)),
                                      (1, self.d2U_edx2_fun(x_sample[-1],
                                                            U_inf, m))))
        U_e = dU_edx.antiderivative()
        U_e.c[-1,:] = U_e.c[-1,:]+self.U_e_fun(x_sample[0], U_inf, m)
        d2U_edx2 = dU_edx.derivative()
        iblb = IBLBaseTest(U_e = U_e, dU_edx = dU_edx)
        
        x=np.linspace(0.1, 5, 21)
        U_e_ref = U_e(x)
        dU_edx_ref = dU_edx(x)
        d2U_edx2_ref = d2U_edx2(x)
        self.assertIsNone(npt.assert_allclose(iblb.U_e(x), U_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.dU_edx(x), dU_edx_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2U_edx2(x), d2U_edx2_ref))
        
        ## set the edge velocity second derivative spline
        x_sample = np.linspace(0.1, 5, 8)
        U_inf = 10
        m = 1.25
        d2U_edx2 = CubicSpline(x_sample, self.d2U_edx2_fun(x_sample, U_inf, m),
                               bc_type=((1, self.d3U_edx3_fun(x_sample[0],
                                                              U_inf, m)),
                                        (1, self.d3U_edx3_fun(x_sample[-1],
                                                              U_inf, m))))
        dU_edx = d2U_edx2.antiderivative()
        dU_edx.c[-1,:] = dU_edx.c[-1,:]+self.dU_edx_fun(x_sample[0], U_inf, m)
        U_e = dU_edx.antiderivative()
        U_e.c[-1,:] = U_e.c[-1,:]+self.U_e_fun(x_sample[0], U_inf, m)
        iblb = IBLBaseTest(U_e = U_e, dU_edx = dU_edx, d2U_edx2 = d2U_edx2)
        
        x=np.linspace(0.1, 5, 21)
        U_e_ref = U_e(x)
        dU_edx_ref = dU_edx(x)
        d2U_edx2_ref = d2U_edx2(x)
        self.assertIsNone(npt.assert_allclose(iblb.U_e(x), U_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.dU_edx(x), dU_edx_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2U_edx2(x), d2U_edx2_ref))
    
    def test_setting_velocity_points(self):
        ## set the edge velocity values
        x_sample = np.linspace(0.1, 5, 8)
        U_inf = 10
        m = 1.25
        U_e = [x_sample, self.U_e_fun(x_sample, U_inf, m)]
        U_e_spline = CubicSpline(x_sample, self.U_e_fun(x_sample, U_inf, m))
        dU_edx_spline = U_e_spline.derivative()
        d2U_edx2_spline = dU_edx_spline.derivative()
        iblb = IBLBaseTest(U_e = U_e)
        
        x=np.linspace(0.1, 5, 21)
        U_e_ref = U_e_spline(x)
        dU_edx_ref = dU_edx_spline(x)
        d2U_edx2_ref = d2U_edx2_spline(x)
        self.assertIsNone(npt.assert_allclose(iblb.U_e(x), U_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.dU_edx(x), dU_edx_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2U_edx2(x), d2U_edx2_ref))
    
    def test_delay_setting_velocity(self):
        ## create test class with all three functions
        U_inf = 10
        m = 0.75
        iblb = IBLBaseTest()
        x=np.linspace(0.1, 5, 21)
        U_e_ref = self.U_e_fun(x, U_inf, m)
        dU_edx_ref = self.dU_edx_fun(x, U_inf, m)
        d2U_edx2_ref = self.d2U_edx2_fun(x, U_inf, m)
        
        with self.assertRaises(ValueError):
            iblb.U_e(x)
        with self.assertRaises(ValueError):
            iblb.dU_edx(x)
        with self.assertRaises(ValueError):
            iblb.d2U_edx2(x)
        
        iblb.set_velocity(U_e = lambda x: self.U_e_fun(x, U_inf, m),
                        dU_edx = lambda x: self.dU_edx_fun(x, U_inf, m),
                        d2U_edx2 = lambda x: self.d2U_edx2_fun(x, U_inf, m))
        self.assertIsNone(npt.assert_allclose(iblb.U_e(x), U_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.dU_edx(x), dU_edx_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2U_edx2(x), d2U_edx2_ref))
    
    def test_terminating_solver(self):
        x_kill = 3
        iblb = IBLBaseTest(x_kill = x_kill)
        
        ## go through the entire xrange
        # NOTE: No need to set the velocity terms because they are not used in
        #       this basic implementation.
        # NOTE: This solves the simple differential equation y'=x
        def ref_fun(x):
            return np.array([0.5*x**2+1])
        
        x_start = 1
        x_end = 2
        y_start = ref_fun(x_start)
        rtn = iblb.solve(x_start, x_end, y_start)
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x_end)
        self.assertIsNone(npt.assert_allclose(rtn.F_end, ref_fun(x_end)))
        
        ## stop because solver terminated early
        x_start = 1
        x_end = x_kill + 1
        y_start = ref_fun(x_start)
        rtn = iblb.solve(x_start, x_end, y_start)
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, -1)
        self.assertEqual(rtn.message, "Separated")
        self.assertEqual(rtn.x_end, x_kill)
        self.assertIsNone(npt.assert_allclose(rtn.F_end, ref_fun(x_kill)))
        
        ## stop because external trigger
        x_start = 1
        x_end = x_kill + 1
        y_start = ref_fun(x_start)
        y_trans = 0.5*(y_start+ref_fun(x_kill))
        x_trans = np.sqrt(2*(y_trans-1))
        rtn = iblb.solve(x_start, x_end, y_start,
                         term_event = IBLBaseTestTransition(y_trans))
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 1)
        self.assertEqual(rtn.message, "Transition")
        self.assertEqual(rtn.x_end, x_trans)
        self.assertIsNone(npt.assert_allclose(rtn.F_end, y_trans))


if (__name__ == "__main__"):
    unittest.main(verbosity=1)

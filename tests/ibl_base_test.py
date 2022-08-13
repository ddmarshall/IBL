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

# these should go with the base class
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod

from pyBL.ibl_base import IBLBase


class IBLResult:
    def __init__(self, x_end = np.inf, F_end = np.inf,
                 status = -99, message = "Not Set", success = False):
        self.x_end = x_end
        self.F_end = F_end
        self.status = status
        self.message = message
        self.success = success


class IBLTermEvent(ABC):
    def __init__(self):
        self.terminal = True
        
    def __call__(self, x, F):
        return self._call_impl(x, F)
    
    @abstractmethod
    def eventInfo(self):
        pass
    
    @abstractmethod
    def _call_impl(self, x, F):
        pass

TERMINATION_MESSAGES = {0: "Completed",
                        -1: "Separated",
                        1: "Transition",
                        -99: "Unknown Event"}


class _IBLBaseTestTermEvent(IBLTermEvent):
    def __init__(self, x_kill):
        self._x_kill = x_kill
        super().__init__()
    
    def _call_impl(self, x, F):
        return x - self._x_kill
    
    def eventInfo(self):
        return -1, ""


class IBLBaseTest(IBLBase):
    """Generic class to test the concrete methods in IBLBase"""
    def __init__(self, U_e = None, dU_edx = None, d2U_edx2 = None, 
                 x_kill = None):
        def fun(x, F):
            return x
        super().__init__(fun, 0, [0], 0,
                         U_e=U_e, dU_edx=dU_edx, d2U_edx2=d2U_edx2)
        self._kill_events = None
        if x_kill is not None:
            self._add_kill_event(_IBLBaseTestTermEvent(x_kill))
    
    def _ode_impl(self, x, F):
        return x
    
    def _add_kill_event(self, ke):
        if self._kill_events is None:
            self._kill_events = [ke]
        else:
            self._kill_events.append(ke)
    
    def solve(self, xrange, y0i, rtol=1e-8, atol=1e-11, term_event = None):
        ## setup the ODE solver
        xrange = np.asarray(xrange)
        y0 = np.asarray(y0i)
        if y0.ndim == 0:
            y0 = [y0i]
        
        kill_events = []
        if self._kill_events is not None:
            kill_events = kill_events + self._kill_events

        if term_event is None:
            if self._kill_events is None:
                kill_events = None
        else:
            if isinstance(term_event, list):
                kill_events = kill_events + term_event
            else:
                kill_events.append(term_event)

        rtn = solve_ivp(fun = self._ode_impl, t_span = xrange, y0 = y0,
                        method = 'RK45', dense_output = True,
                        events = kill_events, rtol = rtol, atol = atol)
        
        # if completed gather info
        self._solution = None
        x_end = xrange[0]
        F_end = y0
        status = -99
        message = rtn.message
        if rtn.success:
            self._solution = rtn.sol
            
            # if terminated on time or early figure out why
            if rtn.status == 0:
                x_end = rtn.t[-1]
                F_end = rtn.sol(x_end)
                status = 0
                message = ""
            elif rtn.status == 1:
                message = "Event not found."
                for i, xe in enumerate(rtn.t_events):
                    if xe.shape[0] > 0:
                        x_end = xe[0]
                        F_end = rtn.sol(x_end)
                        status, message = kill_events[i].eventInfo()
                        break
            else:
                status = -99

        if len(message)> 0:
            message = "{}: {}".format(TERMINATION_MESSAGES.get(status), message)
        else:
            message = TERMINATION_MESSAGES.get(status)
        return IBLResult(x_end = x_end, F_end = F_end, status = status,
                         message = message, success = rtn.success)
    
    def U_n(self, x):
        return np.zeros_like(x)
    
    def delta_d(self, x):
        return np.zeros_like(x)
    
    def delta_m(self, x):
        return np.zeros_like(x)
    
    def H(self, x):
        return np.zeros_like(x)
    
    def tau_w(self, x):
        return np.zeros_like(x)


class IBLBaseTestTransition(IBLTermEvent):
    def __init__(self, F_kill):
        self._F_kill = F_kill
        super().__init__()
    
    def _call_impl(self, x, F):
        return F - self._F_kill
    
    def eventInfo(self):
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
        
        self.assertRaises(ValueError, iblb.U_e, x)
        self.assertRaises(ValueError, iblb.dU_edx, x)
        self.assertRaises(ValueError, iblb.d2U_edx2, x)
        
        iblb.setVelocity(U_e = lambda x: self.U_e_fun(x, U_inf, m),
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
        rtn = iblb.solve([x_start, x_end], y_start)
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x_end)
        self.assertIsNone(npt.assert_allclose(rtn.F_end, ref_fun(x_end)))
        
        ## stop because solver terminated early
        x_start = 1
        x_end = x_kill + 1
        y_start = ref_fun(x_start)
        rtn = iblb.solve([x_start, x_end], y_start)
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
        rtn = iblb.solve([x_start, x_end], y_start,
                         term_event = IBLBaseTestTransition(y_trans))
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 1)
        self.assertEqual(rtn.message, "Transition")
        self.assertEqual(rtn.x_end, x_trans)
        self.assertIsNone(npt.assert_allclose(rtn.F_end, y_trans))


if (__name__ == "__main__"):
    unittest.main(verbosity=1)

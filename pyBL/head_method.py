#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:15:11 2022

@author: ddmarsha
"""

import numpy as np

from pyBL.ibl_base import IBLBase
from pyBL.ibl_base import IBLTermEventBase
from pyBL.skin_friction import c_f_LudwiegTillman as c_f_fun


class HeadMethod(IBLBase):
    """
    Models a turbulent bondary layer using Head's Method (1958) when provided
    the edge velocity profile.
    
    Attributes
    ----------
        _delta_m0: Momentum thickness at start location
        _H_d0: Displacement shape factor at start location
        _nu: Kinematic viscosity
    """
    def __init__(self, U_e = None, dU_edx = None, d2U_edx2 = None,
                 H_d_crit = 2.4):
        super().__init__(U_e, dU_edx, d2U_edx2)
        self.set_H_d_critical(H_d_crit)
    
    def set_H_d_critical(self, H_d_crit):
        self._set_kill_event(_HeadSeparationEvent(H_d_crit))
    
    def set_solution_parameters(self, x0, x_end, delta_m0, H_d0, nu):
        """
        Set the parameters needed for the solver to propagate
        
        Args
        ----
            x0: location to start integration
            x_end: location to end integration
            delta_m0: Momentum thickness at start location
            nu: Kinematic viscosity
        
        Throws
        ------
            ValueError if negative viscosity provided, or invalid initial
            conditions
        """
        if nu < 0:
            raise ValueError("Viscosity must be positive")
        else:
            self._nu = nu
        if delta_m0 < 0:
            raise ValueError("Initial momentum thickness must be positive")
        else:
            self._delta_m0 = delta_m0
        if H_d0 <= 1:
            raise ValueError("Initial displacement shape factor must be "
                             "greater than one")
        else:
            self._H_d0 = H_d0
        self._set_x_range(x0, x_end)
    
    def nu(self):
        """Getter for kinematic viscosity"""
        return self._nu
    
    def solve(self, term_event = None):
        return self._solve_impl([self._delta_m0, self._H_d0],
                                term_event = term_event)
    
    def U_n(self, x):
        """
        Calculate the transpiration velocity
        
        Args
        ----
            x: Streamwise loations to calculate this property
        
        Returns
        -------
            Desired property at the specified locations
        """
        yp = self._ode_impl(x, self._solution(x))
        H_d = self.H_d(x)
        U_e = self.U_e(x)
        dU_edx = self.dU_edx(x)
        delta_m = self.delta_m(x)
        return (dU_edx*H_d*delta_m + U_e*yp[1]*delta_m + U_e*H_d*yp[0])
    
    def delta_d(self, x):
        """
        Calculate the displacement thickness
        
        Args
        ----
            x: Streamwise loations to calculate this property
        
        Returns
        -------
            Desired property at the specified locations
        """
        return self.delta_m(x)*self.H_d(x)
    
    def delta_m(self, x):
        """
        Calcualte the momentum thickness
        
        Args
        ----
            x: Streamwise loations to calculate this property
        
        Returns
        -------
            Desired property at the specified locations
        """
        return self._solution(x)[0]
    
    def H_d(self, x):
        """
        Calculate the shape factor
        
        Args
        ----
            x: Streamwise loations to calculate this property
        
        Returns
        -------
            Desired property at the specified locations
        """
        return self._solution(x)[1]
    
    def tau_w(self, x, rho):
        """
        Calculate the wall shear stress
        
        Args
        ----
            x: Streamwise loations to calculate this property
            rho: Freestream density
        
        Returns
        -------
            Desired property at the specified locations
        """
        delta_m = self._solution(x)[0]
        H_d = self._solution(x)[1]
        U_e = self.U_e(x)
        U_e[np.abs(U_e)<0.001] = 0.001
        Re_delta_m = U_e*delta_m/self._nu
        c_f = c_f_fun(Re_delta_m, H_d)
        return 0.5*rho*U_e**2*c_f
    
    def _ode_impl(self, x, y):
        """
        This is the right-hand-side of the ODE representing Head's method.
        
        Args
        ----
            x: x-location of current step
            y: current step solution vector of momentum thickness and H_d
        """
        yp = np.zeros_like(y)
        delta_m = y[0]
        H_d = np.asarray(y[1])
        if (H_d < 1.11).any():
            H_d[H_d < 1.11] = 1.11
        U_e = self.U_e(x)
        U_e[np.abs(U_e)<0.001] = 0.001
        dU_edx = self.dU_edx(x)
        Re_delta_m = U_e*delta_m/self._nu
        c_f = c_f_fun(Re_delta_m, H_d)
        H1 = self._H1(H_d)
        H1p = self._H1p(H_d)
        yp[0] = 0.5*c_f-delta_m*(2+H_d)*dU_edx/U_e
        yp[1] = (U_e*self._S(H1) - U_e*yp[0]*H1 - dU_edx*delta_m*H1)/(H1p*U_e*delta_m)
        return yp
    
    @staticmethod
    def _H1(H_d):
        H_d = np.asarray(H_d)
        if (H_d <= 1.1).any():
            H_d[H_d <= 1.1] = 1.1001
#            raise ValueError("Cannot pass displacement shape factor less than "
#                             "1.1: {}".format(np.amin(H_d)))
        def H1_low(H_d):
            a = 0.8234
            b = 1.1
            c = 1.287
            d = 3.3
            return d + a/(H_d - b)**c
        def H1_high(H_d):
            a = 1.5501
            b = 0.6778
            c = 3.064
            d = 3.32254659218600974
            return d + a/(H_d - b)**c
        return np.piecewise(H_d, [H_d<=1.6, H_d>1.6], [H1_low, H1_high])
    
    @staticmethod
    def _H1p(H_d):
        H_d = np.asarray(H_d)
        if (H_d <= 1.1).any():
            H_d[H_d <= 1.1] = 1.1001
#            raise ValueError("Cannot pass displacement shape factor less than "
#                             "1.1: {}".format(np.amin(H_d)))
        def H1_low(H_d):
            a = 0.8234
            b = 1.1
            c = 1.287
            return -a*c/(H_d - b)**(c+1)
        def H1_high(H_d):
            a = 1.5501
            b = 0.6778
            c = 3.064
            return -a*c/(H_d - b)**(c+1)
        return np.piecewise(H_d, [H_d<=1.6, H_d>1.6], [H1_low, H1_high])
    
    @staticmethod
    def _H_d(H1):
        H1 = np.asarray(H1)
        if (H1 <= 3.32254659218600974).any():
            raise ValueError("Cannot pass entrainment shape factor less than "
                             "3.323: {}".format(np.amin(H1)))
        def H_d_low(H1):
            a = 1.5501
            b = 0.6778
            c = 3.064
            d = 3.32254659218600974
            return b + (a/(H1 - d))**(1/c)
        def H_d_high(H1):
            a = 0.8234
            b = 1.1
            c = 1.287
            d = 3.3
            return b + (a/(H1 - d))**(1/c)
        H1_break = HeadMethod._H1(1.6)
        return np.piecewise(H1, [H1<=H1_break, H1>H1_break],
                            [H_d_low, H_d_high])
    
    @staticmethod
    def _S(H1):
        H1 = np.asarray(H1)
        if (H1 <= 3).any():
            H1[H1 <= 3] = 3.001
#            raise ValueError("Cannot pass entrainment shape factor less than "
#                             " 3: {}".format(np.amin(H1)))
        return 0.0306/(H1-3)**0.6169


class _HeadSeparationEvent(IBLTermEventBase):
    """
    This class detects separation and will terminate integration when it occurs.
    
    This is a callable object that the ODE integrator will use to determine if
    the integration should terminate before the end location.
    
    Attributes
    ----------
        H_d_crit: Displacement shape factor value that indicates separation
    """
    def __init__(self, H_d_crit):
        super().__init__()
        self._H_d_crit = H_d_crit
    
    def _call_impl(self, x, y):
        """
        Information used to determine if Head method integrator should 
        terminate.
        
        This will terminate once the displacement shape factor becomes greater 
        than critical H_d.
        
        Args
        ----
            x: Current x-location of the integration
            y: Current step momentum thickness and displacement shape factor
        
        Returns
        -------
            Current value of the shear function.
        """
        return self._H_d_crit - y[1]
    
    def event_info(self):
        return -1, ""

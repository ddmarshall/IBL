#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:42:04 2022

@author: ddmarsha
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.misc import derivative as fd

from pyBL.ibl_base import IBLBase
from pyBL.ibl_base import IBLTermEventBase


#def _stagnation_y0(iblsimdata,x0):
#    #From Moran
#      return .075*iblsimdata.nu/iblsimdata.du_edx(x0)


class ThwaitesMethod(IBLBase):
    """
    Models a laminar boundary layer using Thwaites Method (1949) when provided 
    the edge velocity profile. There are a few different ways of modeling the 
    tabular data from Thwaites original work that can be set.
    
    This class solves for \frac{\delta_m^2}{\nu} use the IBLBase ODE solver
    using the linear approximation for the differential equation relationship.
    
    Attributes
    ----------
        _delta_m0: Momentum thickness at start location
        _nu: Kinematic viscosity
        _model: Collection of functions for S, H, and H'
    """
    def __init__(self, U_e = None, dU_edx = None, d2U_edx2 = None,
                 data_fits="Spline"):
        super().__init__(U_e, dU_edx, d2U_edx2)
        self.set_data_fits(data_fits)
        self._nu = None
        self._delta_m0 = None
    
    def set_solution_parameters(self, x0, x_end, delta_m0, nu):
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
            ValueError if negative viscosity provided
        
        """
        if nu < 0:
            raise ValueError("Viscosity must be positive")
        else:
            self._nu = nu
        if delta_m0 < 0:
            raise ValueError("Initial momentum thickness must be positive")
        else:
            self._delta_m0 = delta_m0
        self._set_x_range(x0, x_end)
    
    def nu(self):
        """Getter for kinematic viscosity"""
        return self._nu
    
    def set_data_fits(self, data_fits):
        """
        Set the functions used for the data fits of the shear function,
        S(\lambda), shape function, H(\lambda), and the slope of the shape
        function dH/d\lambda.
        
        Args
        ----
            data_fits: * either a 2-tuple or 3-tuple of functions that each take
                         one parameter, lambda, where the first is the shear 
                         function, the second is the shape function and if 
                         provided, the third is the derivative of the shape
                         function; or 
                       * a string for representing one of the three internal 
                         implementations:
                           * "Spline" - Spline fits of Thwaites original 
                             data (Edland 2022)
                           * "White" - Curve fits from White (2011)
                           * "Cebeci-Bradshaw" - Curve fits from 
                             Cebeci-Bradshaw 1977"
        
        Throws
        ------
            ValueError if an invalid fit name or unusable 2-tuple or
            3-tuple provided
        """
        # data_fits can either be string or 2-tuple of callables
        self._model = None
        if type(data_fits) is str:
            if data_fits == "Spline":
                self._model = _ThwaitesFunctionsSpline()
            elif data_fits == "White":
                self._model = _ThwaitesFunctionsWhite()
            elif data_fits == "Cebeci-Bradshaw":
                self._model = _ThwaitesFunctionsCebeciBradshaw()
            else:
                raise ValueError("Unknown fitting function name: ", data_fits)
        else:
            # check to make sure have two callables
            if (type(data_fits) is tuple):
                if len(data_fits)==3:
                    if callable(data_fits[0]) and callable(data_fits[1]) \
                            and callable(data_fits[2]):
                        self._model = _ThwaitesFunctionsBase("Custom",
                                                             data_fits[0],
                                                             data_fits[1],
                                                             data_fits[2],
                                                             -np.inf, np.inf)
                    else:
                        raise ValueError("Need to pass callable objects for "
                                         "fit functions")
                elif len(data_fits)==2:
                    if callable(data_fits[0]) and callable(data_fits[1]):
                        def Hp_fun(lam):
                            return fd(self._model.H, lam, 1e-5, n=1, order=3)
                        self._model = _ThwaitesFunctionsBase("Custom",
                                                             data_fits[0],
                                                             data_fits[1],
                                                             Hp_fun,
                                                             -np.inf, np.inf)
                    else:
                        raise ValueError("Need to pass callable objects for "
                                         "fit functions")
                else:
                    raise ValueError("Need to pass two or three callable "
                                     "objects for fit functions")
            else:
                raise ValueError("Need to pass a 2-tuple for fit functions")
        
        self._set_kill_event(_ThwaitesSeparationEvent(self._calc_lambda,
                                                      self._model.S))
    
    def solve(self, term_event = None):
        return self._solve_impl(self._delta_m0**2/self._nu,
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
        U_e = self.U_e(x)
        dU_edx = self.dU_edx(x)
        delta_m2_on_nu = self._solution(x)[0]
        term1 = dU_edx*self.delta_d(x)
        term2 = np.sqrt(self._nu/delta_m2_on_nu)
        dsol_dx = self._ode_impl(x, delta_m2_on_nu)
        term3 = 0.5*U_e*self.H_d(x)*dsol_dx
        term4 = (U_e*delta_m2_on_nu
                *self._model.Hp(self._calc_lambda(x,delta_m2_on_nu)))
        term5 = dU_edx*dsol_dx+self.d2U_edx2(x)*delta_m2_on_nu
        return term1 + term2*(term3+term4*term5)
    
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
        return np.sqrt(self._solution(x)[0]*self._nu)
    
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
        lam = self._calc_lambda(x, self._solution(x)[0])
        return self._model.H(lam)
    
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
        lam = self._calc_lambda(x, self._solution(x)[0])
        return rho*self._nu*self.U_e(x)*self._model.S(lam)/self.delta_m(x)
    
    def _ode_impl(self, x, delta_m2_on_nu):
        """
        This is the right-hand-side of the ODE representing Thwaites method.
        
        Args
        ----
            x: x-location of current step
            delta_m2_on_nu: current step square of momentum thickness divided
            by the kinematic viscosity
        """
        a = 0.45
        b = 6
        lam = self._calc_lambda(x, delta_m2_on_nu)
        F = (a-b*lam)
        
        return F/self.U_e(x)
    
    def _calc_lambda(self, x, delta_m2_on_nu):
        return delta_m2_on_nu*self.dU_edx(x)


class _ThwaitesFunctionsBase:
    """Base class for curve fits for Thwaites data"""
    def __init__(self, name, S_fun, H_fun, Hp_fun, lambda_min, lambda_max):
        self._range = [lambda_min, lambda_max]
        self._name = name
        self._H_fun = H_fun
        self._Hp_fun = Hp_fun
        self._S_fun = S_fun
    
    def range(self):
        return self._range[0], self._range[1]
    
    def H(self, lam):
        self._check_range(lam)
        return self._H_fun(lam)
    
    def Hp(self, lam):
        self._check_range(lam)
        return self._Hp_fun(lam)
    
    def S(self, lam):
        self._check_range(lam)
        return self._S_fun(lam)
    
    def F(self, lam):
        self._check_range(lam)
        return 2*(self.S(lam)-lam*(self.H(lam)+2))
    
    def get_name(self):
        return self._name
    
    def _check_range(self, lam):
        lam_min, lam_max = self.range()
        
        if (np.array(lam) < lam_min).any():
            raise ValueError('cannot pass value less than {} into this function: {}'.format(lam_min, lam))
        elif (np.array(lam) > lam_max).any():
            raise ValueError('cannot pass value greater than {} into this function: {}'.format(lam_max, lam))


class _ThwaitesFunctionsWhite(_ThwaitesFunctionsBase):
    """Returns White's calculation of Thwaites functions from 2011 book"""
    def __init__(self):
        def S(lam):
            return pow(lam + 0.09, 0.62)
        def H(lam):
            z = 0.25-lam
            return 2 + z*(4.14 + z*(-83.5 + z*(854 + z*(-3337 + z*4576))))
        def Hp(lam):
            z = 0.25-lam
            return -(4.14 + z*(-2*83.5 + z*(3*854 + z*(-4*3337 + z*5*4576))))
        
        super().__init__("White", S, H, Hp, -0.09, np.inf)


class _ThwaitesFunctionsCebeciBradshaw(_ThwaitesFunctionsBase):
    """
    Returns Cebeci and Bradshaw's calculation of Thwaites functions from
    1977 book
    """
    def __init__(self):
        def S(lam):
            return np.piecewise(lam, [lam<0, lam>=0],
                                [lambda lam: (0.22 + 1.402*lam
                                              + 0.018*lam/(0.107 + lam)),
                                 lambda lam: 0.22 + 1.57*lam - 1.8*lam**2])
        def H(lam):
            # NOTE: C&B's H function is not continuous at lam=0,
            #       so using second interval
            return np.piecewise(lam, [lam<0, lam>=0],
                                [lambda lam: 2.088 + 0.0731/(0.14 + lam),
                                 lambda lam: 2.61 - 3.75*lam + 5.24*lam**2])
        def Hp(lam):
            # NOTE: C&B's H function is not continuous at lam=0,
            #       so using second interval
            return np.piecewise(lam, [lam<0, lam>=0],
                                [lambda lam: -0.0731/(0.14 + lam)**2,
                                 lambda lam: -3.75 + 2*5.24*lam])
        
        super().__init__("Cebeci and Bradshaw", S, H, Hp, -0.1, 0.1)

class _ThwaitesFunctionsSpline(_ThwaitesFunctionsBase):
    """Returns cubic splines of Thwaites original tables based on Edland 2021"""
    def __init__(self):
        # Spline fits to Thwaites original data Edland
        S = CubicSpline(self._tab_lambda, self._tab_S)
        H = CubicSpline(self._tab_lambda, self._tab_H)
        Hp = H.derivative()
        
        super().__init__("Thwaites Splines", S, H, Hp, np.min(self._tab_lambda),
                         np.max(self._tab_lambda))

    # Tabular data section
    _tab_F = np.array([0.938, 0.953, 0.956, 0.962, 0.967, 0.969, 0.971, 0.970, 
                       0.963, 0.952, 0.936, 0.919, 0.902, 0.886, 0.854, 0.825,
                       0.797, 0.770, 0.744, 0.691, 0.640, 0.590, 0.539, 0.490,
                       0.440, 0.342, 0.249, 0.156, 0.064,-0.028,-0.138,-0.251,
                      -0.362,-0.702,-1.000])
    _tab_S = np.array([0.000, 0.011, 0.016, 0.024, 0.030, 0.035, 0.039, 0.049,
                       0.055, 0.067, 0.076, 0.083, 0.089, 0.094, 0.104, 0.113,
                       0.122, 0.130, 0.138, 0.153, 0.168, 0.182, 0.195, 0.208,
                       0.220, 0.244, 0.268, 0.291, 0.313, 0.333, 0.359, 0.382,
                       0.404, 0.463, 0.500])
    _tab_H = np.array([3.70, 3.69, 3.66, 3.63, 3.61, 3.59, 3.58, 3.52, 3.47,
                       3.38, 3.30, 3.23, 3.17, 3.13, 3.05, 2.99, 2.94, 2.90,
                       2.87, 2.81, 2.75, 2.71, 2.67, 2.64, 2.61, 2.55, 2.49,
                       2.44, 2.39, 2.34, 2.28, 2.23, 2.18, 2.07,  2.00])
    _tab_lambda = np.array([-0.082,-0.0818,-0.0816,-0.0812,-0.0808,-0.0804,
                            -0.080,-0.079, -0.078, -0.076, -0.074, -0.072,
                            -0.070,-0.068, -0.064, -0.060, -0.056, -0.052,
                            -0.048,-0.040, -0.032, -0.024, -0.016, -0.008,
                             0.000, 0.016,  0.032,  0.048,  0.064,  0.080,
                             0.10,  0.12,   0.14,   0.20,   0.25])
    

class _ThwaitesSeparationEvent(IBLTermEventBase):
    """
    This class detects separation and will terminate integration when it occurs.
    
    This is a callable object that the ODE integrator will use to determine if
    the integration should terminate before the end location.
    
    Attributes
    ----------
        _x_kill: x-location that the integrator should stop.
    """
    def __init__(self, calc_lam, S_fun):
        super().__init__()
        self._calc_lam = calc_lam
        self._S_fun = S_fun
    
    def _call_impl(self, x, delta_m2_on_nu):
        """
        Information used to determine if Thwaites method integrator should 
        terminate.
        
        This will terminate once the shear function goes negative.
        
        Args
        ----
            x: Current x-location of the integration
            delta_m2_on_nu: Current step square of momentum thickness divided
                            by the kinematic viscosity
        
        Returns
        -------
            Current value of the shear function.
        """
        return self._S_fun(self._calc_lam(x, delta_m2_on_nu))
    
    def event_info(self):
        return -1, ""


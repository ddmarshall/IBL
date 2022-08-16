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


#class ThwaitesSimData(IBLSimData):
#    def __init__(self,
#                 x_vec,
#                 u_e_vec,
#                 u_inf,
#                 nu,
#                 re,
#                 x0,
#                 theta0=None,
#                 s=None,
#                 h=None,
#                 hp=None,
#                 linearize=False):
#        super().__init__(x_vec,
#                         u_e_vec,
#                         u_inf,
#                         nu)
#        self.x0 = x0
#        self.theta0 = theta0
#        self.re = re
#        #these go through the setters
#        self.s_lam = s
#        self.h_lam = h
#        self.hp_lam = hp
#        self._linearize=linearize
#
#
#
#    h_lam = property(fget=lambda self: self._h,
#                 fset=lambda self, f: setattr(self,
#                                              '_h',
#                                              _function_of_lambda_property_setter(f)))
#
#    hp_lam = property(fget=lambda self: self._hp,
#                 fset=lambda self, f: setattr(self,
#                                              '_hp',
#                                              _function_of_lambda_property_setter(f)))
#
#    s_lam = property(fget=lambda self: self._s,
#                 fset=lambda self, f: setattr(self,
#                                              '_s',
#                                              _function_of_lambda_property_setter(f)))
#
#    re = property(fget=lambda self: self._re,
#                  fset=lambda self, new_re: setattr(self,
#                                                    '_re',
#                                                    new_re))

                                                                           
  
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
        _S_fun: Shear function used in algorithm
        _H_fun: Shape function used in algorithm
        _Hp_fun: Derivative of the Shape function

        _tab_lam: Original tabular data for lambda from Thwaites
        _tab_S: Original tabular data for S from Thwaites
        _tab_H: Original tabular data for H from Thwaites
        _tab_F: Original tabular data for F=2(S-\lambda(2+H)) from Thwaites
        _S_lam_spline: Cubic spline through Thwaites tabular data for S
        _H_lam_spline: Cubic spline through Thwaites tabular data for H
        _Hp_lam_spline: Derivative of the cubic spline through H
        
    """
    def __init__(self, U_e = None, dU_edx = None, d2U_edx2 = None,
                 data_fits="Thwaites"):
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
        S(\lambda), and the shape function, H(\lambda).
        
        Args
        ----
            data_fits: either a 2-tuple of functions that each take one 
                       parameter, lambda where the first is the shear 
                       function and the second is the shape function; or 
                       a string for representing one of the three internal 
                       implementations:
                           * "Thwaites" - Spline fits of Thwaites original 
                             data (Edland 2022)
                           * "White" - Curve fits from White (2011)
                           * "Cebeci-Bradshaw" - Curve fits from 
                             Cebeci-Bradshaw 1977"
        
        Throws
        ------
            ValueError if an invalid fit name or unusable 2-tuple provided
        """
        # data_fits can either be string or 2-tuple of callables
        self._H_fun = None
        self._Hp_fun = None
        self._S_fun = None
        if type(data_fits) is str:
            if data_fits == "Thwaites":
                self._H_fun = self._spline_H
                self._Hp_fun = self._spline_Hp
                self._S_fun = self._spline_S
            elif data_fits == "White":
                self._H_fun = self._white_H
                self._Hp_fun = self._white_Hp
                self._S_fun = self._white_S
            elif data_fits == "Cebeci-Bradshaw":
                self._H_fun = self._cb_H
                self._Hp_fun = self._cb_Hp
                self._S_fun = self._cb_S
            else:
                raise ValueError("Unknown fitting function name: ", data_fits)
        else:
            # check to make sure have two callables
            if (type(data_fits) is tuple):
                if len(data_fits)==3:
                    if callable(data_fits[0]) and callable(data_fits[1]) \
                            and callable(data_fits[2]):
                        self._S_fun = data_fits[0]
                        self._H_fun = data_fits[1]
                        self._Hp_fun = data_fits[2]
                    else:
                        raise ValueError("Need to pass callable objects for "
                                         "fit functions")
                elif len(data_fits)==2:
                    if callable(data_fits[0]) and callable(data_fits[1]):
                        self._S_fun = data_fits[0]
                        self._H_fun = data_fits[1]
                        self._Hp_fun = lambda lam: fd(self._H_fun, lam, 1e-5, 
                                                      n=1, order=3)
                    else:
                        raise ValueError("Need to pass callable objects for "
                                         "fit functions")
                else:
                    raise ValueError("Need to pass two or three callable "
                                     "objects for fit functions")
            else:
                raise ValueError("Need to pass a 2-tuple for fit functions")
        
        self._set_kill_event(_ThwaitesSeparationEvent(self._calc_lambda,
                                                      self._S_fun))
    
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
        term3 = 0.5*U_e*self.H(x)*dsol_dx
        term4 = (U_e*delta_m2_on_nu
                *self._Hp_fun(self._calc_lambda(x,delta_m2_on_nu)))
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
        return self.delta_m(x)*self.H(x)
    
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
    
    def H(self, x):
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
        return self._H_fun(lam)
    
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
        return rho*self._nu*self.U_e(x)*self._S_fun(lam)/self.delta_m(x)
    
#    def lam(self,x):
#        return (np.transpose(self.y(x))[0,:] *self.du_edx(x) /self.nu)
#    #     self._lam_vec = (pow(self._theta_vec, 2) *np.gradient(thwaites_sim_data.u_e, thwaites_sim_data.x) /thwaites_sim_data.nu)
    
#    #     self._h_vec = np.array([thwaites_sim_data.h(lam) for lam in self._lam_vec],dtype=np.float)
#    #     self._s_vec = np.array([thwaites_sim_data.s(lam) for lam in self._lam_vec],dtype=np.float)
    
#    def dhdx(self,x):
#        #h function (not shape factor) as a function of x (simulation completed)
#        return np.array([self.h_lam(lam) for lam in self.lam(x)])
    
#    def rtheta(self,x):
#        return self.u_e(x)*self.theta(x)/self.nu  
    
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
    _tab_lam = np.array([-0.082,-0.0818,-0.0816,-0.0812,-0.0808,-0.0804,-0.080,
                         -0.079,-0.078, -0.076, -0.074, -0.072, -0.070, -0.068,
                         -0.064,-0.060, -0.056, -0.052, -0.048, -0.040, -0.032,
                         -0.024,-0.016, -0.008,  0.000,  0.016,  0.032,  0.048,
                          0.064,  0.080,  0.10,  0.12,   0.14,   0.20,   0.25])
    
    @staticmethod
    def _tabular_range():
        """Returns the minimum and maximum lambda for tabular data"""
        return (np.amin(ThwaitesMethod._tab_lam), np.amax(ThwaitesMethod._tab_lam))

    @staticmethod
    def _tabular_data():
        """Returns the tabulated values of F from Thwaites' Table I of 1949 paper"""
        return ThwaitesMethod._tab_lam, ThwaitesMethod._tab_H, ThwaitesMethod._tab_S, ThwaitesMethod._tab_F

    @staticmethod
    def _white_range():
        """Returns the minimum and maximum lambda for White from 2011 book"""
        return (-0.09, np.inf)
    
    # Whites curve fits section
    @staticmethod
    def _white_S(lam):
        """
        Returns White's calculation of S from 2011 book
        
        Args
        ----
            lam: lambda parameter(s) at which evaluation is wanted
            
        Returns
        -------
            Returns the S(\lambda) value(s)
        
        Throws
        ------
            ValueError if an invalid fit name or unusable 2-tuple provided
        """
        # case when lambda is out of range
        lam_min, lam_max = ThwaitesMethod._white_range()
        
        if (np.array(lam) < lam_min).any():
            raise ValueError('cannot pass value less than {} into this function'.format(lam_min))
        elif (np.array(lam) > lam_max).any():
            raise ValueError('cannot pass value greater than {} into this function'.format(lam_max))

        return pow(lam + 0.09, 0.62)
    
    @staticmethod
    def _white_H(lam):
        """
        Returns White's calculation of H from 2011 book
        
        Args
        ----
            lam: lambda parameter(s) at which evaluation is wanted
            
        Returns
        -------
            Returns the H(\lambda) value(s)
        
        Throws
        ------
            ValueError if an invalid fit name or unusable 2-tuple provided
        """
        # case when lambda is out of range
        lam_min, lam_max = ThwaitesMethod._white_range()
        
        if (np.array(lam) < lam_min).any():
            raise ValueError('cannot pass value less than {} into this function'.format(lam_min))
        elif (np.array(lam) > lam_max).any():
            raise ValueError('cannot pass value greater than {} into this function'.format(lam_max))

        z = 0.25-lam
        return 2 + z*(4.14 + z*(-83.5 + z*(854 + z*(-3337 + z*4576))))
    
    @staticmethod
    def _white_Hp(lam):
        """
        Returns the derivative of White's calculation of H from 2011 book
        
        Args
        ----
            lam: lambda parameter(s) at which evaluation is wanted
            
        Returns
        -------
            Returns the dH/d\lambda value(s)
        
        Throws
        ------
            ValueError if an invalid fit name or unusable 2-tuple provided
        """
        # case when lambda is out of range
        lam_min, lam_max = ThwaitesMethod._white_range()
        
        if (np.array(lam) < lam_min).any():
            raise ValueError('cannot pass value less than {} into this function'.format(lam_min))
        elif (np.array(lam) > lam_max).any():
            raise ValueError('cannot pass value greater than {} into this function'.format(lam_max))

        z = 0.25-lam
        return -(4.14 + z*(-2*83.5 + z*(3*854 + z*(-4*3337 + z*5*4576))))

    # Cebeci & Bradshaw curve fits section
    @staticmethod
    def _cb_range():
        """Returns the minimum and maximum lambda for Cebeci and Bradshaw from 1977 book"""
        return (-0.1, 0.1)
    
    @staticmethod
    def _cb_S(lam):
        """
        Returns Cebeci and Bradshaw's calculation of S from 1977 book
        
        Args
        ----
            lam: lambda parameter(s) at which evaluation is wanted
            
        Returns
        -------
            Returns the S(\lambda) value(s)
        
        Throws
        ------
            ValueError if an invalid fit name or unusable 2-tuple provided
        """
        # case when lambda is out of range
        lam_min, lam_max = ThwaitesMethod._cb_range()
        
        if (np.array(lam) < lam_min).any():
            raise ValueError('cannot pass value less than {} into this function'.format(lam_min))
        elif (np.array(lam) > lam_max).any():
            raise ValueError('cannot pass value greater than {} into this function'.format(lam_max))
        
        # case when lambda fits first interval
        return np.piecewise(lam, [lam<0, lam>=0],
                            [lambda lam: 0.22 + 1.402*lam + 0.018*lam/(0.107 + lam),
                             lambda lam: 0.22 + 1.57*lam - 1.8*lam**2])
    
    @staticmethod
    def _cb_H(lam):
        """
        Returns Cebeci and Bradshaw's calculation of H from 1977 book
        
        Args
        ----
            lam: lambda parameter(s) at which evaluation is wanted
            
        Returns
        -------
            Returns the H(\lambda) value(s)
        
        Throws
        ------
            ValueError if an invalid fit name or unusable 2-tuple provided
        """
        # case when lambda is out of range
        lam_min, lam_max = ThwaitesMethod._cb_range()
        
        if (np.array(lam) < lam_min).any():
            raise ValueError('cannot pass value less than {} into this function'.format(lam_min))
        elif (np.array(lam) > lam_max).any():
            raise ValueError('cannot pass value greater than {} into this function'.format(lam_max))
        

        # case when lambda fits first interval
        # NOTE: C&B's H function is not continuous at lam=0, so using second interval
        return np.piecewise(lam, [lam<0, lam>=0],
                            [lambda lam: 2.088 + 0.0731/(0.14 + lam),
                             lambda lam: 2.61 - 3.75*lam + 5.24*lam**2])
    
    @staticmethod
    def _cb_Hp(lam):
        """
        Returns the derivative of Cebeci and Bradshaw's calculation of H from 1977 book
        
        Args
        ----
            lam: lambda parameter(s) at which evaluation is wanted
            
        Returns
        -------
            Returns the dH/d\lambda value(s)
        
        Throws
        ------
            ValueError if an invalid fit name or unusable 2-tuple provided
        """
        # case when lambda is out of range
        lam_min, lam_max = ThwaitesMethod._cb_range()
        
        if (np.array(lam) < lam_min).any():
            raise ValueError('cannot pass value less than -0.1 into this function')
        elif (np.array(lam) > lam_max).any():
            raise ValueError('cannot pass value greater than 0.1 into this function')
        

        # case when lambda fits first interval
        # NOTE: C&B's H function is not continuous at lam=0, so using second interval
        return np.piecewise(lam, [lam<0, lam>=0],
                            [lambda lam: -0.0731/(0.14 + lam)**2,
                             lambda lam: -3.75 + 2*5.24*lam])

    # Spline fits to Thwaites original data Edland
    _S_lam_spline = CubicSpline(_tab_lam, _tab_S)
    _H_lam_spline = CubicSpline(_tab_lam, _tab_H)
    _Hp_lam_spline = _H_lam_spline.derivative()
    
    @staticmethod
    def _spline_range():
        return ThwaitesMethod._tabular_range()
    
    @staticmethod
    def _spline_S(lam):
        """
        Returns the spline of S from Edland 2021
        
        Args
        ----
            lam: lambda parameter(s) at which evaluation is wanted
            
        Returns
        -------
            Returns the S(\lambda) value(s)
        
        Throws
        ------
            ValueError if an invalid fit name or unusable 2-tuple provided
        """
        # case when lambda is out of range
        lam_min, lam_max = ThwaitesMethod._spline_range()
        
        if (np.array(lam) < lam_min).any():
            raise ValueError('cannot pass value less than {} into this function'.format(lam_min))
        elif (np.array(lam) > lam_max).any():
            raise ValueError('cannot pass value greater than {} into this function'.format(lam_max))

        return ThwaitesMethod._S_lam_spline(lam)

    @staticmethod
    def _spline_H(lam):
        """
        Returns the spline of H from Edland 2021
        
        Args
        ----
            lam: lambda parameter(s) at which evaluation is wanted
            
        Returns
        -------
            Returns the H(\lambda) value(s)
        
        Throws
        ------
            ValueError if an invalid fit name or unusable 2-tuple provided
        """
        # case when lambda is out of range
        lam_min, lam_max = ThwaitesMethod._spline_range()
        
        if (np.array(lam) < lam_min).any():
            raise ValueError('cannot pass value less than {} into this function'.format(lam_min))
        elif (np.array(lam) > lam_max).any():
            raise ValueError('cannot pass value greater than {} into this function'.format(lam_max))

        return ThwaitesMethod._H_lam_spline(lam)

    @staticmethod
    def _spline_Hp(lam):
        """
        Returns the derivative of the spline of H from Edland 2021
        
        Args
        ----
            lam: lambda parameter(s) at which evaluation is wanted
            
        Returns
        -------
            Returns the dH/d\lambda value(s)
        
        Throws
        ------
            ValueError if an invalid fit name or unusable 2-tuple provided
        """
        # case when lambda is out of range
        lam_min, lam_max = ThwaitesMethod._spline_range()
        
        if (np.array(lam) < lam_min).any():
            raise ValueError('cannot pass value less than {} into this function'.format(lam_min))
        elif (np.array(lam) > lam_max).any():
            raise ValueError('cannot pass value greater than {} into this function'.format(lam_max))

        return ThwaitesMethod._Hp_lam_spline(lam)


#class ThwaitesSeparation(SeparationModel):
#    def __init__(self,thwaitessim,buffer=0):
#        def lambda_difference(thwaitessim,x=None):
#            if type(x)!=np.ndarray and x ==None:
#                x = thwaitessim.x_vec
#            return -thwaitessim.lam(x)-.0842 # @ -.0842, separation
#        super().__init__(thwaitessim,lambda_difference,buffer)

        
        
#Thwaites Default Functions       
#def _function_of_lambda_property_setter(f):
#    try:
#        sig = inspect.signature(f)
#    except Exception as e:
#        print('Must be a function of lambda.')
#        print(e)
#    else:
#        if len(sig.parameters) != 1:
#            raise Exception('Must take one argument, lambda.')
#        else:
#            return f


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


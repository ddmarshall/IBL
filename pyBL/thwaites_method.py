#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:42:04 2022

@author: ddmarsha
"""

import numpy as np
from scipy.interpolate import CubicSpline
import inspect    # used to return source code of h,s

from pyBL.ibl_base import IBLBase


def _stagnation_y0(iblsimdata,x0):
    #From Moran
      return .075*iblsimdata.nu/iblsimdata.du_edx(x0)


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
    
    This class solves for delta_m^2 use the IBLBase ODE solver using the linear
    approximation for the differential equation relationship .
    
    Attributes
    ----------
        _tab_lam: Original tabular data for lambda from Thwaites
        _tab_S: Original tabular data for S from Thwaites
        _tab_H: Original tabular data for H from Thwaites
        _tab_F: Original tabular data for F=2(S-\lambda(2+H)) from Thwaites
        _S_lam_spline: Cubic spline through Thwaites tabular data for S
        _H_lam_spline: Cubic spline through Thwaites tabular data for H
        _Hp_lam_spline: Derivative of the cubic spline through H
        
    """
    def __init__(self, U_e = None, dU_edx = None, d2U_edx2 = None, data_fits="Thwaites"):
        super().__init__(U_e, dU_edx, d2U_edx2)
        self.set_data_fits(data_fits)
#        #note - f's(lambda) aren't actually used in solver
#        self.u_e = thwaites_sim_data.u_e #f(x)
#        self.u_inf = thwaites_sim_data.u_inf
#        self.re = thwaites_sim_data.re
#        self.x0 = thwaites_sim_data.x0
#        self.theta0 = thwaites_sim_data.theta0
#        self.s_lam = thwaites_sim_data.s_lam
#        self.h_lam = thwaites_sim_data.h_lam
#        self.hp_lam = thwaites_sim_data.hp_lam
#        self.nu = thwaites_sim_data.nu
#        
#        self._x_tr = None
    
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
        self._S_fun = None
        if type(data_fits) is str:
            if data_fits == "Thwaites":
                self._H_fun = self._spline_H
                self._S_fun = self._spline_S
            elif data_fits == "White":
                self._H_fun = self._white_H
                self._S_fun = self._white_S
            elif data_fits == "Cebeci-Bradshaw":
                self._H_fun = self._cb_H
                self._S_fun = self._cb_S
            else:
                raise ValueError("Unknown fitting function name: ", data_fits)
        else:
            # check to make sure have two callables
            if (type(data_fits) is tuple) and (len(data_fits)==2):
                if callable(data_fits[0]) and callable(data_fits[1]):
                    self._H_fun = data_fits[1]
                    self._S_fun = data_fits[0]
                else:
                    raise ValueError("Need to pass callable objects for fit "
                                     "functions")
            else:
                raise ValueError("Need to pass a 2-tuple for fit functions")
            
#        def derivatives(t,y):
#            #modified derivatives to use s and h, define y as theta^2
#            x=t
#            lam = np.clip(y*thwaites_sim_data.du_edx(x)/self.nu, -0.5, 0.5)
#            # if (lam<= (-0.0842)):
#            #     lam =np.array([(-0.0842)])
#            if abs(self.u_e(x))<np.array([1E-8]):
#                return np.array([1E-8])
#            else:
#                #Check whether to assume .45-6lam for 2(s-(2+H)*lam)
#                if thwaites_sim_data._linearize==True:
#                    return np.array([self.nu*(.45-6*lam)/self.u_e(x)])
#                else:
#                    return np.array([2*self.nu*(self.s_lam(lam)-(2+self.h_lam(lam))*lam)/self.u_e(x)])

#        #Probably user changeable eventually
#        #self.x0 = thwaites_sim_data.x_vec[0]
#        #self.x0=x0
#        #self.y0 = np.array([5*pow(thwaites_sim_data.u_e(self.x0),4)])
#        if self.theta0 is not None:
#            self.y0 = np.array([pow(self.theta0,2)])
#        else:
#            self.y0 = [_stagnation_y0(thwaites_sim_data,self.x0)]
#        self.x_bound = thwaites_sim_data.x_vec[-1] 
#        
#        super().__init__(thwaites_sim_data, derivatives, self.x0, self.y0, self.x_bound)
#        self.u_e = thwaites_sim_data.u_e
    
#    def theta(self,x):
#        #momentum thickness
#        #return pow(self.eq5_6_16(x)*self.nu, .5)
#        return np.sqrt(np.transpose(self.y(x))[0,:])
#    #     self._theta_vec = pow(self._eq5_6_16_vec * thwaites_sim_data.nu, .5)
#    def lam(self,x):
#        return (np.transpose(self.y(x))[0,:] *self.du_edx(x) /self.nu)
#    #     self._lam_vec = (pow(self._theta_vec, 2) *np.gradient(thwaites_sim_data.u_e, thwaites_sim_data.x) /thwaites_sim_data.nu)
    
#    #     self._h_vec = np.array([thwaites_sim_data.h(lam) for lam in self._lam_vec],dtype=np.float)
#    #     self._s_vec = np.array([thwaites_sim_data.s(lam) for lam in self._lam_vec],dtype=np.float)
#    def h(self,x):
#        #h function (not shape factor) as a function of x (simulation completed)
#        return np.array([self.h_lam(lam) for lam in self.lam(x)])
    
#    def dhdx(self,x):
#        #h function (not shape factor) as a function of x (simulation completed)
#        return np.array([self.h_lam(lam) for lam in self.lam(x)])
    
#    def s(self,x):
#        #s as a function of x (simulation completed)
#        return np.array([self.s_lam(lam) for lam in self.lam(x)])
    
#    def c_f(self,x):
#        #q - scalar
#        #skin friction
#        #return 2 *self.nu*self.s(self.lam(x))/(self.u_e(x)*self.theta(x))
#        return 2 *self.nu*self.s(x) / (self.u_e(x)*self.theta(x))
#        #gives s a single value at a time
#        #return 2 *self.nu*np.array([self.s(lam) for lam in self.lam(x)]) / (self.u_e(x)*self.theta(x))
#    #     self._cf_vec = (2 *
#    #                     thwaites_sim_data.nu *
#    #                     self._s_vec /
#    #                     (thwaites_sim_data.u_e *
#    #                     self._theta_vec))
#    #     self._del_star_vec = self._h_vec*self._theta_vec
    
#    def del_star(self,x):
#        return self.h(x)*self.theta(x)
#        #return np.array([self.h(lam) for lam in self.lam(x)])*self.theta(x)
#    #     self._wall_shear_vec = (thwaites_sim_data.nu * 
#    #                             self._s_vec * 
#    #                             pow(thwaites_sim_data.u_e / 
#    #                                 thwaites_sim_data.u_inf, 
#    #                                 2) / 
#    #                             (thwaites_sim_data.u_e*self._theta_vec))
    
#    def rtheta(self,x):
#        return self.u_e(x)*self.theta(x)/self.nu  
    
#    def Un(self, x):
#        theta2 = np.transpose(self.y(x))[0,:]
#        return (self.du_edx(x)*self.del_star(x)
#               + 0.5*self.u_e(x)*self.h(x)*self.up(x)[:,0]/self.theta(x)
#               + (self.u_e(x)*self.theta(x).self.hp_lam(x)/self.nu)*(self.up(x)[:,0]*self.du_edx(x)+theta2*self.d2u_edx2(x)))
    
    def _ode_impl(self, x, delta_m2):
        """
        This is the right-hand-side of the ODE representing Thwaites method.
        
        Args
        ----
            x: x-location of current step
            delta_m2: current step square of momentum thickness
        """
        a = 0.45
        b = 6
        lam = self._calc_lambda(x, delta_m2)
        F = a-b*lam
        
        return self._nu*F/self.U_e(x)
    
    def _calc_lambda(self, x, delta_m2):
        return delta_m2*self.dU_edx(x)/self.nu
    
    def U_n(self, x):
        raise NotImplementedError("Need to implement this")
        return np.zeros_like(x)
    
    def delta_d(self, x):
        raise NotImplementedError("Need to implement this")
        return np.zeros_like(x)
    
    def delta_m(self, x):
        raise NotImplementedError("Need to implement this")
        return np.zeros_like(x)
    
    def H(self, x):
        raise NotImplementedError("Need to implement this")
        return np.zeros_like(x)
    
    def tau_w(self, x):
        raise NotImplementedError("Need to implement this")
        return np.zeros_like(x)
    
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
def _function_of_lambda_property_setter(f):
    try:
        sig = inspect.signature(f)
    except Exception as e:
        print('Must be a function of lambda.')
        print(e)
    else:
        if len(sig.parameters) != 1:
            raise Exception('Must take one argument, lambda.')
        else:
            return f


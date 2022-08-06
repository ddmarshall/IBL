#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:42:04 2022

@author: ddmarsha
"""

import numpy as np
from scipy.interpolate import CubicSpline
import inspect    # used to return source code of h,s

from pyBL.pyBL_base import IBLSimData, IBLSim, SeparationModel

    
#bringing in thwaites' tabulated values
f_tab = np.array([.938,.953,.956,.962,.967,.969,.971,.970,.963,.952,.936,.919,.902,.886,.854,.825,.797,.770,.744,.691,.640,.590,.539,.490,.440,.342,.249,.156,.064,-.028,-.138,-.251,-.362,-.702,-1])
m_tab = np.array([.082,.0818,.0816,.0812,.0808,.0804,.08,.079,.078,.076,.074,.072,.07,.068,.064,.06,.056,.052,.048,.04,.032,.024,.016,.008,0,-0.016,-.032,-.048,-.064,-.08,-.1,-.12,-.14,-.2,-.25])
s_tab = np.array([0,.011,.016,.024,.03,.035,.039,.049,.055,.067,.076,.083,.089,.094,.104,.113,.122,.13,.138,.153,.168,.182,.195,.208,.22,.244,.268,.291,.313,.333,.359,.382,.404,.463,.5])
h_tab = np.array([3.7,3.69,3.66,3.63,3.61,3.59,3.58,3.52,3.47,3.38,3.3,3.23,3.17,3.13,3.05,2.99,2.94,2.9,2.87,2.81,2.75,2.71,2.67,2.64,2.61,2.55,2.49,2.44,2.39,2.34,2.28,2.23,2.18,2.07,2])

lam_tab = -m_tab

s_lam_spline = CubicSpline(lam_tab, s_tab)
h_lam_spline = CubicSpline(lam_tab, h_tab)
hp_lam_spline = h_lam_spline.derivative()

#redundant (f can be calculated from other values):
f_lam_spline = CubicSpline(lam_tab,f_tab)

def spline_h(lam):
    return h_lam_spline(lam)

def spline_hp(lam):
    return hp_lam_spline(lam)

def spline_s(lam):
    return s_lam_spline(lam)

def cebeci_s(lam):
    try:
        lam = min([max([lam,-.1]),.1]) #gives back lambda at endpoints of fit (if outside fit)
    except:
        pass
    if lam >= 0 and lam <= .1:
        return .22 + 1.57 * lam - 1.8 * pow(lam, 2)
    elif lam >= -.1 and lam <= 0:
        return .22 + 1.402 * lam + (.018 * lam) / (.107 + lam)
    else:
        return np.nan #pass  # I'll deal with this later

def cebeci_h(lam):
    try:
        
        lam = min([max([lam,-.1]),.1]) #gives back lambda at endpoints of fit (if outside fit)
    except:
        pass
    if lam >= 0 and lam <= .1:
        return 2.61-3.75*lam+5.24*pow(lam, 2)
    elif lam >= -.1 and lam <= 0:
        return (.0731)/(.14+lam) + 2.088
    else:
        return np.nan #pass  # Returned if lambda fails > or <

# TODO: Implement this
def cebeci_hp(lam):
    return 0*lam

def white_s(lam):
    return pow(lam+.09,.62)
    
def white_h(lam):
    z = .25-lam
    return 2+4.14*z-83.5*pow(z,2) +854*pow(z,3) -3337*pow(z,4) +4576*pow(z,5)

# TODO: Implement this
def white_hp(lam):
    return 0*lam

def _stagnation_y0(iblsimdata,x0):
    #From Moran
      return .075*iblsimdata.nu/iblsimdata.du_edx(x0)


class ThwaitesSimData(IBLSimData):
    def __init__(self,
                 x_vec,
                 u_e_vec,
                 u_inf,
                 nu,
                 re,
                 x0,
                 theta0=None,
                 s=spline_s,
                 h=spline_h,
                 hp=spline_hp,
                 linearize=False):
        super().__init__(x_vec,
                         u_e_vec,
                         u_inf,
                         nu)
        self.x0 = x0
        self.theta0 = theta0
        self.re = re
        #these go through the setters
        self.s_lam = s
        self.h_lam = h
        self.hp_lam = hp
        self._linearize=linearize



    h_lam = property(fget=lambda self: self._h,
                 fset=lambda self, f: setattr(self,
                                              '_h',
                                              _function_of_lambda_property_setter(f)))

    hp_lam = property(fget=lambda self: self._hp,
                 fset=lambda self, f: setattr(self,
                                              '_hp',
                                              _function_of_lambda_property_setter(f)))

    s_lam = property(fget=lambda self: self._s,
                 fset=lambda self, f: setattr(self,
                                              '_s',
                                              _function_of_lambda_property_setter(f)))

    re = property(fget=lambda self: self._re,
                  fset=lambda self, new_re: setattr(self,
                                                    '_re',
                                                    new_re))

                                                                           
  
class ThwaitesSim(IBLSim):
    def __init__(self, thwaites_sim_data):
        #note - f's(lambda) aren't actually used in solver
        self.u_e = thwaites_sim_data.u_e #f(x)
        self.u_inf = thwaites_sim_data.u_inf
        self.re = thwaites_sim_data.re
        self.x0 = thwaites_sim_data.x0
        self.theta0 = thwaites_sim_data.theta0
        self.s_lam = thwaites_sim_data.s_lam
        self.h_lam = thwaites_sim_data.h_lam
        self.hp_lam = thwaites_sim_data.hp_lam
        self.nu = thwaites_sim_data.nu
#        self.char_length = thwaites_sim_data.char_length
        
        self._x_tr = None 
        def derivatives(t,y):
            #modified derivatives to use s and h, define y as theta^2
            x=t
            lam = np.clip(y*thwaites_sim_data.du_edx(x)/self.nu, -0.5, 0.5)
            # if (lam<= (-0.0842)):
            #     lam =np.array([(-0.0842)])
            if abs(self.u_e(x))<np.array([1E-8]):
                return np.array([1E-8])
            else:
                #Check whether to assume .45-6lam for 2(s-(2+H)*lam)
                if thwaites_sim_data._linearize==True:
                    return np.array([self.nu*(.45-6*lam)/self.u_e(x)])
                else:
                    return np.array([2*self.nu*(self.s_lam(lam)-(2+self.h_lam(lam))*lam)/self.u_e(x)])

        #Probably user changeable eventually
        #self.x0 = thwaites_sim_data.x_vec[0]
        #self.x0=x0
        #self.y0 = np.array([5*pow(thwaites_sim_data.u_e(self.x0),4)])
        if self.theta0 is not None:
            self.y0 = np.array([pow(self.theta0,2)])
        else:
            self.y0 = [_stagnation_y0(thwaites_sim_data,self.x0)]
        self.x_bound = thwaites_sim_data.x_vec[-1] 
        
        super().__init__(thwaites_sim_data, derivatives, self.x0, self.y0, self.x_bound)
        self.u_e = thwaites_sim_data.u_e

    

    
    #def u_e_star(self,x):
        #return self.u_e(x)/self.u_inf
        #u_e_star_series = pd.Series(thwaites_sim_data.u_e) / thwaites_sim_data.u_inf
    #def x_star(self,x):
        #return x/thwaites_sim_data.char_length
    
        #x_star_series = pd.Series(thwaites_sim_data.x) / thwaites_sim_data.char_length
        #with warnings.catch_warnings():
            #warnings.simplefilter("ignore")
        
       # integrand_vec = pow(thwaites_sim_data.u_e,5)
       # sim = RK45(fun=derivatives, t0=x0 , y0=5*,t_bound = self.x[-1],max_step=.1)
       # integral_vec = cumtrapz(integrand_vec, thwaites_sim_data.x, initial=0)
    #def eq5_6_16(self,x):
        #return np.nan_to_num((.45 / pow(self.u_e(x),6))*np.transpose(self.y(x))[0,:]) #back down to (m,) array
      
    #     self._eq5_6_16_vec  = (.45 / pow(thwaites_sim_data.u_e, 6)) *integral_vec
    def theta(self,x):
        #momentum thickness
        #return pow(self.eq5_6_16(x)*self.nu, .5)
        return np.sqrt(np.transpose(self.y(x))[0,:])
    #     self._theta_vec = pow(self._eq5_6_16_vec * thwaites_sim_data.nu, .5)
    def lam(self,x):
        return (np.transpose(self.y(x))[0,:] *self.du_edx(x) /self.nu)
    #     self._lam_vec = (pow(self._theta_vec, 2) *np.gradient(thwaites_sim_data.u_e, thwaites_sim_data.x) /thwaites_sim_data.nu)
    
    #     self._h_vec = np.array([thwaites_sim_data.h(lam) for lam in self._lam_vec],dtype=np.float)
    #     self._s_vec = np.array([thwaites_sim_data.s(lam) for lam in self._lam_vec],dtype=np.float)
    def h(self,x):
        #h function (not shape factor) as a function of x (simulation completed)
        return np.array([self.h_lam(lam) for lam in self.lam(x)])
        
    def dhdx(self,x):
        #h function (not shape factor) as a function of x (simulation completed)
        return np.array([self.h_lam(lam) for lam in self.lam(x)])
        
    def s(self,x):
        #s as a function of x (simulation completed)
        return np.array([self.s_lam(lam) for lam in self.lam(x)])
    
    def c_f(self,x):
        #q - scalar
        #skin friction
        #return 2 *self.nu*self.s(self.lam(x))/(self.u_e(x)*self.theta(x))
        return 2 *self.nu*self.s(x) / (self.u_e(x)*self.theta(x))
        #gives s a single value at a time
        #return 2 *self.nu*np.array([self.s(lam) for lam in self.lam(x)]) / (self.u_e(x)*self.theta(x))
    #     self._cf_vec = (2 *
    #                     thwaites_sim_data.nu *
    #                     self._s_vec /
    #                     (thwaites_sim_data.u_e *
    #                     self._theta_vec))
    #     self._del_star_vec = self._h_vec*self._theta_vec
    def del_star(self,x):
        return self.h(x)*self.theta(x)
        #return np.array([self.h(lam) for lam in self.lam(x)])*self.theta(x)
    #     self._wall_shear_vec = (thwaites_sim_data.nu * 
    #                             self._s_vec * 
    #                             pow(thwaites_sim_data.u_e / 
    #                                 thwaites_sim_data.u_inf, 
    #                                 2) / 
    #                             (thwaites_sim_data.u_e*self._theta_vec))
    def rtheta(self,x):
        return self.u_e(x)*self.theta(x)/self.nu  
    
    def Un(self, x):
        theta2 = np.transpose(self.y(x))[0,:]
        return (self.du_edx(x)*self.del_star(x)
               + 0.5*self.u_e(x)*self.h(x)*self.up(x)[:,0]/self.theta(x)
               + (self.u_e(x)*self.theta(x).self.hp_lam(x)/self.nu)*(self.up(x)[:,0]*self.du_edx(x)+theta2*self.d2u_edx2(x)))
        
        
class ThwaitesSeparation(SeparationModel):
    def __init__(self,thwaitessim,buffer=0):
        def lambda_difference(thwaitessim,x=None):
            if type(x)!=np.ndarray and x ==None:
                x = thwaitessim.x_vec
            return -thwaitessim.lam(x)-.0842 # @ -.0842, separation
        super().__init__(thwaitessim,lambda_difference,buffer)

        
        
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



# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 18:48:15 2020

@author: Malachi
"""

import inspect    # used to return source code of h,s
import pandas as pd
from scipy.integrate import cumtrapz
import numpy as np
import warnings

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


def _default_s(lam):
    if lam >= 0 and lam <= .1:
        return .22 + 1.57 * lam - 1.8 * pow(lam, 2)
    elif lam >= -.1 and lam <= 0:
        return .22 + 1.402 * lam + (.018 * lam) / (.107 + lam)
    else:
        pass  # I'll deal with this later


def _default_h(lam):
    if lam >= 0 and lam <= .1:
        return 2.61-3.75*lam+5.24*pow(lam, 2)
    elif lam >= -.1 and lam <= 0:
        return (.0731)/(.14+lam) + 2.088
    else:
        pass  # I'll deal with this later

# def _rect_int(x,y):
    

class ThwaitesSimData:
    def __init__(self,
                 x,
                 u_e,
                 u_inf,
                 nu,
                 re,
                 s=_default_s,
                 h=_default_h,
                 char_length=None):
        self.x = x
        self.u_e = u_e
        self.u_inf = u_inf
        self.nu = nu
        self.re = re
        self.s = s
        self.h = h

        if char_length is None:
            self.char_length = 1
        else:
            self.char_length = char_length

    u_e = property(fget=lambda self: self._u_e,
                   fset=lambda self, new_u_e: setattr(self,
                                                      '_u_e',
                                                      new_u_e))

    u_inf = property(fget=lambda self: self._u_inf,
                     fset=lambda self, new_u_inf: setattr(self,
                                                          '_u_inf',
                                                          new_u_inf))
    nu = property(fget=lambda self: self._nu,
                  fset=lambda self, new_nu: setattr(self,
                                                    '_nu',
                                                    new_nu))

    h = property(fget=lambda self: self._h,
                 fset=lambda self, f: setattr(self,
                                              '_h',
                                              _function_of_lambda_property_setter(f)))

    s = property(fget=lambda self: self._s,
                 fset=lambda self, f: setattr(self,
                                              '_s',
                                              _function_of_lambda_property_setter(f)))

    re = property(fget=lambda self: self._re,
                  fset=lambda self, new_re: setattr(self,
                                                    '_re',
                                                    new_re))

    char_length = property(fget=lambda self: self._char_length,
                           fset=lambda self, new_char_length: setattr(self,
                                                                      '_char_length',
                                                                      new_char_length))


class ThwaitesSim:
    def __init__(self, thwaites_sim_data):

        #u_e_star_series = pd.Series(thwaites_sim_data.u_e) / thwaites_sim_data.u_inf
        #x_star_series = pd.Series(thwaites_sim_data.x) / thwaites_sim_data.char_length
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            integrand_vec = pow(thwaites_sim_data.u_e,5)
            integral_vec = cumtrapz(integrand_vec, thwaites_sim_data.x, initial=0)
            
            self._eq5_6_16_vec  = (.45 / pow(thwaites_sim_data.u_e, 6)) *integral_vec
            self._theta_vec = pow(self._eq5_6_16_vec * thwaites_sim_data.nu, .5)
            self._lam_vec = (pow(self._theta_vec, 2) *np.gradient(thwaites_sim_data.u_e, thwaites_sim_data.x) /thwaites_sim_data.nu)
            self._h_vec = np.array([thwaites_sim_data.h(lam) for lam in self._lam_vec],dtype=np.float)
            self._s_vec = np.array([thwaites_sim_data.s(lam) for lam in self._lam_vec],dtype=np.float)
            
            self._cf_vec = (2 *
                            thwaites_sim_data.nu *
                            self._s_vec /
                            (thwaites_sim_data.u_e *
                            self._theta_vec))
            self._del_star_vec = self._h_vec*self._theta_vec
            self._wall_shear_vec = (thwaites_sim_data.nu * 
                                    self._s_vec * 
                                    pow(thwaites_sim_data.u_e / 
                                        thwaites_sim_data.u_inf, 
                                        2) / 
                                    (thwaites_sim_data.u_e*self._theta_vec))
        
    #eq5_6_16_vec = property(fget=lambda self: self._eq5_6_16_vec)
    theta_vec = property(fget=lambda self: self._theta_vec)
    lam_vec = property(fget=lambda self: self._lam_vec)
    h_vec = property(fget=lambda self: self._h_vec)
    s_vec = property(fget=lambda self: self._s_vec)
    cf_vec = property(fget=lambda self: self._cf_vec)
    del_star_vec = property(fget=lambda self: self._del_star_vec)
    wall_shear_vec =  property(fget=lambda self: self._wall_shear_vec)

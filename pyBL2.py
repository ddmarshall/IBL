
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 18:48:15 2020

@author: Malachi
"""

import inspect    # used to return source code of h,s
import pandas as pd
from scipy.integrate import cumtrapz
import numpy as np


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
        return .22+1.57*lam-1.8*pow(lam, 2)
    elif lam >= -.1 and lam <= 0:
        return .22+1.402*lam+(.018*lam)/(.107+lam)
    else:
        pass  # I'll deal with this later


def _default_h(lam):
    if lam >= 0 and lam <= .1:
        return 2.61-3.75*lam+5.24*pow(lam, 2)
    elif lam >= -.1 and lam <= 0:
        return (.0731)/(.14+lam) + 2.088
    else:
        pass  # I'll deal with this later


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
        
        u_e_star_series = pd.Series(thwaites_sim_data.u_e) / thwaites_sim_data.u_inf
        x_star_series = pd.Series(thwaites_sim_data.x) / thwaites_sim_data.char_length

        integrand_series = pd.Series(pow(u_e_star_series,5))
        integral_series = pd.Series(cumtrapz(integrand_series,x_star_series,initial=0))
        
        self._eq5_6_18_series  = (.45 / pow(u_e_star_series, 6)) *integral_series
        self._theta_series = (thwaites_sim_data.char_length *
                              pow(self._eq5_6_18_series / thwaites_sim_data.re,
                                  .5))
        self._lam_series = (pow(self._theta_series, 2) *np.gradient(thwaites_sim_data.u_e, thwaites_sim_data.x) /thwaites_sim_data.nu)
        #self._lam_series = self._eq5_6_18_series*np.gradient(thwaites_sim_data.u_e,thwaites_sim_data.x)
        h_tuple=()
        s_tuple=()
        for lam in self._lam_series:
            h_tuple += (thwaites_sim_data.h(lam),)
            s_tuple += (thwaites_sim_data.s(lam),)
        self._h_series = pd.Series(h_tuple)
        self._s_series = pd.Series(s_tuple)
        self._cf_series = (2 *
                           thwaites_sim_data.nu *
                           self._s_series /
                           (thwaites_sim_data.u_e *
                            self._theta_series))
        self._del_star_series = self._h_series*thwaites_sim_data.char_length*pow(self._eq5_6_18_series,.5)/pow(thwaites_sim_data.re,.5)
        
    eq5_6_18_series = property(fget=lambda self: self._eq5_6_18_series)
    theta_series = property(fget=lambda self: self._theta_series)
    lam_series = property(fget=lambda self: self._lam_series)
    h_series = property(fget=lambda self: self._h_series)
    s_series = property(fget=lambda self: self._s_series)
    cf_series = property(fget=lambda self: self._cf_series)
    del_star_series = property(fget=lambda self: self._del_star_series)

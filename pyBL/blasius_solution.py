#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 18:01:43 2022

@author: ddmarshall
"""

import numpy as np
from scipy.integrate import solve_ivp


class BlasiusSolution:
    def __init__(self, U_ref, nu, fpp0 = None, eta_inf = 10):
        self._U_ref = U_ref
        self._nu = nu
        self._eta_range = [0, eta_inf]
        
        self._set_boundary_condition(fpp0)
        self._set_solution()
    
    def _set_boundary_condition(self, fpp0 = None):
        if fpp0 is None:
            raise NotImplementedError("Not implemented ability to search for "
                                      "boundary condition value")
        else:
            self._fpp0 = fpp0
    
    def _set_solution(self):
        F0 = [0, 0, self._fpp0]
        
        def fun(eta, F):
            Fp = np.zeros_like(F)
            
            Fp[0] = F[1]
            Fp[1] = F[2]
            Fp[2] = -F[0]*F[2]
            
            return Fp
            
        rtn = solve_ivp(fun = fun, t_span = self._eta_range, y0 = F0,
                        method = 'RK45', dense_output = True, events = None,
                        rtol = 1e-8, atol = 1e-11)
        
        self._F = None
        if rtn.success:
            self._F=rtn.sol
        else:
            raise ValueError("Initial condition for solver, f\'\'(0)={0:.6f}, "
                             "did not produce converged solution."
                             "".format(self._fpp0))
    
    def f(self, eta):
        return self._F(eta)[0, :]
    
    def fp(self, eta):
        return self._F(eta)[1, :]
    
    def fpp(self, eta):
        return self._F(eta)[2, :]
    
    def eta_d(self):
        pass
    
    def eta_m(self):
        pass
    
    def eta_k(self):
        pass
    
    def eta_s(self):
        pass
    
    def eta(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return y*np.sqrt(0.5*self._U_ref/(self._nu*x))
    
    def u(self, *args):
        if len(args) == 1:
            eta = args[0]
        elif len(args) == 2:
            eta = self.eta(args[0], args[1])
        else:
            raise ValueError("Can only pass either eta or x,y into this method")
        return self.U_ref*self.fp(eta);
    
    def v(self, *args):
        if len(args) == 1:
            eta = args[0]
        elif len(args) == 2:
            eta = self.eta(args[0], args[1])
        else:
            raise ValueError("Can only pass either eta or x,y into this method")
        pass
    
    def U_e(self, x):
        x = np.asarray(x)
        return self._U_ref*np.ones_like(x)
    
    def V_e(self, x):
        pass
    
    def delta_d(self, x):
        pass
    
    def delta_m(self, x):
        pass
    
    def delta_s(self, x):
        pass
    
    def H(self, x):
        return self.delta_m(x)/self.delta_d(x)
    
    def tau_w(self, x, rho):
        pass

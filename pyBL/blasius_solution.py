#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 18:01:43 2022

@author: ddmarshall
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quadrature
from scipy.optimize import root_scalar


class BlasiusSolution:
    def __init__(self, U_ref, nu, fpp0 = None, eta_inf = 10):
        self._U_ref = U_ref
        self._nu = nu
        self._eta_inf = eta_inf
        
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
            
        rtn = solve_ivp(fun = fun, t_span = [0, self._eta_inf], y0 = F0,
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
        return self._F(eta)[0]
    
    def fp(self, eta):
        return self._F(eta)[1]
    
    def fpp(self, eta):
        return self._F(eta)[2]
    
    def eta_d(self):
        return self._eta_inf-self.f(self._eta_inf)
    
    def eta_m(self):
        return self.fpp(0)
    
    def eta_k(self):
        def fun(eta):
            return 2*np.prod(self._F(eta), axis=0)
        
        val = quadrature(fun, 0, self._eta_inf)
        return val[0]
    
    def eta_s(self):
        def fun(eta):
            return 0.99-self.fp(eta)
        
        sol = root_scalar(fun, method = "bisect", bracket = [0, 10])
        if sol.converged:
            return sol.root
        else:
            raise ValueError("Could not find shear thickness with error: "
                             + sol.flag)
    
    def eta(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return y*np.sqrt(0.5*self._U_ref/(self._nu*x))
    
    def u(self, x, y):
        eta = self.eta(x, y)
        return self._U_ref*self.fp(eta);
    
    def v(self, x, y):
        eta = self.eta(x, y)
        return self._nu*self._g(x)*(eta*self.fp(eta)-self.f(eta))
    
    def U_e(self, x):
        x = np.asarray(x)
        return self._U_ref*np.ones_like(x)
    
    def V_e(self, x):
        return self._nu*self._g(x)*self.eta_d()
    
    def delta_d(self, x):
        return self.eta_d()/self._g(x)
    
    def delta_m(self, x):
        return self.eta_m()/self._g(x)
    
    def delta_k(self, x):
        return self.eta_k()/self._g(x)
    
    def delta_s(self, x):
        return self.eta_s()/self._g(x)
    
    def H_d(self, x):
        x = np.asarray(x)
        return (self.eta_d()/self.eta_m())*np.ones_like(x)
    
    def H_k(self, x):
        x = np.asarray(x)
        return (self.eta_k()/self.eta_m())*np.ones_like(x)
    
    def tau_w(self, x, rho):
        return rho*self._nu*self._U_ref*self._g(x)*self.fpp(0)
    
    def D(self, x, rho):
        return 0.5*rho*self._nu*self._g(x)*self._U_ref**2*self.eta_k()
    
    def _g(self, x):
        return np.sqrt(0.5*self._U_ref/(self._nu*x))

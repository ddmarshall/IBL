#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 01:00:26 2022

@author: ddmarshall
"""


import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quadrature
from scipy.optimize import root_scalar


class FalknerSkanSolution:
    def __init__(self, U_ref, nu, m = 0, eta_inf = 10):
        self._U_ref = U_ref
        self._nu = nu
        self._eta_inf = eta_inf
        
        self._set_boundary_condition(m)
    
    def f(self, eta):
        return self._F(eta)[0]
    
    def fp(self, eta):
        return self._F(eta)[1]
    
    def fpp(self, eta):
        return self._F(eta)[2]
    
    def eta_d(self):
        return self._eta_inf-self.f(self._eta_inf)
    
    def eta_m(self):
        beta = self._beta()
        return (self.fpp(0)-beta*self.eta_d())/(1+beta)
    
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
        return y*self._g(x)
    
    def u(self, x, y):
        eta = self.eta(x, y)
        return self.U_e(x)*self.fp(eta);
    
    def v(self, x, y):
        eta = self.eta(x, y)
        return self._nu*self._g(x)*(eta*self.fp(eta)-self.f(eta))
    
    def U_e(self, x):
        x = np.asarray(x)
        return self._U_ref*x**self._m*np.ones_like(x)
    
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
        return rho*self._nu*self.U_e(x)*self._g(x)*self.fpp(0)
    
    def D(self, x, rho):
        beta = self._beta()
        D_term = (0.5*(1+3*beta)*self.eta_k()
                 +beta*(1+beta)*self.eta_m()+beta**2*self.eta_d())
        return rho*self._nu*self._g(x)*self.U_e(x)**2*D_term
    
    def _beta(self):
        if self._m == np.inf:
            beta = 2
        else:
            beta = 2*self._m/(1+self._m)
        return beta
    
    def _g(self, x):
        return np.sqrt(0.5*(self._m+1)*self.U_e(x)/(self._nu*x))
    
    def _set_boundary_condition(self, m = 0):
        self._m = m
        def fun(fpp0):
            class bc_event:
                self.terminal = True
                def __call__(self, x, F):
                    return F[1] - 1
                
            F0 = [0, 0, fpp0]
            rtn = solve_ivp(fun = self._ode_fun,
                            t_span = [0, self._eta_inf], y0 = F0,
                            method = 'RK45', dense_output = False,
                            events = bc_event(), rtol = 1e-8, atol = 1e-11)
            if not rtn.success:
                raise ValueError("Could not find boundary condition")
            
            # hack to get beta at separation to work
            val = 1-rtn.y[1, -1]
            if (m < 0) and (val > -2e-6) and (val < 0):
                val = 0
            return val
        
        # These ranges were found via trial and error. Is there a more robust
        # way of finding a suitable initial range?
        if m <= -0.905:
            raise ValueError("Value of m is below separation value")
        elif m <= -0.09:
            rng = [0, 0.1]
        elif m <= -0.08:
            rng = [0.1, 0.2]
        elif m <= -0.05:
            rng = [0.2, 0.4]
        elif m <= 0.00:
            rng = [0.4, 0.5]
        elif m <= 0.05:
            rng = [0.5, 0.6]
        elif m <= 0.1:
            rng = [0.6, 0.7]
        elif m <= 0.2:
            rng = [0.7, 0.9]
        elif m <= 0.3:
            rng = [0.9, 1.0]
        elif m <= 0.5:
            rng = [1.0, 1.05]
        elif m <= 0.7:
            rng = [1.05, 1.15]
        elif m <= 0.9:
            rng = [1.15, 1.23]
        elif m <= 1.0:
            rng = [1.23, 1.25]
        elif m <= 1.1:
            rng = [1.25, 1.26]
        elif m <= 1.5:
            rng = [1.26, 1.3]
        elif m <= 1.7:
            rng = [1.3, 138]
        elif m <= 2.0:
            rng = [1.38, 1.41]
        elif m <= 2.5:
            rng = [1.41, 1.45]
        elif m <= 3.0:
            rng = [1.45, 1.48]
        elif m <= 4.0:
            rng = [1.48, 1.525]
        else:
            raise ValueError("Value of m is out of range for this solver")
        sol = root_scalar(fun, method = "bisect", bracket = rng)
        if sol.converged:
            self._fpp0 = sol.root
        else:
            raise ValueError("Root finded could not find boundary "
                             "condition")
        self._set_solution()
    
    def _set_solution(self):
        F0 = [0, 0, self._fpp0]
        
        rtn = solve_ivp(fun = self._ode_fun, t_span = [0, self._eta_inf],
                        y0 = F0, method = 'RK45', dense_output = True,
                        events = None, rtol = 1e-8, atol = 1e-11)
        
        self._F = None
        if rtn.success:
            self._F=rtn.sol
        else:
            raise ValueError("Initial condition for solver, f\'\'(0)={0:.6f}, "
                             "did not produce converged solution."
                             "".format(self._fpp0))
    
    def _ode_fun(self, eta, F):
        Fp = np.zeros_like(F)
        
        Fp[0] = F[1]
        Fp[1] = F[2]
        Fp[2] = -F[0]*F[2]-self._beta()*(1-F[1]**2)
        
        return Fp

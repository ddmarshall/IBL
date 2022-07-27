#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:37:25 2022

@author: ddmarshall
"""

import numpy as np

from pyBL.thwaites_method import spline_h, spline_s

from examples.falkner_skan import falkner_skan

def get_falkner_skan_results(m, U_inf, nu, x):
    """
    Calculates the Falkner-Skan results and returns various boundary layer 
    parameters
    
    Args
    ----
        m: Velocity parameter used to define specific case
        U_inf: Scale of the freestream velocity
        nu: Kinematic viscosity
        x(numpy.array): Array of x-locations along surface to return values
    
    Returns
    -------
        delta_star(numpy.array): Displacement thickness at each location
        theta(numpy.array): Momentum thickness at each location
        cf(numpy.array): Skin friction coefficient at each location
        H(numpy.array): Shape factor at each location
    """
    beta=2*m/(1+m)
    U_e = U_inf*(x**m)
    eta, f0, f1, f2 = falkner_skan(n_points=71,m=m)  # each returned value is a ndarray
    eta_star = eta[-1]-f0[-1]
    theta_star = (f2[0]-beta*eta_star)/(1+beta)
    g = np.sqrt(U_e/((2-beta)*nu*x))
    delta_star = eta_star/g
    theta = theta_star/g
    H = delta_star/theta
    cf=2*nu*g*f2[0]/U_e
    
    return delta_star, theta, cf, H


def get_thwaites_falkner_skan_results(m, U_inf, nu, x):
    
    K = np.sqrt(0.45/(5*m+1))
    Rex_sqrt = np.sqrt(U_inf*x**(m+1)/nu)
    
    if (m==0):
        lam = 0
    else:
        lam = m*K**2
    S_fun = spline_s(lam)
    H_fun = spline_h(lam)
    theta = x*K/Rex_sqrt
    cf = 2*S_fun/(K*Rex_sqrt)
    delta_star = theta*H_fun
    H = delta_star/theta
    
    return delta_star, theta, cf, H


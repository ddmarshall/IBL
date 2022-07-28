#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:40:25 2022

@author: ddmarshall
"""


import numpy as np

from pyBL.thwaites_method import ThwaitesSimData, ThwaitesSim, spline_h, spline_s

from falkner_skan_analysis import get_falkner_skan_results
from falkner_skan_analysis import get_thwaites_falkner_skan_results
from falkner_skan_analysis import plot_falkner_skan_comparison

def axisymmetric_stagnation_flow_case():
    """
    Create the axisymmetric stagnation flow comparison, figures 3.7-3.9 in Edland's thesis
    
    Args
    ----
        None
    
    Returns
    -------
        None
    """

    ## Set the common values for the analysis
    npts_in=201 # number of points used as input for edge velocities
    npts_out=81 # number of points used for results
    c=1 # length of plate
    U_inf=1 # edge velocity at x/c=1 (used as a reference)
    nu_e=1.45e-5 # kinematic viscosity
    m=1/3 # Falkner-Skan parameter specifying the edge velocity profile
    xmin=1e-6 # avoid divide by zero at leading edge

    x_in=(np.logspace(xmin, np.log10(2), npts_in)-1)*c
    x_out=(np.logspace(xmin, np.log10(2), npts_out)-1)*c
#    x_in=np.linspace(xmin, c, npts_in)
#    x_out=np.linspace(xmin, c, npts_out)
    
    ## Get the Blasius solution for comparison
    delta_star_exact, theta_exact, cf_exact, H_exact = get_falkner_skan_results(m=m, U_inf=U_inf, nu=nu_e, x=x_out)
    
    ## Get the Analytic solution for comparison
    delta_star_analytic, theta_analytic, cf_analytic, H_analytic = get_thwaites_falkner_skan_results(m=m, U_inf=U_inf, nu=nu_e, x=x_out)
    
    ## Get the results from Thwaites method
    
    # Set up the simulation parameters
    U_e = U_inf*x_in**m;
    Re_c = U_inf*c/nu_e;
    
    # Get the solution using the linearized function
    tsd_linear = ThwaitesSimData(x_in,U_e,U_inf,nu_e,Re_c,x_in[0],theta_analytic[0],linearize=True, h=spline_h, s=spline_s)
    ts_linear = ThwaitesSim(tsd_linear) 
    while ts_linear.status=='running':
        ts_linear.step()
    delta_star_linear = ts_linear.del_star(x_out)
    theta_linear = ts_linear.theta(x_out)
    cf_linear = ts_linear.c_f(x_out)
    H_linear = ts_linear.h(x_out)
    
    # Get the solution using the nonlinear function
    tsd_nonlinear = ThwaitesSimData(x_in,U_e,U_inf,nu_e,Re_c,x_in[0],theta_analytic[0],linearize=False, h=spline_h, s=spline_s)
    ts_nonlinear = ThwaitesSim(tsd_nonlinear)
    while ts_nonlinear.status=='running':
        ts_nonlinear.step()
    delta_star_nonlinear = ts_nonlinear.del_star(x_out)
    theta_nonlinear = ts_nonlinear.theta(x_out)
    cf_nonlinear = ts_nonlinear.c_f(x_out)
    H_nonlinear = ts_nonlinear.h(x_out)
    
    ## Plot results for figures 3.1, 3.2, and 3.3
    plot_falkner_skan_comparison(x_out, theta_exact, theta_analytic, theta_linear, theta_nonlinear,
                                 delta_star_exact, delta_star_analytic, delta_star_linear, delta_star_nonlinear,
                                 cf_exact, cf_analytic, cf_linear, cf_nonlinear,
                                 H_exact, H_analytic, H_linear, H_nonlinear,
                                 [0.000, 0.002], [0.00, 0.005], [0.0, 0.025], [2.25, 2.45])


if (__name__ == "__main__"):
    axisymmetric_stagnation_flow_case()

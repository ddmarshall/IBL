# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:25:18 2022

@author: ddmarshall
"""

import numpy as np
import matplotlib.pyplot as plt

from pyBL.thwaites_method import ThwaitesSimData, ThwaitesSim

from falkner_skan_analysis import get_falkner_skan_results, get_thwaites_falkner_skan_results

def blasius_case():
    """
    Create the Blasius comparison, figures 3.1-3.3 in Edland's thesis
    
    Args
    ----
        None
    
    Returns
    -------
        None
    """

    ## Set the common values for the analysis
    npts_in=11 # number of points used as input for edge velocities
    npts_out=41 # number of points used for results
    c=1 # length of plate
    U_inf=1 # edge velocity at x/c=1 (used as a reference)
    nu_e=1.45e-5 # kinematic viscosity
    m=0 # Falkner-Skan parameter specifying the edge velocity profile

    x_in=np.linspace(0, c, npts_in)
    x_out=np.linspace(0, c, npts_out)
    x_in[0]=1e-6 # avoid divide by zero at leading edge
    x_out[0]=1e-6 # avoid divide by zero at leading edge
    
    ## Get the Blasius solution for comparison
    delta_star_exact, theta_exact, cf_exact, H_exact = get_falkner_skan_results(m=m, U_inf=U_inf, nu=nu_e, x=x_out)
    
    ## Get the Analytic solution for comparison
    delta_star_analytic, theta_analytic, cf_analytic, H_analytic = get_thwaites_falkner_skan_results(m=m, U_inf=U_inf, nu=nu_e, x=x_out)
    
    ## Get the results from Thwaites method
    
    # Set up the simulation parameters
    U_e = U_inf*x_in**m;
    Re_c = U_inf*c/nu_e;
    
    # Get the solution using the linearized function
    tsd_linear = ThwaitesSimData(x_in,U_e,U_inf,nu_e,Re_c,x_in[0],theta_analytic[0],linearize=True)
    ts_linear = ThwaitesSim(tsd_linear) 
    while ts_linear.status=='running':
        ts_linear.step()
    delta_star_linear = ts_linear.del_star(x_out)
    theta_linear = ts_linear.theta(x_out)
    cf_linear = ts_linear.c_f(x_out)
    H_linear = ts_linear.h(x_out)
    
    # Get the solution using the nonlinear function
    tsd_nonlinear = ThwaitesSimData(x_in,U_e,U_inf,nu_e,Re_c,x_in[0],theta_analytic[0],linearize=False)
    ts_nonlinear = ThwaitesSim(tsd_nonlinear)
    while ts_nonlinear.status=='running':
        ts_nonlinear.step()
    delta_star_nonlinear = ts_nonlinear.del_star(x_out)
    theta_nonlinear = ts_nonlinear.theta(x_out)
    cf_nonlinear = ts_nonlinear.c_f(x_out)
    H_nonlinear = ts_nonlinear.h(x_out)
    
    ## Plot results
    exact_color = 'black'
    exact_label = 'Falkner-Skan'
    analytic_color = 'green'
    analytic_label = 'Thwaites (Analytic)'
    linear_color = 'blue'
    linear_label = 'Thwaites (Standard)'
    nonlinear_color = 'cyan'
    nonlinear_label = 'Thwaites (Improved)'
    theta_label = r'$\theta$'
    delta_star_label = r'$\delta^*$'
    cf_label = r'$c_f$'
    H_label = r'$H$'
    
    plt.rcParams['figure.figsize'] = [8, 5]

    # Plot figure 3.1, the quad plot with boundary layer parameters
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='all', constrained_layout=True)
    
    # Momentum thickness in 0,0
    i=0
    j=0
    exact_curve = ax[i][j].plot(x_out, theta_exact, color=exact_color)
    analytic_curve = ax[i][j].plot(x_out, theta_analytic, color=analytic_color)
    linear_curve = ax[i][j].plot(x_out, theta_linear, color=linear_color)
    nonlinear_curve = ax[i][j].plot(x_out, theta_nonlinear, color=nonlinear_color)
    ax[i][j].set_xlabel(r'$x$ (m)')
    ax[i][j].set_ylabel(r'$\theta$ (m)')
    ax[i][j].grid(True)
    
    # Displacement thickness in 0,1
    i=0
    j=1
    ax[i][j].plot(x_out, delta_star_exact, color=exact_color)
    ax[i][j].plot(x_out, delta_star_analytic, color=analytic_color)
    ax[i][j].plot(x_out, delta_star_linear, color=linear_color)
    ax[i][j].plot(x_out, delta_star_nonlinear, color=nonlinear_color)
    ax[i][j].set_xlabel(r'$x$ (m)')
    ax[i][j].set_ylabel(r'$\delta$ (m)')
    ax[i][j].grid(True)
    
    # Skin friction coefficient in 1,0
    i=1
    j=0
    ax[i][j].plot(x_out, cf_exact, color=exact_color)
    ax[i][j].plot(x_out, cf_analytic, color=analytic_color)
    ax[i][j].plot(x_out, cf_linear, color=linear_color)
    ax[i][j].plot(x_out, cf_nonlinear, color=nonlinear_color)
    ax[i][j].set_ylim(0, 1e-2)
    ax[i][j].set_xlabel(r'$x$ (m)')
    ax[i][j].set_ylabel(r'$c_f$')
    ax[i][j].grid(True)
    
    # Shape factor in 1,1
    i=1
    j=1
    ax[i][j].plot(x_out, H_exact, color=exact_color)
    ax[i][j].plot(x_out, H_analytic, color=analytic_color)
    ax[i][j].plot(x_out, H_linear, color=linear_color)
    ax[i][j].plot(x_out, H_nonlinear, color=nonlinear_color)
    ax[i][j].set_ylim(2.5, 2.7)
    ax[i][j].set_xlabel(r'$x$ (m)')
    ax[i][j].set_ylabel(r'$H$')
    ax[i][j].grid(True)

    # Based on example from: https://riptutorial.com/matplotlib/example/10473/single-legend-shared-across-multiple-subplots
    fig.legend([exact_curve, analytic_curve, linear_curve, nonlinear_curve], 
               labels=[exact_label, analytic_label, linear_label, nonlinear_label],
               loc="lower center", ncol=2, borderaxespad=0.1)
    fig.set_figwidth(8)
    fig.set_figheight(8)
    plt.subplots_adjust(bottom=0.1, wspace=0.3)
    plt.show()
    
    # Plot figure 3.2, error compared to the Thwaites analytic result
    plt.figure()
    plt.plot(x_out, np.abs(1-theta_linear/theta_analytic), label=theta_label);
    plt.plot(x_out, np.abs(1-delta_star_linear/delta_star_analytic), label=delta_star_label);
    plt.plot(x_out, np.abs(1-cf_linear/cf_analytic), label=cf_label);
    plt.plot(x_out, np.abs(1-H_linear/H_analytic), label=H_label);
    plt.xlabel(r'$x$ (m)')
    plt.ylabel('Relative Error')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot figure 3.3, error compared to the Falkner-Skan result
    plt.figure()
    plt.plot(x_out, np.abs(1-theta_analytic/theta_exact), label=theta_label);
    plt.plot(x_out, np.abs(1-delta_star_analytic/delta_star_exact), label=delta_star_label);
    plt.plot(x_out, np.abs(1-cf_analytic/cf_exact), label=cf_label);
    plt.plot(x_out, np.abs(1-H_analytic/H_exact), label=H_label);
    plt.xlabel(r'$x$ (m)')
    plt.ylabel('Relative Error')
    plt.ylim([.00001,5])
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()


if (__name__ == "__main__"):
    blasius_case()


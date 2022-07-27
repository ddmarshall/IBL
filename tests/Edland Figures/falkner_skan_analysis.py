#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:37:25 2022

@author: ddmarshall
"""

import numpy as np
import matplotlib.pyplot as plt

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


def  plot_falkner_skan_comparison(x_out, theta_exact, theta_analytic, theta_linear, theta_nonlinear,
                                  delta_star_exact, delta_star_analytic, delta_star_linear, delta_star_nonlinear,
                                  cf_exact, cf_analytic, cf_linear, cf_nonlinear,
                                  H_exact, H_analytic, H_linear, H_nonlinear,
                                  theta_range, delta_star_range, cf_range, H_range):
    # Set the plotting parameters
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

    # Plot the quad plot with boundary layer parameters
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='all')
    
    # Momentum thickness in 0,0
    i=0
    j=0
    exact_curve = ax[i][j].plot(x_out, theta_exact, color=exact_color)
    analytic_curve = ax[i][j].plot(x_out, theta_analytic, color=analytic_color)
    linear_curve = ax[i][j].plot(x_out, theta_linear, color=linear_color)
    nonlinear_curve = ax[i][j].plot(x_out, theta_nonlinear, color=nonlinear_color)
    ax[i][j].set_ylim(theta_range[0], theta_range[1])
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
    ax[i][j].set_ylim(delta_star_range[0], delta_star_range[1])
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
    ax[i][j].set_ylim(cf_range[0], cf_range[1])
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
    ax[i][j].set_ylim(H_range[0], H_range[1])
    ax[i][j].set_xlabel(r'$x$ (m)')
    ax[i][j].set_ylabel(r'$H$')
    ax[i][j].grid(True)

    # Based on example from: https://riptutorial.com/matplotlib/example/10473/single-legend-shared-across-multiple-subplots
    fig.legend(handles=[exact_curve[0], analytic_curve[0], linear_curve[0], nonlinear_curve[0]], 
               labels=[exact_label, analytic_label, linear_label, nonlinear_label],
               loc="lower center", ncol=2, borderaxespad=0.1)
    fig.set_figwidth(8)
    fig.set_figheight(8)
    plt.subplots_adjust(bottom=0.15, wspace=0.35)
    plt.show()
    
    # Plot error compared to the Thwaites analytic result
    plt.figure()
    plt.plot(x_out, np.abs(1-theta_linear/theta_analytic), label=theta_label+": "+linear_label);
    plt.plot(x_out, np.abs(1-delta_star_linear/delta_star_analytic), label=delta_star_label+": "+linear_label);
    plt.plot(x_out, np.abs(1-cf_linear/cf_analytic), label=cf_label+": "+linear_label);
    plt.plot(x_out, np.abs(1-H_linear/H_analytic), label=H_label+": "+linear_label);
    plt.xlabel(r'$x$ (m)')
    plt.ylabel('Relative Error')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot error compared to the Falkner-Skan result
    plt.figure()
    plt.plot(x_out, np.abs(1-theta_linear/theta_exact), label=theta_label+": "+linear_label);
    plt.plot(x_out, np.abs(1-delta_star_linear/delta_star_exact), label=delta_star_label+": "+linear_label);
    plt.plot(x_out, np.abs(1-cf_linear/cf_exact), label=cf_label+": "+linear_label);
    plt.plot(x_out, np.abs(1-H_linear/H_exact), label=H_label+": "+linear_label);
    plt.plot(x_out, np.abs(1-theta_nonlinear/theta_exact), label=theta_label+": "+nonlinear_label);
    plt.plot(x_out, np.abs(1-delta_star_nonlinear/delta_star_exact), label=delta_star_label+": "+nonlinear_label);
    plt.plot(x_out, np.abs(1-cf_nonlinear/cf_exact), label=cf_label+": "+nonlinear_label);
    plt.plot(x_out, np.abs(1-H_nonlinear/H_exact), label=H_label+": "+nonlinear_label);
    plt.xlabel(r'$x$ (m)')
    plt.ylabel('Relative Error')
    plt.ylim([.00001,10])
    plt.yscale('log')
    plt.grid(True)
    plt.legend(ncol=2)
    plt.show()


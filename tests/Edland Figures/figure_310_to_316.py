#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:05:15 2022

@author: ddmarshall
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline #for smoothed derivative experiment

from pyBL.head_method import HeadSimData, HeadSim

def mild_adverse_pressure_gradient_case():
    """
    Create the mild adverse pressure gradient comparison, figures 3.10-3.16 in Edland's thesis
    
    Args
    ----
        None
    
    Returns
    -------
        None
    """

    ## Case configuration
    npts = 161 # number of points for plotting
    
    ## Load the experimental data
    #tabular data:
    x_vec=    np.array([0.782, 1.282, 1.782, 2.282, 2.782, 3.132, 3.332, 3.532, 3.732, 3.932, 4.132, 4.332])
    u_e_vec = np.array([33.90, 32.60, 30.70, 28.60, 27.10, 26.05, 25.75,24.85,24.50, 24.05, 23.60, 23.10])
#    du_edx_tab = np.array([-2.3,-3.35,-4.32,-3.580,-3,-2.74,-2.6,-2.5,-2.4,-2.3,-2.25,-2.18]) #from tabulated values
#    approx_du_edx = (u_e_vec[:-1]-u_e_vec[1:])/(x_vec[:-1]-x_vec[1:])
    #from plot:
    smooth_x = np.linspace(.5,4.5,17)
    smooth_u_e = np.array([34.41,33.98,33.38,32.63,31.79,30.78,29.66,28.78,27.9,27.15,26.42,25.7,25.1,24.45,23.85,23.38,22.79])
    smooth_du_edx = -1 - .1*np.array([6.9,12.5,17.5,22.9,28.2,33.3,30,26.2,22.8,20.5,18.2,16.5,15.2,13.8,12.9,12.1,11.3])
    theta_tab = np.array([.276,.413,.606,.811,1.074,1.276,1.432,1.614,1.773,2.005,2.246,2.528])/100
    cf_tab = np.array([.00285,.00249,.00221,.00205,.00180,.00168,.00162,.00150,.00141,.00133,.00124,.00117])
#    cf_lt_tab = np.array([.00276,.00246,.00222,.00202,.00181,.00167,.00161,.00151,.00142,.00133,.00124,.00117])
    H_tab = np.array([1.381,1.394,1.402,1.427,1.457,1.485,1.492,1.519,1.542,1.566,1.594,1.618])
    delta_star_tab = H_tab*theta_tab
    nu = .15500/(100*100) #cm^2/sec
    
    ## Construct the three simulations and get results
    x_o = np.linspace(x_vec[0], x_vec[-1], npts)
    
    #Simulation using tabulated values
    hsd = HeadSimData(x_vec, u_e_vec, u_e_vec[0], nu, x_vec[0], theta_tab[0], H_tab[0])
    hs = HeadSim(hsd)
    while hs.status=='running': # and hs.x_vec[-1]<3.4:
        hs.step()
    theta_spline = hs.theta(x_o)
    delta_star_spline = hs.del_star(x_o)
    cf_spline = hs.c_f(x_o)
    H_spline = hs.h(x_o)
    Un_spline = hs.Un(x_o)
    
    #Simulation using values from smoothed plot    
    hsd_smooth = HeadSimData(smooth_x, smooth_u_e, smooth_u_e[0], nu, x_vec[0], theta_tab[0], H_tab[0])
    hs_smooth = HeadSim(hsd_smooth)
    while hs_smooth.status=='running': # and hs_smooth.x_vec[-1]<3.4:
        hs_smooth.step()
    theta_smooth = hs_smooth.theta(x_o)
    delta_star_smooth = hs_smooth.del_star(x_o)
    cf_smooth = hs_smooth.c_f(x_o)
    H_smooth = hs_smooth.h(x_o)
    Un_smooth = hs_smooth.Un(x_o)

    #Smooth Derivative to establish spline
    smooth_due_dx_spline = CubicSpline(smooth_x,smooth_du_edx)
    smooth_due_dx_spline_antiderivative = smooth_due_dx_spline.antiderivative()
    
    #Create same simulation using smoothed u_e
    hsd_smooth_der = HeadSimData(smooth_x, smooth_u_e, smooth_u_e[0], nu, x_vec[0], theta_tab[0], H_tab[0])
    hsd_smooth_der.u_e = lambda x: smooth_due_dx_spline_antiderivative(x)+smooth_u_e[0]
    hsd_smooth_der.du_edx = lambda x: smooth_due_dx_spline(x)
    hs_smooth_der = HeadSim(hsd_smooth_der)
    while hs_smooth_der.status=='running': # and hs_smooth.x_vec[-1]<3.4:
        hs_smooth_der.step()
    theta_smooth_der = hs_smooth_der.theta(x_o)
    delta_star_smooth_der = hs_smooth_der.del_star(x_o)
    cf_smooth_der = hs_smooth_der.c_f(x_o)
    H_smooth_der = hs_smooth_der.h(x_o)
    Un_smooth_der = hs_smooth_der.Un(x_o)

    ## Plot the velocity comparisons
    spline_label = '$U_e$'
    spline_color = 'blue'
    smooth_label = 'Smoothed $U_e$'
    smooth_color = 'green'
    smooth_der_label = 'Smoothed $U_e\'$'
    smooth_der_color = 'orange'
    ref_label = 'Kline et al.'
    ref_color = 'black'
    ref_marker = 'o'
    ref_linestyle = 'None'

    x_s = np.linspace(x_vec[0], smooth_x[-1], npts)
    
    plt.rcParams['figure.figsize'] = [8, 5]
    
    # Plot three velocity profiles, figure 3.10
    plt.figure()
    plt.plot(x_o, hs.u_e(x_o),label=spline_label,color=spline_color)
    plt.plot(x_s, hs_smooth.u_e(x_s),label=smooth_label,color=smooth_color)
    plt.plot(x_s, hs_smooth_der.u_e(x_s),label=smooth_der_label,color=smooth_der_color)
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$u_e$ (m/s)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot three velocity derivative profiles, figure 3.11
    plt.figure()
    plt.plot(x_o, hs.du_edx(x_o),label=spline_label,color=spline_color)
    plt.plot(x_s, hs_smooth.du_edx(x_s),label=smooth_label,color=smooth_color)
    plt.plot(x_s, hs_smooth_der.du_edx(x_s),label=smooth_der_label,color=smooth_der_color)
    plt.xlabel(r'$x$ (m)')
    plt.ylabel('$u_e\'$ (m/s)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    ## Plot the results comparisons
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='all')
    
    # Momentum thickness in 0,0
    i=0
    j=0
    ref_curve = ax[i][j].plot(x_vec, theta_tab, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    spline_curve = ax[i][j].plot(x_o, theta_spline, color=spline_color)
    smooth_curve = ax[i][j].plot(x_o, theta_smooth, color=smooth_color)
    smooth_der_curve = ax[i][j].plot(x_o, theta_smooth_der, color=smooth_der_color)
    ax[i][j].set_ylim(0, 0.03)
    ax[i][j].set_xlabel(r'$x$ (m)')
    ax[i][j].set_ylabel(r'$\theta$ (m)')
    ax[i][j].grid(True)

    # Displacement thickness in 0,1
    i=0
    j=1
    ax[i][j].plot(x_vec, delta_star_tab, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(x_o, delta_star_spline, color=spline_color)
    ax[i][j].plot(x_o, delta_star_smooth, color=smooth_color)
    ax[i][j].plot(x_o, delta_star_smooth_der, color=smooth_der_color)
    ax[i][j].set_ylim(0, 0.05)
    ax[i][j].set_xlabel(r'$x$ (m)')
    ax[i][j].set_ylabel(r'$\delta^*$ (m)')
    ax[i][j].grid(True)
    
    # Skin friction coefficient in 1,0
    i=1
    j=0
    ax[i][j].plot(x_vec, cf_tab, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(x_o, cf_spline, color=spline_color)
    ax[i][j].plot(x_o, cf_smooth, color=smooth_color)
    ax[i][j].plot(x_o, cf_smooth_der, color=smooth_der_color)
    ax[i][j].set_ylim(0.001, 0.003)
    ax[i][j].set_xlabel(r'$x$ (m)')
    ax[i][j].set_ylabel(r'$c_f$')
    ax[i][j].grid(True)
    
    # Shape factor in 1,1
    i=1
    j=1
    ax[i][j].plot(x_vec, H_tab, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(x_o, H_spline, color=spline_color)
    ax[i][j].plot(x_o, H_smooth, color=smooth_color)
    ax[i][j].plot(x_o, H_smooth_der, color=smooth_der_color)
    ax[i][j].set_ylim(1.35, 1.65)
    ax[i][j].set_xlabel(r'$x$ (m)')
    ax[i][j].set_ylabel(r'$H$')
    ax[i][j].grid(True)

    # Based on example from: https://riptutorial.com/matplotlib/example/10473/single-legend-shared-across-multiple-subplots
    fig.legend(handles=[ref_curve[0], spline_curve[0], smooth_curve[0], smooth_der_curve[0]], 
               labels=[ref_label, spline_label, smooth_label, smooth_der_label],
               loc="lower center", ncol=2, borderaxespad=0.1)
    fig.set_figwidth(8)
    fig.set_figheight(8)
    plt.subplots_adjust(bottom=0.15, wspace=0.35)
    plt.show()

    # Plot errors compared to the experimental data
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='all')
    
    # Momentum thickness in 0,0
    i=0
    j=0
    spline_curve = ax[i][j].plot(x_vec, np.abs(1-hs.theta(x_vec)/theta_tab), color=spline_color);
    smooth_curve = ax[i][j].plot(x_vec, np.abs(1-hs_smooth.theta(x_vec)/theta_tab), color=smooth_color);
    smooth_der_curve = ax[i][j].plot(x_vec, np.abs(1-hs_smooth_der.theta(x_vec)/theta_tab), color=smooth_der_color);
    ax[i][j].set_ylim(7e-3, 1.3e-0)
    ax[i][j].set_xlabel(r'$x$ (m)')
    ax[i][j].set_ylabel(r'Relative Error in $\theta$')
    ax[i][j].set_yscale('log')
    ax[i][j].grid(True)

    # Displacement thickness in 0,1
    i=0
    j=1
    ax[i][j].plot(x_vec, np.abs(1-hs.del_star(x_vec)/delta_star_tab), color=spline_color);
    ax[i][j].plot(x_vec, np.abs(1-hs_smooth.del_star(x_vec)/delta_star_tab), color=smooth_color);
    ax[i][j].plot(x_vec, np.abs(1-hs_smooth_der.del_star(x_vec)/delta_star_tab), color=smooth_der_color);
    ax[i][j].set_ylim(7e-3, 1.3)
    ax[i][j].set_xlabel(r'$x$ (m)')
    ax[i][j].set_ylabel(r'Relative Error in $\delta^*$')
    ax[i][j].set_yscale('log')
    ax[i][j].grid(True)
    
    # Skin friction coefficient in 1,0
    i=1
    j=0
    ax[i][j].plot(x_vec, np.abs(1-hs.c_f(x_vec)/cf_tab), color=spline_color);
    ax[i][j].plot(x_vec, np.abs(1-hs_smooth.c_f(x_vec)/cf_tab), color=smooth_color);
    ax[i][j].plot(x_vec, np.abs(1-hs_smooth_der.c_f(x_vec)/cf_tab), color=smooth_der_color);
    ax[i][j].set_ylim(7e-4, 1.3e-1)
    ax[i][j].set_xlabel(r'$x$ (m)')
    ax[i][j].set_ylabel(r'Relative Error in $c_f$')
    ax[i][j].set_yscale('log')
    ax[i][j].grid(True)
    
    # Shape factor in 1,1
    i=1
    j=1
    ax[i][j].plot(x_vec, np.abs(1-hs.h(x_vec)/H_tab), color=spline_color);
    ax[i][j].plot(x_vec, np.abs(1-hs_smooth.h(x_vec)/H_tab), color=smooth_color);
    ax[i][j].plot(x_vec, np.abs(1-hs_smooth_der.h(x_vec)/H_tab), color=smooth_der_color);
    ax[i][j].set_ylim(7e-5, 1.3e-1)
    ax[i][j].set_xlabel(r'$x$ (m)')
    ax[i][j].set_ylabel(r'Relative Error in $c_f$')
    ax[i][j].set_yscale('log')
    ax[i][j].grid(True)

    # Based on example from: https://riptutorial.com/matplotlib/example/10473/single-legend-shared-across-multiple-subplots
    fig.legend(handles=[spline_curve[0], smooth_curve[0], smooth_der_curve[0]], 
               labels=[spline_label, smooth_label, smooth_der_label],
               loc="lower center", ncol=3, borderaxespad=0.1)
    fig.set_figwidth(8)
    fig.set_figheight(8)
    plt.subplots_adjust(bottom=0.1, wspace=0.35)
    plt.show()
    
    # plot the transpiration velocity
    plt.figure()
    plt.plot(x_o, Un_spline, label=spline_label, color=spline_color)
    plt.plot(x_o, Un_smooth, label=smooth_label, color=smooth_color)
    plt.plot(x_o, Un_smooth_der, label=smooth_der_label, color=smooth_der_color)
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$U_n$ (m/s)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # plot the transpiration velocity derivative
    Unp_spline = hs.u_e(x_o)*hs.del_star(x_o)
    Unp_smooth = hs_smooth.u_e(x_o)*hs_smooth.del_star(x_o)
    Unp_smooth_der = hs_smooth_der.u_e(x_o)*hs_smooth_der.del_star(x_o)
    
    plt.figure()
    plt.plot(x_o, Unp_spline, label=spline_label, color=spline_color)
    plt.plot(x_o, Unp_smooth, label=smooth_label, color=smooth_color)
    plt.plot(x_o, Unp_smooth_der, label=smooth_der_label, color=smooth_der_color)
    plt.xlabel(r'$x$ (m)')
    plt.ylabel('$U_n\'$ (m/s)')
    plt.grid(True)
    plt.legend()
    plt.show()


if (__name__ == "__main__"):
    mild_adverse_pressure_gradient_case()


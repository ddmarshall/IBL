#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 23:29:32 2022

@author: ddmarshall
"""

import numpy as np
import matplotlib.pyplot as plt

from pyBL.head_method import HeadSimData, HeadSim

from xfoil_interface import read_xfoil_dump_file

def turbulent_xfoil_comparison():
    """
    Create the laminar airfoil comparison, figures 3.17-3.19 in Edland's thesis
    
    Args
    ----
        None
    
    Returns
    -------
        None
    """
    
    ## Load the XFoil data for the case
    # parameters use to generate case
    # NACA 0012
    # aoa =0
    # x_trans = 0.001 (upper and lower)
    # Transition model n = 9
    c = 1 #m
    U_inf = 20 #m/s
    Re = 1e6
    xfoil_upper, xfoil_lower, _ = read_xfoil_dump_file('xfoil_turbulent_dump.txt', U_inf)
    nu = U_inf*c/Re

    ## Create the simulations for the upper and lower surfaces
    lower_range=slice(1,np.size(xfoil_lower.s))
    hsd_l = HeadSimData(xfoil_lower.s[lower_range],xfoil_lower.Ue[lower_range],U_inf,nu,xfoil_lower.s[lower_range.start],theta0=xfoil_lower.delta_m[lower_range.start], h0=xfoil_lower.H[lower_range.start])
    hs_l = HeadSim(hsd_l) 
    while hs_l.status=='running':
        hs_l.step()
    upper_range=slice(1,np.size(xfoil_upper.s))
    hsd_u = HeadSimData(xfoil_upper.s[upper_range],xfoil_upper.Ue[upper_range],U_inf,nu,xfoil_upper.s[upper_range.start],theta0=xfoil_upper.delta_m[upper_range.start], h0=xfoil_upper.H[upper_range.start])
    hs_u = HeadSim(hsd_u) 
    while hs_u.status=='running':
        hs_u.step()

    ## Plot the comparisons
    spline_label = 'PyBL'
    spline_color = 'green'
    ref_label = 'XFoil (laminar)'
    ref_color = 'black'
    ref_marker = ''
    ref_linestyle = '-'

    plt.rcParams['figure.figsize'] = [8, 5]
    
    # Plot the results comparisons
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='all')
    
    # Momentum thickness in 0,0
    i=0
    j=0
    ref_curve = ax[i][j].plot(xfoil_lower.x/c, xfoil_lower.delta_m/c, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(xfoil_upper.x/c, xfoil_upper.delta_m/c, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    spline_curve = ax[i][j].plot(xfoil_lower.x[lower_range], hs_l.theta(xfoil_lower.s[lower_range])/c, color=spline_color)
    ax[i][j].plot(xfoil_upper.x[upper_range]/c, hs_u.theta(xfoil_upper.s[upper_range])/c, color=spline_color)
    ax[i][j].set_ylim(0, 0.0043)
    ax[i][j].set_xlabel(r'$x/c$')
    ax[i][j].set_ylabel(r'$\theta/c$')
    ax[i][j].grid(True)

    # Displacement thickness in 0,1
    i=0
    j=1
    ax[i][j].plot(xfoil_lower.x/c, xfoil_lower.delta_d/c, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(xfoil_upper.x/c, xfoil_upper.delta_d/c, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(xfoil_lower.x[lower_range], hs_l.del_star(xfoil_lower.s[lower_range])/c, color=spline_color)
    ax[i][j].plot(xfoil_upper.x[upper_range], hs_u.del_star(xfoil_upper.s[upper_range])/c, color=spline_color)
    ax[i][j].set_ylim(0, 0.007)
    ax[i][j].set_xlabel(r'$x/c$')
    ax[i][j].set_ylabel(r'$\delta^*/c$')
    ax[i][j].grid(True)
    
    # Skin friction coefficient in 1,0
    i=1
    j=0
    ax[i][j].plot(xfoil_lower.x/c, xfoil_lower.cf, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(xfoil_upper.x/c, xfoil_upper.cf, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(xfoil_lower.x[lower_range], hs_l.c_f(xfoil_lower.s[lower_range]), color=spline_color)
    ax[i][j].plot(xfoil_upper.x[upper_range], hs_u.c_f(xfoil_upper.s[upper_range]), color=spline_color)
    ax[i][j].set_ylim(0, 0.025)
    ax[i][j].set_xlabel(r'$x/c$')
    ax[i][j].set_ylabel(r'$c_f$')
    ax[i][j].grid(True)
    
    # Shape factor in 1,1
    i=1
    j=1
    ax[i][j].plot(xfoil_lower.x/c, xfoil_lower.H, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(xfoil_upper.x/c, xfoil_upper.H, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(xfoil_lower.x[lower_range], hs_l.h(xfoil_lower.s[lower_range]), color='orange')
    ax[i][j].plot(xfoil_upper.x[upper_range], hs_u.h(xfoil_upper.s[upper_range]), color=spline_color)
    ax[i][j].set_ylim(1.4, 2.4)
    ax[i][j].set_xlabel(r'$x/c$')
    ax[i][j].set_ylabel(r'$H$')
    ax[i][j].grid(True)

    # Based on example from: https://riptutorial.com/matplotlib/example/10473/single-legend-shared-across-multiple-subplots
    fig.legend(handles=[ref_curve[0], spline_curve[0]], labels=[ref_label, spline_label],
               loc="lower center", ncol=2, borderaxespad=0.1)
    fig.set_figwidth(8)
    fig.set_figheight(8)
    plt.subplots_adjust(bottom=0.1, wspace=0.35)
    plt.show()

    # Plot difference compared to the XFoil results
    plt.figure()
    plt.plot(xfoil_lower.x[lower_range]/c, np.abs(1-hs_l.theta(xfoil_lower.s[lower_range])/xfoil_lower.delta_m[lower_range]), label=r'$\theta$');
    plt.plot(xfoil_lower.x[lower_range]/c, np.abs(1-hs_l.del_star(xfoil_lower.s[lower_range])/xfoil_lower.delta_d[lower_range]), label=r'$\delta^*$');
    plt.plot(xfoil_lower.x[lower_range]/c, np.abs(1-hs_l.c_f(xfoil_lower.s[lower_range])/xfoil_lower.cf[lower_range]), label='$c_f$');
    plt.plot(xfoil_lower.x[lower_range]/c, np.abs(1-hs_l.h(xfoil_lower.s[lower_range])/xfoil_lower.H[lower_range]), label='$H$');
    plt.xlabel(r'$x/c$')
    plt.ylabel('Relative Difference')
    plt.ylim([.00001,10])
    plt.yscale('log')
    plt.grid(True)
    plt.legend(ncol=2)
    plt.show()


if (__name__ == "__main__"):
    turbulent_xfoil_comparison()


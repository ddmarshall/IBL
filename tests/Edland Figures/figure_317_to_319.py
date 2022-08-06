#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:58:33 2022

@author: ddmarshall
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import namedtuple

from pyBL.thwaites_method import ThwaitesSimData, ThwaitesSim

def laminar_xfoil_comparison():
    """
    Create the laminar airfoil comparison, figures 3.17-3.19 in Edland's thesis
    
    Args
    ----
        None
    
    Returns
    -------
        None
    """
    
    ## Case configuration
#    npts = 161 # number of points for plotting
    
    ## Load the XFoil data for the case
    c, U_inf, Re, xfoil_upper, xfoil_lower, _ = get_laminar_xfoil_data('xfoil_laminar_dump.txt')
    nu = U_inf*c/Re

    ## Create the simulations for the upper and lower surfaces
    lower_range=slice(0,np.size(xfoil_lower.s))
    tsd_l = ThwaitesSimData(xfoil_lower.s[lower_range],xfoil_lower.Ue[lower_range],U_inf,nu,Re,xfoil_lower.s[lower_range.start],theta0=xfoil_lower.delta_m[lower_range.start])
    ts_l = ThwaitesSim(tsd_l) 
    while ts_l.status=='running':
        ts_l.step()
    upper_range=slice(0,np.size(xfoil_upper.s))
    tsd_u = ThwaitesSimData(xfoil_upper.s[upper_range],xfoil_upper.Ue[upper_range],U_inf,nu,Re,xfoil_upper.s[upper_range.start],theta0=xfoil_upper.delta_m[upper_range.start])
    ts_u = ThwaitesSim(tsd_u) 
    while ts_u.status=='running':
        ts_u.step()

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
    spline_curve = ax[i][j].plot(xfoil_lower.x[lower_range], ts_l.theta(xfoil_lower.s[lower_range])/c, color=spline_color)
    ax[i][j].plot(xfoil_upper.x[upper_range]/c, ts_u.theta(xfoil_upper.s[upper_range])/c, color=spline_color)
    ax[i][j].set_ylim(0, 0.025)
    ax[i][j].set_xlabel(r'$x/c$')
    ax[i][j].set_ylabel(r'$\theta/c$')
    ax[i][j].grid(True)

    # Displacement thickness in 0,1
    i=0
    j=1
    ax[i][j].plot(xfoil_lower.x/c, xfoil_lower.delta_d/c, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(xfoil_upper.x/c, xfoil_upper.delta_d/c, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(xfoil_lower.x[lower_range], ts_l.del_star(xfoil_lower.s[lower_range])/c, color=spline_color)
    ax[i][j].plot(xfoil_upper.x[upper_range], ts_u.del_star(xfoil_upper.s[upper_range])/c, color=spline_color)
    ax[i][j].set_ylim(0, 0.075)
    ax[i][j].set_xlabel(r'$x/c$')
    ax[i][j].set_ylabel(r'$\delta^*/c$')
    ax[i][j].grid(True)
    
    # Skin friction coefficient in 1,0
    i=1
    j=0
    ax[i][j].plot(xfoil_lower.x/c, xfoil_lower.cf, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(xfoil_upper.x/c, xfoil_upper.cf, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(xfoil_lower.x[lower_range], ts_l.c_f(xfoil_lower.s[lower_range]), color=spline_color)
    ax[i][j].plot(xfoil_upper.x[upper_range], ts_u.c_f(xfoil_upper.s[upper_range]), color=spline_color)
    ax[i][j].set_ylim(0, 0.3)
    ax[i][j].set_xlabel(r'$x/c$')
    ax[i][j].set_ylabel(r'$c_f$')
    ax[i][j].grid(True)
    
    # Shape factor in 1,1
    i=1
    j=1
    ax[i][j].plot(xfoil_lower.x/c, xfoil_lower.H, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(xfoil_upper.x/c, xfoil_upper.H, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
    ax[i][j].plot(xfoil_lower.x[lower_range], ts_l.h(xfoil_lower.s[lower_range]+1e-6), color='orange')
    ax[i][j].plot(xfoil_upper.x[upper_range], ts_u.h(xfoil_upper.s[upper_range]+1e-6), color=spline_color)
    ax[i][j].set_ylim(2.2, 2.7)
    ax[i][j].set_xlabel(r'$x$')
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
    plt.plot(xfoil_lower.x/c, np.abs(1-ts_l.theta(xfoil_lower.s[lower_range])/xfoil_lower.delta_m), label=r'$\theta$');
    plt.plot(xfoil_lower.x/c, np.abs(1-ts_l.del_star(xfoil_lower.s[lower_range])/xfoil_lower.delta_d), label=r'$\delta^*$');
    plt.plot(xfoil_lower.x/c, np.abs(1-ts_l.c_f(xfoil_lower.s[lower_range])/xfoil_lower.cf), label='$c_f$');
    plt.plot(xfoil_lower.x/c, np.abs(1-ts_l.h(xfoil_lower.s[lower_range])/xfoil_lower.H), label='$H$');
    plt.xlabel(r'$x/c$')
    plt.ylabel('Relative Difference')
    plt.ylim([.00001,10])
    plt.yscale('log')
    plt.grid(True)
    plt.legend(ncol=2)
    plt.show()
    
    fig,axs = plt.subplots(ncols=2)
    ax = axs[0]
    ax.plot(xfoil_lower.x[xfoil_lower.x>0.6], ts_l.u_e(xfoil_lower.s[xfoil_lower.x>0.6]),label='XFOIL',color='k')
    ax.set(xlabel=('$x/c$'))
    ax.set(ylabel=(r'$u_e$ (m/s)'))
    ax.grid(True)

    ax=axs[1]
    ax.plot(xfoil_lower.x[xfoil_lower.x>0.6], ts_l.du_edx(xfoil_lower.s[xfoil_lower.x>0.6]),label='XFOIL',color='k')
    ax.set(xlabel=('x(m)'))
    ax.set(ylabel=(r'$\frac{du_e}{dx}$ (m/s$^2$)'))
    ax.grid(True)
    
    fig.set_figwidth(8)
    fig.set_figheight(8)
    plt.subplots_adjust(bottom=0.1, wspace=0.35)
    plt.show()


def get_laminar_xfoil_data(xfoil_file):
    """
    Return the laminar results from XFoil for a NACA 0003 airfoil at -3 deg. 
    angle of attack.
    
    Args
    ----
        xfoi_file - file name containing XFoil dump data
        
    Returns
    -------
        c - chord length
        U_inf - freestream velocity
        Re - frestream Reynolds number
        s_lower - arclength distance from stagnation point on lower surface
        s_upper - arclength distance from stagnation point on upper surface
        Ue_lower - lower surface velocity
        Ue_upper - upper surface velocity
        del_star_lower - lower surface displacement thickness
        del_star_upper - upper surface displacement thickness
        theta_lower - lower surface momentum thickness
        theta_upper - upper surface momentum thickness
        cf_lower - lower surface skin friction coefficient
        cf_upper - upper surface skin friction coefficient
        H_lower - lower surface shape factor
        H_upper - upper surface shape factor
    """
    ## Load the data from XFoil viscous dump file
    # parameters use to generate case
    # NACA 0003
    # aoa =0
    # x_trans = 1 (upper and lower)
    # Transition model n = 9
    c = 1 #m
    U_inf = 20 #m/s 
    Re = 1000
    
    PanelValues = namedtuple('PanelValues', ['s', 'x', 'y', 'Ue', 'delta_d', 'delta_m', 'cf', 'H', 'Hstar', 'P', 'm', 'K'])
    WakeValues = namedtuple('PanelValues', ['s', 'x', 'y', 'Ue', 'delta_d', 'delta_m', 'H'])
    
    xfoil_file = open(xfoil_file, 'r')
    xfoil_reader = csv.reader(xfoil_file)
    panel_terms = []
    wake_terms = []
    for row in xfoil_reader:
        if (row[0][0] != '#'):
            col = np.array(row[0].split(), 'float')
            
            if (np.size(col) == 12):
                panel_terms.append(col)
            elif (np.size(col) == 8):
                wake_terms.append(col)
            else:
                raise Exception('Invalid number of columns for XFoil import')
                    
    ## Find stagnation point and the total arclength of airfoil
    panel_terms = np.asarray(panel_terms)
    last_upper_idx = np.where(np.sign(panel_terms[:-1, 3]) != np.sign(panel_terms[1:, 3]))[0][0]
    s_af = panel_terms[-1, 0]
    
    # interpolate stagnation point
    dU = panel_terms[last_upper_idx+1, 3]-panel_terms[last_upper_idx, 3]
    frac = -panel_terms[last_upper_idx, 3]/dU
    stag_col = frac*panel_terms[last_upper_idx+1,:]+(1-frac)*panel_terms[last_upper_idx,:]

    ## Split into upper and lower surface terms
    # upper panel needs to be reversed since XFoil writes out starting with 
    #   the upper surface trailing edge to lower surface trailing edge.
    panel_upper_terms = panel_terms[last_upper_idx::-1, :]
    # insert the stagnation point
    panel_upper_terms = np.insert(panel_upper_terms, 0, stag_col, 0)
    # adjust distance is based on arclength distance from upper trailing edge.
    panel_upper_terms[:,0] = panel_upper_terms[0,0] - panel_upper_terms[:,0]
    panel_upper_terms[:,3] = U_inf*panel_upper_terms[:,3]
    upper_values = PanelValues(*panel_upper_terms.T)
    
    # lower panel
    panel_lower_terms = panel_terms[last_upper_idx+1:, :]
    # insert the stagnation point
    panel_lower_terms = np.insert(panel_lower_terms, 0, stag_col, 0)
    # adjust distance (see upper panel)
    panel_lower_terms[:,0] = panel_lower_terms[:,0] - panel_lower_terms[0,0]
    # adjust the signs of the lower surface terms
    panel_lower_terms[:,3] = -panel_lower_terms[:,3]
    panel_lower_terms[:,[6,10,11]] = np.abs(panel_lower_terms[:,[6,10,11]])
    # remove scale of velocities
    panel_lower_terms[:,3] = U_inf*panel_lower_terms[:,3]
    lower_values = PanelValues(*panel_lower_terms.T)
    
    # wake
    # adjust distance (base it on trailing edge)
    wake_terms = np.asarray(wake_terms)
    wake_terms[:,0] = wake_terms[:,0] - s_af
    # remove zeros from skin friction coefficient
    wake_terms = np.delete(wake_terms, 6, 1)
    wake_values = WakeValues(*wake_terms.T)
    
    return c, U_inf, Re, upper_values, lower_values, wake_values


if (__name__ == "__main__"):
    laminar_xfoil_comparison()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:05:15 2022

@author: ddmarshall
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline #for smoothed derivative experiment

from pyBL.heads_method import HeadSimData, HeadSim

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
    c_f_tab = np.array([.00285,.00249,.00221,.00205,.00180,.00168,.00162,.00150,.00141,.00133,.00124,.00117])
    c_f_lt_tab = np.array([.00276,.00246,.00222,.00202,.00181,.00167,.00161,.00151,.00142,.00133,.00124,.00117])
    h_tab = np.array([1.381,1.394,1.402,1.427,1.457,1.485,1.492,1.519,1.542,1.566,1.594,1.618])
    del_tab = h_tab*theta_tab
    nu = .15500/(100*100) #cm^2/sec
    
    ## Construct the three simulations and get results
    #Simulation using tabulated values
    hsd = HeadSimData(x_vec, u_e_vec, u_e_vec[0], nu, x_vec[0], theta_tab[0], h_tab[0])
    hs = HeadSim(hsd)
    
    #Simulation using values from smoothed plot    
    hsd_smooth = HeadSimData(smooth_x, smooth_u_e, smooth_u_e[0], nu, smooth_x[0], theta_tab[0], h_tab[0])
    hs_smooth = HeadSim(hsd_smooth)

    #Smooth Derivative to establish spline
    smooth_due_dx_spline = CubicSpline(smooth_x,smooth_du_edx)
    smooth_due_dx_spline_antiderivative = smooth_due_dx_spline.antiderivative()
    
    #Create same simulation using smoothed u_e
    hsd_smooth_der = HeadSimData(smooth_x, smooth_u_e, smooth_u_e[0], nu, smooth_x[0], theta_tab[0], h_tab[0])
    hsd_smooth_der.u_e = lambda x: smooth_due_dx_spline_antiderivative(x)+smooth_u_e[0]
    hsd_smooth_der.du_edx = lambda x: smooth_due_dx_spline(x)
    hs_smooth_der = HeadSim(hsd_smooth_der)

    ## Plot the velocity comparisons
    npts = 41
    spline_label = '$u_e$'
    spline_color = 'blue'
    smooth_label = 'Smoothed $u_e$'
    smooth_color = 'green'
    der_label = 'Smoothed $u_e\'$'
    der_color = 'orange'
    
    plt.rcParams['figure.figsize'] = [8, 5]
    
    x_o = np.linspace(x_vec[0], x_vec[-1], npts)
    x_s = np.linspace(smooth_x[0], smooth_x[-1], npts)
    
    # Plot three velocity profiles, figure 3.10
    plt.figure()
    plt.plot(x_o, hs.u_e(x_o),label=spline_label,color=spline_color)
    plt.plot(x_s, hs_smooth.u_e(x_s),label=smooth_label,color=smooth_color)
    plt.plot(x_s, hs_smooth_der.u_e(x_s),label=der_label,color=der_color)
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$u_e$ (m/s)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot three velocity derivative profiles, figure 3.11
    plt.figure()
    plt.plot(x_o, hs.du_edx(x_o),label=spline_label,color=spline_color)
    plt.plot(x_s, hs_smooth.du_edx(x_s),label=smooth_label,color=smooth_color)
    plt.plot(x_s, hs_smooth_der.du_edx(x_s),label=der_label,color=der_color)
    plt.xlabel(r'$x$ (m)')
    plt.ylabel('$u_e\'$ (m/s)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    ## Plot the results comparisons


if (__name__ == "__main__"):
    mild_adverse_pressure_gradient_case()


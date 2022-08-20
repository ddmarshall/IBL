#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example shows a comparison between various forms of Thwaites and the
Blasius solution to laminar flat plate boundary layer flows.

Created on Wed Aug 17 15:03:07 2022

@author: ddmarshall
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pyBL.blasius_solution import BlasiusSolution
from pyBL.thwaites_method import ThwaitesMethod


def compare_blasius_solution():
    ## Set flow parameters
    U_inf = 10
    nu_inf = 1.45e-5
    rho_inf = 1.2
    c = 2
    npts = 101
    
    ## Set up the velocity functions
    def U_e_fun(x):
        x = np.asarray(x)
        return U_inf*np.ones_like(x)
    def dU_edx_fun(x):
        x = np.asarray(x)
        return np.zeros_like(x)
    def d2U_edx2_fun(x):
        x = np.asarray(x)
        return np.zeros_like(x)
    
    ## Get the solutions for comparisons
    x = np.linspace(1e-6, c, npts)
    bs = BlasiusSolution(U_ref = U_inf, nu = nu_inf, fpp0 = 0.46960)
    tm = ThwaitesMethod(U_e = U_e_fun, dU_edx = dU_edx_fun,
                        d2U_edx2 = d2U_edx2_fun, data_fits = "Spline")
    tm.set_solution_parameters(x0 = x[0], x_end = x[-1],
                               delta_m0 = bs.delta_m(x[0]), nu = nu_inf)
    rtn = tm.solve()
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    
    ## Calculate the boundary layer parameters
    delta_d_exact = bs.delta_d(x)
    delta_d_standard = tm.delta_d(x)
    delta_m_exact = bs.delta_m(x)
    delta_m_standard = tm.delta_m(x)
    c_f_exact = bs.tau_w(x, rho_inf)/(0.5*rho_inf*U_inf**2)
    c_f_standard = tm.tau_w(x, rho_inf)/(0.5*rho_inf*U_inf**2)
    H_d_exact = bs.H_d(x)
    H_d_standard = tm.H_d(x)
    U_n_exact = bs.V_e(x)
    U_n_standard = tm.U_n(x)
    
    ## plot functions
    plt.rcParams['figure.figsize'] = [8, 5]
    
    # Plot the quad plot with boundary layer parameters
    fig = plt.figure(constrained_layout = True)
    fig.set_figwidth(8)
    fig.set_figheight(10)
    gs = GridSpec(3, 2, figure = fig)
    axis_delta_m = fig.add_subplot(gs[0, 0])
    axis_delta_d = fig.add_subplot(gs[0, 1])
    axis_c_f = fig.add_subplot(gs[1, 0])
    axis_H_d = fig.add_subplot(gs[1, 1])
    axis_error = fig.add_subplot(gs[2, :])
    
    exact_color = "black"
    exact_label = "Blasius"
    standard_color = "green"
    standard_label = "Standard"
    
    # Momentum thickness in 0,0
    ax = axis_delta_m
    exact_curve = ax.plot(x/c, delta_m_exact/c, color=exact_color)
    standard_curve = ax.plot(x/c, delta_m_standard/c, color=standard_color)
    ax.set_ylim(0, 0.0006)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$\delta_m/c$")
    ax.grid(True)
    
    # Displacement thickness in 0,1
    ax = axis_delta_d
    ax.plot(x/c, delta_d_exact/c, color=exact_color)
    ax.plot(x/c, delta_d_standard/c, color=standard_color)
    ax.set_ylim(0, 0.0015)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$\delta_d/c$")
    ax.grid(True)
    
    # Skin friction coefficient in 1,0
    ax = axis_c_f
    ax.plot(x/c, c_f_exact, color=exact_color)
    ax.plot(x/c, c_f_standard, color=standard_color)
    ax.set_ylim(0, 0.01)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$c_f$")
    ax.grid(True)
    
    # Shape factor in 1,1
    ax = axis_H_d
    ax.plot(x/c, H_d_exact, color=exact_color)
    ax.plot(x/c, H_d_standard, color=standard_color)
    ax.set_ylim(2.5, 2.7)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$H_d$")
    ax.grid(True)
    
    fig.legend(handles = [exact_curve[0], standard_curve[0]], 
               labels = [exact_label, standard_label],
               loc = "upper center", bbox_to_anchor = (0.55, 0.36),
               ncol = 2, borderaxespad = 0.1)
    
    # Plot error compared to the Blasius result
    ax = axis_error
    ax.plot(x/c, np.abs(1-delta_m_standard/delta_m_exact),
            label = r"$\delta_m$: " + standard_label);
    ax.plot(x/c, np.abs(1-delta_d_standard/delta_d_exact),
            label = r"$\delta_d$: " + standard_label);
    ax.plot(x/c, np.abs(1-c_f_standard/c_f_exact),
            label = r"$c_f$: " + standard_label);
    ax.plot(x/c, np.abs(1-H_d_standard/H_d_exact),
            label = r"$H_d$: " + standard_label);
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel("Relative Error")
    ax.set_ylim([1e-4,1])
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend(ncol=4, loc = "lower center")
    plt.show()
    
    # Plot Transpiration velocity
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='all')
    fig.set_figwidth(9)
    fig.set_figheight(7)
    
    ax[0].plot(x/c, U_n_exact/U_inf, label = exact_label, color = exact_color);
    ax[0].plot(x/c, U_n_standard/U_inf, label = standard_label, color = standard_color);
    ax[0].set_xlabel(r"$x/c$")
    ax[0].set_ylabel(r"$U_n/U_e$")
    ax[0].set_ylim(0, 0.01)
    ax[0].grid(True)
    
    ax[1].plot(x/c, np.abs(1-U_n_standard/U_n_exact))
    ax[1].set_xlabel(r"$x/c$")
    ax[1].set_ylabel("Relative Error")
    ax[1].set_ylim([1e-4,1])
    ax[1].set_yscale('log')
    ax[1].grid(True)
    plt.show()


if (__name__ == "__main__"):
    compare_blasius_solution()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example shows a comparison between various forms of Thwaites and the
Falkner-Skan solution to laminar flat plate boundary layer wedge flows at angle
of pi/4. It shows similar results to Figures 3.7 to 3.9 in Edland thesis.

Created on Wed Aug 17 15:03:07 2022

@author: ddmarshall
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pyBL.falkner_skan import FalknerSkanSolution
from pyBL.thwaites_method import ThwaitesMethodLinear
from pyBL.thwaites_method import ThwaitesMethodNonlinear


def compare_stagnation_solution():
    ## Set flow parameters
    U_inf = 10
    m = 1/3
    nu_inf = 1.45e-5
    rho_inf = 1.2
    c = 2
    npts = 201
    
    ## Set up the velocity functions
    def U_e_fun(x):
        x = np.asarray(x)
        return U_inf*x**m
    def dU_edx_fun(x):
        x = np.asarray(x)
        if m == 0:
            return np.zeros_like(x)
        else:
            return m*U_inf*x**(m-1)
    def d2U_edx2_fun(x):
        x = np.asarray(x)
        if (m==0) or (m== 1):
            return np.zeros_like(x)
        else:
            return m*(m-1)*U_inf*x**(m-2)
    
    ## Get the solutions for comparisons
    x = np.linspace(1e-6, c, npts)
    fs = FalknerSkanSolution(U_ref = U_inf, m=m, nu = nu_inf)
    tml = ThwaitesMethodLinear(U_e = U_e_fun, dU_edx = dU_edx_fun,
                               d2U_edx2 = d2U_edx2_fun, data_fits = "Spline")
    tml.set_solution_parameters(x0 = x[0], x_end = x[-1],
                                delta_m0 = fs.delta_m(x[0]), nu = nu_inf)
    rtn = tml.solve()
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    
    tmn = ThwaitesMethodNonlinear(U_e = U_e_fun, dU_edx = dU_edx_fun,
                                  d2U_edx2 = d2U_edx2_fun,
                                  data_fits = "Spline")
    tmn.set_solution_parameters(x0 = x[0], x_end = x[-1],
                                delta_m0 = fs.delta_m(x[0]), nu = nu_inf)
    rtn = tmn.solve()
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    
    ## Calculate the boundary layer parameters
    delta_d_exact = fs.delta_d(x)
    delta_d_standard = tml.delta_d(x)
    delta_d_nonlinear = tmn.delta_d(x)
    delta_m_exact = fs.delta_m(x)
    delta_m_standard = tml.delta_m(x)
    delta_m_nonlinear = tmn.delta_m(x)
    c_f_exact = fs.tau_w(x, rho_inf)/(0.5*rho_inf*U_inf**2)
    c_f_standard = tml.tau_w(x, rho_inf)/(0.5*rho_inf*U_inf**2)
    c_f_nonlinear = tmn.tau_w(x, rho_inf)/(0.5*rho_inf*U_inf**2)
    H_d_exact = fs.H_d(x)
    H_d_standard = tml.H_d(x)
    H_d_nonlinear = tmn.H_d(x)
    V_e_exact = fs.V_e(x)
    V_e_standard = tml.V_e(x)
    V_e_nonlinear = tmn.V_e(x)
    
    ## plot functions
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(15)
    gs = GridSpec(5, 2, figure = fig)
    axis_delta_d = fig.add_subplot(gs[0, 0])
    axis_delta_d_error = fig.add_subplot(gs[0, 1])
    axis_delta_m = fig.add_subplot(gs[1, 0])
    axis_delta_m_error = fig.add_subplot(gs[1, 1])
    axis_H_d = fig.add_subplot(gs[2, 0])
    axis_H_d_error = fig.add_subplot(gs[2, 1])
    axis_c_f = fig.add_subplot(gs[3, 0])
    axis_c_f_error = fig.add_subplot(gs[3, 1])
    axis_V_e = fig.add_subplot(gs[4, 0])
    axis_V_e_error = fig.add_subplot(gs[4, 1])
    
    exact_color = "black"
    exact_label = "Blasius"
    standard_color = "green"
    standard_label = "Standard"
    nonlinear_color = "red"
    nonlinear_label = "Nonlinear"
    
    # Displacement thickness in 0,:
    ax = axis_delta_d
    exact_curve = ax.plot(x/c, delta_d_exact/c, color = exact_color,
                          label = exact_label)
    standard_curve = ax.plot(x/c, delta_d_standard/c, color = standard_color,
            label = standard_label)
    nonlinear_curve = ax.plot(x/c, delta_d_nonlinear/c, color = nonlinear_color,
            label = nonlinear_label)
    ax.set_ylim(0, 0.0008)
    ax.set_ylabel(r"$\delta_d/c$")
    ax.grid(True)
    
    ax = axis_delta_d_error
    ax.plot(x/c, np.abs(1-delta_d_standard/delta_d_exact),
            color = standard_color)
    ax.plot(x/c, np.abs(1-delta_d_nonlinear/delta_d_exact),
            color = nonlinear_color)
    ax.set_ylabel("Relative Error")
    ax.set_ylim([1e-4,1])
    ax.set_yscale('log')
    ax.grid(True)
    
    # Momentum thickness in 1,:
    ax = axis_delta_m
    ax.plot(x/c, delta_m_exact/c, color = exact_color)
    ax.plot(x/c, delta_m_standard/c, color = standard_color)
    ax.plot(x/c, delta_m_nonlinear/c, color = nonlinear_color)
    ax.set_ylim(0, 0.0004)
    ax.set_ylabel(r"$\delta_m/c$")
    ax.grid(True)
    
    ax = axis_delta_m_error
    ax.plot(x/c, np.abs(1-delta_m_standard/delta_m_exact),
            color = standard_color)
    ax.plot(x/c, np.abs(1-delta_m_nonlinear/delta_m_exact),
            color = nonlinear_color)
    ax.set_ylabel("Relative Error")
    ax.set_ylim([1e-4,1])
    ax.set_yscale('log')
    ax.grid(True)
    
    # Displacement shape factor in 2,:
    ax = axis_H_d
    ax.plot(x/c, H_d_exact, color = exact_color)
    ax.plot(x/c, H_d_standard, color = standard_color)
    ax.plot(x/c, H_d_nonlinear, color = nonlinear_color)
    ax.set_ylim(2.2, 2.5)
    ax.set_ylabel(r"$H_d$")
    ax.grid(True)
    
    ax = axis_H_d_error
    ax.plot(x/c, np.abs(1-H_d_standard/H_d_exact),
            color = standard_color)
    ax.plot(x/c, np.abs(1-H_d_nonlinear/H_d_exact),
            color = nonlinear_color)
    ax.set_ylabel("Relative Error")
    ax.set_ylim([1e-4,1])
    ax.set_yscale('log')
    ax.grid(True)
    
    # Skin friction coefficient in 3,:
    ax = axis_c_f
    ax.plot(x/c, c_f_exact, color = exact_color)
    ax.plot(x/c, c_f_standard, color = standard_color)
    ax.plot(x/c, c_f_nonlinear, color = nonlinear_color)
    ax.set_ylim(0.001, 0.002)
    ax.set_ylabel(r"$c_f$")
    ax.grid(True)
    
    ax = axis_c_f_error
    ax.plot(x/c, np.abs(1-c_f_standard/c_f_exact),
            color = standard_color)
    ax.plot(x/c, np.abs(1-c_f_nonlinear/c_f_exact),
            color = nonlinear_color)
    ax.set_ylabel("Relative Error")
    ax.set_ylim([1e-4,1])
    ax.set_yscale('log')
    ax.grid(True)
    
    # Transpiration velocity in 4,:
    ax = axis_V_e
    ax.plot(x/c, V_e_exact/U_inf, color = exact_color)
    ax.plot(x/c, V_e_standard/U_inf, color = standard_color)
    ax.plot(x/c, V_e_nonlinear/U_inf, color = nonlinear_color)
    ax.set_ylim(0, 0.01)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$V_e/U_\infty$")
    ax.grid(True)
    
    ax = axis_V_e_error
    ax.plot(x/c, np.abs(1-V_e_standard/V_e_exact),
            color = standard_color)
    ax.plot(x/c, np.abs(1-V_e_nonlinear/V_e_exact),
            color = nonlinear_color)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel("Relative Error")
    ax.set_ylim([1e-4,1])
    ax.set_yscale('log')
    ax.grid(True)
    
    fig.subplots_adjust(bottom=0.075, wspace=0.5)
    fig.legend(handles = [exact_curve[0], standard_curve[0],
                          nonlinear_curve[0]], 
               labels = [exact_label, standard_label, nonlinear_label],
               loc = "upper center", bbox_to_anchor = (0.45, 0.03),
               ncol = 3, borderaxespad = 0.1)


if (__name__ == "__main__"):
    compare_stagnation_solution()

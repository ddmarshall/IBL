#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example shows a comparison between Head's method and case 1100 from the 
1968 Stanford Olympics from Luwieg and Tillman. It shows similar results to 
Figures 3.10 to 3.16 in Edland thesis.

Created on Tue Aug 16 16:20:20 2022

@author: ddmarshall
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pyBL.head_method import HeadMethod
from pyBL.stanford_olympics import StanfordOlympics1968


def compare_case1100():
    so1100 = StanfordOlympics1968("1100")
    x, U_e, _ = so1100.get_smooth_velocity()
    rho = 1.2
    
    hm = HeadMethod(U_e = [x, U_e])
    hm.set_solution_parameters(x0 = so1100.get_x()[0],
                               x_end = so1100.get_x()[-1],
                               delta_m0 = so1100.get_delta_m()[0],
                               H_d0 = so1100.get_H_d()[0],
                               nu = so1100.nu)
    rtn = hm.solve()
    if not rtn.success:
        print("Could not get solution for Head method: " + rtn.message)
        return
    
    ## Calculate the boundary layer parameters
    x_ref = so1100.get_x()
    delta_d_ref = so1100.get_delta_d()
    delta_m_ref = so1100.get_delta_m()
    H_d_ref = so1100.get_H_d()
    c_f_ref = so1100.get_c_f()
    
    temp = hm.delta_d(x_ref)
    
    # TODO: Replace this with actual calculations
    x = np.linspace(x_ref[0], x_ref[-1], 101)
    delta_d_head = hm.delta_d(x)
    delta_m_head = hm.delta_m(x)
    H_d_head = hm.H_d(x)
#    c_f_head = 2*hm.tau_w(x, rho)/(rho*hm.U_e(x))
    
    ## Plot results
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(15)
    gs = GridSpec(4, 2, figure = fig)
    axis_delta_d = fig.add_subplot(gs[0, 0])
    axis_delta_d_diff = fig.add_subplot(gs[0, 1])
    axis_delta_m = fig.add_subplot(gs[1, 0])
    axis_delta_m_diff = fig.add_subplot(gs[1, 1])
    axis_H_d = fig.add_subplot(gs[2, 0])
    axis_H_d_diff = fig.add_subplot(gs[2, 1])
    axis_c_f = fig.add_subplot(gs[3, 0])
    axis_c_f_diff = fig.add_subplot(gs[3, 1])
#    axis_U_n = fig.add_subplot(gs[4, 0])
#    axis_U_n_diff = fig.add_subplot(gs[4, 1])
    
    ref_color = "black"
    ref_label = "Ludwieg & Tillman"
    head_color = "green"
    head_label = "Head"
    
    # Displacement thickness in 0,:
    ax = axis_delta_d
    ref_curve = ax.plot(x_ref, delta_d_ref, color = ref_color,
                          label = ref_label)
    head_curve = ax.plot(x, delta_d_head, color = head_color,
            label = head_label)
    ax.set_ylim(0, 0.040)
    ax.set_ylabel(r"$\delta_d$ (m)")
    ax.grid(True)
    
    ax = axis_delta_d_diff
    ax.plot(x_ref, np.abs(1-hm.delta_d(x_ref)/delta_d_ref), color = head_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-3,1])
    ax.set_yscale('log')
    ax.grid(True)
    
    # Momentum thickness in 1,:
    ax = axis_delta_m
    ax.plot(x_ref, delta_m_ref, color = ref_color)
    ax.plot(x, delta_m_head, color = head_color)
    ax.set_ylim(0, 0.025)
    ax.set_ylabel(r"$\delta_m$ (m)")
    ax.grid(True)
    
    ax = axis_delta_m_diff
    ax.plot(x_ref, np.abs(1-hm.delta_m(x_ref)/delta_m_ref), color = head_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-3,1])
    ax.set_yscale('log')
    ax.grid(True)
    
    # Displacement shape factor in 2,:
    ax = axis_H_d
    ax.plot(x_ref, H_d_ref, color = ref_color)
    ax.plot(x, H_d_head, color = head_color)
    ax.set_ylim(1, 2)
    ax.set_ylabel(r"$H_d$")
    ax.grid(True)
    
    ax = axis_H_d_diff
    ax.plot(x_ref, np.abs(1-hm.H_d(x_ref)/H_d_ref), color = head_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-3,1])
    ax.set_yscale('log')
    ax.grid(True)
    
#    # Skin friction coefficient in 3,:
#    ax = axis_c_f
#    ax.plot(x_ref, c_f_ref, color = ref_color)
#    ax.plot(x_ref, c_f_head, color = head_color)
#    ax.set_ylim(0, 0.003)
#    ax.set_ylabel(r"$c_f$")
#    ax.grid(True)
#    
#    ax = axis_c_f_error
#    ax.plot(x/c, np.abs(1-c_f_standard/c_f_exact), color = head_color)
#    ax.set_ylabel("Relative Difference")
#    ax.set_ylim([1e-3,1])
#    ax.set_yscale('log')
#    ax.grid(True)
    
#    # Transpiration velocity in 4,:
#    ax = axis_U_n
#    ax.plot(x, U_n_exact/U_inf, color = exact_color)
#    ax.plot(x, U_n_standard/U_inf, color = standard_color)
#    ax.set_ylim(0, 0.01)
#    ax.set_xlabel(r"$x$ (m)")
#    ax.set_ylabel(r"$U_n/U_\infty$")
#    ax.grid(True)
#    
#    ax = axis_U_n_error
#    ax.plot(x/c, np.abs(1-U_n_standard/U_n_exact),
#            color = standard_color)
#    ax.plot(x/c, np.abs(1-U_n_nonlinear/U_n_exact),
#            color = nonlinear_color)
#    ax.set_xlabel(r"$x/c$")
#    ax.set_ylabel("Relative Error")
#    ax.set_ylim([1e-4,1])
#    ax.set_yscale('log')
#    ax.grid(True)
#    
    fig.subplots_adjust(bottom=0.075, wspace=0.5)
    fig.legend(handles = [ref_curve[0], head_curve[0]], 
               labels = [ref_label, head_label],
               loc = "upper center", bbox_to_anchor = (0.45, 0.03),
               ncol = 2, borderaxespad = 0.1)


if (__name__ == "__main__"):
    compare_case1100()

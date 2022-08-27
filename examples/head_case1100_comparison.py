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
    so68 = StanfordOlympics1968("1100")
    x, U_e, dU_edx = so68.get_velocity()
    x_sm, U_e_sm, dU_edx_sm = so68.get_smooth_velocity()
    rho = 1.2
    
    hm_reg = HeadMethod(U_e = [x, U_e])
    hm_reg.set_solution_parameters(x0 = so68.get_x()[0],
                                   x_end = so68.get_x()[-1],
                                   delta_m0 = so68.get_delta_m()[0],
                                   H_d0 = so68.get_H_d()[0],
                                   nu = so68.nu)
    rtn = hm_reg.solve()
    if not rtn.success:
        print("Could not get solution for Head method: " + rtn.message)
        return
    
    hm_sm = HeadMethod(U_e = [x_sm, U_e_sm])
    hm_sm.set_solution_parameters(x0 = so68.get_x()[0],
                                 x_end = so68.get_x()[-1],
                                 delta_m0 = so68.get_delta_m()[0],
                                 H_d0 = so68.get_H_d()[0],
                                 nu = so68.nu)
    rtn = hm_sm.solve()
    if not rtn.success:
        print("Could not get solution for Head method: " + rtn.message)
        return
    
    hm_sm2 = HeadMethod(U_e = U_e[0], dU_edx = [x, dU_edx])
    hm_sm2.set_solution_parameters(x0 = so68.get_x()[0],
                                  x_end = so68.get_x()[-1],
                                  delta_m0 = so68.get_delta_m()[0],
                                  H_d0 = so68.get_H_d()[0],
                                  nu = so68.nu)
    rtn = hm_sm2.solve()
    if not rtn.success:
        print("Could not get solution for Head method: " + rtn.message)
        return
    
    ## Calculate the boundary layer parameters
    x_ref = so68.get_x()
    delta_d_ref = so68.get_delta_d()
    delta_m_ref = so68.get_delta_m()
    H_d_ref = so68.get_H_d()
    c_f_ref = so68.get_c_f()
    U_e_ref = so68.get_U_e()
    
    x = np.linspace(x_ref[0], x_ref[-1], 101)
    delta_d_head_reg = hm_reg.delta_d(x)
    delta_m_head_reg = hm_reg.delta_m(x)
    H_d_head_reg = hm_reg.H_d(x)
    c_f_head_reg = 2*hm_sm.tau_w(x, rho)/(rho*hm_reg.U_e(x)**2)
    U_n_head_reg = hm_reg.U_n(x)
    delta_d_head_sm = hm_sm.delta_d(x)
    delta_m_head_sm = hm_sm.delta_m(x)
    H_d_head_sm = hm_sm.H_d(x)
    c_f_head_sm = 2*hm_sm.tau_w(x, rho)/(rho*hm_sm.U_e(x)**2)
    U_n_head_sm = hm_sm.U_n(x)
    delta_d_head_sm2 = hm_sm2.delta_d(x)
    delta_m_head_sm2 = hm_sm2.delta_m(x)
    H_d_head_sm2 = hm_sm2.H_d(x)
    c_f_head_sm2 = 2*hm_sm2.tau_w(x, rho)/(rho*hm_sm2.U_e(x)**2)
    U_n_head_sm2 = hm_sm2.U_n(x)
    
    ## Plot results
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(15)
    gs = GridSpec(6, 2, figure = fig)
    axis_delta_d = fig.add_subplot(gs[0, 0])
    axis_delta_d_diff = fig.add_subplot(gs[0, 1])
    axis_delta_m = fig.add_subplot(gs[1, 0])
    axis_delta_m_diff = fig.add_subplot(gs[1, 1])
    axis_H_d = fig.add_subplot(gs[2, 0])
    axis_H_d_diff = fig.add_subplot(gs[2, 1])
    axis_c_f = fig.add_subplot(gs[3, 0])
    axis_c_f_diff = fig.add_subplot(gs[3, 1])
    axis_U_e = fig.add_subplot(gs[4, 0])
    axis_U_e_diff = fig.add_subplot(gs[4, 1])
    axis_dU_edx = fig.add_subplot(gs[5, 0])
    axis_U_n = fig.add_subplot(gs[5, 1])
    
    ref_color = "black"
    ref_label = "Ludwieg & Tillman"
    head_reg_color = "red"
    head_reg_label = "Head"
    head_sm_color = "green"
    head_sm_label = "Head (Smooth $U_e$)"
    head_sm2_color = "orange"
    head_sm2_label = "Head (Smooth d$U_e/$d$x$)"
    
    # Displacement thickness in 0,:
    ax = axis_delta_d
    ref_curve = ax.plot(x_ref, delta_d_ref, color = ref_color, linestyle = "",
                        marker = "o", label = ref_label)
    head_reg_curve = ax.plot(x, delta_d_head_reg, color = head_reg_color,
                             label = head_reg_label)
    head_sm_curve = ax.plot(x, delta_d_head_sm, color = head_sm_color,
                            label = head_sm_label)
    head_sm2_curve = ax.plot(x, delta_d_head_sm2, color = head_sm2_color,
                             label = head_sm2_label)
    ax.set_ylim(0, 0.05)
    ax.set_ylabel(r"$\delta_d$ (m)")
    ax.grid(True)
    
    ax = axis_delta_d_diff
    ax.plot(x_ref, np.abs(1-hm_reg.delta_d(x_ref)/delta_d_ref),
            color = head_reg_color)
    ax.plot(x_ref, np.abs(1-hm_sm.delta_d(x_ref)/delta_d_ref),
            color = head_sm_color)
    ax.plot(x_ref, np.abs(1-hm_sm2.delta_d(x_ref)/delta_d_ref),
            color = head_sm2_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-3,1])
    ax.set_yscale('log')
    ax.grid(True)
    
    # Momentum thickness in 1,:
    ax = axis_delta_m
    ax.plot(x_ref, delta_m_ref, color = ref_color, linestyle = "", marker = "o")
    ax.plot(x, delta_m_head_reg, color = head_reg_color)
    ax.plot(x, delta_m_head_sm, color = head_sm_color)
    ax.plot(x, delta_m_head_sm2, color = head_sm2_color)
    ax.set_ylim(0, 0.03)
    ax.set_ylabel(r"$\delta_m$ (m)")
    ax.grid(True)
    
    ax = axis_delta_m_diff
    ax.plot(x_ref, np.abs(1-hm_reg.delta_m(x_ref)/delta_m_ref),
            color = head_reg_color)
    ax.plot(x_ref, np.abs(1-hm_sm.delta_m(x_ref)/delta_m_ref),
            color = head_sm_color)
    ax.plot(x_ref, np.abs(1-hm_sm2.delta_m(x_ref)/delta_m_ref),
            color = head_sm2_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-3,1])
    ax.set_yscale('log')
    ax.grid(True)
    
    # Displacement shape factor in 2,:
    ax = axis_H_d
    ax.plot(x_ref, H_d_ref, color = ref_color, linestyle = "", marker = "o")
    ax.plot(x, H_d_head_reg, color = head_reg_color)
    ax.plot(x, H_d_head_sm, color = head_sm_color)
    ax.plot(x, H_d_head_sm2, color = head_sm2_color)
    ax.set_ylim(1, 2)
    ax.set_ylabel(r"$H_d$")
    ax.grid(True)
    
    ax = axis_H_d_diff
    ax.plot(x_ref, np.abs(1-hm_reg.H_d(x_ref)/H_d_ref), color = head_reg_color)
    ax.plot(x_ref, np.abs(1-hm_sm.H_d(x_ref)/H_d_ref), color = head_sm_color)
    ax.plot(x_ref, np.abs(1-hm_sm2.H_d(x_ref)/H_d_ref), color = head_sm2_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-3,1])
    ax.set_yscale('log')
    ax.grid(True)
    
    # Skin friction coefficient in 3,:
    ax = axis_c_f
    ax.plot(x_ref, c_f_ref, color = ref_color, linestyle = "", marker = "o")
    ax.plot(x, c_f_head_reg, color = head_reg_color)
    ax.plot(x, c_f_head_sm, color = head_sm_color)
    ax.plot(x, c_f_head_sm2, color = head_sm2_color)
    ax.set_ylim(0, 0.003)
    ax.set_ylabel(r"$c_f$")
    ax.grid(True)
    
    ax = axis_c_f_diff
    ax.plot(x_ref, np.abs(1-2*hm_reg.tau_w(x_ref, rho)/(rho*hm_reg.U_e(x_ref)**2)/c_f_ref),
            color = head_reg_color)
    ax.plot(x_ref, np.abs(1-2*hm_sm.tau_w(x_ref, rho)/(rho*hm_sm.U_e(x_ref)**2)/c_f_ref),
            color = head_sm_color)
    ax.plot(x_ref, np.abs(1-2*hm_sm2.tau_w(x_ref, rho)/(rho*hm_sm2.U_e(x_ref)**2)/c_f_ref),
            color = head_sm2_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-3,1])
    ax.set_yscale('log')
    ax.grid(True)
    
    # Edge velocity in 4,:
    ax = axis_U_e
    ax.plot(x_sm, U_e_sm, color = ref_color, linestyle = "", marker = "o")
    ax.plot(x, hm_reg.U_e(x), color = head_reg_color)
    ax.plot(x, hm_sm.U_e(x), color = head_sm_color)
    ax.plot(x, hm_sm2.U_e(x), color = head_sm2_color)
    ax.set_ylim(20, 35)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$U_e$ (m/s)")
    ax.grid(True)
    
    ax = axis_U_e_diff
    ax.plot(x_ref, np.abs(1-hm_reg.U_e(x_ref)/U_e_ref), color = head_reg_color)
    ax.plot(x_ref, np.abs(1-hm_sm.U_e(x_ref)/U_e_ref), color = head_sm_color)
    ax.plot(x_ref, np.abs(1-hm_sm2.U_e(x_ref)/U_e_ref), color = head_sm2_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-3,1])
    ax.set_yscale('log')
    ax.grid(True)

    # Transpiration velocity in 5,:
    ax = axis_dU_edx
    ax.plot(x_sm, dU_edx_sm, color = ref_color, linestyle = "", marker = "o")
    ax.plot(x, hm_reg.dU_edx(x), color = head_reg_color)
    ax.plot(x, hm_sm.dU_edx(x), color = head_sm_color)
    ax.plot(x, hm_sm2.dU_edx(x), color = head_sm2_color)
    ax.set_ylim(-1, -5)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"d$U_e/$d$x$ (1/s)")
    ax.grid(True)
    
    ax = axis_U_n
    ax.plot(x, U_n_head_reg, color = head_reg_color)
    ax.plot(x, U_n_head_sm, color = head_sm_color)
    ax.plot(x, U_n_head_sm2, color = head_sm2_color)
    ax.set_ylim(0, 0.4)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$U_n$ (m/s)")
    ax.grid(True)
    
    fig.subplots_adjust(bottom=0.075, wspace=0.5)
    fig.legend(handles = [ref_curve[0], head_reg_curve[0], head_sm_curve[0],
                          head_sm2_curve[0]], 
               labels = [ref_label, head_reg_label, head_sm_label,
                         head_sm2_label],
               loc = "upper center", bbox_to_anchor = (0.45, 0.03),
               ncol = 4, borderaxespad = 0.1)


if (__name__ == "__main__"):
    compare_case1100()

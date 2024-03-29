#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparing Head's method solution against XFoil case.

This example shows a comparison between Head's method and XFoil turbulent
results for a NACA 0012 airfoil using the XFoil edge velocity profile. It shows
similar results to Figures 3.20 to 3.21 in Edland thesis.
"""

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pyBL.head_method import HeadMethod
from pyBL.xfoil_reader import XFoilReader


def compare_xfoil_turbulent():
    """Compare the Head's method results to XFoil results."""
    # Read in XFoil data
    inv_filename = "../data/xfoil_0012_inviscid_dump.txt"
    visc_filename = "../data/xfoil_0012_turbulent_dump.txt"
    airfoil_name = "NACA 0012"
    alpha = 0
    c = 1  # (m)
    U_inf = 20  # (m/s)
    Re = 1e6
    rho_inf = 1.2
    nu_inf = U_inf*c/Re
    x_trans = 0.001
    n_trans = 9
    xfoil_inv = XFoilReader(inv_filename, airfoil=airfoil_name,
                            alpha=alpha, c=c)
    xfoil_visc = XFoilReader(visc_filename, airfoil=airfoil_name,
                             alpha=alpha, c=c, Re=Re, x_trans=x_trans,
                             n_trans=n_trans)

    s_ref = np.array(xfoil_inv.s_upper())
    U_e_inv = U_inf*np.array(xfoil_inv.U_e_upper())
    U_e_visc = U_inf*np.array(xfoil_visc.U_e_upper())
    s = np.linspace(s_ref[0], s_ref[-1], 101)

    # Setup Head methods
    delta_m0 = xfoil_visc.delta_m_upper()[0]
    H_d0 = xfoil_visc.H_d_upper()[0]
    hm_visc = HeadMethod(nu=nu_inf, U_e=[s_ref, U_e_visc])
    hm_visc.set_initial_parameters(delta_m0=delta_m0, H_d0=H_d0)
    rtn = hm_visc.solve(x0=s[0], x_end=s[-1])
    if not rtn.success:
        print("Could not get solution for Head method: " + rtn.message)
        return
    s_sep_visc = np.inf
    if rtn.status == -1:
        s_sep_visc = rtn.x_end

    hm_inv = HeadMethod(nu=nu_inf, U_e=[s_ref, U_e_inv])
    delta_m0 = np.sqrt(0.075*nu_inf/hm_inv.dU_edx(s[0]))  # Moran's method
    H_d0 = 2.35  # Thwaites method predicts this value for stagnation flow
    hm_inv.set_initial_parameters(delta_m0=delta_m0, H_d0=H_d0)
    rtn = hm_inv.solve(x0=s[0], x_end=s[-1])
    if not rtn.success:
        print("Could not get solution for Head method: " + rtn.message)
        return
    s_sep_inv = np.inf
    if rtn.status == -1:
        s_sep_inv = rtn.x_end

    # Calculate the boundary layer parameters
    delta_d_ref = np.array(xfoil_visc.delta_d_upper())
    delta_m_ref = np.array(xfoil_visc.delta_m_upper())
    H_d_ref = np.array(xfoil_visc.H_d_upper())
    c_f_ref = np.array(xfoil_visc.c_f_upper())

    s_ref_visc = s_ref[s_ref < s_sep_visc]
    delta_d_ref_visc = delta_d_ref[s_ref < s_sep_visc]
    delta_m_ref_visc = delta_m_ref[s_ref < s_sep_visc]
    H_d_ref_visc = H_d_ref[s_ref < s_sep_visc]
    c_f_ref_visc = c_f_ref[s_ref < s_sep_visc]

    s_ref_inv = s_ref[s_ref < s_sep_inv]
    delta_d_ref_inv = delta_d_ref[s_ref < s_sep_inv]
    delta_m_ref_inv = delta_m_ref[s_ref < s_sep_inv]
    H_d_ref_inv = H_d_ref[s_ref < s_sep_inv]
    c_f_ref_inv = c_f_ref[s_ref < s_sep_inv]

    s_visc = np.linspace(s_ref[0], min(s_ref[-1], s_sep_visc), 101)
    delta_d_visc = hm_visc.delta_d(s_visc)
    delta_m_visc = hm_visc.delta_m(s_visc)
    H_d_visc = hm_visc.H_d(s_visc)
    c_f_visc = 2*hm_visc.tau_w(s_visc, rho_inf)/(rho_inf*U_inf**2)
    V_e_visc = hm_visc.V_e(s_visc)
    dU_edx_visc = hm_visc.dU_edx(s_visc)
    d2U_edx2_visc = hm_visc.d2U_edx2(s_visc)

    s_inv = np.linspace(s_ref[0], min(s_ref[-1], s_sep_inv), 101)
    delta_d_inv = hm_inv.delta_d(s_inv)
    delta_m_inv = hm_inv.delta_m(s_inv)
    H_d_inv = hm_inv.H_d(s_inv)
    c_f_inv = 2*hm_inv.tau_w(s_inv, rho_inf)/(rho_inf*U_inf**2)
    V_e_inv = hm_inv.V_e(s_inv)
    dU_edx_inv = hm_inv.dU_edx(s_inv)
    d2U_edx2_inv = hm_inv.d2U_edx2(s_inv)

    # Plot results
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(15)
    gs = GridSpec(6, 2, figure=fig)
    axis_delta_d = fig.add_subplot(gs[0, 0])
    axis_delta_d_diff = fig.add_subplot(gs[0, 1])
    axis_delta_m = fig.add_subplot(gs[1, 0])
    axis_delta_m_diff = fig.add_subplot(gs[1, 1])
    axis_H_d = fig.add_subplot(gs[2, 0])
    axis_H_d_diff = fig.add_subplot(gs[2, 1])
    axis_c_f = fig.add_subplot(gs[3, 0])
    axis_c_f_diff = fig.add_subplot(gs[3, 1])
    axis_U_e = fig.add_subplot(gs[4, 0])
    axis_dU_edx = fig.add_subplot(gs[4, 1])
    axis_d2U_edx2 = fig.add_subplot(gs[5, 0])
    axis_V_e = fig.add_subplot(gs[5, 1])

    ref_color = "black"
    ref_label = "XFoil"
    thwaites_visc_color = "red"
    thwaites_visc_label = "Head (Viscous $U_e$)"
    thwaites_inv_color = "blue"
    thwaites_inv_label = "Head (Inviscid $U_e$)"

    # Displacement thickness in 0,:
    ax = axis_delta_d
    ref_curve = ax.plot(s_ref/c, delta_d_ref/c, color=ref_color,
                        label=ref_label)
    thwaites_visc_curve = ax.plot(s_visc/c, delta_d_visc/c,
                                  color=thwaites_visc_color,
                                  label=thwaites_visc_label)
    thwaites_inv_curve = ax.plot(s_inv/c, delta_d_inv/c,
                                 color=thwaites_inv_color,
                                 label=thwaites_inv_label)
    ax.set_ylim(0, 0.007)
    ax.set_ylabel(r"$\delta_d/c$")
    ax.grid(True)

    ax = axis_delta_d_diff
    ax.plot(s_ref_visc/c,
            np.abs(1-hm_visc.delta_d(s_ref_visc)/delta_d_ref_visc),
            color=thwaites_visc_color)
    ax.plot(s_ref_inv/c, np.abs(1-hm_inv.delta_d(s_ref_inv)/delta_d_ref_inv),
            color=thwaites_inv_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-3,1])
    ax.set_yscale('log')
    ax.grid(True)

    # Momentum thickness in 1,:
    ax = axis_delta_m
    ax.plot(s_ref/c, delta_m_ref/c, color=ref_color)
    ax.plot(s_visc/c, delta_m_visc/c, color=thwaites_visc_color)
    ax.plot(s_inv/c, delta_m_inv/c, color=thwaites_inv_color)
    ax.set_ylim(0, 0.0043)
    ax.set_ylabel(r"$\delta_m/c$")
    ax.grid(True)

    ax = axis_delta_m_diff
    ax.plot(s_ref_visc/c,
            np.abs(1-hm_visc.delta_m(s_ref_visc)/delta_m_ref_visc),
            color=thwaites_visc_color)
    ax.plot(s_ref_inv/c, np.abs(1-hm_inv.delta_m(s_ref_inv)/delta_m_ref_inv),
            color=thwaites_inv_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-4,1])
    ax.set_yscale('log')
    ax.grid(True)

    # Displacement shape factor in 2,:
    ax = axis_H_d
    ax.plot(s_ref/c, H_d_ref, color=ref_color)
    ax.plot(s_visc/c, H_d_visc, color=thwaites_visc_color)
    ax.plot(s_inv/c, H_d_inv, color=thwaites_inv_color)
    ax.set_ylim(1.4, 2.4)
    ax.set_ylabel(r"$H_d$")
    ax.grid(True)

    ax = axis_H_d_diff
    ax.plot(s_ref_visc/c, np.abs(1-hm_visc.H_d(s_ref_visc)/H_d_ref_visc),
            color=thwaites_visc_color)
    ax.plot(s_ref_inv/c, np.abs(1-hm_inv.H_d(s_ref_inv)/H_d_ref_inv),
            color=thwaites_inv_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-3,1])
    ax.set_yscale('log')
    ax.grid(True)

    # Skin friction coefficient in 3,:
    ax = axis_c_f
    ax.plot(s_ref/c, c_f_ref, color=ref_color)
    ax.plot(s_visc/c, c_f_visc, color=thwaites_visc_color)
    ax.plot(s_inv/c, c_f_inv, color=thwaites_inv_color)
    ax.set_ylim(0, 0.01)
    ax.set_ylabel(r"$c_f$")
    ax.grid(True)

    ax = axis_c_f_diff
    ax.plot(s_ref_visc/c,
            np.abs(1-2*hm_visc.tau_w(s_ref_visc,
                                     rho_inf)/(rho_inf*U_inf**2)/c_f_ref_visc),
            color=thwaites_visc_color)
    ax.plot(s_ref_inv/c,
            np.abs(1-2*hm_inv.tau_w(s_ref_inv,
                                    rho_inf)/(rho_inf*U_inf**2)/c_f_ref_inv),
            color=thwaites_inv_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-3,1])
    ax.set_yscale('log')
    ax.grid(True)

    # Edge velocity in 4,:
    ax = axis_U_e
    ax.plot(s_ref/c, U_e_visc/U_inf, color=thwaites_visc_color)
    ax.plot(s_ref/c, U_e_inv/U_inf, color=thwaites_inv_color)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel(r"$U_e/U_\infty$")
    ax.grid(True)

    ax = axis_dU_edx
    ax.plot(s_visc/c, dU_edx_visc, color=thwaites_visc_color)
    ax.plot(s_inv/c, dU_edx_inv, color=thwaites_inv_color)
    ax.set_ylim(-10, 0)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"d$U_e/$d$x$ (1/s)")
    ax.grid(True)

    # Transpiration velocity in 5,:
    ax = axis_d2U_edx2
    ax.plot(s_visc/c, d2U_edx2_visc, color=thwaites_visc_color)
    ax.plot(s_inv/c, d2U_edx2_inv, color=thwaites_inv_color)
    ax.set_ylim(-20, 5)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"d$^2U_e/$d$x^2$ (1/(m$\cdot$s)")
    ax.grid(True)

    ax = axis_V_e
    ax.plot(s_visc/c, V_e_visc/U_inf, color=thwaites_visc_color)
    ax.plot(s_inv/c, V_e_inv/U_inf, color=thwaites_inv_color)
    ax.set_ylim(0, 0.01)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$V_e/U_e$")
    ax.grid(True)

    fig.subplots_adjust(bottom=0.075, wspace=0.5)
    fig.legend(handles=[ref_curve[0], thwaites_visc_curve[0],
                        thwaites_inv_curve[0]],
               labels=[ref_label, thwaites_visc_label, thwaites_inv_label],
               loc="upper center", bbox_to_anchor=(0.45, 0.03), ncol=4,
               borderaxespad=0.1)


if __name__ == "__main__":
    compare_xfoil_turbulent()

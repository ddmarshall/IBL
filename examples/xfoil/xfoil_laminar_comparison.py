#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparing Thwaites' method solution against XFoil case.

This example shows a comparison between Thwaites' method and XFoil laminar
results for a NACA 0003 airfoil using the XFoil edge velocity profile. It shows
similar results to Figures 3.17 to 3.19 in Edland thesis.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ibl.thwaites_method import ThwaitesMethodNonlinear
from ibl.reference import XFoilReader


def compare_xfoil_laminar() -> None:
    """Compare the Thwaites' method results to XFoil results."""
    # Read in XFoil data
    data_dir = Path(__file__).resolve().parent.parent.joinpath("data")
    if not data_dir.exists():
        raise IOError(f"Cannot find data directory: {data_dir}")

    inv_file = data_dir.joinpath("xfoil_0003_inviscid_dump.txt")
    visc_file = data_dir.joinpath("xfoil_0003_laminar_dump.txt")
    airfoil_name = "NACA 0003"
    alpha = 0
    c = 1  # (m)
    u_inf = 20  # (m/s)
    re = 1000
    rho_inf = 1.2
    nu_inf = u_inf*c/re
    x_trans = 1
    n_trans = 9
    xfoil_inv = XFoilReader(str(inv_file))
    xfoil_inv.name = airfoil_name
    xfoil_inv.alpha = alpha
    xfoil_inv.c = c
    xfoil_inv.u_ref = u_inf
    xfoil_visc = XFoilReader(str(visc_file))
    xfoil_visc.name = airfoil_name
    xfoil_visc.alpha = alpha
    xfoil_visc.c = c
    xfoil_visc.u_ref = u_inf
    xfoil_visc.reynolds = re
    xfoil_visc.x_trans_lower = x_trans
    xfoil_visc.x_trans_upper = x_trans
    xfoil_visc.n_trans = n_trans

    s_ref = xfoil_inv.s_upper()
    u_e_inv = xfoil_inv.u_e_upper()
    u_e_visc = xfoil_visc.u_e_upper()
    s = np.linspace(s_ref[0], s_ref[-1], 101)

    # Setup Thwaites methods
    tm_visc = ThwaitesMethodNonlinear(nu=nu_inf, U_e=[s_ref, u_e_visc],
                                      data_fits="Spline")
    rtn = tm_visc.solve(x0=s[0], x_end=s[-1])
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    s_sep_visc = np.inf
    if rtn.status == -1:
        s_sep_visc = rtn.x_end

    tm_inv = ThwaitesMethodNonlinear(nu=nu_inf, U_e=[s_ref, u_e_inv],
                                     data_fits="Spline")
    rtn = tm_inv.solve(x0=s[0], x_end=s[-1])
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
        return
    s_sep_inv = np.inf
    if rtn.status == -1:
        s_sep_inv = rtn.x_end

    # Calculate the boundary layer parameters
    delta_d_ref = xfoil_visc.delta_d_upper()
    delta_m_ref = xfoil_visc.delta_m_upper()
    shape_d_ref = xfoil_visc.shape_d_upper()
    c_f_ref = xfoil_visc.c_f_upper()

    s_ref_visc = s_ref[s_ref < s_sep_visc]
    delta_d_ref_visc = delta_d_ref[s_ref < s_sep_visc]
    delta_m_ref_visc = delta_m_ref[s_ref < s_sep_visc]
    shape_d_ref_visc = shape_d_ref[s_ref < s_sep_visc]
    c_f_ref_visc = c_f_ref[s_ref < s_sep_visc]

    s_ref_inv = s_ref[s_ref < s_sep_inv]
    delta_d_ref_inv = delta_d_ref[s_ref < s_sep_inv]
    delta_m_ref_inv = delta_m_ref[s_ref < s_sep_inv]
    shape_d_ref_inv = shape_d_ref[s_ref < s_sep_inv]
    c_f_ref_inv = c_f_ref[s_ref < s_sep_inv]

    s_visc = np.linspace(s_ref[0], min(s_ref[-1], s_sep_visc), 101)
    delta_d_visc = tm_visc.delta_d(s_visc)
    delta_m_visc = tm_visc.delta_m(s_visc)
    shape_d_visc = tm_visc.shape_d(s_visc)
    c_f_visc = 2*tm_visc.tau_w(s_visc, rho_inf)/(rho_inf*u_inf**2)
    v_e_visc = tm_visc.v_e(s_visc)
    du_e_visc = tm_visc.du_e(s_visc)
    d2u_e_visc = tm_visc.d2u_e(s_visc)

    s_inv = np.linspace(s_ref[0], min(s_ref[-1], s_sep_inv), 101)
    delta_d_inv = tm_inv.delta_d(s_inv)
    delta_m_inv = tm_inv.delta_m(s_inv)
    shape_d_inv = tm_inv.shape_d(s_inv)
    c_f_inv = 2*tm_inv.tau_w(s_inv, rho_inf)/(rho_inf*u_inf**2)
    v_e_inv = tm_inv.v_e(s_inv)
    du_e_inv = tm_inv.du_e(s_inv)
    d2u_e_inv = tm_inv.d2u_e(s_inv)

    # Plot results
    # pylint: disable=duplicate-code
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(15)
    gs = GridSpec(6, 2, figure=fig)
    axis_delta_d = fig.add_subplot(gs[0, 0])
    axis_delta_d_diff = fig.add_subplot(gs[0, 1])
    axis_delta_m = fig.add_subplot(gs[1, 0])
    axis_delta_m_diff = fig.add_subplot(gs[1, 1])
    axis_shape_d = fig.add_subplot(gs[2, 0])
    axis_shape_d_diff = fig.add_subplot(gs[2, 1])
    axis_c_f = fig.add_subplot(gs[3, 0])
    axis_c_f_diff = fig.add_subplot(gs[3, 1])
    axis_u_e = fig.add_subplot(gs[4, 0])
    axis_du_e = fig.add_subplot(gs[4, 1])
    axis_d2u_e = fig.add_subplot(gs[5, 0])
    axis_v_e = fig.add_subplot(gs[5, 1])

    ref_color = "black"
    ref_label = "XFoil"
    thwaites_visc_color = "red"
    thwaites_visc_label = "Thwaites (Viscous $U_e$)"
    thwaites_inv_color = "blue"
    thwaites_inv_label = "Thwaites (Inviscid $U_e$)"
    # pylint: enable=duplicate-code

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
    ax.set_ylim(0, 0.06)
    ax.set_ylabel(r"$\delta_d/c$")
    ax.grid(True)

    ax = axis_delta_d_diff
    ax.plot(s_ref_visc/c,
            np.abs(1-tm_visc.delta_d(s_ref_visc)/delta_d_ref_visc),
            color=thwaites_visc_color)
    ax.plot(s_ref_inv/c, np.abs(1-tm_inv.delta_d(s_ref_inv)/delta_d_ref_inv),
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
    ax.set_ylim(0, 0.025)
    ax.set_ylabel(r"$\delta_m/c$")
    ax.grid(True)

    ax = axis_delta_m_diff
    ax.plot(s_ref_visc/c,
            np.abs(1-tm_visc.delta_m(s_ref_visc)/delta_m_ref_visc),
            color=thwaites_visc_color)
    ax.plot(s_ref_inv/c, np.abs(1-tm_inv.delta_m(s_ref_inv)/delta_m_ref_inv),
            color=thwaites_inv_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-5,1])
    ax.set_yscale('log')
    ax.grid(True)

    # Displacement shape factor in 2,:
    ax = axis_shape_d
    ax.plot(s_ref/c, shape_d_ref, color=ref_color)
    ax.plot(s_visc/c, shape_d_visc, color=thwaites_visc_color)
    ax.plot(s_inv/c, shape_d_inv, color=thwaites_inv_color)
    ax.set_ylim(2.2, 3)
    ax.set_ylabel(r"$H_d$")
    ax.grid(True)

    ax = axis_shape_d_diff
    ax.plot(s_ref_visc/c, np.abs(1-tm_visc.shape_d(s_ref_visc)
                                 / shape_d_ref_visc),
            color=thwaites_visc_color)
    ax.plot(s_ref_inv/c, np.abs(1-tm_inv.shape_d(s_ref_inv)/shape_d_ref_inv),
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
    ax.set_ylabel(r"$c_f$")
    ax.grid(True)

    ax = axis_c_f_diff
    ax.plot(s_ref_visc/c,
            np.abs(1-2*tm_visc.tau_w(s_ref_visc,
                                     rho_inf)/(rho_inf*u_inf**2)/c_f_ref_visc),
            color=thwaites_visc_color)
    ax.plot(s_ref_inv/c,
            np.abs(1-2*tm_inv.tau_w(s_ref_inv,
                                    rho_inf)/(rho_inf*u_inf**2)/c_f_ref_inv),
            color=thwaites_inv_color)
    ax.set_ylabel("Relative Difference")
    ax.set_ylim([1e-4,1])
    ax.set_yscale('log')
    ax.grid(True)

    # Edge velocity in 4,:
    ax = axis_u_e
    ax.plot(s_ref/c, u_e_visc/u_inf, color=thwaites_visc_color)
    ax.plot(s_ref/c, u_e_inv/u_inf, color=thwaites_inv_color)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(r"$U_e/U_\infty$")
    ax.grid(True)

    ax = axis_du_e
    ax.plot(s_visc/c, du_e_visc, color=thwaites_visc_color)
    ax.plot(s_inv/c, du_e_inv, color=thwaites_inv_color)
    ax.set_ylim(-2, 0.5)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"d$U_e/$d$x$ (1/s)")
    ax.grid(True)

    # Transpiration velocity in 5,:
    ax = axis_d2u_e
    ax.plot(s_visc/c, d2u_e_visc, color=thwaites_visc_color)
    ax.plot(s_inv/c, d2u_e_inv, color=thwaites_inv_color)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"d$^2U_e/$d$x^2$ (1/(m$\cdot$s)")
    ax.grid(True)

    ax = axis_v_e
    ax.plot(s_visc/c, v_e_visc/u_inf, color=thwaites_visc_color)
    ax.plot(s_inv/c, v_e_inv/u_inf, color=thwaites_inv_color)
    ax.set_ylim(0, 0.1)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$V_e/U_e$")
    ax.grid(True)

    fig.subplots_adjust(bottom=0.075, wspace=0.5)
    fig.legend(handles=[ref_curve[0], thwaites_visc_curve[0],
                        thwaites_inv_curve[0]],
               labels=[ref_label, thwaites_visc_label, thwaites_inv_label],
               loc="upper center", bbox_to_anchor=(0.45, 0.03), ncol=4,
               borderaxespad=0.1)
    plt.show()


if __name__ == "__main__":
    compare_xfoil_laminar()

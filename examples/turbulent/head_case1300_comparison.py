"""
Comparing Head's method solution for accelerating flow case.

This example shows a comparison between Head's method and case 1300 from the
1968 Stanford Olympics from Luwieg and Tillman.
"""

# pylint: disable=too-many-statements,too-many-locals

# pylint: disable=duplicate-code
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ibl.head_method import HeadMethod
from ibl.reference import StanfordOlympics1968


def compare_case1300() -> None:
    """Compare the Head method results to expiremental data."""
    so68 = StanfordOlympics1968("1300")
    x = so68.x()
    u_e = so68.u_e()
    du_e = so68.du_e()
    x_sm = so68.x_smooth()
    u_e_sm = so68.u_e_smooth()
    du_e_sm = so68.du_e_smooth()
    rho = 1.2

    hm_reg = HeadMethod(nu=so68.nu_ref, U_e=[x, u_e])
    hm_reg.initial_delta_m = so68.delta_m()[0]
    hm_reg.initial_shape_d = so68.shape_d()[0]
    rtn = hm_reg.solve(x0=so68.x()[0], x_end=so68.x()[-1])
    if not rtn.success:
        print("Could not get solution for Head method: " + rtn.message)
        return

    hm_sm = HeadMethod(nu=so68.nu_ref, U_e=[x_sm, u_e_sm])
    hm_sm.initial_delta_m = so68.delta_m()[0]
    hm_sm.initial_shape_d = so68.shape_d()[0]
    rtn = hm_sm.solve(x0=so68.x()[0], x_end=so68.x()[-1])
    if not rtn.success:
        print("Could not get solution for Head method: " + rtn.message)
        return

    hm_sm2 = HeadMethod(nu=so68.nu_ref, U_e=u_e[0], dU_edx=[x, du_e])
    hm_sm2.initial_delta_m = so68.delta_m()[0]
    hm_sm2.initial_shape_d = so68.shape_d()[0]
    rtn = hm_sm2.solve(x0=so68.x()[0], x_end=so68.x()[-1])
    if not rtn.success:
        print("Could not get solution for Head method: " + rtn.message)
        return

    # Calculate the boundary layer parameters
    x_ref = so68.x()
    delta_d_ref = so68.delta_d()
    delta_m_ref = so68.delta_m()
    shape_d_ref = so68.shape_d()
    c_f_ref = so68.c_f()
    u_e_ref = so68.u_e()

    x = np.linspace(x_ref[0], x_ref[-1], 101)
    delta_d_head_reg = hm_reg.delta_d(x)
    delta_m_head_reg = hm_reg.delta_m(x)
    shape_d_head_reg = hm_reg.shape_d(x)
    c_f_head_reg = 2*hm_sm.tau_w(x, rho)/(rho*hm_reg.u_e(x)**2)
    v_e_head_reg = hm_reg.v_e(x)
    delta_d_head_sm = hm_sm.delta_d(x)
    delta_m_head_sm = hm_sm.delta_m(x)
    shape_d_head_sm = hm_sm.shape_d(x)
    c_f_head_sm = 2*hm_sm.tau_w(x, rho)/(rho*hm_sm.u_e(x)**2)
    v_e_head_sm = hm_sm.v_e(x)
    delta_d_head_sm2 = hm_sm2.delta_d(x)
    delta_m_head_sm2 = hm_sm2.delta_m(x)
    shape_d_head_sm2 = hm_sm2.shape_d(x)
    c_f_head_sm2 = 2*hm_sm2.tau_w(x, rho)/(rho*hm_sm2.u_e(x)**2)
    v_e_head_sm2 = hm_sm2.v_e(x)

    # Plot results
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
    axis_u_e_diff = fig.add_subplot(gs[4, 1])
    axis_du_e = fig.add_subplot(gs[5, 0])
    axis_v_e = fig.add_subplot(gs[5, 1])

    ref_color = "black"
    ref_label = "Ludwieg & Tillman"
    head_reg_color = "red"
    head_reg_label = "Head"
    head_sm_color = "green"
    head_sm_label = "Head (Smooth $U_e$)"
    head_sm2_color = "orange"
    head_sm2_label = "Head (Mono. d$U_e/$d$x$)"

    # Displacement thickness in 0,:
    ax = axis_delta_d
    ref_curve = ax.plot(x_ref, delta_d_ref, color=ref_color, linestyle="",
                        marker="o", label=ref_label)
    head_reg_curve = ax.plot(x, delta_d_head_reg, color=head_reg_color,
                             label=head_reg_label)
    head_sm_curve = ax.plot(x, delta_d_head_sm, color=head_sm_color,
                            label=head_sm_label)
    head_sm2_curve = ax.plot(x, delta_d_head_sm2, color=head_sm2_color,
                             label=head_sm2_label)
    _ = ax.set_ylim((0, 0.008))
    _ = ax.set_ylabel(r"$\delta_d$ (m)")
    ax.grid(True)

    ax = axis_delta_d_diff
    _ = ax.plot(x_ref, np.abs(1-hm_reg.delta_d(x_ref)/delta_d_ref),
                color=head_reg_color)
    _ = ax.plot(x_ref, np.abs(1-hm_sm.delta_d(x_ref)/delta_d_ref),
                color=head_sm_color)
    _ = ax.plot(x_ref, np.abs(1-hm_sm2.delta_d(x_ref)/delta_d_ref),
                color=head_sm2_color)
    _ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Momentum thickness in 1,:
    ax = axis_delta_m
    _ = ax.plot(x_ref, delta_m_ref, color=ref_color, linestyle="", marker="o")
    _ = ax.plot(x, delta_m_head_reg, color=head_reg_color)
    _ = ax.plot(x, delta_m_head_sm, color=head_sm_color)
    _ = ax.plot(x, delta_m_head_sm2, color=head_sm2_color)
    _ = ax.set_ylim((0, 0.003))
    _ = ax.set_ylabel(r"$\delta_m$ (m)")
    ax.grid(True)

    ax = axis_delta_m_diff
    _ = ax.plot(x_ref, np.abs(1-hm_reg.delta_m(x_ref)/delta_m_ref),
                color=head_reg_color)
    _ = ax.plot(x_ref, np.abs(1-hm_sm.delta_m(x_ref)/delta_m_ref),
                color=head_sm_color)
    _ = ax.plot(x_ref, np.abs(1-hm_sm2.delta_m(x_ref)/delta_m_ref),
                color=head_sm2_color)
    _ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Displacement shape factor in 2,:
    ax = axis_shape_d
    _ = ax.plot(x_ref, shape_d_ref, color=ref_color, linestyle="", marker="o")
    _ = ax.plot(x, shape_d_head_reg, color=head_reg_color)
    _ = ax.plot(x, shape_d_head_sm, color=head_sm_color)
    _ = ax.plot(x, shape_d_head_sm2, color=head_sm2_color)
    _ = ax.set_ylim(1, 3)
    _ = ax.set_ylabel(r"$H_d$")
    ax.grid(True)

    ax = axis_shape_d_diff
    _ = ax.plot(x_ref, np.abs(1-hm_reg.shape_d(x_ref)/shape_d_ref),
                color=head_reg_color)
    _ = ax.plot(x_ref, np.abs(1-hm_sm.shape_d(x_ref)/shape_d_ref),
                color=head_sm_color)
    _ = ax.plot(x_ref, np.abs(1-hm_sm2.shape_d(x_ref)/shape_d_ref),
                color=head_sm2_color)
    _ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Skin friction coefficient in 3,:
    ax = axis_c_f
    _ = ax.plot(x_ref, c_f_ref, color=ref_color, linestyle="", marker="o")
    _ = ax.plot(x, c_f_head_reg, color=head_reg_color)
    _ = ax.plot(x, c_f_head_sm, color=head_sm_color)
    _ = ax.plot(x, c_f_head_sm2, color=head_sm2_color)
    _ = ax.set_ylim((0, 0.005))
    _ = ax.set_ylabel(r"$c_f$")
    ax.grid(True)

    ax = axis_c_f_diff
    temp = 2*hm_reg.tau_w(x_ref, rho)/(rho*hm_reg.u_e(x_ref)**2)
    _ = ax.plot(x_ref, np.abs(1-temp/c_f_ref),
                color=head_reg_color)
    temp = 2*hm_sm.tau_w(x_ref, rho)/(rho*hm_sm.u_e(x_ref)**2)
    _ = ax.plot(x_ref, np.abs(1-temp/c_f_ref),
                color=head_sm_color)
    temp = 2*hm_sm2.tau_w(x_ref, rho)/(rho*hm_sm2.u_e(x_ref)**2)
    _ = ax.plot(x_ref, np.abs(1-temp/c_f_ref),
                color=head_sm2_color)
    _ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Edge velocity in 4,:
    ax = axis_u_e
    _ = ax.plot(x_sm, u_e_sm, color=ref_color, linestyle="", marker="o")
    _ = ax.plot(x, hm_reg.u_e(x), color=head_reg_color)
    _ = ax.plot(x, hm_sm.u_e(x), color=head_sm_color)
    _ = ax.plot(x, hm_sm2.u_e(x), color=head_sm2_color)
    _ = ax.set_ylim((10, 30))
    _ = ax.set_xlabel(r"$x$ (m)")
    _ = ax.set_ylabel(r"$U_e$ (m/s)")
    ax.grid(True)

    ax = axis_u_e_diff
    _ = ax.plot(x_ref, np.abs(1-hm_reg.u_e(x_ref)/u_e_ref),
                color=head_reg_color)
    _ = ax.plot(x_ref, np.abs(1-hm_sm.u_e(x_ref)/u_e_ref),
                color=head_sm_color)
    _ = ax.plot(x_ref, np.abs(1-hm_sm2.u_e(x_ref)/u_e_ref),
                color=head_sm2_color)
    _ = ax.set_ylabel("Relative Difference")
    _ = ax.set_ylim((1e-3,1))
    ax.set_yscale('log')
    ax.grid(True)

    # Transpiration velocity in 5,:
    ax = axis_du_e
    _ = ax.plot(x_sm, du_e_sm, color=ref_color, linestyle="", marker="o")
    _ = ax.plot(x, hm_reg.du_e(x), color=head_reg_color)
    _ = ax.plot(x, hm_sm.du_e(x), color=head_sm_color)
    _ = ax.plot(x, hm_sm2.du_e(x), color=head_sm2_color)
    _ = ax.set_ylim((3, 6))
    _ = ax.set_xlabel(r"$x$ (m)")
    _ = ax.set_ylabel(r"d$U_e/$d$x$ (1/s)")
    ax.grid(True)

    ax = axis_v_e
    _ = ax.plot(x, v_e_head_reg, color=head_reg_color)
    _ = ax.plot(x, v_e_head_sm, color=head_sm_color)
    _ = ax.plot(x, v_e_head_sm2, color=head_sm2_color)
    _ = ax.set_ylim((0, 0.05))
    _ = ax.set_xlabel(r"$x$ (m)")
    _ = ax.set_ylabel(r"$V_e$ (m/s)")
    ax.grid(True)

    fig.subplots_adjust(bottom=0.075, wspace=0.5)
    _ = fig.legend(handles=[ref_curve[0], head_reg_curve[0], head_sm_curve[0],
                            head_sm2_curve[0]],
                   labels=[ref_label, head_reg_label, head_sm_label,
                           head_sm2_label],
                   loc="upper center", bbox_to_anchor=(0.45, 0.03), ncol=4,
                   borderaxespad=0.1)
    plt.show()


if __name__ == "__main__":
    compare_case1300()

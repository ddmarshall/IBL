"""
Comparing Thwaites' method solutions for wedge flow case.

This example shows a comparison between various forms of Thwaites and the
Falkner-Skan solution to laminar flat plate boundary layer wedge flows at angle
of pi/4. It shows similar results to Figures 3.7 to 3.9 in Edland thesis.
"""

# pylint: disable=duplicate-code
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ibl.analytic import FalknerSkan
from ibl.thwaites_method import ThwaitesMethodLinear
from ibl.thwaites_method import ThwaitesMethodNonlinear
from ibl.typing import InputParam


def compare_stagnation_solution() -> None:
    """Compare the various solutions to the Falkner-Skan solution."""
    # pylint: disable=too-many-locals, too-many-statements
    # Set flow parameters
    u_inf = 10
    m = 1/3
    nu_inf = 1.45e-5
    rho_inf = 1.2
    c = 2
    npts = 201
    x = np.linspace(1e-6, c, npts)

    # Set up the velocity functions
    def u_e_fun(x: InputParam) -> npt.NDArray:
        x = np.asarray(x)
        return u_inf*x**m

    def du_e_fun(x: InputParam) -> npt.NDArray:
        x = np.asarray(x)
        if m == 0:
            return np.zeros_like(x)
        return m*u_inf*x**(m-1)

    def d2u_e_fun(x: InputParam) -> npt.NDArray:
        x = np.asarray(x)
        if m in (0, 1):
            return np.zeros_like(x)
        return m*(m-1)*u_inf*x**(m-2)

    # setup plot functions
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(15)
    gs = GridSpec(5, 2, figure=fig)
    axis_delta_d = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axis_delta_m = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    axis_shape_d = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    axis_c_f = [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]
    axis_v_e = [fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 1])]

    # extract the Blasius (exact) solution
    fs = FalknerSkan(beta=0.5, u_ref=u_inf, nu_ref=nu_inf)
    delta_d_exact = fs.delta_d(x)
    delta_m_exact = fs.delta_m(x)
    c_f_exact = fs.tau_w(x, rho_inf)/(0.5*rho_inf*u_inf**2)
    shape_d_exact = fs.shape_d(x)
    v_e_exact = fs.v_e(x)

    exact_color = "black"
    curve_handles = [axis_delta_d[0].plot(x/c, delta_d_exact/c,
                                          color=exact_color)[0]]
    axis_delta_m[0].plot(x/c, delta_m_exact/c, color=exact_color)
    axis_shape_d[0].plot(x/c, shape_d_exact, color=exact_color)
    axis_c_f[0].plot(x/c, c_f_exact, color=exact_color)
    axis_v_e[0].plot(x/c, v_e_exact/u_inf, color=exact_color)

    # create the various models
    colors = ["green", "red"]
    labels = ["Falkner-Skan", "Standard", "Nonlinear"]

    models = [ThwaitesMethodLinear(nu=nu_inf, U_e=u_e_fun, dU_edx=du_e_fun,
                                   d2U_edx2=d2u_e_fun, data_fits="Spline"),
              ThwaitesMethodNonlinear(nu=nu_inf, U_e=u_e_fun, dU_edx=du_e_fun,
                                      d2U_edx2=d2u_e_fun, data_fits="Spline")]

    for model, color in zip(models, colors):
        model.initial_delta_m = float(fs.delta_m(x[0]))
        rtn = model.solve(x0=x[0], x_end=x[-1])
        if not rtn.success:
            print("Could not get solution for Thwaites method: " + rtn.message)
            return
        delta_d = model.delta_d(x)
        delta_m = model.delta_m(x)
        c_f = model.tau_w(x, rho_inf)/(0.5*rho_inf*u_inf**2)
        shape_d = model.shape_d(x)
        v_e = model.v_e(x)

        curve_handles.append(axis_delta_d[0].plot(x/c, delta_d/c,
                                                  color=color)[0])
        axis_delta_d[1].plot(x/c, np.abs(1-delta_d/delta_d_exact), color=color)
        axis_delta_m[0].plot(x/c, delta_m/c, color=color)
        axis_delta_m[1].plot(x/c, np.abs(1-delta_m/delta_m_exact), color=color)
        axis_shape_d[0].plot(x/c, shape_d, color=color)
        axis_shape_d[1].plot(x/c, np.abs(1-shape_d/shape_d_exact), color=color)
        axis_c_f[0].plot(x/c, c_f, color=color)
        axis_c_f[1].plot(x/c, np.abs(1-c_f/c_f_exact), color=color)
        axis_v_e[0].plot(x/c, v_e/u_inf, color=color)
        axis_v_e[1].plot(x/c, np.abs(1-v_e/v_e_exact), color=color)

    # Displacement thickness in 0,:
    axis_delta_d[0].set_ylim(0, 0.0008)
    axis_delta_d[0].set_ylabel(r"$\delta_d/c$")
    axis_delta_d[0].grid(True)

    axis_delta_d[1].set_ylabel("Relative Error")
    axis_delta_d[1].set_ylim((1e-4,1))
    axis_delta_d[1].set_yscale('log')
    axis_delta_d[1].grid(True)

    # Momentum thickness in 1,:
    axis_delta_m[0].set_ylim(0, 0.0004)
    axis_delta_m[0].set_ylabel(r"$\delta_m/c$")
    axis_delta_m[0].grid(True)

    axis_delta_m[1].set_ylabel("Relative Error")
    axis_delta_m[1].set_ylim((1e-4,1))
    axis_delta_m[1].set_yscale('log')
    axis_delta_m[1].grid(True)

    # Displacement shape factor in 2,:
    axis_shape_d[0].set_ylim(2.2, 2.5)
    axis_shape_d[0].set_ylabel(r"$H_d$")
    axis_shape_d[0].grid(True)

    axis_shape_d[1].set_ylabel("Relative Error")
    axis_shape_d[1].set_ylim((1e-4,1))
    axis_shape_d[1].set_yscale('log')
    axis_shape_d[1].grid(True)

    # Skin friction coefficient in 3,:
    axis_c_f[0].set_ylim(0.001, 0.002)
    axis_c_f[0].set_ylabel(r"$c_f$")
    axis_c_f[0].grid(True)

    axis_c_f[1].set_ylabel("Relative Error")
    axis_c_f[1].set_ylim((1e-4,1))
    axis_c_f[1].set_yscale('log')
    axis_c_f[1].grid(True)

    # Transpiration velocity in 4,:
    axis_v_e[0].set_ylim(0, 0.01)
    axis_v_e[0].set_xlabel(r"$x/c$")
    axis_v_e[0].set_ylabel(r"$V_e/U_\infty$")
    axis_v_e[0].grid(True)

    axis_v_e[1].set_xlabel(r"$x/c$")
    axis_v_e[1].set_ylabel("Relative Error")
    axis_v_e[1].set_ylim((1e-4,1))
    axis_v_e[1].set_yscale('log')
    axis_v_e[1].grid(True)

    fig.subplots_adjust(bottom=0.075, wspace=0.5)
    fig.legend(handles=curve_handles, labels=labels, loc="upper center",
               bbox_to_anchor=(0.45, 0.03), ncol=3, borderaxespad=0.1)
    plt.show()


if __name__ == "__main__":
    compare_stagnation_solution()

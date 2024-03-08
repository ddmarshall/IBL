r"""
Comparing Thwaites' tabular data approximations to Falkner-Skan values.

This example shows a comparison between the tabular data from Thwaites (1949)
and the equivalent values from Falkner-Skan solution.
"""


from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ibl.thwaites_method import _ThwaitesFunctionsWhite
#from ibl.thwaites_method import _ThwaitesFunctionsCebeciBradshaw
from ibl.thwaites_method import _ThwaitesFunctionsDrela
from ibl.thwaites_method import _ThwaitesFunctionsSpline
#from ibl.typing import InputParam
from ibl.analytic import FalknerSkan


def recreate_drela_table() -> None:
    """Recreate Table 4.1 from Drela (2014)"""
    u_ref = 1.0
    nu_ref = 1.0
    rho_ref = 1.0
    fs = FalknerSkan(beta=0.0, u_ref=u_ref, nu_ref=nu_ref)

    m_values = [2.0, 1.0, 0.6, 0.3, 0.1, 0.0, -0.05, -0.08, -0.09042860905]
    row_format = "{:^9}"*10
    s = 1.0
    print(row_format.format("m", "delta_d", "delta_m", "H_d", "H_k", "Shear",
                            "Diss.", "lambda", "S", "F"))
    for m in m_values:
        fs.reset_m(m=m)
        u_e = fs.u_e(s)
        u_es = m*u_e/s
        delta_fs = np.sqrt(nu_ref*s/u_e)
        # print(delta_fs - np.sqrt(0.5*(m+1))/fs._g(s))
        re_s = u_e*s/nu_ref
        delta_d = fs.delta_d(s)
        delta_m = fs.delta_m(s)
        shape_d = fs.shape_d(s)
        shape_k = fs.shape_k(s)
        shear_nondim = (np.sqrt(re_s)*fs.tau_w(s, rho_ref=rho_ref)
                        / (rho_ref*u_e**2))
        diss_nondim = (np.sqrt(re_s)*fs.dissipation(s, rho_ref=rho_ref)
                       / (rho_ref*u_e**3))
        lam_thwaites = fs.beta*fs.eta_m()**2
        # print(lam_thwaites - delta_m**2*u_es/nu_ref)
        s_thwaites = delta_m*fs.tau_w(s, rho_ref=rho_ref)/(rho_ref*nu_ref*u_e)
        f_thwaites = 2*(s_thwaites-(shape_d+2)*lam_thwaites)

        row_format = "{:>8.5f} "*10
        print(delta_d/delta_fs - fs.eta_d()*np.sqrt(2/(1+m)))
        print(row_format.format(m, delta_d/delta_fs, delta_m/delta_fs, shape_d,
                                shape_k, shear_nondim, diss_nondim,
                                lam_thwaites, s_thwaites, f_thwaites))


def compare_shape_d() -> None:
    """Compare the available displacement shape factor fits."""
    # Set common plot properties
    plt.rcParams["figure.figsize"] = [8, 5]
    npts = 101

    # extract the original Thwaites tabular data for comparisons
    tab_lambda = _ThwaitesFunctionsSpline._tab_lambda  # pylint: disable=protected-access
    tab_shape = _ThwaitesFunctionsSpline._tab_shape  # pylint: disable=protected-access
    tab_shear = _ThwaitesFunctionsSpline._tab_shear  # pylint: disable=protected-access

    # Create the various fit models
    white = _ThwaitesFunctionsWhite()
#    cb = _ThwaitesFunctionsCebeciBradshaw()
#    spline = _ThwaitesFunctionsSpline()
    drela = _ThwaitesFunctionsDrela()

    # get the m values to needed to calculate the Falkner-Skan values
    m = np.array([2.0, 1.0, 0.6, 0.3, 0.1, 0.0, -0.05, -0.08, -0.09042860905])
    beta = 2*m/(m+1)
    # beta = np.linspace(-0.19883785, 1.999999, 5)

    it = np.nditer([beta, None, None, None])
    with it:
        for beta_val, lam_val, shape_val, s_val in it:
            fs = FalknerSkan(beta=float(beta_val), u_ref=1, nu_ref=1)
            eta_m = fs.eta_m()
            fpp_w = float(fs.f_pp0)
            shape_d = float(fs.eta_d()/fs.eta_m())
            s = 1.0
            lam = np.sqrt(2)*fs.delta_m(s)**2*(fs.m*fs.u_e(s)/s)/fs.nu_ref
            print(lam - beta_val*eta_m**2)
            lam_val[...] = lam
            shape_val[...] = fs.shape_d(s)
            s_val[...] = eta_m*fpp_w

        fs_lambda = it.operands[1]
        fs_shape = it.operands[2]
        fs_shear = it.operands[3]

    lambda_min = -0.1
    lambda_max = 0.25
    # would like same range of lambdas, but each scheme has different range on
    # which it is defined.
    white_lambda = np.linspace(np.maximum(white.range()[0], lambda_min),
                               np.minimum(white.range()[1], lambda_max), npts)
#    cb_lambda = np.linspace(np.maximum(cb.range()[0], lambda_min),
#                            np.minimum(cb.range()[1], lambda_max), npts)
    drela_lambda = np.linspace(np.maximum(drela.range()[0], lambda_min),
                               np.minimum(drela.range()[1], lambda_max), npts)
#    spline_lambda = np.linspace(np.maximum(spline.range()[0], lambda_min),
#                                np.minimum(spline.range()[1], lambda_max),
#                                npts)

    # calculate the corresponding values
    white_shape = white.shape(white_lambda)
    white_shear = white.shear(white_lambda)
#    cb_shape = cb.shape(cb_lambda)
    drela_shape = drela.shape(drela_lambda)
    drela_shear = drela.shear(drela_lambda)
#    spline_shape = spline.shape(spline_lambda)

    # plot functions
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(15)
    gs = GridSpec(3, 1, figure=fig)
    axis_plot = fig.add_subplot(gs[0, 0])
    axis_zoom = fig.add_subplot(gs[1, 0])
    axis_err = fig.add_subplot(gs[2, 0])

    # Plot functions compared to the Thwaites tabulated values
    ax = axis_plot
    ax.plot(tab_lambda, tab_shape, marker='o', linestyle='', color="black",
            label=r"Thwaites Data")
    ax.plot(fs_lambda, fs_shape, marker='o', linestyle='', color="cyan",
            label=r"Falkner-Skan")
    ax.plot(white_lambda, white_shape, color="red", label=r"White")
#    ax.plot(cb_lambda, cb_shape, color="orange", label=r"Cebeci & Bradshaw")
#    ax.plot(spline_lambda, spline_shape, color="purple", label=r"Spline")
    ax.plot(drela_lambda, drela_shape, color="green", label=r"Drela")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$H\left(\lambda\right)$")
    ax.set_ybound(lower=0, upper=4)
    ax.grid(True)
    ax.legend()

    # plot the shear function
    ax = axis_zoom
    ax.plot(tab_lambda, tab_shear, marker='o', linestyle='', color="black",
            label=r"Thwaites Data")
    ax.plot(fs_lambda, fs_shear, marker='o', linestyle='', color="cyan",
            label=r"Falkner-Skan")
    ax.plot(white_lambda, white_shear, color="red", label=r"White")
#    ax.plot(cb_lambda, cb_shape, color="orange", label=r"Cebeci & Bradshaw")
#    ax.plot(spline_lambda, spline_shape, color="purple", label=r"Spline")
    ax.plot(drela_lambda, drela_shear, color="green", label=r"Drela")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$S\left(\lambda\right)$")
    ax.grid(True)
    ax.legend()

#     # Zoom in on separation region of previous plot
#     lambda_zoom_min = -0.10
#     lambda_zoom_max = -0.05
#     # Still need to set different minimums for each model's lambda range
# #    linear_lambda_zoom = np.linspace(lambda_zoom_min, lambda_zoom_max, npts)
# #    white_lambda_zoom = np.linspace(np.maximum(white.range()[0],
# #                                               lambda_zoom_min),
# #                                    lambda_zoom_max, npts)
# #    cb_lambda_zoom = np.linspace(np.maximum(cb.range()[0], lambda_zoom_min),
# #                                 lambda_zoom_max, npts)
# #    spline_lambda_zoom = np.linspace(np.maximum(spline.range()[0],
# #                                                lambda_zoom_min),
# #                                     lambda_zoom_max, npts)

#     # calculate the corresponding F values
# #    linear_f_zoom = f_linear(linear_lambda_zoom)
# #    white_f_zoom = white.f(white_lambda_zoom)
# #    cb_f_zoom = cb.f(cb_lambda_zoom)
# #    spline_f_zoom = spline.f(spline_lambda_zoom)

# #     # extract the zoomed original Thwaites tabular data for comparisons
# #     tab_lambda_zoom = tab_lambda[tab_lambda < lambda_zoom_max]
# #     tab_shape_zoom = tab_shape[tab_lambda < lambda_zoom_max]

#     # Plot the zoomed in region
#     ax = axis_zoom

#     ax.plot(tab_lambda_zoom, tab_shape_zoom, marker='o', linestyle='',
#             color="black", label=r"Thwaites Original")
# #    ax.plot(linear_lambda_zoom, linear_f_zoom, color="green", label=r"Linear")
# #    ax.plot(white_lambda_zoom, white_f_zoom, color="red", label=r"White")
# #    ax.plot(cb_lambda_zoom, cb_f_zoom, color="orange",
# #            label=r"Cebeci & Bradshaw")
# #    ax.plot(spline_lambda_zoom, spline_f_zoom, color="purple",
# #            label=r"Spline")
#     ax.set_xlabel(r"$\lambda$")
#     ax.set_ylabel(r"$H\left(\lambda\right)$")
#     ax.grid(True)
#     ax.legend()

#     # Calculate the errors between tabular data and fits
#     #
#     # Notes: * Only Cebeci & Bradshaw cannot compare all tabulated values
#     #        * The spline fit has errors because the spline uses the exact
#     #          equation for the F, while the tabulated values are given in
#     #          3 sig. figs. and roundoff error appears to be in Thwaites
#     #          original data for F
#     linear_error = np.abs(1 - f_linear(tab_lambda)/tab_f)
#     white_error = np.abs(1 - white.f(tab_lambda)/tab_f)
#     cb_lambda_error = tab_lambda[(tab_lambda >= cb.range()[0])
#                                  & (tab_lambda <= cb.range()[1])]
#     cb_error = np.abs(1 - cb.f(cb_lambda_error)/tab_f[(tab_lambda
#                                                        >= cb.range()[0])
#                                                       & (tab_lambda
#                                                          <= cb.range()[1])])
#     spline_error = np.abs(1 - spline.f(tab_lambda)/tab_f)

#     # Show relative errors
#     ax = axis_err
#     ax.plot(tab_lambda, linear_error, color="green", label=r"Linear")
#     ax.plot(tab_lambda, white_error, color="red", label=r"White")
#     ax.plot(cb_lambda_error, cb_error, color="orange",
#             label=r"Cebeci & Bradshaw")
#     ax.plot(tab_lambda, spline_error, color="purple", label=r"Spline")
#     ax.set_xlabel(r"$\lambda$")
#     ax.set_ylabel("Relative Error")
#     ax.set_xlim([-0.10, 0.25])
#     ax.set_ylim([.00001,1])
#     ax.set_yscale("log")
#     ax.grid(True)
#     ax.legend()

    plt.show()


if __name__ == "__main__":
    recreate_drela_table()
    # compare_shape_d()

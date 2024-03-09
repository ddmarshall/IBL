r"""
Comparing Thwaites' tabular data approximations to Falkner-Skan values.

This example shows a comparison between the tabular data from Thwaites (1949)
and the equivalent values from Falkner-Skan solution.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ibl.thwaites_method import _ThwaitesFunctionsWhite
from ibl.thwaites_method import _ThwaitesFunctionsCebeciBradshaw
from ibl.thwaites_method import _ThwaitesFunctionsDrela
from ibl.thwaites_method import _ThwaitesFunctionsSpline
from ibl.analytic import FalknerSkan


def recreate_drela_table() -> None:
    """Recreate Table 4.1 from Drela (2014)"""
    u_ref = 1.0
    nu_ref = 1.0
    fs = FalknerSkan(beta=0.0, u_ref=u_ref, nu_ref=nu_ref)

    m_values = [2.0, 1.0, 0.6, 0.3, 0.1, 0.0, -0.05, -0.08, -0.09042860905]
    row_format = "{:^9}"*10
    # s = 1.0
    print(row_format.format("m", "delta_d", "delta_m", "H_d", "H_k", "Shear",
                            "Diss.", "lambda", "S", "F"))
    for m in m_values:
        fs.reset_m(m=m)
        beta = fs.beta
        eta_m = fs.eta_m()
        eta_d = fs.eta_d()
        eta_k = fs.eta_k()
        fw_pp = fs.f_pp0

        # s-free calculations
        disp_term = eta_d*np.sqrt(2/(1+m))
        mom_term = eta_m*np.sqrt(2/(1+m))
        shape_d = eta_d/eta_m
        shape_k = eta_k/eta_m
        shear_term = fw_pp*np.sqrt((m+1)/2)
        diss_term = np.sqrt(0.5/(1+m))*0.5*(5*m+1)*eta_k
        lam_thwaites = beta*eta_m**2
        s_thwaites = eta_m*fw_pp
        f_thwaites = 2*(s_thwaites-(shape_d+2)*lam_thwaites)

        # # original calculations
        # u_e = fs.u_e(s)
        # u_es = m*u_e/s
        # delta_fs = np.sqrt(0.5*(m+1))/fs._g(s)
        # re_s = u_e*s/nu_ref
        # delta_d = fs.delta_d(s)
        # delta_m = fs.delta_m(s)
        # shape_d_orig = fs.shape_d(s)
        # shape_k_orig = fs.shape_k(s)
        # disp_term_orig = delta_d/delta_fs
        # mom_term_orig = delta_m/delta_fs

        # shear_term_orig = (np.sqrt(re_s)*fs.tau_w(s, rho_ref=rho_ref)
        #                   / (rho_ref*u_e**2))
        # diss_term_orig = (np.sqrt(re_s)*fs.dissipation(s, rho_ref=rho_ref)
        #                  / (rho_ref*u_e**3))
        # lam_thwaites_orig = delta_m**2*u_es/nu_ref
        # s_thwaites_orig = delta_m*fs.tau_w(s, rho_ref=rho_ref)/(rho_ref*nu_ref*u_e)
        # f_thwaites_orig = 2*(s_thwaites-(shape_d+2)*lam_thwaites)

        # # print out temp test results
        # print(f"{disp_term - disp_term_orig:.5e}\t"
        #       f"{lam_thwaites - lam_thwaites_orig:.5e}\t"
        #       f"{mom_term - mom_term_orig:.5e}\t"
        #       f"{shape_d - shape_d_orig:.5e}\t"
        #       f"{shape_k - shape_k_orig:.5e}\t"
        #       f"{shear_term - shear_term_orig:.5e}\t"
        #       f"{diss_term - diss_term_orig:.5e}\t"
        #       f"{s_thwaites - s_thwaites_orig:.5e}\t"
        #       f"{f_thwaites - f_thwaites_orig:.5e}\t")

        row_format = "{:>8.5f} "*10
        print(row_format.format(m, disp_term, mom_term, shape_d,
                                shape_k, shear_term, diss_term,
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
    tab_f = _ThwaitesFunctionsSpline._tab_f  # pylint: disable=protected-access

    # Create the various fit models
    white = _ThwaitesFunctionsWhite()
    cb = _ThwaitesFunctionsCebeciBradshaw()
    spline = _ThwaitesFunctionsSpline()
    drela = _ThwaitesFunctionsDrela()

    # get the m values to needed to calculate the Falkner-Skan values
    m = np.array([2.0, 1.0, 0.6, 0.3, 0.1, 0.0, -0.05, -0.08, -0.09042860905])
#    beta = 2*m/(m+1)
    # beta = np.linspace(-0.19883785, 1.999999, 5)
    fs = FalknerSkan(beta=0, u_ref=1, nu_ref=1)

    it = np.nditer([m, None, None, None])
    with it:
        for m_val, lam_val, shape_val, s_val in it:
            fs.reset_m(m=float(m_val))
            beta = fs.beta
            eta_d = fs.eta_d()
            eta_m = fs.eta_m()
            fpp_w = fs.f_pp0
            shape_d = eta_d/eta_m
            lam = beta*eta_m**2
            lam_val[...] = lam
            shape_val[...] = shape_d
            s_val[...] = eta_m*fpp_w

        fs_lambda = it.operands[1]
        fs_shape = it.operands[2]
        fs_shear = it.operands[3]

    fs_f = 2*(fs_shear - fs_lambda*(fs_shape + 2))
    lambda_min = -0.1
    lambda_max = 0.25
    # would like same range of lambdas, but each scheme has different range on
    # which it is defined.
    white_lambda = np.linspace(np.maximum(white.range()[0], lambda_min),
                               np.minimum(white.range()[1], lambda_max), npts)
    cb_lambda = np.linspace(np.maximum(cb.range()[0], lambda_min),
                            np.minimum(cb.range()[1], lambda_max), npts)
    drela_lambda = np.linspace(np.maximum(drela.range()[0], lambda_min),
                               np.minimum(drela.range()[1], lambda_max), npts)
    spline_lambda = np.linspace(np.maximum(spline.range()[0], lambda_min),
                                np.minimum(spline.range()[1], lambda_max),
                                npts)

    # calculate the corresponding values
    white_shape = white.shape(white_lambda)
    white_shear = white.shear(white_lambda)
    white_f = white.f(white_lambda)
    cb_shape = cb.shape(cb_lambda)
    cb_shear = cb.shear(cb_lambda)
    cb_f = cb.f(cb_lambda)
    drela_shape = drela.shape(drela_lambda)
    drela_shear = drela.shear(drela_lambda)
    drela_f = drela.f(drela_lambda)
    spline_shape = spline.shape(spline_lambda)
    spline_shear = spline.shear(spline_lambda)
    spline_f = spline.f(spline_lambda)

    # plot functions
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(15)
    gs = GridSpec(3, 1, figure=fig)
    axis_shape = fig.add_subplot(gs[0, 0])
    axis_shear = fig.add_subplot(gs[1, 0])
    axis_f = fig.add_subplot(gs[2, 0])

    # Plot functions compared to the Thwaites tabulated values
    ax = axis_shape
    ax.plot(tab_lambda, tab_shape, marker='o', linestyle='', color="black",
            label=r"Thwaites Data")
    ax.plot(fs_lambda, fs_shape, marker='o', linestyle='', color="cyan",
            label=r"Falkner-Skan")
    ax.plot(white_lambda, white_shape, color="red", label=r"White")
    ax.plot(cb_lambda, cb_shape, color="orange", label=r"Cebeci & Bradshaw")
    ax.plot(spline_lambda, spline_shape, color="purple", label=r"Spline")
    ax.plot(drela_lambda, drela_shape, color="green", label=r"Drela")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$H\left(\lambda\right)$")
    ax.grid(True)
    ax.legend()

    # plot the shear function
    ax = axis_shear
    ax.plot(tab_lambda, tab_shear, marker='o', linestyle='', color="black",
            label=r"Thwaites Data")
    ax.plot(fs_lambda, fs_shear, marker='o', linestyle='', color="cyan",
            label=r"Falkner-Skan")
    ax.plot(white_lambda, white_shear, color="red", label=r"White")
    ax.plot(cb_lambda, cb_shear, color="orange", label=r"Cebeci & Bradshaw")
    ax.plot(spline_lambda, spline_shear, color="purple", label=r"Spline")
    ax.plot(drela_lambda, drela_shear, color="green", label=r"Drela")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$S\left(\lambda\right)$")
    ax.grid(True)
    ax.legend()

    # plot the f function
    ax = axis_f
    ax.plot(tab_lambda, tab_f, marker='o', linestyle='', color="black",
            label=r"Thwaites Data")
    ax.plot(fs_lambda, fs_f, marker='o', linestyle='', color="cyan",
            label=r"Falkner-Skan")
    ax.plot(white_lambda, white_f, color="red", label=r"White")
    ax.plot(cb_lambda, cb_f, color="orange", label=r"Cebeci & Bradshaw")
    ax.plot(spline_lambda, spline_f, color="purple", label=r"Spline")
    ax.plot(drela_lambda, drela_f, color="green", label=r"Drela")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$F\left(\lambda\right)$")
    ax.grid(True)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    # recreate_drela_table()
    compare_shape_d()

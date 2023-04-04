#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Comparing approximations to Thwaites' tabular data.

This example shows a comparison between the various approximations to the
tabular data from Thwaites for the function :math:`F(\lambda)`. It shows
similar results to Figures 1.2 and 1.3 in Edland thesis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ibl.thwaites_method import _ThwaitesFunctionsWhite
from ibl.thwaites_method import _ThwaitesFunctionsCebeciBradshaw
from ibl.thwaites_method import _ThwaitesFunctionsSpline


def compare_thwaites_fits():
    """Compare the available fits to Thwaites' data."""
    # Set common plot properties
    plt.rcParams["figure.figsize"] = [8, 5]
    npts = 101

    # Create the various fit models
    white = _ThwaitesFunctionsWhite()
    cb = _ThwaitesFunctionsCebeciBradshaw()
    spline = _ThwaitesFunctionsSpline()

    def f_linear(lam):
        return 0.45-6*lam

    # Plot the various fits
    lambda_min = -0.1
    lambda_max = 0.25
    # would like same range of lambdas, but each scheme has different range on
    # which it is defined.
    linear_lambda = np.linspace(lambda_min, lambda_max, npts)
    white_lambda = np.linspace(np.maximum(white.range()[0], lambda_min),
                               np.minimum(white.range()[1], lambda_max), npts)
    cb_lambda = np.linspace(np.maximum(cb.range()[0], lambda_min),
                            np.minimum(cb.range()[1], lambda_max), npts)
    spline_lambda = np.linspace(np.maximum(spline.range()[0], lambda_min),
                                np.minimum(spline.range()[1], lambda_max),
                                npts)

    # calculate the corresponding F values
    linear_f = f_linear(linear_lambda)
    white_f = white.f(white_lambda)
    cb_f = cb.f(cb_lambda)
    spline_f = spline.f(spline_lambda)

    # extract the original Thwaites tabular data for comparisons
    tab_lambda = spline._tab_lambda  # pylint: disable=protected-access
    tab_f = spline._tab_f  # pylint: disable=protected-access

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
    ax.plot(tab_lambda, tab_f, marker='o', linestyle='', color="black",
            label=r"Thwaites Original")
    ax.plot(linear_lambda, linear_f, color="green", label=r"Linear")
    ax.plot(white_lambda, white_f, color="red", label=r"White")
    ax.plot(cb_lambda, cb_f, color="orange", label=r"Cebeci & Bradshaw")
    ax.plot(spline_lambda, spline_f, color="purple", label=r"Spline")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$F\left(\lambda\right)$")
    ax.grid(True)
    ax.legend()

    # Zoom in on separation region of previous plot
    lambda_zoom_min = -0.10
    lambda_zoom_max = -0.05
    # Still need to set different minimums for each model's lambda range
    linear_lambda_zoom = np.linspace(lambda_zoom_min, lambda_zoom_max, npts)
    white_lambda_zoom = np.linspace(np.maximum(white.range()[0],
                                               lambda_zoom_min),
                                    lambda_zoom_max, npts)
    cb_lambda_zoom = np.linspace(np.maximum(cb.range()[0], lambda_zoom_min),
                                 lambda_zoom_max, npts)
    spline_lambda_zoom = np.linspace(np.maximum(spline.range()[0],
                                                lambda_zoom_min),
                                     lambda_zoom_max, npts)

    # calculate the corresponding F values
    linear_f_zoom = f_linear(linear_lambda_zoom)
    white_f_zoom = white.f(white_lambda_zoom)
    cb_f_zoom = cb.f(cb_lambda_zoom)
    spline_f_zoom = spline.f(spline_lambda_zoom)

    # extract the zoomed original Thwaites tabular data for comparisons
    tab_lambda_zoom = tab_lambda[tab_lambda < lambda_zoom_max]
    tab_f_zoom = tab_f[tab_lambda < lambda_zoom_max]

    # Plot the zoomed in region
    ax = axis_zoom
    ax.plot(tab_lambda_zoom, tab_f_zoom, marker='o', linestyle='',
            color="black", label=r"Thwaites Original")
    ax.plot(linear_lambda_zoom, linear_f_zoom, color="green", label=r"Linear")
    ax.plot(white_lambda_zoom, white_f_zoom, color="red", label=r"White")
    ax.plot(cb_lambda_zoom, cb_f_zoom, color="orange",
            label=r"Cebeci & Bradshaw")
    ax.plot(spline_lambda_zoom, spline_f_zoom, color="purple",
            label=r"Spline")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$F\left(\lambda\right)$")
    ax.grid(True)
    ax.legend()

    # Calculate the errors between tabular data and fits
    #
    # Notes: * Only Cebeci & Bradshaw cannot compare all tabulated values
    #        * The spline fit has errors because the spline uses the exact
    #          equation for the F, while the tabulated values are given in
    #          3 sig. figs. and roundoff error appears to be in Thwaites
    #          original data for F
    linear_error = np.abs(1 - f_linear(tab_lambda)/tab_f)
    white_error = np.abs(1 - white.f(tab_lambda)/tab_f)
    cb_lambda_error = tab_lambda[(tab_lambda >= cb.range()[0])
                                 & (tab_lambda <= cb.range()[1])]
    cb_error = np.abs(1 - cb.f(cb_lambda_error)/tab_f[(tab_lambda
                                                       >= cb.range()[0])
                                                      & (tab_lambda
                                                         <= cb.range()[1])])
    spline_error = np.abs(1 - spline.f(tab_lambda)/tab_f)

    # Show relative errors
    ax = axis_err
    ax.plot(tab_lambda, linear_error, color="green", label=r"Linear")
    ax.plot(tab_lambda, white_error, color="red", label=r"White")
    ax.plot(cb_lambda_error, cb_error, color="orange",
            label=r"Cebeci & Bradshaw")
    ax.plot(tab_lambda, spline_error, color="purple", label=r"Spline")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Relative Error")
    ax.set_xlim([-0.10, 0.25])
    ax.set_ylim([.00001,1])
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    compare_thwaites_fits()

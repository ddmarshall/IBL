#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Comparing approximations to Thwaites' tabular data.

This example shows a comparison between the various approximations to the
tabular data from Thwaites for the function :math:`F(\lambda)`. It shows
similar results to Figures 1.2 and 1.3 in Edland thesis.
"""

# pylint: disable=protected-access
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
import numpy as np
import matplotlib.pyplot as plt

from pyBL.thwaites_method import _ThwaitesFunctionsWhite
from pyBL.thwaites_method import _ThwaitesFunctionsCebeciBradshaw
from pyBL.thwaites_method import _ThwaitesFunctionsSpline


def compare_thwaites_fits():
    """Compare the available fits to Thwaites' data."""
    # Set common plot properties
    plt.rcParams["figure.figsize"] = [8, 5]
    npts = 101

    # Create the various fit models
    white = _ThwaitesFunctionsWhite()
    cb = _ThwaitesFunctionsCebeciBradshaw()
    spline = _ThwaitesFunctionsSpline()

    def F_linear(lam):
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
    linear_F = F_linear(linear_lambda)
    white_F = white.F(white_lambda)
    cb_F = cb.F(cb_lambda)
    spline_F = spline.F(spline_lambda)

    # extract the original Thwaites tabular data for comparisons
    tab_lambda = spline._tab_lambda
    tab_F = spline._tab_F

    # Plot functions compared to the Thwaites tabulated values
    plt.figure()
    plt.plot(tab_lambda, tab_F, marker='o', linestyle='', color="black",
             label=r"Thwaites Original")
    plt.plot(linear_lambda, linear_F, color="green", label=r"Linear")
    plt.plot(white_lambda, white_F, color="red", label=r"White")
    plt.plot(cb_lambda, cb_F, color="orange", label=r"Cebeci & Bradshaw")
    plt.plot(spline_lambda, spline_F, color="purple", label=r"Spline")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$F\left(\lambda\right)$")
    plt.grid(True)
    plt.legend()
    plt.show()

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
    linear_F_zoom = F_linear(linear_lambda_zoom)
    white_F_zoom = white.F(white_lambda_zoom)
    cb_F_zoom = cb.F(cb_lambda_zoom)
    spline_F_zoom = spline.F(spline_lambda_zoom)

    # extract the zoomed original Thwaites tabular data for comparisons
    tab_lambda_zoom = tab_lambda[tab_lambda < lambda_zoom_max]
    tab_F_zoom = tab_F[tab_lambda < lambda_zoom_max]

    # Plot the zoomed in region
    plt.figure()
    plt.plot(tab_lambda_zoom, tab_F_zoom, marker='o', linestyle='',
             color="black", label=r"Thwaites Original")
    plt.plot(linear_lambda_zoom, linear_F_zoom, color="green", label=r"Linear")
    plt.plot(white_lambda_zoom, white_F_zoom, color="red", label=r"White")
    plt.plot(cb_lambda_zoom, cb_F_zoom, color="orange",
             label=r"Cebeci & Bradshaw")
    plt.plot(spline_lambda_zoom, spline_F_zoom, color="purple",
             label=r"Spline")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$F\left(\lambda\right)$")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Calculate the errors between tabular data and fits
    #
    # Notes: * Only Cebeci & Bradshaw cannot compare all tabulated values
    #        * The spline fit has errors because the spline uses the exact
    #          equation for the F, while the tabulated values are given in
    #          3 sig. figs. and roundoff error appears to be in Thwaites
    #          original data for F
    linear_error = np.abs(1 - F_linear(tab_lambda)/tab_F)
    white_error = np.abs(1 - white.F(tab_lambda)/tab_F)
    cb_lambda_error = tab_lambda[(tab_lambda >= cb.range()[0])
                                 & (tab_lambda <= cb.range()[1])]
    cb_error = np.abs(1 - cb.F(cb_lambda_error)/tab_F[(tab_lambda
                                                       >= cb.range()[0])
                                                      & (tab_lambda
                                                         <= cb.range()[1])])
    spline_error = np.abs(1 - spline.F(tab_lambda)/tab_F)

    # Show relative errors
    plt.figure()
    plt.plot(tab_lambda, linear_error, color="green", label=r"Linear")
    plt.plot(tab_lambda, white_error, color="red", label=r"White")
    plt.plot(cb_lambda_error, cb_error, color="orange",
             label=r"Cebeci & Bradshaw")
    plt.plot(tab_lambda, spline_error, color="purple", label=r"Spline")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Relative Error")
    plt.xlim([-0.10, 0.25])
    plt.ylim([.00001,1])
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    compare_thwaites_fits()

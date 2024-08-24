r"""
Comparing approximations to Thwaites' tabular data.

This example shows a comparison between the various approximations to the
tabular data from Thwaites for the function :math:`F(\lambda)`. It shows
similar results to Figures 1.2 and 1.3 in Edland thesis.
"""

# pyright: reportPrivateUsage=false
# pylint: disable=protected-access

from typing import Tuple

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ibl.thwaites_method import _ThwaitesFunctionsWhite
from ibl.thwaites_method import _ThwaitesFunctionsCebeciBradshaw
from ibl.thwaites_method import _ThwaitesFunctionsDrela
from ibl.thwaites_method import _ThwaitesFunctionsSpline
from ibl.thwaites_method import _ThwaitesFunctions


def compare_thwaites_fits() -> None:
    """Compare the available fits to Thwaites' data."""
    # Set common plot properties
    plt.rcParams["figure.figsize"] = [8, 5]
    npts = 101

    # Create the various fit models
    models = [_ThwaitesFunctionsWhite(), _ThwaitesFunctionsCebeciBradshaw(),
              _ThwaitesFunctionsDrela(), _ThwaitesFunctionsSpline()]
    colors = ["red", "orange", "blue", "purple"]
    labels = ["White", "Cebeci & Bradshaw", "Drela", "Spline"]

    # prepare plot functions
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(15)
    gs = GridSpec(3, 1, figure=fig)
    axis_plot = fig.add_subplot(gs[0, 0])
    axis_zoom = fig.add_subplot(gs[1, 0])
    axis_err = fig.add_subplot(gs[2, 0])

    lambda_min, lambda_max = -0.1, 0.25

    # plot the reference Thwaites tabular data for comparisons
    lam, f = get_thwaites_table_data(lambda_min=lambda_min,
                                     lambda_max=lambda_max)
    axis_plot.plot(lam, f, marker='o', linestyle='', color="black",
                   label="Thwaites Original")

    # plot the linear fit
    lam = np.linspace(lambda_min, lambda_max, npts)
    f = f_linear(lam)
    axis_plot.plot(lam, f, color="green", label="Linear")

    # plot the models
    for model, color, label in zip(models, colors, labels):
        lam, f = calculate_thwaites_values(lambda_min=lambda_min,
                                           lambda_max=lambda_max, npts=npts,
                                           model=model)
        axis_plot.plot(lam, f, color=color, label=label)

    # Plot functions compared to the Thwaites tabulated values
    axis_plot.set_xlabel(r"$\lambda$")
    axis_plot.set_ylabel(r"$F\left(\lambda\right)$")
    axis_plot.grid(True)
    axis_plot.legend()

    # Zoom in on separation region of previous plot
    lambda_zoom_max = -0.05

    # extract the zoomed original Thwaites tabular data for comparisons
    lam, f = get_thwaites_table_data(lambda_min=lambda_min,
                                     lambda_max=lambda_zoom_max)
    axis_zoom.plot(lam, f, marker='o', linestyle='',
                   color="black", label="Thwaites Original")

    # plot the linear fit
    lam = np.linspace(lambda_min, lambda_zoom_max, npts)
    f = f_linear(lam)
    axis_zoom.plot(lam, f, color="green", label="Linear")

    # plot the models
    for model, color, label in zip(models, colors, labels):
        lam, f = calculate_thwaites_values(lambda_min=lambda_min,
                                           lambda_max=lambda_zoom_max,
                                           npts=npts, model=model)
        axis_zoom.plot(lam, f, color=color, label=label)

    # Plot the zoomed in region
    axis_zoom.set_xlabel(r"$\lambda$")
    axis_zoom.set_ylabel(r"$F\left(\lambda\right)$")
    axis_zoom.grid(True)
    axis_zoom.legend()

    # Calculate the errors between tabular data and fits
    #
    # Notes: * The spline fit has errors because the spline uses the exact
    #          equation for the F, while the tabulated values are given in
    #          3 sig. figs. and roundoff error appears to be in Thwaites
    #          original data for F

    # get the reference data for comparison
    lam_ref, f_ref = get_thwaites_table_data(lambda_min=lambda_min,
                                             lambda_max=lambda_max)

    # plot the linear fit
    f_err = np.abs(1 - f_linear(lam_ref)/f_ref)
    axis_err.plot(lam_ref, f_err, color="green", label="Linear")

    # plot the models
    for model, color, label in zip(models, colors, labels):
        f_err = np.abs(1 - model.f(lam_ref)/f_ref)
        axis_err.plot(lam_ref, f_err, color=color, label=label)

    # Show relative errors
    axis_err.set_xlabel(r"$\lambda$")
    axis_err.set_ylabel("Relative Error")
    axis_err.set_xlim((-0.10, 0.25))
    axis_err.set_ylim((.00001,1))
    axis_err.set_yscale("log")
    axis_err.grid(True)
    axis_err.legend()

    plt.show()


def f_linear(lam: npt.NDArray) -> npt.NDArray:
    """Linear form of F function."""
    return 0.45-6*lam


def calculate_thwaites_values(lambda_min: float, lambda_max: float, npts: int,
                              model: _ThwaitesFunctions) -> Tuple[npt.NDArray,
                                                                  npt.NDArray]:
    """Calculate the Thwaites values."""
    lam = np.linspace(np.maximum(model.range()[0], lambda_min),
                      np.minimum(model.range()[1], lambda_max), npts)
    f = model.f(lam)
    return lam, f


def get_thwaites_table_data(lambda_min: float,
                            lambda_max: float) -> Tuple[npt.NDArray,
                                                        npt.NDArray]:
    """Return the Thwaites tabular data."""

    model = _ThwaitesFunctionsSpline()
    lam = model.lambda_values
    idx = np.nonzero((lam <= lambda_max) & (lam >= lambda_min))
    lam = lam[idx]
    f = model.f_values[idx]
    return lam, f


if __name__ == "__main__":
    compare_thwaites_fits()

"""
Comparing Thwaites' tabular data approximations to Falkner-Skan values.

This example shows a comparison between the tabular data from Thwaites (1949)
and the equivalent values from Falkner-Skan solution.
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
from ibl.analytic import FalknerSkan


def compare_shape_d() -> None:
    """Compare the available displacement shape factor fits."""
    # Set common plot properties
    plt.rcParams["figure.figsize"] = [8, 5]
    npts = 101

    # setup plot functions
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(15)
    gs = GridSpec(3, 1, figure=fig)
    axis_shape = fig.add_subplot(gs[0, 0])
    axis_shear = fig.add_subplot(gs[1, 0])
    axis_f = fig.add_subplot(gs[2, 0])

    # extract the original Thwaites tabular data for comparisons
    lam, shape, shear, f = get_thwaites_tabular_data()
    _ = axis_shape.plot(lam, shape, marker="o", linestyle="",
                        color="black", label="Thwaites")
    _ = axis_shear.plot(lam, shear, marker="o", linestyle="",
                        color="black", label="Thwaites")
    _ = axis_f.plot(lam, f, marker="o", linestyle="",
                    color="black", label="Thwaites")

    # get the m values to needed to calculate the Falkner-Skan values
    m = np.array([2.0, 1.0, 0.6, 0.3, 0.1, 0.0, -0.05, -0.08, -0.09042860905])
    lam, shape, shear, f = calculate_falkner_skan_values(m)
    _ = axis_shape.plot(lam, shape, marker="o", linestyle="",
                        color="cyan", label="Falkner-Skan")
    _ = axis_shear.plot(lam, shear, marker="o", linestyle="",
                        color="cyan", label="Falkner-Skan")
    _ = axis_f.plot(lam, f, marker="o", linestyle="",
                    color="cyan", label="Falkner-Skan")

    # Create the various fit models
    lambda_min = -0.1
    lambda_max = 0.25
    models = [_ThwaitesFunctionsWhite(), _ThwaitesFunctionsCebeciBradshaw(),
              _ThwaitesFunctionsDrela(), _ThwaitesFunctionsSpline()]
    colors = ["red", "orange", "green", "purple"]
    labels = ["White", "Cebeci & Bradshaw", "Drela", "Spline"]

    for model, color, label in zip(models, colors, labels):
        lam, shape, shear, f = calculate_thwaites_values(lambda_min,
                                                         lambda_max,
                                                         npts, model)
        _ = axis_shape.plot(lam, shape, color=color, label=label)
        _ = axis_shear.plot(lam, shear, color=color, label=label)
        _ = axis_f.plot(lam, f, color=color, label=label)

    # Plot functions compared to the Thwaites tabulated values
    ax = axis_shape
    _ = ax.set_xlabel(r"$\lambda$")
    _ = ax.set_ylabel(r"$H\left(\lambda\right)$")
    ax.grid(True)
    _ = ax.legend()

    # plot the shear function
    ax = axis_shear
    _ = ax.set_xlabel(r"$\lambda$")
    _ = ax.set_ylabel(r"$S\left(\lambda\right)$")
    ax.grid(True)
    _ = ax.legend()

    # plot the f function
    ax = axis_f
    _ = ax.set_xlabel(r"$\lambda$")
    _ = ax.set_ylabel(r"$F\left(\lambda\right)$")
    ax.grid(True)
    _ = ax.legend()

    plt.show()


def calculate_falkner_skan_values(m: npt.NDArray) -> Tuple[npt.NDArray,
                                                           npt.NDArray,
                                                           npt.NDArray,
                                                           npt.NDArray]:
    """Calculate the Falkner-Skan value for the comparison"""
    fs = FalknerSkan(beta=0, u_ref=1, nu_ref=1)

    it = np.nditer([m, None, None, None, None])  # type: ignore[arg-type]
    with it:
        for m_val, lam_val, shape_val, s_val, f_val in it:
            fs.reset_m(m=float(m_val))
            beta = fs.beta
            eta_d = fs.eta_d
            eta_m = fs.eta_m
            fpp_w = fs.fw_pp
            shape_d = eta_d/eta_m
            lam = beta*eta_m**2
            lam_val[...] = lam
            shape_val[...] = shape_d
            s_val[...] = eta_m*fpp_w
            f_val[...] = 2*(eta_m*fpp_w - lam*(shape_d + 2))

        return it.operands[1], it.operands[2], it.operands[3], it.operands[4]


def calculate_thwaites_values(lambda_min: float, lambda_max: float, npts: int,
                              model: _ThwaitesFunctions) -> Tuple[npt.NDArray,
                                                                  npt.NDArray,
                                                                  npt.NDArray,
                                                                  npt.NDArray]:
    """Calculate the Thwaites parameters for given model."""
    # would like same range of lambdas, but each scheme has different range on
    # which it is defined.
    lam = np.linspace(np.maximum(model.range()[0], lambda_min),
                      np.minimum(model.range()[1], lambda_max), npts)

    # calculate the corresponding values
    shape = model.shape(lam)
    shear = model.shear(lam)
    f = model.f(lam)

    return lam, shape, shear, f


def get_thwaites_tabular_data() -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray,
                                         npt.NDArray]:
    """Return the Thwaites tabular data."""
    model = _ThwaitesFunctionsSpline()
    return (model.lambda_values, model.shape_values, model.shear_values,
            model.f_values)


if __name__ == "__main__":
    compare_shape_d()

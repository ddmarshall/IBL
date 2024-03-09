r"""
Comparing Thwaites' tabular data approximations to Falkner-Skan values.

This example shows a comparison between the tabular data from Thwaites (1949)
and the equivalent values from Falkner-Skan solution.
"""


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
    # pylint: disable-next=protected-access
    lam = _ThwaitesFunctionsSpline._tab_lambda
    # pylint: disable-next=protected-access
    shape = _ThwaitesFunctionsSpline._tab_shape
    # pylint: disable-next=protected-access
    shear = _ThwaitesFunctionsSpline._tab_shear
    # pylint: disable-next=protected-access
    f = _ThwaitesFunctionsSpline._tab_f
    axis_shape.plot(lam, shape, marker="o", linestyle="",
                    color="black", label="Thwaites")
    axis_shear.plot(lam, shear, marker="o", linestyle="",
                    color="black", label="Thwaites")
    axis_f.plot(lam, f, marker="o", linestyle="",
                color="black", label="Thwaites")

    # get the m values to needed to calculate the Falkner-Skan values
    m = np.array([2.0, 1.0, 0.6, 0.3, 0.1, 0.0, -0.05, -0.08, -0.09042860905])
    lam, shape, shear, f = calculate_falkner_skan_values(m)
    axis_shape.plot(lam, shape, marker="o", linestyle="",
                    color="cyan", label="Falkner-Skan")
    axis_shear.plot(lam, shear, marker="o", linestyle="",
                    color="cyan", label="Falkner-Skan")
    axis_f.plot(lam, f, marker="o", linestyle="",
                color="cyan", label="Falkner-Skan")

    # Create the various fit models
    lambda_min = -0.1
    lambda_max = 0.25
    models = [_ThwaitesFunctionsWhite(), _ThwaitesFunctionsCebeciBradshaw(),
              _ThwaitesFunctionsDrela(), _ThwaitesFunctionsSpline()]
    colors = ["red", "orange", "green", "purple"]
    labels = ["White", "Cebeci & Bradshaw", "drela", "Spline"]

    for model, color, label in zip(models, colors, labels):
        lam, shape, shear, f = calculate_thwaites_values(lambda_min,
                                                         lambda_max,
                                                         npts, model)
        axis_shape.plot(lam, shape, color=color, label=label)
        axis_shear.plot(lam, shear, color=color, label=label)
        axis_f.plot(lam, f, color=color, label=label)

    # Plot functions compared to the Thwaites tabulated values
    ax = axis_shape
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$H\left(\lambda\right)$")
    ax.grid(True)
    ax.legend()

    # plot the shear function
    ax = axis_shear
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$S\left(\lambda\right)$")
    ax.grid(True)
    ax.legend()

    # plot the f function
    ax = axis_f
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$F\left(\lambda\right)$")
    ax.grid(True)
    ax.legend()

    plt.show()


def calculate_falkner_skan_values(m: npt.NDArray) -> Tuple[npt.NDArray,
                                                           npt.NDArray,
                                                           npt.NDArray,
                                                           npt.NDArray]:
    """Calculate the Falkner-Skan value for the comparison"""
    fs = FalknerSkan(beta=0, u_ref=1, nu_ref=1)

    it = np.nditer([m, None, None, None, None])
    with it:
        for m_val, lam_val, shape_val, s_val, f_val in it:
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


if __name__ == "__main__":
    compare_shape_d()

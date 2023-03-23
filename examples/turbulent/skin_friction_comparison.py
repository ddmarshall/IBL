#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison between skin friction approximations.

This example shows a comparison between the various approximations for the
skin friction on the surface of a body in a incompressible, turbulent boundary
layer.
"""

import numpy as np
import matplotlib.pyplot as plt

from ibl.skin_friction import ludwieg_tillman
from ibl.skin_friction import felsch
from ibl.skin_friction import white


def compare_skin_friction_relations() -> None:
    """Compare skin friction relations."""
    # create the input vectors
    shape_d = np.array([1.4, 1.7, 2.0, 2.3, 2.6, 2.9])
    re_delta_m = np.logspace(2, 5, 101)

    # create the output matrices
    c_f_lt = np.zeros([np.shape(shape_d)[0], np.shape(re_delta_m)[0]])
    c_f_f = np.zeros_like(c_f_lt)
    c_f_w = np.zeros_like(c_f_lt)

    # calculate skin friction values
    for i, sdi in enumerate(shape_d):
        c_f_lt[i, :] = ludwieg_tillman(re_delta_m, sdi)
        c_f_f[i, :] = felsch(re_delta_m, sdi)
        c_f_w[i, :] = white(re_delta_m, sdi)

    # plot the results
    plt.rcParams['figure.figsize'] = [8, 5]
    colors = ["green", "orange", "blue", "purple", "red", "cyan"]

    plt.figure()
    for i, _ in enumerate(shape_d):
        plt.plot(re_delta_m, c_f_lt[i, :], linestyle="-", color=colors[i])
        plt.plot(re_delta_m, c_f_f[i, :], linestyle="--",  color=colors[i])
        plt.plot(re_delta_m, c_f_w[i, :], linestyle=":", color=colors[i])
    plt.xlabel(r"Re$_{\lambda_d}$")
    plt.ylabel(r"$c_f$")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    compare_skin_friction_relations()

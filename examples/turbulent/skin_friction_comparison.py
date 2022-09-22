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

from pyBL.skin_friction import c_f_LudwiegTillman
from pyBL.skin_friction import c_f_Felsch
from pyBL.skin_friction import c_f_White


def compare_skin_friction_relations():
    """Compare skin friction relations."""
    # create the input vectors
    H_d = np.array([1.4, 1.7, 2.0, 2.3, 2.6, 2.9])
    Re_delta_m = np.logspace(2, 5, 101)

    # create the output matrices
    c_f_LT = np.zeros([np.shape(H_d)[0], np.shape(Re_delta_m)[0]])
    c_f_F = np.zeros_like(c_f_LT)
    c_f_W = np.zeros_like(c_f_LT)

    # calculate skin friction values
    for i, H_di in enumerate(H_d):
        c_f_LT[i, :] = c_f_LudwiegTillman(Re_delta_m, H_di)
        c_f_F[i, :] = c_f_Felsch(Re_delta_m, H_di)
        c_f_W[i, :] = c_f_White(Re_delta_m, H_di)

    # plot the results
    plt.rcParams['figure.figsize'] = [8, 5]
    colors = ["green", "orange", "blue", "purple", "red", "cyan"]

    plt.figure()
    for i, H_di in enumerate(H_d):
        plt.plot(Re_delta_m, c_f_LT[i, :], linestyle="-", color=colors[i])
        plt.plot(Re_delta_m, c_f_F[i, :], linestyle="--",  color=colors[i])
        plt.plot(Re_delta_m, c_f_W[i, :], linestyle=":", color=colors[i])
    plt.xlabel(r"Re$_{\lambda_d}$")
    plt.ylabel(r"$c_f$")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    compare_skin_friction_relations()

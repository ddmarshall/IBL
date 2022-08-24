#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example shows a comparison between the various approximations to the 
tabular data from Thwaites for the function F(\lambda). It shows similar
results to Figures 1.2 and 1.3 in Edland thesis.

Created on Tue Aug 16 16:20:20 2022

@author: ddmarshall
"""

import numpy as np
import matplotlib.pyplot as plt

from pyBL.skin_friction import c_f_LudwiegTillman
from pyBL.skin_friction import c_f_Felsch
from pyBL.skin_friction import c_f_White


def compare_skin_friction_relations():
    ## create the input vectors
    H_d = np.array([1.4, 1.7, 2.0, 2.3, 2.6, 2.9])
    Re_delta_m = np.logspace(2, 5, 101)
    
    ## create the output matrices
    c_f_LT = np.zeros([np.shape(H_d)[0], np.shape(Re_delta_m)[0]])
    c_f_F = np.zeros_like(c_f_LT)
    c_f_W = np.zeros_like(c_f_LT)
    
    ## calculate skin friction values
    for i, H_di in enumerate(H_d):
        c_f_LT[i, :] = c_f_LudwiegTillman(Re_delta_m, H_di)
        c_f_F[i, :] = c_f_Felsch(Re_delta_m, H_di)
        c_f_W[i, :] = c_f_White(Re_delta_m, H_di)
    
    ## plot the results
    plt.rcParams['figure.figsize'] = [8, 5]
    colors = ["green", "orange", "blue", "purple", "red", "cyan"]
    
    plt.figure()
    for i, H_di in enumerate(H_d):
        plt.plot(Re_delta_m, c_f_LT[i, :], linestyle = "-", color = colors[i])
        plt.plot(Re_delta_m, c_f_F[i, :], linestyle = "--",  color = colors[i])
        plt.plot(Re_delta_m, c_f_W[i, :], linestyle = ":", color = colors[i])
    plt.xlabel(r"Re$_{\lambda_d}$")
    plt.ylabel(r"$c_f$")
#    plt.xlim([-0.10, 0.25])
#    plt.ylim([.00001,1])
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
#    plt.legend()
    plt.show()


if (__name__ == "__main__"):
    compare_skin_friction_relations()

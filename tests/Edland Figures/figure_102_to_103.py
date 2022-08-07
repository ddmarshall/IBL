#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 03:18:35 2022

@author: ddmarshall
"""

import numpy as np
import matplotlib.pyplot as plt

from pyBL.thwaites_method import lam_tab, f_tab, white_s, white_h, cebeci_s, cebeci_h, s_lam_spline, h_lam_spline

def thwaites_f_comparison():
    """
    Create the $F(\lambda)$ comparison, figures 1.2-1.3 in Edland's thesis
    
    Args
    ----
        None
    
    Returns
    -------
        None
    """
    
    ## Generate the values of F
    # generate the values for the plots of F
    lam = np.linspace(-0.1, 0.25, 101)
    f_linear = 0.45-6*lam
    f_white = 2*(white_s(lam)-lam*(2+white_h(lam)))
    lam_cebeci = lam[lam<=0.1]
    f_cebeci = 2*(cebeci_s(lam_cebeci)-lam_cebeci*(2+cebeci_h(lam_cebeci)))
    lam_spline = lam[lam>=np.min(lam_tab)]
    f_spline = 2*(s_lam_spline(lam_spline)-lam_spline*(2+h_lam_spline(lam_spline)))
    
    # generate the values for error comparison
    f_tab_linear = 0.45-6*lam_tab
    f_tab_white = 2*(white_s(lam_tab)-lam_tab*(2+white_h(lam_tab)))
    lam_tab_cebeci = lam_tab[lam_tab<=0.1]
    f_tab_cebeci = 2*(cebeci_s(lam_tab_cebeci)-lam_tab_cebeci*(2+cebeci_h(lam_tab_cebeci)))
    
    ## Plot the comparisons
    plt.rcParams['figure.figsize'] = [8, 5]
    
    # Plot function compared to the Thwaites tabulated values
    plt.figure()
    plt.plot(lam_tab, f_tab, marker='o', linestyle='', color='black', label=r'Thwaites Original')
    plt.plot(lam, f_linear, color='green', label=r'Linear')
    plt.plot(lam, f_white, color='red', label=r'White')
    plt.plot(lam_cebeci, f_cebeci, color='orange', label=r'Cebeci & Bradshaw')
    plt.plot(lam_spline, f_spline, color='purple', label=r'Spline')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$F\left(\lambda\right)$')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot error compared to Thwaites tabulated values
    plt.figure()
    plt.plot(lam_tab, np.abs(1-f_tab_linear/f_tab), color='green', label=r'Linear')
    plt.plot(lam_tab, np.abs(1-f_tab_white/f_tab), color='red', label=r'White')
    plt.plot(lam_tab_cebeci, np.abs(1-f_tab_cebeci/f_tab[lam_tab<=0.1]), color='orange', label=r'Cebeci & Bradshaw')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Relative Error')
    plt.xlim([-0.10, 0.25])
    plt.ylim([.00001,1])
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()


if (__name__ == "__main__"):
    thwaites_f_comparison()

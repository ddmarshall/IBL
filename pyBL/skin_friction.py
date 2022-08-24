#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:36:59 2022

@author: ddmarshall
"""

import numpy as np

def c_f_LudwiegTillman(Re_delta_m, H_d):
    """
    Calculate the skin friction coefficient using the Ludwieg-Tillman (1950)
    relation.
    
    Args
    ----
        Re_delta_m: Reynolds number based on the momentum thickness
        H_d: Displacement shape factor
    
    Returns
    -------
        Corresponding skin friction coefficient
    """
    return 0.246/(Re_delta_m**0.268*10**(0.678*H_d))

def c_f_Felsch(Re_delta_m, H_d):
    """
    Calculate the skin friction coefficient using the Felsch (1968) relation.
    
    Args
    ----
        Re_delta_m: Reynolds number based on the momentum thickness
        H_d: Displacement shape factor
    
    Returns
    -------
        Corresponding skin friction coefficient
    """
    H_d_sep = 2.9986313485
    return 0.058*(0.93 - 1.95*np.log10(np.clip(H_d, 1, H_d_sep)))**1.705/(Re_delta_m**0.268)

def c_f_White(Re_delta_m, H_d):
    """
    Calculate the skin friction coefficient using the White (2011) relation.
    
    Args
    ----
        Re_delta_m: Reynolds number based on the momentum thickness
        H_d: Displacement shape factor
    
    Returns
    -------
        Corresponding skin friction coefficient
    """
    return 0.3*np.exp(-1.33*H_d)/(np.log10(Re_delta_m))**(1.74+0.31*H_d)

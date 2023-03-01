"""
Functions to calculate skin friction coefficient for turbulent boundary layers.

This module provides functions to calculate the skin friction coefficient for
turbulent boundary layers  from the momentum thickness Reynolds number and
displacement thickness.
"""

import numpy as np


def c_f_LudwiegTillman(Re_delta_m, H_d):
    """
    Calculate skin friction coefficient using Ludwieg-Tillman (1950) relation.

    This function calculate the skin friction given the momentum thickness
    Reynolds number and displacement shape factor and is based on experimental
    data on turbulent boundary layers from Ludwieg and Tillman.

    If both parameters are array-like, they have to be the same shape.

    Parameters
    ----------
    Re_delta_m: float or array-like
        Reynolds number based on the momentum thickness
    H_d: float or array-like
        Displacement shape factor

    Returns
    -------
    array-like same shape as array-like input
        Corresponding skin friction coefficient
    """
    return 0.246/(Re_delta_m**0.268*10**(0.678*H_d))


def c_f_Felsch(Re_delta_m, H_d):
    """
    Calculate skin friction coefficient using Felsch (1968) relation.

    This function calculate the skin friction given the momentum thickness
    Reynolds number and displacement shape factor and is based on experimental
    data on turbulent boundary layers from Felsch.

    If both parameters are array-like, they have to be the same shape.

    Parameters
    ----------
    Re_delta_m: float or array-like
        Reynolds number based on the momentum thickness
    H_d: float or array-like
        Displacement shape factor

    Returns
    -------
    array-like same shape as array-like input
        Corresponding skin friction coefficient
    """
    H_d_sep = 2.9986313485
    term1 = 1.95*np.log10(np.clip(H_d, 1, H_d_sep))
    return 0.058*(0.93 - term1)**1.705/(Re_delta_m**0.268)


def c_f_White(Re_delta_m, H_d):
    """
    Calculate skin friction coefficient using White (2011) relation.

    This function calculate the skin friction given the momentum thickness
    Reynolds number and displacement shape factor and is based on experimental
    data on turbulent boundary layers from White.

    If both parameters are array-like, they have to be the same shape.

    Parameters
    ----------
    Re_delta_m: float or array-like
        Reynolds number based on the momentum thickness
    H_d: float or array-like
        Displacement shape factor

    Returns
    -------
    array-like same shape as array-like input
        Corresponding skin friction coefficient
    """
    return 0.3*np.exp(-1.33*H_d)/(np.log10(Re_delta_m))**(1.74+0.31*H_d)

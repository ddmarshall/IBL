"""
Functions to calculate skin friction coefficient for turbulent boundary layers.

This module provides functions to calculate the skin friction coefficient for
turbulent boundary layers  from the momentum thickness Reynolds number and
displacement thickness.
"""

import numpy as np

from ibl.typing import InputParam


def ludwieg_tillman(re_delta_m: InputParam, shape_d: InputParam) -> InputParam:
    """
    Calculate skin friction coefficient using Ludwieg-Tillman (1950) relation.

    This function calculate the skin friction given the momentum thickness
    Reynolds number and displacement shape factor and is based on experimental
    data on turbulent boundary layers from Ludwieg and Tillman.

    If both parameters are numpy arrays, they have to be the same shape.

    Parameters
    ----------
    re_delta_m: InputParam
        Reynolds number based on the momentum thickness
    shape_d: InputParam
        Displacement shape factor

    Returns
    -------
    InputParam
        Corresponding skin friction coefficient
    """
    return 0.246/(re_delta_m**0.268*10**(0.678*shape_d))


def felsch(re_delta_m: InputParam, shape_d: InputParam) -> InputParam:
    """
    Calculate skin friction coefficient using Felsch (1968) relation.

    This function calculate the skin friction given the momentum thickness
    Reynolds number and displacement shape factor and is based on experimental
    data on turbulent boundary layers from Felsch.

    If both parameters are array-like, they have to be the same shape.

    Parameters
    ----------
    re_delta_m: InputParam
        Reynolds number based on the momentum thickness
    shape_d: InputParam
        Displacement shape factor

    Returns
    -------
    InputParam
        Corresponding skin friction coefficient
    """
    shape_d_sep = 2.9986313485
    term1 = 1.95*np.log10(np.clip(shape_d, 1, shape_d_sep))
    return 0.058*(0.93 - term1)**1.705/(re_delta_m**0.268)


def white(re_delta_m: InputParam, shape_d: InputParam) -> InputParam:
    """
    Calculate skin friction coefficient using White (2011) relation.

    This function calculate the skin friction given the momentum thickness
    Reynolds number and displacement shape factor and is based on experimental
    data on turbulent boundary layers from White.

    If both parameters are array-like, they have to be the same shape.

    Parameters
    ----------
    re_delta_m: InputParam
        Reynolds number based on the momentum thickness
    shape_d: InputParam
        Displacement shape factor

    Returns
    -------
    InputParam
        Corresponding skin friction coefficient
    """
    return 0.3*np.exp(-1.33*shape_d)/(np.log10(re_delta_m))**(1.74
                                                              + 0.31*shape_d)

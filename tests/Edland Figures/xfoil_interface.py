# pylint: skip-file

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 22:56:30 2022

@author: ddmarshall
"""

import numpy as np
import csv
from collections import namedtuple


def read_xfoil_dump_file(xfoil_file, U_inf=1):
    """
    Return the results from XFoil dump file.
    
    Args
    ----
        xfoi_file - file name containing XFoil dump data
        
    Returns
    -------
        s_lower - arclength distance from stagnation point on lower surface
        s_upper - arclength distance from stagnation point on upper surface
        Ue_lower - lower surface velocity
        Ue_upper - upper surface velocity
        del_star_lower - lower surface displacement thickness
        del_star_upper - upper surface displacement thickness
        theta_lower - lower surface momentum thickness
        theta_upper - upper surface momentum thickness
        cf_lower - lower surface skin friction coefficient
        cf_upper - upper surface skin friction coefficient
        H_lower - lower surface shape factor
        H_upper - upper surface shape factor
    """
    
    PanelValues = namedtuple('PanelValues', ['s', 'x', 'y', 'Ue', 'delta_d', 'delta_m', 'cf', 'H', 'Hstar', 'P', 'm', 'K'])
    WakeValues = namedtuple('PanelValues', ['s', 'x', 'y', 'Ue', 'delta_d', 'delta_m', 'H'])
    
    xfoil_file = open(xfoil_file, 'r')
    xfoil_reader = csv.reader(xfoil_file)
    panel_terms = []
    wake_terms = []
    for row in xfoil_reader:
        if (row[0][0] != '#'):
            col = np.array(row[0].split(), 'float')
            
            if (np.size(col) == 12):
                panel_terms.append(col)
            elif (np.size(col) == 8):
                wake_terms.append(col)
            else:
                raise Exception('Invalid number of columns for XFoil import')
                    
    ## Find stagnation point and the total arclength of airfoil
    panel_terms = np.asarray(panel_terms)
    last_upper_idx = np.where(np.sign(panel_terms[:-1, 3]) != np.sign(panel_terms[1:, 3]))[0][0]
    s_af = panel_terms[-1, 0]
    
    # interpolate stagnation point
    dU = panel_terms[last_upper_idx+1, 3]-panel_terms[last_upper_idx, 3]
    frac = -panel_terms[last_upper_idx, 3]/dU
    stag_col = frac*panel_terms[last_upper_idx+1,:]+(1-frac)*panel_terms[last_upper_idx,:]

    ## Split into upper and lower surface terms
    # upper panel needs to be reversed since XFoil writes out starting with 
    #   the upper surface trailing edge to lower surface trailing edge.
    panel_upper_terms = panel_terms[last_upper_idx::-1, :]
    # insert the stagnation point
    panel_upper_terms = np.insert(panel_upper_terms, 0, stag_col, 0)
    # adjust distance is based on arclength distance from upper trailing edge.
    panel_upper_terms[:,0] = panel_upper_terms[0,0] - panel_upper_terms[:,0]
    panel_upper_terms[:,3] = U_inf*panel_upper_terms[:,3]
    upper_values = PanelValues(*panel_upper_terms.T)
    
    # lower panel
    panel_lower_terms = panel_terms[last_upper_idx+1:, :]
    # insert the stagnation point
    panel_lower_terms = np.insert(panel_lower_terms, 0, stag_col, 0)
    # adjust distance (see upper panel)
    panel_lower_terms[:,0] = panel_lower_terms[:,0] - panel_lower_terms[0,0]
    # adjust the signs of the lower surface terms
    panel_lower_terms[:,3] = -panel_lower_terms[:,3]
    panel_lower_terms[:,[6,10,11]] = np.abs(panel_lower_terms[:,[6,10,11]])
    # remove scale of velocities
    panel_lower_terms[:,3] = U_inf*panel_lower_terms[:,3]
    lower_values = PanelValues(*panel_lower_terms.T)
    
    # wake
    # adjust distance (base it on trailing edge)
    wake_terms = np.asarray(wake_terms)
    wake_terms[:,0] = wake_terms[:,0] - s_af
    # remove zeros from skin friction coefficient
    wake_terms = np.delete(wake_terms, 6, 1)
    wake_values = WakeValues(*wake_terms.T)
    
    return upper_values, lower_values, wake_values



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:13:14 2022

@author: ddmarsha
"""
import os
import subprocess
import glob
from sys import platform
import numpy as np

def get_xfoil_data(airfoil, aoa, v_inf, re, s_trans):

    # executable name
    if (platform == 'linux'):
        xfoil_name = 'xfoil'
    elif (platform == 'win32'):
        xfoil_name = 'xfoil.exe'
    else:
        xfoil_name = 'noname'

    n_iter = 1
    n_transition = 9 #for xfoil e^N, 9 is default
    
    invfile = 'inv.txt'
    invcpfile = 'invcp.txt'
    viscfile =  'visc.txt'
    
    invisc_command_list = ("""NACA """+str(airfoil)+"""
    OPER
    a """+str(aoa)+"""
    dump """+invfile+"""
    cpwr """+invcpfile+"""
    
    quit
    """).encode('utf-8')
    
    process = subprocess.Popen([xfoil_name],
                  stdin=subprocess.PIPE,
                  stdout=None,
                  stderr=None)
    process.communicate(invisc_command_list)
    process.wait()

    #Get the inviscid sim u_e and coordinates
    invdata = np.loadtxt(invfile)
    s_data = invdata[:,0]
    u_e_over_v_inf_data = invdata[:,3]
    
    #Get the inviscid cp
    invcpdata = np.loadtxt(invcpfile,skiprows=2)
    cp = invcpdata[:,1]
    #Extract estimation of stagnation point
    stagnation_ind = np.where(abs(cp-1) == min(abs(cp-1)))[0][0] -1 #stagnation based on cp being close to 1
    # stagnation_ind = int(np.where(x_data==min(x_data))[0][0]) #location of minimum x coordinate
    # stagnation_ind = np.where(u_e_over_v_inf_data==abs(u_e_over_v_inf_data))[-1][-1] #last place velocity is positive (first place on chord)
    
    #flip the inviscid results
    # s = np.flip(s_data[leading_edge_ind]-s_data[0:leading_edge_ind+1]) #
    # u_e = np.flip(u_e_over_v_inf_data[0:leading_edge_ind+1])*v_inf
    c = s_data[stagnation_ind]-s_data[0];
    s = np.flip(s_data[stagnation_ind]-s_data) #s=0 is stagnation point
    u_e = np.flip(u_e_over_v_inf_data)*v_inf

    #Run viscous xfoil
    visc_command_list = ("""NACA """+str(airfoil)+"""
    OPER
    VISC
    """+str(re)+"""
    ITER 
    """+str(n_iter)+"""
    vpar
    n
    """+str(n_transition)+"""
    xtr
    """+str(s_trans)+"""
    1
    
    a """+str(aoa)+"""
    dump """+viscfile+"""
    
    quit
    """).encode('utf-8')
    
    # visc_command_list = ("""NACA """+str(airfoil)+"""
    # OPER
    # VISC
    # """+str(re)+"""
    # ITER 
    # """+str(n_iter)+"""
    # vpar
    # n
    # """+str(n_transition)+"""
    # xtr
    # """+str(float(michel.x_tr))+"""
    # 1
    
    # a """+str(aoa)+"""
    # dump """+viscfile+"""
    
    # quit
    # """).encode('utf-8')
    
    process = subprocess.Popen([xfoil_name],
                  stdin=subprocess.PIPE,
                  stdout=None,
                  stderr=None)
    process.communicate(visc_command_list)
    process.wait()
    
    #truncate viscous data (avoid added points)
    viscdata = np.loadtxt(viscfile, usecols=(0,1,2,3,4,5,6,7))
    invlength = invdata.shape[0]
    del_star_data = viscdata[0:invlength,4]
    theta_data = viscdata[0:invlength,5]
    c_f_data = viscdata[0:invlength,6]
    h_data = viscdata[0:invlength,7]

    #flip the viscous results
    del_star = np.flip(del_star_data)
    theta = np.flip(theta_data)
    c_f = np.flip(c_f_data)*(v_inf**2)/(u_e**2)
    h = np.flip(h_data)
    theta0=float(theta_data[stagnation_ind])
    h0=float(h_data[stagnation_ind])
    
    # remove the xfoil dumped files
    os.remove(invfile)
    os.remove(invcpfile)
    os.remove(viscfile)
    filelist = glob.glob('*.bl')
    for filepath in filelist:
        os.remove(filepath)
    
    return c, theta0, h0, s, u_e, del_star, theta, c_f, h


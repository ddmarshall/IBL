# pylint: skip-file

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sat Aug  6 02:46:06 2022

# @author: ddmarshall
# """


# import numpy as np
# import matplotlib.pyplot as plt

# from pyBL.thwaites_method import ThwaitesSimData, ThwaitesSim
# from pyBL.head_method import HeadSimData, HeadSim
# from pyBL.pyBL import Forced

# from xfoil_interface import read_xfoil_dump_file

# def forced_xfoil_comparison():
#     """
#     Create the forced transition airfoil comparison, figures 3.24-3.25 in Edland's thesis
    
#     Args
#     ----
#         None
    
#     Returns
#     -------
#         None
#     """
    
#     ## Load the XFoil data for the case
#     # parameters use to generate case
#     # NACA 009
#     # aoa =0
#     # x_trans = 1 (upper and lower)
#     # Transition model n = 9
#     # Natural transition occurs at x=0.6600
#     c = 1 #m
#     U_inf = 20 #m/s
#     Re = 2e6
#     x_tr = 0.6600
#     xfoil_upper, xfoil_lower, _ = read_xfoil_dump_file('xfoil_natural_dump.txt', U_inf)
#     nu = U_inf*c/Re

#     ## Create the simulations for the upper and lower surfaces
#     lower_range=slice(1,np.size(xfoil_lower.s))
#     # do laminar calculation
#     tsd_l = ThwaitesSimData(xfoil_lower.s[lower_range],xfoil_lower.Ue[lower_range],U_inf,nu,Re,xfoil_lower.s[lower_range.start],theta0=xfoil_lower.delta_m[lower_range.start])
#     ts_l = ThwaitesSim(tsd_l) 
#     while ts_l.status=='running':
#         ts_l.step()

#     upper_range=slice(1,np.size(xfoil_upper.s))
#     tsd_u = ThwaitesSimData(xfoil_upper.s[upper_range],xfoil_upper.Ue[upper_range],U_inf,nu,Re,xfoil_upper.s[upper_range.start],theta0=xfoil_upper.delta_m[upper_range.start])
#     ts_u = ThwaitesSim(tsd_u) 
#     while ts_u.status=='running':
#         ts_u.step()

#     # find transition
#     forced_l = Forced(ts_l, x_tr, buffer = 0)
#     forced_u = Forced(ts_u, x_tr, buffer = 0)
#     print('lower: x_tr = ', np.float(forced_l.x_tr))
#     print('upper: x_tr = ', np.float(forced_u.x_tr))

#     # continue with turbulent boundary layer
#     try:
#         hsd_l = HeadSimData(xfoil_lower.s[lower_range],xfoil_lower.Ue[lower_range],U_inf,nu,float(forced_l.x_tr),theta0=float(ts_l.theta(forced_l.x_tr)), h0=forced_l.h0)
#         hs_l = HeadSim(hsd_l) 
#         while hs_l.status=='running':
#             hs_l.step()

#         # extract the laminar and turbulent results into arrays
#         forced_idx = np.where(xfoil_lower.s<=forced_l.x_tr)[0][-1]
#         lam_slice=slice(lower_range.start, forced_idx+1)
#         turb_slice=slice(forced_idx+1, lower_range.stop)
#         s_lam = xfoil_lower.s[lam_slice]
#         s_turb = xfoil_lower.s[turb_slice]
#         x_l = np.append(xfoil_lower.x[lam_slice], xfoil_lower.x[turb_slice])
#         theta_l = np.append(ts_l.theta(s_lam), hs_l.theta(s_turb))
#         del_star_l = np.append(ts_l.del_star(s_lam), hs_l.del_star(s_turb))
#         cf_l = np.append(ts_l.c_f(s_lam), hs_l.c_f(s_turb))
#         H_l = np.append(ts_l.h(s_lam), hs_l.h(s_turb))
#     except TypeError: #happens when flow has not transitioned
#         print('Why did this not transition?')
#     try:
#         hsd_u = HeadSimData(xfoil_upper.s[upper_range],xfoil_upper.Ue[upper_range],U_inf,nu,float(forced_u.x_tr),theta0=float(ts_u.theta(forced_u.x_tr)), h0=forced_u.h0)
#         hs_u = HeadSim(hsd_u) 
#         while hs_u.status=='running':
#             hs_u.step()

#         # extract the laminar and turbulent results into arrays
#         forced_idx = np.where(xfoil_upper.s<=forced_u.x_tr)[0][-1]
#         lam_slice=slice(upper_range.start, forced_idx+1)
#         turb_slice=slice(forced_idx+1, upper_range.stop)
#         s_lam = xfoil_upper.s[lam_slice]
#         s_turb = xfoil_upper.s[turb_slice]
#         x_u = np.append(xfoil_upper.x[lam_slice], xfoil_upper.x[turb_slice])
#         theta_u = np.append(ts_u.theta(s_lam), hs_u.theta(s_turb))
#         del_star_u = np.append(ts_u.del_star(s_lam), hs_u.del_star(s_turb))
#         cf_u = np.append(ts_u.c_f(s_lam), hs_u.c_f(s_turb))
#         H_u = np.append(ts_u.h(s_lam), hs_u.h(s_turb))
#     except TypeError: #happens when flow has not transitioned
#         print('Why did this not transition?')

#     ## Plot the comparisons
#     spline_label = 'PyBL'
#     spline_color = 'green'
#     ref_label = 'XFoil (laminar)'
#     ref_color = 'black'
#     ref_marker = ''
#     ref_linestyle = '-'

#     plt.rcParams['figure.figsize'] = [8, 5]
    
#     # Plot the results comparisons
#     fig, ax = plt.subplots(nrows=2, ncols=2, sharex='all')
    
#     # Momentum thickness in 0,0
#     i=0
#     j=0
#     ref_curve = ax[i][j].plot(xfoil_lower.x/c, xfoil_lower.delta_m/c, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
#     ax[i][j].plot(xfoil_upper.x/c, xfoil_upper.delta_m/c, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
#     spline_curve = ax[i][j].plot(x_l/c, theta_l/c, color=spline_color)
#     ax[i][j].plot(x_u/c, theta_u/c, color=spline_color)
#     ax[i][j].set_ylim(0, 0.002)
#     ax[i][j].set_xlabel(r'$x/c$')
#     ax[i][j].set_ylabel(r'$\theta/c$')
#     ax[i][j].grid(True)

#     # Displacement thickness in 0,1
#     i=0
#     j=1
#     ax[i][j].plot(xfoil_lower.x/c, xfoil_lower.delta_d/c, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
#     ax[i][j].plot(xfoil_upper.x/c, xfoil_upper.delta_d/c, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
#     ax[i][j].plot(x_l/c, del_star_l/c, color=spline_color)
#     ax[i][j].plot(x_u/c, del_star_u/c, color=spline_color)
#     ax[i][j].set_ylim(0, 0.003)
#     ax[i][j].set_xlabel(r'$x/c$')
#     ax[i][j].set_ylabel(r'$\delta^*/c$')
#     ax[i][j].grid(True)
    
#     # Skin friction coefficient in 1,0
#     i=1
#     j=0
#     ax[i][j].plot(xfoil_lower.x/c, xfoil_lower.cf, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
#     ax[i][j].plot(xfoil_upper.x/c, xfoil_upper.cf, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
#     ax[i][j].plot(x_l/c, cf_l, color=spline_color)
#     ax[i][j].plot(x_u/c, cf_u, color=spline_color)
#     ax[i][j].set_ylim(0, 0.01)
#     ax[i][j].set_xlabel(r'$x/c$')
#     ax[i][j].set_ylabel(r'$c_f$')
#     ax[i][j].grid(True)
    
#     # Shape factor in 1,1
#     i=1
#     j=1
#     ax[i][j].plot(xfoil_lower.x/c, xfoil_lower.H, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
#     ax[i][j].plot(xfoil_upper.x/c, xfoil_upper.H, color=ref_color, marker=ref_marker, linestyle=ref_linestyle)
#     ax[i][j].plot(x_l/c, H_l, color=spline_color)
#     ax[i][j].plot(x_u/c, H_u, color=spline_color)
#     ax[i][j].set_ylim(1, 3)
#     ax[i][j].set_xlabel(r'$x/c$')
#     ax[i][j].set_ylabel(r'$H$')
#     ax[i][j].grid(True)

#     # Based on example from: https://riptutorial.com/matplotlib/example/10473/single-legend-shared-across-multiple-subplots
#     fig.legend(handles=[ref_curve[0], spline_curve[0]], labels=[ref_label, spline_label],
#                loc="lower center", ncol=2, borderaxespad=0.1)
#     fig.set_figwidth(8)
#     fig.set_figheight(8)
#     plt.subplots_adjust(bottom=0.1, wspace=0.35)
#     plt.show()

#     # Plot difference compared to the XFoil results
#     plt.figure()
#     plt.plot(xfoil_lower.x[lower_range]/c, np.abs(1-theta_l/xfoil_lower.delta_m[lower_range]), label=r'$\theta$')
#     plt.plot(xfoil_lower.x[lower_range]/c, np.abs(1-del_star_l/xfoil_lower.delta_d[lower_range]), label=r'$\delta^*$')
#     plt.plot(xfoil_lower.x[lower_range]/c, np.abs(1-cf_l/xfoil_lower.cf[lower_range]), label='$c_f$')
#     plt.plot(xfoil_lower.x[lower_range]/c, np.abs(1-H_l/xfoil_lower.H[lower_range]), label='$H$')
#     plt.xlabel(r'$x/c$')
#     plt.ylabel('Relative Difference')
#     plt.ylim([.00001,30])
#     plt.yscale('log')
#     plt.grid(True)
#     plt.legend(ncol=2)
#     plt.show()


# if (__name__ == "__main__"):
#     forced_xfoil_comparison()


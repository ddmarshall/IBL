# -*- coding: utf-8 -*-
"""
@author: blond
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
     
from pyBL.head_method import HeadSimData, HeadSim, HeadSeparation #, TransitionModel
from scipy.interpolate import CubicSpline #for smoothed derivative experiment
import tikzplotlib

from plot_BL_params import theta_linestyle,del_linestyle,c_f_linestyle,h_linestyle
from plot_BL_params import plot_BL_params,spline_label,spline_linestyle,smooth_label,smooth_linestyle,der_label,der_linestyle,data_marker,data_label

#Keeping colors straight
tabcolor = 'tab:blue'
smoothcolor = 'tab:red'
smoothdcolor = 'tab:green'

thetacolor = 'tab:blue'
hcolor = 'tab:orange'
delcolor = 'tab:green'
cfcolor = 'tab:red'

def head_fp_mpg():

    #tabular data:
    x_vec=    np.array([.782,  1.282, 1.782, 2.282, 2.782, 3.132, 3.332, 3.532, 3.732, 3.932, 4.132, 4.332])
    u_e_vec = np.array([33.90, 32.60, 30.70, 28.60, 27.10, 26.05, 25.75,24.85,24.50, 24.05, 23.60, 23.10])
    du_edx_tab = np.array([-2.3,-3.35,-4.32,-3.580,-3,-2.74,-2.6,-2.5,-2.4,-2.3,-2.25,-2.18]) #from tabulated values
    approx_du_edx = (u_e_vec[:-1]-u_e_vec[1:])/(x_vec[:-1]-x_vec[1:])
    #from plot:
    smooth_x = np.linspace(.5,4.5,17)
    smooth_u_e = np.array([34.41,33.98,33.38,32.63,31.79,30.78,29.66,28.78,27.9,27.15,26.42,25.7,25.1,24.45,23.85,23.38,22.79])
    smooth_du_edx = -1 - .1*np.array([6.9,12.5,17.5,22.9,28.2,33.3,30,26.2,22.8,20.5,18.2,16.5,15.2,13.8,12.9,12.1,11.3])
    approx_du_edx_smooth = (smooth_u_e[:-1]-smooth_u_e[1:])/(smooth_x[:-1]-smooth_x[1:])
    theta_tab = np.array([.276,.413,.606,.811,1.074,1.276,1.432,1.614,1.773,2.005,2.246,2.528])/100
    c_f_tab = np.array([.00285,.00249,.00221,.00205,.00180,.00168,.00162,.00150,.00141,.00133,.00124,.00117])
    c_f_lt_tab = np.array([.00276,.00246,.00222,.00202,.00181,.00167,.00161,.00151,.00142,.00133,.00124,.00117])
    h_tab = np.array([1.381,1.394,1.402,1.427,1.457,1.485,1.492,1.519,1.542,1.566,1.594,1.618])
    del_tab = h_tab*theta_tab
    nu = .15500/(100*100) #cm^2/sec
    
    start_index = 0
    x0 = x_vec[start_index]
    u_inf = 1 #not used
    h0 = h_tab[start_index]
    #theta0 = .276/100
    theta0 = theta_tab[start_index]
    
    
    
    #Simulation using tabulated values
    hsd = HeadSimData(x_vec,
                     u_e_vec,
                     u_inf,
                     nu,
                     x0,
                     theta0,
                     h0)
    hs = HeadSim(hsd)
    hs_sep = HeadSeparation(hs)
    #while hs.status=='running' and hs.x_vec[-1]<x_vec[-1]:
    while hs.status=='running' and hs.x_vec[-1]<3.4:
        hs.step()
    
    #Now Simulation using values from smoothed plot    
    hsd_smooth = HeadSimData(smooth_x,
                     smooth_u_e,
                     u_inf,
                     nu,
                     x0,
                     theta0,
                     h0)
    hs_smooth = HeadSim(hsd_smooth)
    while hs_smooth.status=='running': # and hs_smooth.x_vec[-1]<3.4:
        hs_smooth.step()
    hs_smooth_sep = HeadSeparation(hs_smooth)
    ###Experiment with Smooth Derivative to establish spline
    smooth_due_dx_spline = CubicSpline(smooth_x,smooth_du_edx)
    smooth_due_dx_spline_antiderivative = smooth_due_dx_spline.antiderivative()
    
    #Create same simulation using smoothed u_e
    hsd_smooth_der = HeadSimData(smooth_x,
                     smooth_u_e,
                     u_inf,
                     nu,
                     x0,
                     theta0,
                     h0)
    # modify the u_e method to be the antiderivative of the duedx data + the first u_e
    hsd_smooth_der.u_e = lambda x: smooth_due_dx_spline_antiderivative(x)+smooth_u_e[0]
    #modify the du_edx method to be a curve fit to the smoothed derivative
    hsd_smooth_der.du_edx = lambda x: smooth_due_dx_spline(x)
    hs_smooth_der = HeadSim(hsd_smooth_der)
    hs_smooth_der_sep= HeadSeparation(hs_smooth_der)
    while hs_smooth_der.status=='running': # and hs_smooth.x_vec[-1]<3.4:
        hs_smooth_der.step()
    
    #Incremented for plotting
    plotx = np.linspace(x_vec[0],x_vec[-1])
    smoothplotx = np.linspace(smooth_x[0],smooth_x[-1]) #different limmits than tabulated data
    
    
    
    #u_e comparison
    
    #Change Font for Plots to Times New Roman
    #plt.rcParams["font.family"] = "Times New Roman"
    #plt.rcParams["mathtext.default"] = "regular" #make math text also tnr (not italic - probably need LaTeX)
    
    
    fig,ax = plt.subplots()
    #Formatting - sizing goes away in tikz
    # fig.set_figheight(6) 
    # fig.set_figwidth(10)
    ax.set(xlabel='x(m)', ylabel='$u_e$ (m/s)')
    
    # plt.rcParams["legend.edgecolor"] = "0" #sold legend edge
    # plt.rcParams["legend.framealpha"] = "1" #legend transparency - solid
    plt.grid(True)
    
    #plt.title(r'$u_e$ values')
    
    #plotx = np.linspace(x_vec[0],x_vec[-1])
    # ax.plot(x_vec,u_e_vec,'o',label=r'$u_e$ tabulated points',color=tabcolor)
    ax.plot(plotx, hs.u_e(plotx),label=spline_label,color='k',linestyle=spline_linestyle)
    # ax.plot(smooth_x,smooth_u_e,'o',label=r'smoothed $u_e$ points',color='k')
    ax.plot(smoothplotx, hs_smooth.u_e(smoothplotx),label=smooth_label,color='k',linestyle=smooth_linestyle)
    ax.plot(smoothplotx, hs_smooth_der.u_e(smoothplotx),label=der_label,color='k',linestyle=der_linestyle)
    # ax.legend(loc='upper right', ncol=1)
    ax.legend()
    #Try exporting to tikz:
    # tikzplotlib.Flavors.latex.preamble()
    #tikzplotlib.clean_figure() #cleans the figure up, not sure if important
    #tikzplotlib.save("velocities.tex")
    
    if (not os.path.isdir('figures')):
        os.mkdir('figures')

    tikzplotlib.save(
        'figures/fp_mapg_velocities.tex',
        axis_height = '\\figH',
        axis_width = '\\figW'
        )
    
    
    #duedx comparison
    fig,ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(10)
    plt.rcParams["legend.edgecolor"] = "0" #sold legend edge
    plt.rcParams["legend.framealpha"] = "1" #legend transparency - solid
    plt.grid(True)
    #plotx = np.linspace(x_vec[0],x_vec[-1])
    midx = (x_vec[1:] + x_vec[:-1]) / 2 #for simple derivative approximation
    midx_smooth = (smooth_x[1:] + smooth_x[:-1]) / 2
    # ax.plot(x_vec,du_edx_tab,'*',label = data_label ,color='k')
    ax.plot(plotx, hs.du_edx(plotx),label=spline_label,color='k',linestyle=spline_linestyle)
    ax.plot(smoothplotx, hs_smooth.du_edx(smoothplotx),label=smooth_label,color='k',linestyle=smooth_linestyle)
    # ax.plot(smooth_x,smooth_du_edx,'+',label=der_label,color='k')
    ax.plot(smoothplotx, hs_smooth_der.du_edx(smoothplotx),label=der_label,color='k',linestyle=der_linestyle)
    
    #ax.plot(midx,approx_du_edx,'o',label='simple derivative approximation')
    #ax.plot(midx_smooth,approx_du_edx_smooth,'o',label=r'simple derivative approximation $\frac{u_{e(i+1)}-u_{e(i)}}{x_{i+1}-x_i}$ (from smooth data)')
    ax.set(xlabel=r'$x$(m)',ylim=[2,-5.2])#, ylabel=r'$\frac{du_e}{dx}$',ylim=[2,-5.2])
    ax.set_ylabel(r'$\frac{du_e}{dx}$',rotation=0,labelpad=15)
    #plt.title(r'$\frac{du_e}{dx}$ Comparison')
    ax.legend(loc='lower right', ncol=1)
    tikzplotlib.save(
        'figures/derivatives.tex',
        axis_height = '\\figH',
        axis_width = '\\figW'
        )
    
    
    
    #Transpiration Velocity Comparison
    #transpiration_velocity_tab = CubicSpline(plotx,hs.u_e(plotx)*hs.h(plotx)*hs.theta(plotx))(plotx,1)
    transpiration_velocity_tab = (hs.du_edx(plotx)*hs.h(plotx)*hs.theta(plotx) + 
                                  hs.u_e(plotx)*hs.yp(plotx)[:,1]*hs.theta(plotx) +
                                  hs.u_e(plotx)*hs.h(plotx)*hs.yp(plotx)[:,0])
    #transpiration_velocity_smooth_u_e = CubicSpline(plotx,hs_smooth.u_e(plotx)*hs_smooth.h(plotx)*hs_smooth.theta(plotx))(plotx,1)
    transpiration_velocity_smooth_u_e = (hs_smooth.du_edx(plotx)*hs_smooth.h(plotx)*hs_smooth.theta(plotx) + 
                                          hs_smooth.u_e(plotx)*hs_smooth.yp(plotx)[:,1]*hs_smooth.theta(plotx) +
                                          hs_smooth.u_e(plotx)*hs_smooth.h(plotx)*hs_smooth.yp(plotx)[:,0])
    
    #transpiration_velocity_smooth_der = CubicSpline(plotx,hs_smooth_der.u_e(plotx)*hs_smooth_der.h(plotx)*hs_smooth_der.theta(plotx))(plotx,1)
    transpiration_velocity_smooth_der = (hs_smooth_der.du_edx(plotx)*hs_smooth_der.h(plotx)*hs_smooth_der.theta(plotx) + 
                                          hs_smooth_der.u_e(plotx)*hs_smooth_der.yp(plotx)[:,1]*hs_smooth_der.theta(plotx) +
                                          hs_smooth_der.u_e(plotx)*hs_smooth_der.h(plotx)*hs_smooth_der.yp(plotx)[:,0])
    fig,ax = plt.subplots()
    ax.ticklabel_format(style='plain')
    #plt.yticks(ticks = plt.yticks(),labels= [str(x) for x in np.arange(0,.45,step=.05)])
    # fig.set_figheight(6)
    # fig.set_figwidth(10)
    # plt.rcParams["legend.edgecolor"] = "0" #sold legend edge
    # plt.rcParams["legend.framealpha"] = "1" #legend transparency - solid
    plt.grid(True)
    ax.plot(plotx,transpiration_velocity_tab,label=spline_label,color='k',linestyle=spline_linestyle)
    ax.plot(plotx,transpiration_velocity_smooth_u_e,label=smooth_label,color='k',linestyle=smooth_linestyle)
    ax.plot(plotx,transpiration_velocity_smooth_der,label=der_label,color='k',linestyle=der_linestyle)
    # ax.legend(loc = 'lower right')
    ax.legend()
    ax.set(xlabel ='x(m)',ylabel='$u_n$ (m/s)') ##need better notation
    #plt.title('Transpiration Velocity Comparison')
    tikzplotlib.save(
        'figures/transpiration.tex',
        axis_height = '\\figH',
        axis_width = '\\figW'
        )
    
    fig,ax = plt.subplots()
    # errors
    # theta_tab = np.array([.276,.413,.606,.811,1.074,1.276,1.432,1.614,1.773,2.005,2.246,2.528])/100
    # c_f_tab = np.array([.00285,.00249,.00221,.00205,.00180,.00168,.00162,.00150,.00141,.00133,.00124,.00117])
    # c_f_lt_tab = np.array([.00276,.00246,.00222,.00202,.00181,.00167,.00161,.00151,.00142,.00133,.00124,.00117])
    # h_tab = np.array([1.381,1.394,1.402,1.427,1.457,1.485,1.492,1.519,1.542,1.566,1.594,1.618])
    # del_tab = h_tab*theta_tab
    
    ax.plot(x_vec,abs((hs.theta(x_vec)-theta_tab)/theta_tab),color='k',label=r'$\theta$',linestyle=theta_linestyle)
    ax.plot(x_vec,abs((hs.del_star(x_vec)-del_tab)/del_tab),color='k',label='$\delta$',linestyle=del_linestyle)
    ax.plot(x_vec,abs((hs.c_f(x_vec)-c_f_tab)/c_f_tab),color='k',label='$c_f$',linestyle=c_f_linestyle)
    ax.plot(x_vec,abs((hs.h(x_vec)-h_tab)/h_tab),color='k',label='$H$',linestyle=h_linestyle)
    
    plt.yscale('log')
    ax.legend()
    plt.grid(True)
    ax.set(xlabel='$x$(m)', ylabel='Relative Error')
    tikzplotlib.save(
        'figures/flatplate_mpg_raw_error.tex',
        axis_height = '\\figH',
        axis_width = '\\figW'
        )
    fig,ax = plt.subplots()
    ax.plot(x_vec,abs((hs_smooth.theta(x_vec)-theta_tab)/theta_tab),color='k',label=r'$\theta$',linestyle=theta_linestyle)
    ax.plot(x_vec,abs((hs_smooth.del_star(x_vec)-del_tab)/del_tab),color='k',label='$\delta$',linestyle=del_linestyle)
    ax.plot(x_vec,abs((hs_smooth.c_f(x_vec)-c_f_tab)/c_f_tab),color='k',label='$c_f$',linestyle=c_f_linestyle)
    ax.plot(x_vec,abs((hs_smooth.h(x_vec)-h_tab)/h_tab),color='k',label='$H$',linestyle=h_linestyle)
    
    plt.yscale('log')
    ax.legend()
    plt.grid(True)
    ax.set(xlabel='$x$(m)', ylabel='Relative Error')
    tikzplotlib.save(
        'figures/flatplate_mpg_smooth_error.tex',
        axis_height = '\\figH',
        axis_width = '\\figW'
        )
    fig,ax = plt.subplots()
    ax.plot(x_vec,abs((hs_smooth_der.theta(x_vec)-theta_tab)/theta_tab),color='k',label=r'$\theta$',linestyle=theta_linestyle)
    ax.plot(x_vec,abs((hs_smooth_der.del_star(x_vec)-del_tab)/del_tab),color='k',label='$\delta$',linestyle=del_linestyle)
    ax.plot(x_vec,abs((hs_smooth_der.c_f(x_vec)-c_f_tab)/c_f_tab),color='k',label='$c_f$',linestyle=c_f_linestyle)
    ax.plot(x_vec,abs((hs_smooth_der.h(x_vec)-h_tab)/h_tab),color='k',label='$H$',linestyle=h_linestyle)
    
    plt.yscale('log')
    ax.legend()
    plt.grid(True)
    ax.set(xlabel='$x$(m)', ylabel='Relative Error')
    tikzplotlib.save(
        'figures/flatplate_mpg_smoothder_error.tex',
        axis_height = '\\figH',
        axis_width = '\\figW'
        )
    
    
    
    
    # #Figure 3a from NACA TM 1285
    # fig,ax = plt.subplots()
    
    # fig3a_smooth_y = np.tile(10,len(x_vec))+np.log10(hs_smooth.c_f(x_vec))
    # fig3a_smooth_x = np.log10(hs_smooth.rtheta(x_vec))
    
    # fig3a_smooth_der_y = np.tile(10,len(x_vec))+np.log10(hs_smooth_der.c_f(x_vec))
    # fig3a_smooth_der_x = np.log10(hs_smooth_der.rtheta(x_vec))
    
    # fig3a_tabx = np.log10(u_e_vec*theta_tab/nu)
    # fig3a_taby = np.tile(10,len(u_e_vec))+ np.log10(c_f_tab)
    # fig3a_taby_lt = np.tile(10,len(u_e_vec))+ np.log10(c_f_lt_tab)
    
    # ax.plot(fig3a_smooth_x,fig3a_smooth_y,label='smooth data')
    # ax.plot(fig3a_smooth_der_x,fig3a_smooth_der_y,label='smooth derivative')
    # ax.plot(fig3a_tabx,fig3a_taby,'o',label='Values from Tabulated Data')
    # ax.plot(fig3a_tabx,fig3a_taby_lt,'o',label='Values from Tabulated Data (CFLT)')
    # ax.set(xlabel='Log10 RTheta',ylabel='10+log10(c_f)',label = 'Vlues ')
    # ax.grid(True, linewidth=0.5,  color = '#000000', linestyle='-') #plt.grid(True, linewidth=0.5, color='#ff0000', linestyle='-')
    # ax.legend(loc='lower left')
    # #Return Theta Derivative
    # fig,ax = plt.subplots()
    # ax.plot(plotx,hs.yp(plotx)[:,0],label='Derivative from simulation')
    # ax.plot(plotx,hs_smooth.yp(plotx)[:,0],label='Derivative from smooth data y spline')
    # ax.plot(plotx,hs_smooth_der.yp(plotx)[:,0],label='Derivative from smooth derivative y spline')
    # approx_dthetadx_tab = (hs.y(plotx[:-1])[:,0]-hs.y(plotx[1:])[:,0])/(plotx[:-1]-plotx[1:])
    # #ax.plot(plotx[:-1]+.5*(plotx[1:]-plotx[:-1]),approx_dthetadx_tab,label='Derivative from simple approximation (tabular data)')
    # approx_dthetadx = (hs_smooth_der.y(plotx[:-1])[:,0]-hs_smooth_der.y(plotx[1:])[:,0])/(plotx[:-1]-plotx[1:])
    # ax.plot(plotx[:-1]+.5*(plotx[1:]-plotx[:-1]),approx_dthetadx,label='Derivative from simple approximation (smooth derivative)')
    # ax.legend(loc = 'lower right')
    # ax.set(xlabel='x(m)', ylabel='dtheta/dx')
    
    # #Figure 3a-b
    # fig,ax = plt.subplots()
    # smooth_c_f = hs_smooth.c_f(x_vec)
    # smooth_rtheta = hs_smooth.rtheta(x_vec)
    # smooth_der_c_f = hs_smooth_der.c_f(x_vec)
    # smooth_der_rtheta = hs_smooth_der.rtheta(x_vec)
    # ax.plot(smooth_rtheta,smooth_c_f,label ='smooth data')
    # ax.plot(smooth_der_rtheta,smooth_der_c_f,label = 'smooth derivative')
    # ax.plot(u_e_vec*theta_tab/nu,c_f_tab,label='tabulated cf')
    # ax.plot(u_e_vec*theta_tab/nu,c_f_lt_tab,label = 'tabulated cflt')
    # ax.legend(loc='lower left')
    # ax.set(xlabel='Rtheta',ylabel='Cf')
    
    
    
    #Return H derivative
    # fig,ax = plt.subplots()
    # fig.set_figheight(6)
    # fig.set_figwidth(10)
    # plt.rcParams["legend.edgecolor"] = "0" #sold legend edge
    # plt.rcParams["legend.framealpha"] = "1" #legend transparency - solid
    # plt.grid(True)
    # ax.plot(plotx,hs_smooth_der.yp(plotx)[:,1],label=r'$\frac{dH}{dx}$ from y spline')
    # approx_dthetadx = (hs_smooth_der.y(plotx[:-1])[:,1]-hs_smooth_der.y(plotx[1:])[:,1])/(plotx[:-1]-plotx[1:])
    # ax.plot(plotx[:-1]+.5*(plotx[1:]-plotx[:-1]),approx_dthetadx,label=r'$\frac{dH}{dx}$ from simple approximation')
    # ax.legend(loc = 'lower right')
    # ax.set(xlabel='x(m)')
    # ax.set_ylabel(r'$\frac{dH}{dx}$',rotation=0)
    # plt.title('Shape Factor Derivative')
    # tikzplotlib.save(
    #     'figures/h.tex',
    #     axis_height = '\\figH',
    #     axis_width = '\\figW'
    #     )
    
    fig,axs = plot_BL_params(theta=theta_tab,
                             c_f=c_f_tab,
                             delta=del_tab,
                             h=h_tab,
                             x=x_vec,
                             marker=data_marker,
                             label=data_label,)
    fig,axs = plot_BL_params(fig=fig,
                             axs=axs,
                             sim=hs,
                             label=spline_label,
                             linestyle=spline_linestyle,
                             x=plotx)
    fig,axs = plot_BL_params(fig=fig,
                             axs=axs,
                             sim=hs_smooth,
                             label=smooth_label,
                             linestyle=smooth_linestyle,
                             x=plotx)
    fig,axs = plot_BL_params(fig=fig,
                             axs=axs,
                             sim=hs_smooth_der,
                             label=der_label,
                             linestyle=der_linestyle,
                             x=plotx,
                             last=True,
                             file = 'flatplate_mapg')

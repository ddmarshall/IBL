# -*- coding: utf-8 -*-
"""
@author: blond
"""
import numpy as np
import matplotlib.pyplot as plt
from pyBL import HeadSimData, HeadSim #, TransitionModel
from scipy.interpolate import CubicSpline #for smoothed derivative experiment


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



#Simmulation using tabulated values
hsd = HeadSimData(x_vec,
                 u_e_vec,
                 u_inf,
                 nu,
                 x0,
                 theta0,
                 h0)
hs = HeadSim(hsd)
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
while hs_smooth_der.status=='running': # and hs_smooth.x_vec[-1]<3.4:
    hs_smooth_der.step()
    
plotx = np.linspace(x_vec[0],x_vec[-1])



#u_e comparison


fig,ax = plt.subplots()
#plotx = np.linspace(x_vec[0],x_vec[-1])
ax.plot(plotx, hs.u_e(plotx),label='u_e spline')
ax.plot(plotx, hs_smooth.u_e(plotx),label='u_e spline from smoothed u_e')
ax.plot(plotx, hs_smooth_der.u_e(plotx),label='u_e spline from smoothed du_e/dx')
ax.plot(x_vec,u_e_vec,'o',label='u_e tabulated points')
ax.plot(smooth_x,smooth_u_e,'o',label='u_e points from plot')
ax.set(xlabel='x(m)', ylabel='u_e')
ax.legend(loc='upper right', ncol=1)

#duedx comparison
fig,ax = plt.subplots()
plotx = np.linspace(x_vec[0],x_vec[-1])
ax.plot(plotx, hs.du_edx(plotx),label='du_e/dx spline from tabulated u_e data')
ax.plot(x_vec,du_edx_tab,'o',label = 'tabulated du_e/dx (DUI)' )
ax.plot(plotx, hs_smooth.du_edx(plotx),label='du_e/dx spline from smoothed u_e')
ax.plot(plotx, hs_smooth_der.du_edx(plotx),label='du_e/dx spline from smoothed du_e/dx')
ax.plot(smooth_x,smooth_du_edx,'o',label='du_e/dx points from plot')
ax.plot(x_vec[:-1],approx_du_edx,'o',label='simple derivative approximation')
ax.plot(smooth_x[:-1],approx_du_edx_smooth,'o',label='simple derivative approximation (from smooth data)')
ax.set(xlabel='x(m)', ylabel='du_e/dx',ylim=[2,-5.2])
ax.legend(loc='lower right', ncol=1)

#Theta Comparison
fig,ax = plt.subplots()
ax.plot(plotx, hs.theta(plotx),label='Simulation Theta (from tabulated data)')
ax.plot(plotx, hs_smooth.theta(plotx),label='Simulation Theta (from smoothed u_e data)')
ax.plot(plotx, hs_smooth_der.theta(plotx),label='Simulation Theta (from smoothed derivative)')
#theta_tab = np.array([.276,.413,.606,.811,1.074,1.276,1.432,1.614,1.773,2.005,2.246,2.528])/100
ax.plot(x_vec,theta_tab,'o',label='Tabulated Theta')
ax.set(xlabel='x(m)', ylabel='theta(m)')
ax.legend(loc='upper left', ncol=1)

#C_f Comparison
fig,ax = plt.subplots()
ax.plot(plotx, hs.c_f(plotx),label='Simulation c_f (from tabulated data)')
ax.plot(plotx, hs_smooth.c_f(plotx),label='Simulation c_f (from smoothed u_e data)')
ax.plot(plotx, hs_smooth_der.c_f(plotx),label='Simulation Theta (from smoothed derivative)')
#c_f_tab = np.array([.00285,.00249,.00221,.00205,.00180,.00168,.00162,.00150,.00141,.00133,.00124,.00117])
ax.plot(x_vec,c_f_tab,'o',label='Tabulated c_f')
#c_f_lt_tab = np.array([.00276,.00246,.00222,.00202,.00181,.00167,.00161,.00151,.00142,.00133,.00124,.00117])
ax.plot(x_vec,c_f_lt_tab,'o',label='Tabulated c_f LT')
ax.set(xlabel='x(m)', ylabel='c_f')
ax.legend(loc='upper right', ncol=1)

#H comparison
fig,ax = plt.subplots()
ax.plot(plotx, hs.h(plotx),label='Simulation H (from tabulated data)')
ax.plot(plotx, hs_smooth.h(plotx),label='Simulation H (from smoothed u_e data)')
ax.plot(plotx, hs_smooth_der.h(plotx),label='Simulation H (from smoothed derivative)')
#c_f_tab = np.array([.00285,.00249,.00221,.00205,.00180,.00168,.00162,.00150,.00141,.00133,.00124,.00117])
ax.plot(x_vec,h_tab,'o',label='Tabulated H')
#c_f_lt_tab = np.array([.00276,.00246,.00222,.00202,.00181,.00167,.00161,.00151,.00142,.00133,.00124,.00117])

ax.set(xlabel='x(m)', ylabel='Shape Factor H')
ax.legend(loc='lower left', ncol=1)

#displacement thickness comparison
fig,ax = plt.subplots()
ax.plot(plotx, hs.h(plotx)*hs.theta(plotx),label='Simulation del (from tabulated data)')
ax.plot(plotx, hs_smooth.h(plotx)*hs_smooth.theta(plotx),label='Simulation del (from smoothed u_e data)')
ax.plot(plotx, hs_smooth_der.h(plotx)*hs_smooth_der.theta(plotx),label='Simulation del (from smoothed derivative)')
#c_f_tab = np.array([.00285,.00249,.00221,.00205,.00180,.00168,.00162,.00150,.00141,.00133,.00124,.00117])
ax.plot(x_vec,del_tab,'o',label='Tabulated Displacement Thickness')
#c_f_lt_tab = np.array([.00276,.00246,.00222,.00202,.00181,.00167,.00161,.00151,.00142,.00133,.00124,.00117])

ax.set(xlabel='x(m)', ylabel='del (m)')
ax.legend(loc='upper left', ncol=1)


#Transpiratoin Velocity Comparison
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
ax.plot(plotx,transpiration_velocity_tab,label='Tabulated')
ax.plot(plotx,transpiration_velocity_smooth_u_e,label='Smooth u_e')
ax.plot(plotx,transpiration_velocity_smooth_der,label='Smooth du_e/dx')
ax.legend(loc = 'lower right')
ax.set(xlabel ='x(m)',ylabel='Transpiration U (m/s)')

#Figure 3a from NACA TM 1285
fig,ax = plt.subplots()

fig3a_smooth_y = np.tile(10,len(x_vec))+np.log10(hs_smooth.c_f(x_vec))
fig3a_smooth_x = np.log10(hs_smooth.rtheta(x_vec))

fig3a_smooth_der_y = np.tile(10,len(x_vec))+np.log10(hs_smooth_der.c_f(x_vec))
fig3a_smooth_der_x = np.log10(hs_smooth_der.rtheta(x_vec))

fig3a_tabx = np.log10(u_e_vec*theta_tab/nu)
fig3a_taby = np.tile(10,len(u_e_vec))+ np.log10(c_f_tab)
fig3a_taby_lt = np.tile(10,len(u_e_vec))+ np.log10(c_f_lt_tab)

ax.plot(fig3a_smooth_x,fig3a_smooth_y,label='smooth data')
ax.plot(fig3a_smooth_der_x,fig3a_smooth_der_y,label='smooth derivative')
ax.plot(fig3a_tabx,fig3a_taby,'o',label='Values from Tabulated Data')
ax.plot(fig3a_tabx,fig3a_taby_lt,'o',label='Values from Tabulated Data (CFLT)')
ax.set(xlabel='Log10 RTheta',ylabel='10+log10(c_f)',label = 'Vlues ')
ax.grid(True, linewidth=0.5,  color = '#000000', linestyle='-') #plt.grid(True, linewidth=0.5, color='#ff0000', linestyle='-')
ax.legend(loc='lower left')
#Return Theta Derivative
fig,ax = plt.subplots()
ax.plot(plotx,hs.yp(plotx)[:,0],label='Derivative from simulation')
ax.plot(plotx,hs_smooth.yp(plotx)[:,0],label='Derivative from smooth data y spline')
ax.plot(plotx,hs_smooth_der.yp(plotx)[:,0],label='Derivative from smooth derivative y spline')
approx_dthetadx_tab = (hs.y(plotx[:-1])[:,0]-hs.y(plotx[1:])[:,0])/(plotx[:-1]-plotx[1:])
#ax.plot(plotx[:-1]+.5*(plotx[1:]-plotx[:-1]),approx_dthetadx_tab,label='Derivative from simple approximation (tabular data)')
approx_dthetadx = (hs_smooth_der.y(plotx[:-1])[:,0]-hs_smooth_der.y(plotx[1:])[:,0])/(plotx[:-1]-plotx[1:])
ax.plot(plotx[:-1]+.5*(plotx[1:]-plotx[:-1]),approx_dthetadx,label='Derivative from simple approximation (smooth derivative)')
ax.legend(loc = 'lower right')
ax.set(xlabel='x(m)', ylabel='dtheta/dx')

#Figure 3a-b
fig,ax = plt.subplots()
smooth_c_f = hs_smooth.c_f(x_vec)
smooth_rtheta = hs_smooth.rtheta(x_vec)
smooth_der_c_f = hs_smooth_der.c_f(x_vec)
smooth_der_rtheta = hs_smooth_der.rtheta(x_vec)
ax.plot(smooth_rtheta,smooth_c_f,label ='smooth data')
ax.plot(smooth_der_rtheta,smooth_der_c_f,label = 'smooth derivative')
ax.plot(u_e_vec*theta_tab/nu,c_f_tab,label='tabulated cf')
ax.plot(u_e_vec*theta_tab/nu,c_f_lt_tab,label = 'tabulated cflt')
ax.legend(loc='lower left')
ax.set(xlabel='Rtheta',ylabel='Cf')



#Return H derivative
fig,ax = plt.subplots()
ax.plot(plotx,hs_smooth_der.yp(plotx)[:,1],label='Derivative from y spline')
approx_dthetadx = (hs_smooth_der.y(plotx[:-1])[:,1]-hs_smooth_der.y(plotx[1:])[:,1])/(plotx[:-1]-plotx[1:])
ax.plot(plotx[:-1]+.5*(plotx[1:]-plotx[:-1]),approx_dthetadx,label='Derivative from simple approximation')
ax.legend(loc = 'lower right')
ax.set(xlabel='x(m)', ylabel='dH/dx')

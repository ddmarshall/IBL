# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 00:26:30 2021

@author: blond
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45


from scipy.optimize import root #for finding optimal blasius eta

from pyBL import ThwaitesSimData, ThwaitesSim, HeadSimData, HeadSim, Michel

#Inviscid Flate Plate
u_inf = 6 #m/sec
x = np.linspace(0,10)
u_e = u_inf*np.ones([len(x)])
#u_e[0] = 0
nu = 1.48E-5
re = u_inf * x[-1] / nu

#Thwaites Simulation
tsd = ThwaitesSimData(x, u_e, u_inf, nu, re, char_length=x[-1])
ts = ThwaitesSim(tsd)
#sim_y_vec = np.array([ts._sim.y])
michel = Michel(ts)
while ts.status=='running':
#while ms.status=='running':
    #ts.step()
    ts.step()
    #print(michel.x_tr)
    #Check for Michel transition criteria
    #if bool(ts.rtheta(np.array([ts._sim.t]))>2.9*pow(ts.u_e(np.array([ts._sim.t]))*np.array([ts._sim.t])/nu,.4)):
        #break
#def michel0(xpt):
    #return float(ts.rtheta(np.array([xpt]))-2.9*pow(ts.u_e(np.array([xpt]))*np.array([xpt])/nu,.4))

x_tr = michel.x_tr #transition x point according to michel
#Test Transition Criteria - Michel
fig,ax = plt.subplots()
ax.plot(x,ts.rtheta(x),label='Momentum Thickness Re')
ax.plot(x,2.9*pow(ts.u_e(x)*x/nu,.4),label='Michel Condition (f(Rex))')
#ax.plot
ax.legend(loc='upper left', ncol=1)

#x_tr = root(michel0,ts._sim.t_old).x

# hsd = HeadSimData(x,
#                   u_e,
#                   u_inf,
#                   nu,
#                   5.6237742,
#                   0.00249848,
#                   2.61)
#h0 = 1.4754/np.log(ts.rtheta(ts.x_tr)) +.9698
#h0 = 1.4754/np.log(ts.rtheta(michel.x_tr)) +.9698
#h0 = michel.h0
hsd = HeadSimData(x,
                  u_e,
                  u_inf,
                  nu,
                  float(x_tr), #x0
                  float(ts.theta(michel.x_tr)),
                  michel.h0)

hs = HeadSim(hsd)
while hs.status=='running':
    hs.step()
    

x_lam = np.linspace(x[0],float(michel.x_tr))
x_turb = np.linspace(float(michel.x_tr),x[-1])
x_tot = np.append(x_lam,x_turb)
fig,ax = plt.subplots()
ax.plot(x_tot,np.append(ts.theta(x_lam),hs.theta(x_turb)),label='Momentum Thickness')

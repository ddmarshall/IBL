# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 00:26:30 2021

@author: blond
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import RK45


#from scipy.optimize import root #for finding optimal blasius eta

from pyBL import ThwaitesSimData, ThwaitesSim, HeadSimData, HeadSim, Michel

#Inviscid Flate Plate
u_inf = 6 #m/sec
x = np.linspace(0,10)
u_e = u_inf*np.ones([len(x)])
nu = 1.48E-5
re = u_inf * x[-1] / nu

#Thwaites Simulation
tsd = ThwaitesSimData(x, u_e, u_inf, nu, re, char_length=x[-1])
ts = ThwaitesSim(tsd)
michel = Michel(ts)
while ts.status=='running':
    ts.step()

x_tr = michel.x_tr #transition x point according to michel
#Test Transition Criteria - Michel
fig,ax = plt.subplots()
ax.plot(x,ts.rtheta(x),label='Momentum Thickness Re')
ax.plot(x,2.9*pow(ts.u_e(x)*x/nu,.4),label='Michel Condition (f(Rex))')
#ax.plot
ax.legend(loc='upper left', ncol=1)


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
plt.title('Momentum Thickness')
ax.plot(x_tot,np.append(ts.theta(x_lam),hs.theta(x_turb)),label='Momentum Thickness')
ax.legend(loc='upper left', ncol=1)

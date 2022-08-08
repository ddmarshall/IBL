#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:15:11 2022

@author: ddmarsha
"""

import numpy as np

from pyBL.pyBL_base import IBLSimData, IBLSim, SeparationModel

class HeadSimData(IBLSimData):
    def __init__(self,
                 x_vec,
                 u_e_vec,
                 u_inf,
                 nu,
                 x0,
                 theta0,
                 h0):
        super().__init__(x_vec,
                         u_e_vec,
                         u_inf,
                         nu)
        self.x0 = x0
        self.theta0 = theta0
        self.h0 = h0
        
class HeadSim(IBLSim):
    def __init__(self, head_sim_data):
        self.u_e = head_sim_data.u_e #f(x)
        self.du_edx = head_sim_data.du_edx #f(x)
        self.u_inf = head_sim_data.u_inf        
        self.nu = head_sim_data.nu 
        self.x0 = head_sim_data.x0
        self.theta0 = head_sim_data.theta0
        self.h0 = head_sim_data.h0
        self.x_bound = head_sim_data.x_vec[-1]
        
        def derivatives(t,y):
            x = t
            theta = y[0]
            h = y[1]
            
            u_e = self.u_e(x)
            du_edx = self.du_edx(x)
            rtheta = u_e*theta/self.nu
            cf = .246*pow(10,-.678*h)*pow(rtheta,-.268)
            dthetadx = cf/2 - (h+2)*theta*du_edx/u_e
            
            if h<=1.6:
                constants = [.8234,1.1,-1.287]
            else:
                constants = [1.5501,.6778,-3.064]
            dh1dh = constants[2]*constants[0]*pow(h-constants[1],constants[2]-1)
            h1 = constants[0]*pow(h-constants[1],constants[2])+3.3
            f1 = .0306*pow(h1-3,-.6169)
            dhdx = ((u_e*f1-theta*h1*du_edx-u_e*h1*dthetadx) / 
                    (u_e*theta*dh1dh))
            return np.array([dthetadx,dhdx])
        
                #Probably user changeable eventually
        #x0 = self.x0   
        self.y0 = np.array([self.theta0,self.h0])
        #x_bound = self.x_bound
        
        super().__init__(head_sim_data,derivatives, self.x0, self.y0, self.x_bound)
        
    def theta(self,x):
        return self.y(x)[:,0]
    def thetap(self,x):
        return self.yp(x)[:,0]
    def h(self,x):
        #shape factor
        return self.y(x)[:,1]
    def hp(self,x):
        return self.yp(x)[:,1]
    
    def rtheta(self,x):
        return self.u_e(x)*self.theta(x)/self.nu
    def c_f(self,x):
        return .246*pow(10,-.678*self.h(x))*pow(self.rtheta(x),-.268)
        re_theta = self.u_inf*self.theta(x)/self.nu
        return.246*pow(10,-.678*self.h(x))*pow(re_theta,-.268)
    def del_star(self,x):
        return self.h(x)*self.theta(x)
    def Un(self, x):
        return (self.du_edx(x)*self.h(x)*self.theta(x) + 
                self.u_e(x)*self.yp(x)[:,1]*self.theta(x) +
                self.u_e(x)*self.h(x)*self.yp(x)[:,0])

class HeadSeparation(SeparationModel):
    def __init__(self,headsim,buffer=0):
        h_crit = 2
        def h_difference(headsim,x=None):
            if type(x)!=np.ndarray and x ==None:
                x = headsim.x_vec
            if len(x.shape)==2:
                x = x[0]
            return headsim.h(x)-h_crit # @ 2.4, separation
        super().__init__(headsim,h_difference,buffer)

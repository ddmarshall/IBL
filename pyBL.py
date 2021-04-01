# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:33:36 2021

@author: blond
"""

import inspect    # used to return source code of h,s
#from scipy.integrate import cumtrapz
import numpy as np
#import warnings
from scipy.interpolate import CubicSpline
from scipy.integrate import RK45

class IBLSimData:
    def __init__(self,
                 x_vec,
                 u_e_vec,
                 u_inf,
                 nu):
        self.x_vec = x_vec
        self.u_e_vec = u_e_vec
        self.u_inf = u_inf
        self.nu = nu
        self._x_u_e_spline = CubicSpline(x_vec, u_e_vec)
        #self.derivatives = derivatives
        #self.profile = profile
    
    #x_vec and u_e_vec: need to add traces to update the spline
    x_vec = property(fget=lambda self: self._x_vec,
                   fset=lambda self, new_x_vec: setattr(self,
                                                      '_x_vec',
                                                      new_x_vec)) #; self._x_u_e_spline = CubicSpline(new_x_vec, 
                                                                                                   #self.u_e_vec))       
    u_e_vec = property(fget=lambda self: self._u_e,
                   fset=lambda self, new_u_e_vec: setattr(self,
                                                      '_u_e_vec',
                                                      new_u_e_vec)) #self._x_u_e_spline = CubicSpline(self.x_vec, 
                                                                                                     #new_u_e_vec))

    u_inf = property(fget=lambda self: self._u_inf,
                     fset=lambda self, new_u_inf: setattr(self,
                                                          '_u_inf',
                                                          new_u_inf))
    nu = property(fget=lambda self: self._nu,
                  fset=lambda self, new_nu: setattr(self,
                                                    '_nu',
                                                    new_nu))
    def u_e(self, x):
        return self._x_u_e_spline(x)
    def du_edx(self, x):
        return self._x_u_e_spline(x,1)
        
        
class IBLSim:
    def __init__(self,iblsimdata,derivatives,x0,y0,x_bound):
        self._sim = RK45(fun=derivatives,t0=x0, t_bound=x_bound, y0=y0 ) #y0=np.array([y0] t_bound = np.array([ x_bound])
        self._x_vec = np.array([self._sim.t])
        self._dense_output_vec = np.array([])
        #self._piecewise_ranges = np.array([lambda x: False])
        #self._piecewise_funs = np.array([])
        #self._status = self._sim.status
        self.u_e = iblsimdata.u_e #holds reference to u_e(x) from IBLSimData
        self.du_edx = iblsimdata.du_edx
        #self._x_u_e_spline = CubicSpline(iblsimdata.x_vec, iblsimdata.u_e_vec)
        
    x_vec = property(fget = lambda self: self._x_vec)
    status = property(fget = lambda self: self._sim.status)
    dense_output_vec = property(fget = lambda self: self._dense_output_vec)
    
    #dense_output_vec = property(fget  = lambda self : self._dense_output_vec)
    # def u_e(self,x):
    #     self._x_u_e_spline(x)
    # def du_edx(self,x):
    #     self._x_u_e_spline(x,1)        
        
    def step(self):
        self._sim.step()
        self._x_vec = np.append(self._x_vec, [self._sim.t])
        self._dense_output_vec = np.append(self.dense_output_vec,[self._sim.dense_output()])
        #self._piecewise_funs = np.append(self._piecewise_funs,[lambda x: self._sim.dense_output()(x)]) #was calling same function for every point
        
       
    def y(self,x):
        #returns m*n array, where m is len(x) and n is length(y)
        x_array = x #must be array
        #x_array = np.array([x])
        y_array = np.zeros([len(x),len(self._sim.y)])
        for i in range(len(x_array)):
            for j in range(len(self.dense_output_vec)): #-1
                if (x_array[i] >= self.x_vec[j]) & (x_array[i] <= self.x_vec[j+1]):
                    #y_array = np.append(y_array, [[self._piecewise_funs[j](x_array[i])]],axis=0)
                    #print(x_array[i])
                    #y_array[i,:] = self._piecewise_funs[j](x_array[i])
                    y_array[i,:] = self.dense_output_vec[j](x_array[i])
                    break
        return y_array
        #conds = np.array([[(x[i] >= self.x_vec[j]) & (x[i] <= self.x_vec[j+1]) for j in range(len(self.x_vec)-1)] for i in range(len(x))])
        
        #return np.array([[self._piecewise_funs[j](x[i]) for j in range(len(self.x_vec)-1) if (x[i] >= self.x_vec[j]) & (x[i] <= self.x_vec[j+1])] for i in range(len(x))])
        

# class IBLSimProfile:
#     def __init__(self,IBLSimData,t,y):
#         self
        
#Thwaites Default Functions

def _function_of_lambda_property_setter(f):
    try:
        sig = inspect.signature(f)
    except Exception as e:
        print('Must be a function of lambda.')
        print(e)
    else:
        if len(sig.parameters) != 1:
            raise Exception('Must take one argument, lambda.')
        else:
            return f


def _default_s(lam):
    if lam >= 0 and lam <= .1:
        return .22 + 1.57 * lam - 1.8 * pow(lam, 2)
    elif lam >= -.1 and lam <= 0:
        return .22 + 1.402 * lam + (.018 * lam) / (.107 + lam)
    else:
        return np.nan #pass  # I'll deal with this later


def _default_h(lam):
    if lam >= 0 and lam <= .1:
        return 2.61-3.75*lam+5.24*pow(lam, 2)
    elif lam >= -.1 and lam <= 0:
        return (.0731)/(.14+lam) + 2.088
    else:
        return np.nan #pass  # I'll deal with this later
        
        
class ThwaitesSimData(IBLSimData):
    def __init__(self,
                 x_vec,
                 u_e_vec,
                 u_inf,
                 nu,
                 re,
                 s=_default_s,
                 h=_default_h,
                 char_length=None):
        super().__init__(x_vec,
                         u_e_vec,
                         u_inf,
                         nu)
        self.re = re
        self.s = s
        self.h = h

        if char_length is None:
            self.char_length = 1
        else:
            self.char_length = char_length



    h = property(fget=lambda self: self._h,
                 fset=lambda self, f: setattr(self,
                                              '_h',
                                              _function_of_lambda_property_setter(f)))

    s = property(fget=lambda self: self._s,
                 fset=lambda self, f: setattr(self,
                                              '_s',
                                              _function_of_lambda_property_setter(f)))

    re = property(fget=lambda self: self._re,
                  fset=lambda self, new_re: setattr(self,
                                                    '_re',
                                                    new_re))

    char_length = property(fget=lambda self: self._char_length,
                           fset=lambda self, new_char_length: setattr(self,
                                                                      '_char_length',
                                                                      new_char_length))
    
    # def derivatives(self,t,y):
    #     x = t
    #     u_e = self.ue(x)
    #     return pow(u_e,5)  
    #def profile(self,y):
                                                                           

# class ThwaitesSim:
#     def __init__(self, thwaites_sim_data):

#         #u_e_star_series = pd.Series(thwaites_sim_data.u_e) / thwaites_sim_data.u_inf
#         #x_star_series = pd.Series(thwaites_sim_data.x) / thwaites_sim_data.char_length
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             integrand_vec = pow(thwaites_sim_data.u_e,5)
#             integral_vec = cumtrapz(integrand_vec, thwaites_sim_data.x, initial=0)
            
#             self._eq5_6_16_vec  = (.45 / pow(thwaites_sim_data.u_e, 6)) *integral_vec
#             self._theta_vec = pow(self._eq5_6_16_vec * thwaites_sim_data.nu, .5)
#             self._lam_vec = (pow(self._theta_vec, 2) *np.gradient(thwaites_sim_data.u_e, thwaites_sim_data.x) /thwaites_sim_data.nu)
#             self._h_vec = np.array([thwaites_sim_data.h(lam) for lam in self._lam_vec],dtype=np.float)
#             self._s_vec = np.array([thwaites_sim_data.s(lam) for lam in self._lam_vec],dtype=np.float)
            
#             self._cf_vec = (2 *
#                             thwaites_sim_data.nu *
#                             self._s_vec /
#                             (thwaites_sim_data.u_e *
#                             self._theta_vec))
#             self._del_star_vec = self._h_vec*self._theta_vec
#             self._wall_shear_vec = (thwaites_sim_data.nu * 
#                                     self._s_vec * 
#                                     pow(thwaites_sim_data.u_e / 
#                                         thwaites_sim_data.u_inf, 
#                                         2) / 
#                                     (thwaites_sim_data.u_e*self._theta_vec))
        
#     #eq5_6_16_vec = property(fget=lambda self: self._eq5_6_16_vec)
#     theta_vec = property(fget=lambda self: self._theta_vec)
#     lam_vec = property(fget=lambda self: self._lam_vec)
#     h_vec = property(fget=lambda self: self._h_vec)
#     s_vec = property(fget=lambda self: self._s_vec)
#     cf_vec = property(fget=lambda self: self._cf_vec)
#     del_star_vec = property(fget=lambda self: self._del_star_vec)
#     wall_shear_vec =  property(fget=lambda self: self._wall_shear_vec)
    
class ThwaitesSim(IBLSim):
    def __init__(self, thwaites_sim_data):
        
        
        self.u_e = thwaites_sim_data.u_e #f(x)
        self.u_inf = thwaites_sim_data.u_inf
        self.re = thwaites_sim_data.re
        self.s = thwaites_sim_data.s
        self.h = thwaites_sim_data.h
        self.nu = thwaites_sim_data.nu
        self.char_length = thwaites_sim_data.char_length
        def derivatives(t,y):
            x = t
            u_e = self.u_e(x)
            return np.array([pow(u_e,5)])
        
        #Probably user changeable eventually
        self.x0 = thwaites_sim_data.x_vec[0]    
        #self.y0 = np.array([5*pow(thwaites_sim_data.u_e(self.x0),4)])
        self.y0 = np.array([0])
        self.x_bound = thwaites_sim_data.x_vec[-1] 
        
        super().__init__(thwaites_sim_data, derivatives, self.x0, self.y0, self.x_bound)
        self.u_e = thwaites_sim_data.u_e
        


    
    def u_e_star(self,x):
        return self.u_e(x)/self.u_inf
        #u_e_star_series = pd.Series(thwaites_sim_data.u_e) / thwaites_sim_data.u_inf
    def x_star(self,x):
        return x/thwaites_sim_data.char_length
    
        #x_star_series = pd.Series(thwaites_sim_data.x) / thwaites_sim_data.char_length
        #with warnings.catch_warnings():
            #warnings.simplefilter("ignore")
        
       # integrand_vec = pow(thwaites_sim_data.u_e,5)
       # sim = RK45(fun=derivatives, t0=x0 , y0=5*,t_bound = self.x[-1],max_step=.1)
       # integral_vec = cumtrapz(integrand_vec, thwaites_sim_data.x, initial=0)
    def eq5_6_16(self,x):
        return (.45 / pow(self.u_e(x),6))*np.transpose(self.y(x))[0,:] #back down to (m,) array
      
    #     self._eq5_6_16_vec  = (.45 / pow(thwaites_sim_data.u_e, 6)) *integral_vec
    def theta(self,x):
        #momentum thickness
        return pow(self.eq5_6_16(x)*self.nu, .5)
    #     self._theta_vec = pow(self._eq5_6_16_vec * thwaites_sim_data.nu, .5)
    def lam(self,x):
        return (pow(self.theta(x), 2) *self.du_edx(x) /self.nu)
    #     self._lam_vec = (pow(self._theta_vec, 2) *np.gradient(thwaites_sim_data.u_e, thwaites_sim_data.x) /thwaites_sim_data.nu)
    
    #     self._h_vec = np.array([thwaites_sim_data.h(lam) for lam in self._lam_vec],dtype=np.float)
    #     self._s_vec = np.array([thwaites_sim_data.s(lam) for lam in self._lam_vec],dtype=np.float)
    def h_x(self,x):
        #h as a function of x (simulation completed)
        return np.array([self.h(lam) for lam in self.lam(x)])
        
    def s_x(self,x):
        #s as a function of x (simulation completed)
        return np.array([self.s(lam) for lam in self.lam(x)])
    def c_f(self,x)   :
        #skin friction
        #return 2 *self.nu*self.s(self.lam(x))/(self.u_e(x)*self.theta(x))
        return 2 *self.nu*self.s_x(x) / (self.u_e(x)*self.theta(x))
        #gives s a single value at a time
        #return 2 *self.nu*np.array([self.s(lam) for lam in self.lam(x)]) / (self.u_e(x)*self.theta(x))
    #     self._cf_vec = (2 *
    #                     thwaites_sim_data.nu *
    #                     self._s_vec /
    #                     (thwaites_sim_data.u_e *
    #                     self._theta_vec))
    #     self._del_star_vec = self._h_vec*self._theta_vec
    def del_star(self,x):
        return self.h_x(x)*self.theta(x)
        #return np.array([self.h(lam) for lam in self.lam(x)])*self.theta(x)
    #     self._wall_shear_vec = (thwaites_sim_data.nu * 
    #                             self._s_vec * 
    #                             pow(thwaites_sim_data.u_e / 
    #                                 thwaites_sim_data.u_inf, 
    #                                 2) / 
    #                             (thwaites_sim_data.u_e*self._theta_vec))
    def rtheta(self,x):
        return self.u_e(x)*self.theta(x)/self.nu    
    # #eq5_6_16_vec = property(fget=lambda self: self._eq5_6_16_vec)
    # theta_vec = property(fget=lambda self: self._theta_vec)
    # lam_vec = property(fget=lambda self: self._lam_vec)
    # h_vec = property(fget=lambda self: self._h_vec)
    # s_vec = property(fget=lambda self: self._s_vec)
    # cf_vec = property(fget=lambda self: self._cf_vec)
    # del_star_vec = property(fget=lambda self: self._del_star_vec)
    # wall_shear_vec =  property(fget=lambda self: self._wall_shear_vec)
    
    def michel(self,x):
        return np.array(self.rtheta(x)>2.9*pow(self.u_e(x)*x/self.nu,.4))
        #more efficient: np.where(x>x_tr)
    
    
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
            dgdh = constants[2]*constants[0]*pow(h-constants[1],constants[2]-1)
            h1 = constants[0]*pow(h-constants[1],constants[2])+3.3
            f = .0306*pow(h1-3,-.6169)
            dhdx = ((u_e*f-theta*h1*du_edx-u_e*h1*dthetadx) / 
                    (u_e*theta*dgdh))
            return np.array([dthetadx,dhdx])
        
                #Probably user changeable eventually
        #x0 = self.x0   
        self.y0 = np.array([self.theta0,self.h0])
        #x_bound = self.x_bound
        
        super().__init__(head_sim_data,derivatives, self.x0, self.y0, self.x_bound)
        
    def theta(self,x):
        return self.y(x)[:,0]
    def h(self,x):
        #shape factor
        return self.y(x)[:,1]
    def rtheta(self,x):
        return self.u_e(x)*self.theta(x)/self.nu
    def c_f(self,x):
        return .246*pow(10,-.678*self.h(x))*pow(self.rtheta(x),-.268)
        

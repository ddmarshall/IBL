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
from scipy.optimize import root #used for x_tr root finding in Thwaites (Michel)



# def input_decorator(func):
#     #this decorator will help all f(x) accept values cleanly - when it works
    
#     def inner(x=None):
#         if type(x)!=np.ndarray and  x==None:
#             return fun(sim.x_vec) 
#         if type(x) == np.ndarray:
#             return fun(x)
#         elif type(x) == list:
#             return fun(np.array(x))
#         elif type(x) == float or type(x) == int:
#             return fun(np.array([x]))
#     return inner


    
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
        #self._x_u_e_spline = CubicSpline(x_vec, u_e_vec,bc_type='natural') #replace line above for 'natural' bc's instead of 'not-a-knot'
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
        self._data = iblsimdata
        self._sim = RK45(fun=derivatives,t0=x0, t_bound=x_bound, y0=y0 ) #y0=np.array([y0] t_bound = np.array([ x_bound])
        self._x_vec = np.array([self._sim.t])
        self._dense_output_vec = np.array([])
        #self._piecewise_ranges = np.array([lambda x: False])
        #self._piecewise_funs = np.array([])
        #self._status = self._sim.status
        self.u_e = iblsimdata.u_e #holds reference to u_e(x) from IBLSimData
        self.du_edx = iblsimdata.du_edx
        #self._x_u_e_spline = CubicSpline(iblsimdata.x_vec, iblsimdata.u_e_vec)
    data = property(fget = lambda self: self._data) 
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
    
    def yp(self,x):
        x_array = x #must be array
        #x_array = np.array([x])
        yp_array = np.zeros([len(x),len(self._sim.y)])
        for i in range(len(x_array)):
            for j in range(len(self.dense_output_vec)): #-1
                if (x_array[i] >= self.x_vec[j]) & (x_array[i] <= self.x_vec[j+1]):
                    #y_array = np.append(y_array, [[self._piecewise_funs[j](x_array[i])]],axis=0)
                    #print(x_array[i])
                    #y_array[i,:] = self._piecewise_funs[j](x_array[i])
                    xdist = (x_array[i] - self.dense_output_vec[j].t_old) / self.dense_output_vec[j].h
                    if np.array(x_array[i]).ndim == 0:
                                #p = np.tile(x, testfit.order + 1)
                                p = np.tile(xdist, self.dense_output_vec[j].order + 1)
                                p = np.cumprod(p)/p
                    else:
                                p = np.tile(xdist, (self.dense_output_vec[j].order + 1, 1))
                                p = np.cumprod(p, axis=0)/p
                    #term1 = self.dense_output_vec[j].h h actually disappears
                    term2 = np.arange(1,self.dense_output_vec[j].order+2)
                    term3 = self.dense_output_vec[j].Q
                    term4 = p
                    #yp_array[i,:] = self.dense_output_vec[j].h * np.dot(np.arange(1,self.dense_output_vec[j].order+2)*self.dense_output_vec[j].Q, p)  
                    #yp_array[i,:] = term1 * np.dot(term2*term3, term4) 
                    yp_array[i,:] = np.dot(term2*term3, term4) 
                                        #yp_array[i,:] = self.dense_output_vec[j](x_array[i])
                    
                    break
        return yp_array        
        #conds = np.array([[(x[i] >= self.x_vec[j]) & (x[i] <= self.x_vec[j+1]) for j in range(len(self.x_vec)-1)] for i in range(len(x))])
        
        #return np.array([[self._piecewise_funs[j](x[i]) for j in range(len(self.x_vec)-1) if (x[i] >= self.x_vec[j]) & (x[i] <= self.x_vec[j+1])] for i in range(len(x))])


    #@staticmethod
    # def input_decorator(fun):
    #     #this decorator will help all f(x) accept values cleanly - when it works
    #     def returnfun(x=None):
    #         if x==None:
    #            return fun(self.x_vec) 
    #         if type(x) == np.ndarray:
    #             return fun(x)
    #         elif type(x) == list:
    #             return fun(np.array(x))
    #         elif type(x) == float or type(x) == int:
    #             return fun(np.array([x]))
    #     return returnfun

# class IBLSimProfile:
#     def __init__(self,IBLSimData,t,y):
#         self
class TransitionModel:
    def __init__(self,iblsim,criteria,h0calc):
        #iblsim: instance of a laminar ibl sim 
        #criteria: f(iblsim), returns difference from criteria at last point. Positive if transitioned.
        self._iblsim = iblsim
        
        self._criteria = lambda x=None: criteria(self.iblsim,x) #x is none by default
        self._h0calc = h0calc
        #self._transitioned = False
        self._x_tr = None
        self._h0 = None
    #criteria = property(fget = lambda self:self._criteria)
    # def criteria(self,x=None):
    #     return self._criteria(x)
    iblsim = property(fget = lambda self:self._iblsim)
    transitioned = property(fget = lambda self:self.x_tr!=None) #returns true if x_tr not none
    status = property(fget = lambda self: self.iblsim.status)
    
    # def step(self): ######Ultimately unnecessary, does nothing unique
    #     self.iblsim.step()
        
    @property
    def x_tr(self):
        if self._x_tr == None and np.any(self._criteria()>0):
            self._transitioned = True
            best_guess = np.argmax(self._criteria()>0)
            find_x_tr = root(lambda xpt:float(self._criteria(np.array([xpt]))),x0=best_guess)
            self._x_tr = find_x_tr.x
        return self._x_tr

    @property
    def h0(self):
        #checks whether x_tr is None ()
        if self.x_tr!=None and self._h0==None: #also checks if h0 has already been calculated
            self._h0 = self._h0calc(self.iblsim,self.x_tr)
            #self._h0= 1.4754/np.log(self.iblsim.rtheta(self.x_tr)) +.9698
        return self._h0
        
    #x_tr = property(fget = _get_x_tr)
    #h0 = property(fget = _get_h0)
                #     if self._x_tr==None and np.any(self.michel(self.x_vec)):
    #         best_guess = np.argmax(self.michel(self.x_vec)>0)
    #         find_x_tr = root(lambda xpt:float(self.michel_difference(np.array([xpt]))),x0=best_guess)
    #         return find_x_tr.x
            
#Thwaites Default Functions
class Michel(TransitionModel):
    def __init__(self,iblsim):
        def michel_difference(iblsim,x=None):
            #michel line for transition prediction
            #returns all points for x = None or no x provided
            if type(x)!=np.ndarray and x ==None:
                x = iblsim.x_vec
            return iblsim.rtheta(x) - 2.9*pow(iblsim.u_e(x)*x/iblsim.nu,.4)
        def h0calc(iblsim,x_tr):
            return 1.4754/np.log(iblsim.rtheta(x_tr)) +.9698
        super().__init__(iblsim,michel_difference,h0calc)
        
# class PMARCTransition(Transition):
#     def __init__(self,iblsim):
        
        
        
        
        
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
        
        self._x_tr = None 
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
    def c_f(self,x,q=None):
        #q - scalar
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
    # def michel_line(self,x):
    #     #michel line for transition prediction
    #     return 2.9*pow(self.u_e(x)*x/self.nu,.4)
    
    # def michel_difference(self,x):
    #     return self.rtheta(x)-self.michel_line(x)
    # #@IBLSim.input_decorator
    # def michel(self,x):
    #     #return np.array(self.rtheta(x)>self.michel_line(x))
    #     return self.michel_difference(x)>0
    #     #more efficient: x>x_tr
    # @property
    # def x_tr(self):
    #     if self._x_tr==None and np.any(self.michel(self.x_vec)):
    #         best_guess = np.argmax(self.michel(self.x_vec)>0)
    #         find_x_tr = root(lambda xpt:float(self.michel_difference(np.array([xpt]))),x0=best_guess)
    #         return find_x_tr.x
    
    
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
    #def c_fp(self,x):
        
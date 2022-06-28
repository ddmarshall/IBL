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
        #self._x_u_e_spline = CubicSpline(x_vec, u_e_vec,bc_type=((1,25),(2,0))) #hack
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
        self._sim = RK45(fun=derivatives,t0=x0, t_bound=x_bound, y0=y0) #y0=np.array([y0] t_bound = np.array([ x_bound])
       #######hack (following line)
        #self._sim = RK45(fun=derivatives,t0=x0, t_bound=x_bound, y0=np.array([pow(5E-4,2)*self._data.re*pow(iblsimdata.u_inf/x0,6)])) #y0=np.array([y0] t_bound = np.array([ x_bound])
        #self._sim = RK45(fun=derivatives,t0=x0, t_bound=x_bound, y0=np.array([.0005]))  #y0=np.array([y0] t_bound = np.array([ x_bound])
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
     
        
    def step(self):
        self._sim.step()
        self._x_vec = np.append(self._x_vec, [self._sim.t])
        self._dense_output_vec = np.append(self.dense_output_vec,[self._sim.dense_output()])
        if self._sim.status!='running':
            print(self._sim.status)
            self._x_vec = np.append(self._x_vec, self.data.x_vec[-1])
        #self._piecewise_funs = np.append(self._piecewise_funs,[lambda x: self._sim.dense_output()(x)]) #was calling same function for every point
        
       
    def y(self,x):
        #returns m*n array, where m is len(x) and n is length(y)
        x_array = x #must be array
        #x_array = np.array([x])
        y_array = np.zeros([len(x),len(self._sim.y)])
        for i in range(len(x_array)):
            for j in range(len(self.dense_output_vec)): #-1
                if (x_array[i] >= self.x_vec[j]) & (x_array[i] <= self.x_vec[j+1]):
                    y_array[i,:] = self.dense_output_vec[j](x_array[i])
                    break
                 
        return y_array
    
    def yp(self,x):
        #Uses Dense Output construct to return derivative with polynomial
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
                    yp_array[i,:] = np.dot(term2*term3, term4) 
                                        #yp_array[i,:] = self.dense_output_vec[j](x_array[i])
                    
                    break
        return yp_array        

class TransitionModel:
    def __init__(self,iblsim,criteria,h0calc,buffer):
        #iblsim: instance of a laminar ibl sim 
        #criteria: f(iblsim), returns difference from criteria at last point. Positive if transitioned.
        self._iblsim = iblsim
        
        self._criteria = lambda x=None: criteria(self.iblsim,x) #x is none by default
        self._h0calc = h0calc
        #self._transitioned = False
        self._x_tr = None
        self._buffer = buffer
        self._h0 = None
    iblsim = property(fget = lambda self:self._iblsim)
    transitioned = property(fget = lambda self:self.x_tr!=None) #returns true if x_tr not none

    @property
    def x_tr(self):
        if self._x_tr == None and np.any(self._criteria(self._iblsim._data.x_vec[self._iblsim._data.x_vec>self._buffer])>0):
            self._transitioned = True
            buffered_x = self._iblsim._data.x_vec[self._iblsim._data.x_vec>self._buffer]
            
            # crits = self._criteria(self._iblsim._data.x_vec)
            crits = self._criteria(buffered_x)
            #best_guess = np.argmax(self._criteria(self._iblsim._data.x_vec)>0)
            # best_guess = self._iblsim._data.x_vec[crits<0][-1] #last occurrence of 
            best_guess = buffered_x[crits>0][0] #furthest upstream occurrence of criteria met
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
        

class Michel(TransitionModel):
    def __init__(self,iblsim,buffer=0):
        def michel_difference(iblsim,x=None):
            #michel line for transition prediction
            #returns all points for x = None or no x provided
            if type(x)!=np.ndarray and x ==None:
                x = iblsim.x_vec
            return iblsim.rtheta(x) - 2.9*pow(iblsim.u_e(x)*x/iblsim.nu,.4)
        def h0calc(iblsim,x_tr):
            return 1.4754/np.log(iblsim.rtheta(x_tr)) +.9698
        super().__init__(iblsim,michel_difference,h0calc,buffer)
        
        
class SeparationModel:
    #when criteria are positive, has separated
    def __init__(self,iblsim,criteria,buffer):
        self._iblsim = iblsim
        self._criteria = lambda x=None: criteria(self._iblsim,x) #if x == None, return crit for all x
        self._x_sep = None
        self._buffer = buffer

    separated = property(fget = lambda self:self.x_sep!=None) #returns true if x_sep not none    
    @property
    def x_sep(self):
        if self._x_sep == None and np.any(self._criteria(self._iblsim._data.x_vec[self._iblsim._data.x_vec>self._buffer])>0):
            self._separated = True
            buffered_x = self._iblsim._data.x_vec[self._iblsim._data.x_vec>self._buffer]
            # crits = self._criteria(self._iblsim._data.x_vec)
            # crits = self._criteria(self._iblsim._data.x_vec[self._iblsim._data.x_vec>self._buffer])
            crits = self._criteria(buffered_x)
            #best_guess = np.argmax(self._criteria(self._iblsim._data.x_vec)>0)
            # best_guess = self._iblsim._data.x_vec[crits>0][0] #furthest upstream occurrence of criteria met
            best_guess = buffered_x[crits>0][0] #furthest upstream occurrence of criteria met
            
            find_x_sep = root(lambda xpt:float(self._criteria(np.array([xpt]))),x0=best_guess)
            self._x_sep = find_x_sep.x
        return self._x_sep

class ThwaitesSeparation(SeparationModel):
    def __init__(self,thwaitessim,buffer=0):
        def lambda_difference(thwaitessim,x=None):
            if type(x)!=np.ndarray and x ==None:
                x = thwaitessim.x_vec
            return -thwaitessim.lam(x)-.0842 # @ -.0842, separation
        super().__init__(thwaitessim,lambda_difference,buffer)

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

#bringing in thwaites' tabulated values
f_tab = np.array([.938,.953,.956,.962,.967,.969,.971,.970,.963,.952,.936,.919,.902,.886,.854,.825,.797,.770,.744,.691,.640,.590,.539,.490,.440,.342,.249,.156,.064,-.028,-.138,-.251,-.362,-.702,-1])
m_tab = np.array([.082,.0818,.0816,.0812,.0808,.0804,.08,.079,.078,.076,.074,.072,.07,.068,.064,.06,.056,.052,.048,.04,.032,.024,.016,.008,0,-0.016,-.032,-.048,-.064,-.08,-.1,-.12,-.14,-.2,-.25])
s_tab = np.array([0,.011,.016,.024,.03,.035,.039,.049,.055,.067,.076,.083,.089,.094,.104,.113,.122,.13,.138,.153,.168,.182,.195,.208,.22,.244,.268,.291,.313,.333,.359,.382,.404,.463,.5])
h_tab = np.array([3.7,3.69,3.66,3.63,3.61,3.59,3.58,3.52,3.47,3.38,3.3,3.23,3.17,3.13,3.05,2.99,2.94,2.9,2.87,2.81,2.75,2.71,2.67,2.64,2.61,2.55,2.49,2.44,2.39,2.34,2.28,2.23,2.18,2.07,2])

lam_tab = -m_tab

s_lam_spline = CubicSpline(lam_tab, s_tab)
h_lam_spline = CubicSpline(lam_tab, h_tab)

#redundant (f can be calculated from other values):
f_lam_spline = CubicSpline(lam_tab,f_tab)

def spline_h(lam):
    return h_lam_spline(lam)

def spline_s(lam):
    return s_lam_spline(lam)

def cebeci_s(lam):
    try:
        lam = min([max([lam,-.1]),.1]) #gives back lambda at endpoints of fit (if outside fit)
    except:
        pass
    if lam >= 0 and lam <= .1:
        return .22 + 1.57 * lam - 1.8 * pow(lam, 2)
    elif lam >= -.1 and lam <= 0:
        return .22 + 1.402 * lam + (.018 * lam) / (.107 + lam)
    else:
        return np.nan #pass  # I'll deal with this later


def cebeci_h(lam):
    try:
        
        lam = min([max([lam,-.1]),.1]) #gives back lambda at endpoints of fit (if outside fit)
    except:
        pass
    if lam >= 0 and lam <= .1:
        return 2.61-3.75*lam+5.24*pow(lam, 2)
    elif lam >= -.1 and lam <= 0:
        return (.0731)/(.14+lam) + 2.088
    else:
        return np.nan #pass  # Returned if lambda fails > or <

def white_s(lam):
    return pow(lam+.09,.62)
    
def white_h(lam):
    z = .25-lam
    return 2+4.14*z-83.5*pow(z,2) +854*pow(z,3) -3337*pow(z,4) +4576*pow(z,5)

def _stagnation_y0(iblsimdata,x0):
    #From Moran
      return .075*iblsimdata.nu/iblsimdata.du_edx(x0)

       
class ThwaitesSimData(IBLSimData):
    def __init__(self,
                 x_vec,
                 u_e_vec,
                 u_inf,
                 nu,
                 re,
                 x0,
                 theta0=None,
                 s=spline_s,
                 h=spline_h,
                 linearize=False):
        super().__init__(x_vec,
                         u_e_vec,
                         u_inf,
                         nu)
        self.x0 = x0
        self.theta0 = theta0
        self.re = re
        #these go through the setters
        self.s_lam = s
        self.h_lam = h
        self._linearize=linearize



    h_lam = property(fget=lambda self: self._h,
                 fset=lambda self, f: setattr(self,
                                              '_h',
                                              _function_of_lambda_property_setter(f)))

    s_lam = property(fget=lambda self: self._s,
                 fset=lambda self, f: setattr(self,
                                              '_s',
                                              _function_of_lambda_property_setter(f)))

    re = property(fget=lambda self: self._re,
                  fset=lambda self, new_re: setattr(self,
                                                    '_re',
                                                    new_re))

                                                                           
  
class ThwaitesSim(IBLSim):
    def __init__(self, thwaites_sim_data):
        #note - f's(lambda) aren't actually used in solver
        self.u_e = thwaites_sim_data.u_e #f(x)
        self.u_inf = thwaites_sim_data.u_inf
        self.re = thwaites_sim_data.re
        self.x0 = thwaites_sim_data.x0
        self.theta0 = thwaites_sim_data.theta0
        self.s_lam = thwaites_sim_data.s_lam
        self.h_lam = thwaites_sim_data.h_lam
        self.nu = thwaites_sim_data.nu
#        self.char_length = thwaites_sim_data.char_length
        
        self._x_tr = None 
        def derivatives(t,y):
            #modified derivatives to use s and h, define y as theta^2
            x=t
            lam = y*thwaites_sim_data.du_edx(x)/self.nu
            # if (lam<= (-0.0842)):
            #     lam =np.array([(-0.0842)])
            if abs(self.u_e(x))<np.array([1E-8]):
                return np.array([1E-8])
            else:
                #Check whether to assume .45-6lam for 2(s-(2+H)*lam)
                if thwaites_sim_data._linearize==True:
                    return np.array([self.nu*(.45-6*lam)/self.u_e(x)])
                else:
                    return np.array([2*self.nu*(self.s_lam(lam)-(2+self.h_lam(lam))*lam)/self.u_e(x)])

        #Probably user changeable eventually
        #self.x0 = thwaites_sim_data.x_vec[0]
        #self.x0=x0
        #self.y0 = np.array([5*pow(thwaites_sim_data.u_e(self.x0),4)])
        if self.theta0 is not None:
            self.y0 = np.array([pow(self.theta0,2)])
        else:
            self.y0 = [_stagnation_y0(thwaites_sim_data,self.x0)]
        self.x_bound = thwaites_sim_data.x_vec[-1] 
        
        super().__init__(thwaites_sim_data, derivatives, self.x0, self.y0, self.x_bound)
        self.u_e = thwaites_sim_data.u_e

    

    
    #def u_e_star(self,x):
        #return self.u_e(x)/self.u_inf
        #u_e_star_series = pd.Series(thwaites_sim_data.u_e) / thwaites_sim_data.u_inf
    #def x_star(self,x):
        #return x/thwaites_sim_data.char_length
    
        #x_star_series = pd.Series(thwaites_sim_data.x) / thwaites_sim_data.char_length
        #with warnings.catch_warnings():
            #warnings.simplefilter("ignore")
        
       # integrand_vec = pow(thwaites_sim_data.u_e,5)
       # sim = RK45(fun=derivatives, t0=x0 , y0=5*,t_bound = self.x[-1],max_step=.1)
       # integral_vec = cumtrapz(integrand_vec, thwaites_sim_data.x, initial=0)
    #def eq5_6_16(self,x):
        #return np.nan_to_num((.45 / pow(self.u_e(x),6))*np.transpose(self.y(x))[0,:]) #back down to (m,) array
      
    #     self._eq5_6_16_vec  = (.45 / pow(thwaites_sim_data.u_e, 6)) *integral_vec
    def theta(self,x):
        #momentum thickness
        #return pow(self.eq5_6_16(x)*self.nu, .5)
        return np.sqrt(np.transpose(self.y(x))[0,:])
    #     self._theta_vec = pow(self._eq5_6_16_vec * thwaites_sim_data.nu, .5)
    def lam(self,x):
        return (np.transpose(self.y(x))[0,:] *self.du_edx(x) /self.nu)
    #     self._lam_vec = (pow(self._theta_vec, 2) *np.gradient(thwaites_sim_data.u_e, thwaites_sim_data.x) /thwaites_sim_data.nu)
    
    #     self._h_vec = np.array([thwaites_sim_data.h(lam) for lam in self._lam_vec],dtype=np.float)
    #     self._s_vec = np.array([thwaites_sim_data.s(lam) for lam in self._lam_vec],dtype=np.float)
    def h(self,x):
        #h function (not shape factor) as a function of x (simulation completed)
        return np.array([self.h_lam(lam) for lam in self.lam(x)])
        
    def s(self,x):
        #s as a function of x (simulation completed)
        return np.array([self.s_lam(lam) for lam in self.lam(x)])
    def c_f(self,x):
        #q - scalar
        #skin friction
        #return 2 *self.nu*self.s(self.lam(x))/(self.u_e(x)*self.theta(x))
        return 2 *self.nu*self.s(x) / (self.u_e(x)*self.theta(x))
        #gives s a single value at a time
        #return 2 *self.nu*np.array([self.s(lam) for lam in self.lam(x)]) / (self.u_e(x)*self.theta(x))
    #     self._cf_vec = (2 *
    #                     thwaites_sim_data.nu *
    #                     self._s_vec /
    #                     (thwaites_sim_data.u_e *
    #                     self._theta_vec))
    #     self._del_star_vec = self._h_vec*self._theta_vec
    def del_star(self,x):
        return self.h(x)*self.theta(x)
        #return np.array([self.h(lam) for lam in self.lam(x)])*self.theta(x)
    #     self._wall_shear_vec = (thwaites_sim_data.nu * 
    #                             self._s_vec * 
    #                             pow(thwaites_sim_data.u_e / 
    #                                 thwaites_sim_data.u_inf, 
    #                                 2) / 
    #                             (thwaites_sim_data.u_e*self._theta_vec))
    def rtheta(self,x):
        return self.u_e(x)*self.theta(x)/self.nu  

    
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
    #def c_fp(self,x):
        
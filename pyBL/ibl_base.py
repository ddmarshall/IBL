#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:17:53 2022

@author: ddmarsha
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import RK45
from scipy.optimize import root
from scipy.misc import derivative as fd
from abc import ABC, abstractmethod

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
        return self._x_u_e_spline(x, 1)
    def d2u_edx2(self, x):
        return self._x_u_e_spline(x, 2)
        
        
class IBLBase(ABC):
    """
    The base class for integral boundary layer classes.
    
    This encapsulates the common features and needed parameters for all IBL 
    methods. At the very least it provides the inteface that is expected for all
    IBL classes.
    
    Attributes
    ----------
        _U_e: Function representing the edge velocity profile
        _dU_edx: Function representing the streamwise derivative of the edge 
                 velocity
        _d2U_edx2: Function representing the streamwise second derivative of the
                   edge velocity
        _ode: Common ODE solver to be used by all IBL methods. Currently is an
              dense output RK45 from scipy.
        _x_vec: Vector 
    
    Raises
    ------
        ValueError: if configuration parameter is invalid (see message)
    """
    def __init__(self, yp, x0, y0, x_end, int_rtol=1e-8, int_atol=1e-11,
                 U_e = None, dU_edx=None, d2U_edx2=None):
#        self._data = iblsimdata
        self._ode = RK45(fun=yp, t0=x0, t_bound=x_end, y0=y0,
                         rtol=int_rtol, atol=int_atol)
        
        ## set the velocity terms
        if U_e is None:
            if (dU_edx is not None):
                raise ValueError("Must specify U_e if specifying dU_edx")
            if (d2U_edx2 is not None):
                raise ValueError("Must specify U_e if specifying d2U_edx2")
            self._U_e = None
            self._dU_edx = None
            self._d2U_edx2 = None
        else:
            self.setVelocity(U_e, dU_edx, d2U_edx2)
        
        self._x_vec = np.array([self._ode.t])
        self._dense_output_vec = np.array([])

    def setVelocity(self, U_e, dU_edx = None, d2U_edx2 = None):
        """
        Set the edge velocity relations.
        
        A number of different ways can be used to set the velocity relation and
        its derivatives.
        * U_e can be a 2-tuple of xpoints and velocity values. In this case a
          cubic spline will be created and the derivative functionss will be
          taken from the cubic spline.
        * U_e and the derivatives can be callable objects.
            * If the first derivative object is provided but not the second 
              derivative object, then if the first derivative object has a
              method called 'derivative' then that method will be used to
              generate the second derivative object. Otherwise the second
              derivative will be approximated by finite differences of the first
              derivative.
            * If neither derivative objects are provided, then if U_e has a
              method called 'derivative' then that method will be used to
              generate both derivative objects. Otherwise the derivative
              objects will be created from finite difference approximations.
        """
        # check if U_e is callable
        if callable(U_e):
            self._U_e = U_e
            
            # if dU_edx not provided then use finite differences
            if dU_edx is None:
                if d2U_edx2 is not None:
                    raise ValueError("Can only pass second derivative if first "
                                     "derivative was specified")
                
                # if U_e has derivative method then use it
                if (hasattr(U_e, 'derivative') and
                        callable(getattr(U_e, 'derivative'))):
                    self._dU_edx = U_e.derivative()
                    self._d2U_edx2 = U_e.derivative(2)
                else:
                    self._dU_edx = lambda x: fd(self._U_e, x, 1e-4,
                                                n=1, order=3)
                    self._d2U_edx2 = lambda x: fd(self._U_e, x, 1e-4,
                                                  n=2, order=3)
            else:
                if not callable(dU_edx):
                    raise ValueError("Must pass in callable object for first "
                                     "derivative if callable U_e given")
                self._dU_edx = dU_edx
                
                # if d2U_edx2 not provied then use finite difference
                if d2U_edx2 is None:
                    # if dU_edx has derivative method then use it
                    if (hasattr(dU_edx, 'derivative') and
                            callable(getattr(dU_edx, 'derivative'))):
                        self._d2U_edx2 = dU_edx.derivative()
                    else:
                        self._d2U_edx2 = lambda x: fd(self._dU_edx, x, 1e-5,
                                                      n=1, order=3)
                else:
                    if not callable(dU_edx):
                        raise ValueError("Must pass in callable object for "
                                         "first derivative if callable U_e "
                                         "given")
                    
                    self._d2U_edx2 = d2U_edx2
        else:
            # if is 2-tuple the assume x, U_e pairs to build Cubic Spline
            if len(U_e) == 2:
                x_pts = np.asarray(U_e[0])
                U_e_pts = np.asarray(U_e[1])
                npts = x_pts.shape[0]
                # check to make sure have two vectors of same length suitable
                #   for building splines
                if (x_pts.ndim != 1):
                    raise ValueError("First element of U_e 2-tuple must be 1D "
                                     "vector of distances")
                if (U_e_pts.ndim != 1):
                    raise ValueError("Second element of U_e 2-tuple must be 1D "
                                     "vector of Velocities")
                if npts != U_e_pts.shape[0]:
                    raise ValueError("Vectors in U_e 2-tuple must be of same "
                                     "length")
                if npts < 2:
                    raise ValueError("Must pass at least two points for edge "
                                     "velocity")
                
                U_e_spline = CubicSpline(x_pts, U_e_pts)
                self.setVelocity(U_e_spline)
            else:
                # otherwise unknown velocity input
                raise ValueError("Don't know how to use {} to initialize "
                                 "velocity".format(U_e))
    
    def U_e(self, x):
        """
        Return the inviscid edge velocity at specified location(s)
        
        Args
        ----
            x: distance along surface
            
        Returns
        -------
            Inviscid edge velocity
        """
        if (self._U_e is not None):
            return self._U_e(x)
        else:
            raise ValueError("U_e was not set")
            return None
    
    def dU_edx(self, x):
        """
        Return the streamwise derivative of the inviscid edge velocity at 
        specified location(s)
        
        Args
        ----
            x: distance along surface
            
        Returns
        -------
            Derivative of inviscid edge velocity
        """
        if (self._dU_edx is not None):
            return self._dU_edx(x)
        else:
            raise ValueError("dU_edx was not set")
            return None
    
    def d2U_edx2(self, x):
        """
        Return the streamwise second derivative of the inviscid edge velocity at 
        specified location(s)
        
        Args
        ----
            x: distance along surface
            
        Returns
        -------
            Second derivative of inviscid edge velocity
        """
        if (self._d2U_edx2 is not None):
            return self._d2U_edx2(x)
        else:
            raise ValueError("d2U_edx2 was not set")
            return None
    
    #TODO: These need to be (re)moved
    data = property(fget = lambda self: self._data) 
    x_vec = property(fget = lambda self: self._x_vec)
    status = property(fget = lambda self: self._ode.status)
    dense_output_vec = property(fget = lambda self: self._dense_output_vec)
     
    def step(self):
        self._ode.step()
        self._x_vec = np.append(self._x_vec, [self._ode.t])
        self._dense_output_vec = np.append(self.dense_output_vec,[self._ode.dense_output()])
        if self._ode.status!='running':
            print(self._ode.status)
            self._x_vec = np.append(self._x_vec, self.data.x_vec[-1])
       
    def y(self,x):
        #returns m*n array, where m is len(x) and n is length(y)
        x_array = x #must be array
        #x_array = np.array([x])
        y_array = np.zeros([len(x),len(self._ode.y)])
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
        yp_array = np.zeros([len(x),len(self._ode.y)])
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
                                # TODO: This produces error when xdist=0 because p becomes vector of zeros
                                #       See issue #21
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
    
    @abstractmethod
    def U_n(self, x):
        pass
    
    @abstractmethod
    def delta_d(self, x):
        pass
    
    @abstractmethod
    def delta_m(self, x):
        pass
    
    @abstractmethod
    def H(self, x):
        pass
    
    @abstractmethod
    def tau_w(self, x):
        pass

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


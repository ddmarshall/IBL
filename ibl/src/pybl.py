# pylint: skip-file

# import numpy as np


# class SeparationModel:
#     #when criteria are positive, has separated
#     def __init__(self,iblsim,criteria,buffer):
#         self._iblsim = iblsim
#         self._criteria = lambda x=None: criteria(self._iblsim,x) #if x == None, return crit for all x
#         self._x_sep = None
#         self._buffer = buffer

#     separated = property(fget = lambda self:self.x_sep!=None) #returns true if x_sep not none
#     @property
#     def x_sep(self):
#         if self._x_sep == None and np.any(self._criteria(self._iblsim._data.x_vec[self._iblsim._data.x_vec>self._buffer])>0):
#             self._separated = True
#             buffered_x = self._iblsim._data.x_vec[self._iblsim._data.x_vec>self._buffer]
#             # crits = self._criteria(self._iblsim._data.x_vec)
#             # crits = self._criteria(self._iblsim._data.x_vec[self._iblsim._data.x_vec>self._buffer])
#             crits = self._criteria(buffered_x)
#             #best_guess = np.argmax(self._criteria(self._iblsim._data.x_vec)>0)
#             # best_guess = self._iblsim._data.x_vec[crits>0][0] #furthest upstream occurrence of criteria met
#             best_guess = buffered_x[crits>0][0] #furthest upstream occurrence of criteria met

#             find_x_sep = root(lambda xpt:float(self._criteria(np.array([xpt]))),x0=best_guess)
#             self._x_sep = find_x_sep.x
#         return self._x_sep

# class TransitionModel:
#     def __init__(self,iblsim,criteria,h0calc,buffer):
#         #iblsim: instance of a laminar ibl sim
#         #criteria: f(iblsim), returns difference from criteria at last point. Positive if transitioned.
#         self._iblsim = iblsim

#         self._criteria = lambda x=None: criteria(self.iblsim,x) #x is none by default
#         self._h0calc = h0calc
#         #self._transitioned = False
#         self._x_tr = None
#         self._buffer = buffer
#         self._h0 = None
#     iblsim = property(fget = lambda self:self._iblsim)
#     transitioned = property(fget = lambda self:self.x_tr!=None) #returns true if x_tr not none

#     @property
#     def x_tr(self):
#         if self._x_tr == None and np.any(self._criteria(self._iblsim._data.x_vec[self._iblsim._data.x_vec>self._buffer])>0):
#             self._transitioned = True
#             buffered_x = self._iblsim._data.x_vec[self._iblsim._data.x_vec>self._buffer]

#             # crits = self._criteria(self._iblsim._data.x_vec)
#             crits = self._criteria(buffered_x)
#             #best_guess = np.argmax(self._criteria(self._iblsim._data.x_vec)>0)
#             # best_guess = self._iblsim._data.x_vec[crits<0][-1] #last occurrence of
#             best_guess = buffered_x[crits>0][0] #furthest upstream occurrence of criteria met
#             find_x_tr = root(lambda xpt:float(self._criteria(np.array([xpt]))),x0=best_guess)
#             self._x_tr = find_x_tr.x
#         return self._x_tr

#     @property
#     def h0(self):
#         #checks whether x_tr is None ()
#         if self.x_tr!=None and self._h0==None: #also checks if h0 has already been calculated
#             self._h0 = self._h0calc(self.iblsim,self.x_tr)
#             #self._h0= 1.4754/np.log(self.iblsim.rtheta(self.x_tr)) +.9698
#         return self._h0


# class Michel(TransitionModel):
#     def __init__(self,iblsim,buffer=0):
#         def michel_difference(iblsim,x=None):
#             #michel line for transition prediction
#             #returns all points for x = None or no x provided
#             if type(x)!=np.ndarray and x ==None:
#                 x = iblsim.x_vec
#             return iblsim.rtheta(x) - 2.9*pow(iblsim.u_e(x)*x/iblsim.nu,.4)
#         def h0calc(iblsim,x_tr):
#             return 1.4754/np.log(iblsim.rtheta(x_tr)) +.9698
#         super().__init__(iblsim,michel_difference,h0calc,buffer)


# class Forced(TransitionModel):
#     def __init__(self,iblsim, x_tr, buffer=0):
#         self._x_forced = x_tr
#         def forced_difference(iblsim,x=None):
#             #returns all points for x = None or no x provided
#             if type(x)!=np.ndarray and x ==None:
#                 x = iblsim.x_vec
#             return x - self._x_forced
#         def h0calc(iblsim,x_tr):
#             return 1.4754/np.log(iblsim.rtheta(x_tr)) +.9698
#         super().__init__(iblsim,forced_difference,h0calc,buffer)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 15:59:40 2022.

@author: ddmarshall
"""

# import numpy as np
# from scipy.interpolate import PPoly


# class RKDenseOutputPPoly:
#     """
#     The collection of dense output interpolating polynomials from the RK ODE
#     solvers from scipy.integrate.

#     Attributes
#     ----------
#     _pp: Piecewise polynomial representation of the collection of dense
#          output
#     _neqs: Number of equations represented in the dense output
#     """
#     def __init__(self, rk_dense_out):
#         coef = self._convert_q_to_c(rk_dense_out.Q, rk_dense_out.h,
#                                     rk_dense_out.y_old)
#         xrange = [rk_dense_out.t_old, rk_dense_out.t]

#         # cycle through each equation to create ppoly
#         self._pp = list()
#         self._neqs = coef.shape[0]
#         for i in range(self._neqs):
#             self._pp.append(PPoly(np.asarray([coef[i, :]]).T, xrange,
#                             extrapolate=False))

#     def extend(self, rk_dense_out):
#         """
#         Extend the piecewise polynomial to include new dense output range

#         Args
#         ----
#             rk_dense_out: Dense output from RK solver

#         Returns
#         -------
#             None
#         """
#         coef = self._convert_q_to_c(rk_dense_out.Q, rk_dense_out.h,
#                                     rk_dense_out.y_old)
#         xend = [rk_dense_out.t]
#         for i, pp in enumerate(self._pp):
#             pp.extend(np.asarray([coef[i, :]]).T, xend)

#     def __call__(self, t):
#         """
#         Evaluate the piecewise interpolant

#         Args
#         ----
#             t: Values at which the interpolants should be evaluted

#         Returns
#         -------
#             Piecewise interpolant evaluated
#         """
#         t = np.asarray(t)
#         if t.ndim > 1:
#             raise ValueError("`t` must be a float or a 1-D array.")

#         if self._neqs == 1:
#             y = self._pp[0](t).T
#         else:
#             if t.ndim == 0:
#                 nrow = 1
#             else:
#                 nrow = t.shape[0]
#             y = np.empty([nrow, self._neqs])

#             for j, pp in enumerate(self._pp):
#                 y[:, j] = pp(t).T

#         return y

#     @staticmethod
#     def _convert_q_to_c(Q, h, y0):
#         """
#         Internal function that converts the Q coefficients from ODE dense
#         output to the coefficients needed for a piecewise polynomial segment.

#         Args
#         ----
#             Q: ODE dense output coefficient vector
#             h: Step size of current ODE dense output
#             y0: Starting value of the function modeled by the dense output

#         Returns
#         -------
#             Equivalent coefficients for piecewise polynomial

#         Notes
#         -----
#         * Q vector of cefficients is polynomial coefficients from x^1 to
#           x^{order+1}
#         * The constant term of polynomial is stored in separate term, y0
#         * The dense output polynomial independent variable, x, always goes
#           from zero to one. This is different than piecewise polynomial by
#           a factor of h^{n-1} for each Q_n term.
#         """
#         ydim, pdim = Q.shape
#         pdim=pdim+1
#         coef=np.zeros([ydim, pdim])
#         p = np.tile(h, pdim - 1)
#         p = np.cumprod(p)/h
#         coef[:, 0:-1]=(Q/p)[:, ::-1]
#         coef[:, -1] = y0
#         return coef

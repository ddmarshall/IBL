#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical solution to Blasius equation.

This module calculates the solution to the laminar, incompressible flow over
a flat plate with no pressure gradient. This solution is known as the Blasius
solution. After the differential equation is solved, a number of properties
can be obtained about the flow in similarity coordinates as well as in
Cartesian coordinates.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quadrature
from scipy.optimize import root_scalar


class BlasiusSolution:
    """
    Solution to Blasius equation.

    This class represents the solution to the Blasius equation. While it can
    be initialized with user defined parameters needed for the solution, the
    default parameters are sufficient to obain an accurate solution.

    Once the solution is obtained, the dense output from the ODE integrator is
    used to report back a wide variety of parameters associated with the
    boundary layer. Both integrated and point properties can be obtained from
    the similarity coordinate or from the corresponding Cartesian coordinates.
    """

    def __init__(self, U_ref, nu, fpp0=0.46959998713136886, eta_inf=10):
        self._U_ref = U_ref
        self._nu = nu
        self._eta_inf = eta_inf
        self._F = None

        self._set_boundary_condition(fpp0)

    def f(self, eta):
        """
        Return the non-dimensional stream function from the solution.

        Parameters
        ----------
        eta : array-like
            Similarity coordinates to calculate the property.

        Returns
        -------
        array-like same shape as `eta`
            Non-dimensional stream function values.
        """
        return self._F(eta)[0]

    def fp(self, eta):
        """
        Return the non-dimensional velocity from the solution.

        Parameters
        ----------
        eta : array-like
            Similarity coordinates to calculate the property.

        Returns
        -------
        array-like same shape as `eta`
            Non-dimensional velocity values.
        """
        return self._F(eta)[1]

    def fpp(self, eta):
        """
        Return the derivative of the non-dimensional velocity from solution.

        Parameters
        ----------
        eta : array-like
            Similarity coordinates to calculate the property.

        Returns
        -------
        array-like same shape as `eta`
            Derivative of the non-dimensional velocity values.
        """
        return self._F(eta)[2]

    def eta_d(self):
        """
        Return the displacement thickness in similarity coordinates.

        Returns
        -------
        float
            Displacement thickness in similarity coordinates.
        """
        return self._eta_inf-self.f(self._eta_inf)

    def eta_m(self):
        """
        Return the momentum thickness in similarity coordinates.

        Returns
        -------
        float
            Momentum thickness in similarity coordinates.
        """
        return self.fpp(0)

    def eta_k(self):
        """
        Return the kinetic energy thickness in similarity coordinates.

        Returns
        -------
        float
            Kinetic energy thickness in similarity coordinates.
        """
        def fun(eta):
            return 2*np.prod(self._F(eta), axis=0)

        val = quadrature(fun, 0, self._eta_inf)
        return val[0]

    def eta_s(self):
        """
        Return the shear thickness in similarity coordinates.

        Returns
        -------
        float
            Shear thickness in similarity coordinates.
        """
        def fun(eta):
            return 0.99-self.fp(eta)

        sol = root_scalar(fun, method="bisect", bracket=[0, 10])
        if not sol.converged:
            raise ValueError("Could not find shear thickness with error: "
                             + sol.flag)
        return sol.root

    def eta(self, x, y):
        """
        Return the similarity coordinate corresponding to coordinates.

        Parameters
        ----------
        x : array-like
            Streamwise coordinates of points of interest.
        y : array-llike
            Coordinates normal to the streamwise direction of points of
            interest.

        Returns
        -------
        array-like same shape as either `x` or `y`
            Similarity coordinate at the Cartesian coordinates.

        Notes
        -----
        Both `x` and `y` must be the same shape.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        return y*self._g(x)

    def u(self, x, y):
        """
        Return the x-velocity that corresponds to the Cartesian coordinates.

        Parameters
        ----------
        x : array-like
            Streamwise coordinates of points of interest.
        y : array-llike
            Coordinates normal to the streamwise direction of points of
            interest.

        Returns
        -------
        array-like same shape as either `x` or `y`
            Velocity component in the x-direction at the Cartesian
            coordinates.

        Notes
        -----
        Both `x` and `y` must be the same shape.
        """
        eta = self.eta(x, y)
        return self._U_ref*self.fp(eta)

    def v(self, x, y):
        """
        Return the y-velocity that corresponds to the Cartesian coordinates.

        Parameters
        ----------
        x : array-like
            Streamwise coordinates of points of interest.
        y : array-llike
            Coordinates normal to the streamwise direction of points of
            interest.

        Returns
        -------
        array-like same shape as either `x` or `y`
            Velocity component in the y-direction at the Cartesian
            coordinates.

        Notes
        -----
        Both `x` and `y` must be the same shape.
        """
        eta = self.eta(x, y)
        return self._nu*self._g(x)*(eta*self.fp(eta)-self.f(eta))

    def U_e(self, x):
        """
        Return the inviscid edge velocity at specified location(s).

        Parameters
        ----------
        x : array-like
            Streamwise location of points of interest.

        Returns
        -------
        array-like same as `x`
            Edge streamwise velocity at streamwise locations.
        """
        x = np.asarray(x)
        return self._U_ref*np.ones_like(x)

    def V_e(self, x):
        """
        Calculate the transpiration velocity.

        Parameters
        ----------
        x: array-like
            Streamwise loations to calculate this property.

        Returns
        -------
        array-like same shape as `x`
            Desired transpiration velocity at the specified locations.
        """
        return self._nu*self._g(x)*self.eta_d()

    def delta_d(self, x):
        """
        Calculate the displacement thickness.

        Parameters
        ----------
        x: array-like
            Streamwise loations to calculate this property.

        Returns
        -------
        array-like same shape as `x`
            Desired displacement thickness at the specified locations.
        """
        return self.eta_d()/self._g(x)

    def delta_m(self, x):
        """
        Calculate the momentum thickness.

        Parameters
        ----------
        x: array-like
            Streamwise loations to calculate this property.

        Returns
        -------
        array-like same shape as `x`
            Desired momentum thickness at the specified locations.
        """
        return self.eta_m()/self._g(x)

    def delta_k(self, x):
        """
        Calculate the kinetic energy thickness.

        Parameters
        ----------
        x: array-like
            Streamwise loations to calculate this property.

        Returns
        -------
        array-like same shape as `x`
            Desired kinetic energy thickness at the specified locations.
        """
        return self.eta_k()/self._g(x)

    def delta_s(self, x):
        """
        Calculate the shear thickness.

        Parameters
        ----------
        x: array-like
            Streamwise loations to calculate this property.

        Returns
        -------
        array-like same shape as `x`
            Desired shear thickness at the specified locations.
        """
        return self.eta_s()/self._g(x)

    def H_d(self, x):
        """
        Calculate the displacement shape factor.

        Parameters
        ----------
        x: array-like
            Streamwise loations to calculate this property.

        Returns
        -------
        array-like same shape as `x`
            Desired displacement shape factor at the specified locations.
        """
        x = np.asarray(x)
        return (self.eta_d()/self.eta_m())*np.ones_like(x)

    def H_k(self, x):
        """
        Calculate the kinetic energy shape factor.

        Parameters
        ----------
        x: array-like
            Streamwise loations to calculate this property.

        Returns
        -------
        array-like same shape as `x`
            Desired kinetic energy shape factor at the specified locations.
        """
        x = np.asarray(x)
        return (self.eta_k()/self.eta_m())*np.ones_like(x)

    def tau_w(self, x, rho):
        """
        Calculate the wall shear stress.

        Parameters
        ----------
        x: array-like
            Streamwise loations to calculate this property.
        rho: float
            Freestream density.

        Returns
        -------
        array-like same shape as `x`
            Desired wall shear stress at the specified locations.
        """
        return rho*self._nu*self._U_ref*self._g(x)*self.fpp(0)

    def D(self, x, rho):
        """
        Calculate the dissipation integral.

        Parameters
        ----------
        x: array-like
            Streamwise loations to calculate this property.
        rho: float
            Freestream density.

        Returns
        -------
        array-like same shape as `x`
            Desired dissipation integral at the specified locations.
        """
        return 0.5*rho*self._nu*self._g(x)*self._U_ref**2*self.eta_k()

    def _set_boundary_condition(self, fpp0=None):
        if fpp0 is None:
            def fun(fpp0):
                F0 = [0, 0, fpp0]
                rtn = solve_ivp(fun=self._ode_fun,
                                t_span=[0, self._eta_inf], y0=F0,
                                method="RK45", dense_output=False,
                                events=None, rtol=1e-8, atol=1e-11)
                if not rtn.success:
                    raise ValueError("Could not find boundary condition")
                return 1 - rtn.y[1, -1]

            sol = root_scalar(fun, method="bisect", bracket=[0.4, 0.5])
            if sol.converged:
                self._fpp0 = sol.root
            else:
                raise ValueError("Root finded could not find boundary "
                                 "condition")
        else:
            self._fpp0 = fpp0
        self._set_solution()

    def _set_solution(self):
        F0 = [0, 0, self._fpp0]

        rtn = solve_ivp(fun=self._ode_fun, t_span=[0, self._eta_inf],
                        y0=F0, method="RK45", dense_output=True,
                        events=None, rtol=1e-8, atol=1e-11)

        self._F = None
        if rtn.success:
            self._F = rtn.sol
        else:
            raise ValueError("Initial condition for solver, "
                             f"f\'\'(0)={self._fpp0:.6f}, did not produce "
                             "converged solution.")

    def _g(self, x):
        return np.sqrt(0.5*self._U_ref/(self._nu*x))

    @staticmethod
    def _ode_fun(eta, F):
        _ = eta  # To avoid pylint unused-argument message
        Fp = np.zeros_like(F)

        Fp[0] = F[1]
        Fp[1] = F[2]
        Fp[2] = -F[0]*F[2]

        return Fp

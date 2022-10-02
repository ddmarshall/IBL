#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of Head's method.

This module contains the necessary classes and data for the implementation of
Head's two equation integral boundary layer method.
"""

import numpy as np

from pyBL.ibl_method import IBLMethod
from pyBL.ibl_method import IBLTermEventBase
from pyBL.skin_friction import c_f_LudwiegTillman as c_f_fun


class HeadMethod(IBLMethod):
    """
    Models a turbulent bondary layer using Head's Method (1958).

    Solves the system of ODEs from Head's method when provided the edge
    velocity profile and other configuration information.
    """

    # Attributes
    # ----------
    #    _delta_m0: Momentum thickness at start location
    #    _H_d0: Displacement shape factor at start location
    #    _nu: Kinematic viscosity
    def __init__(self, U_e=None, dU_edx=None, d2U_edx2=None, H_d_crit=2.4):
        super().__init__(U_e, dU_edx, d2U_edx2)

        self._nu = None
        self._H_d0 = None
        self._delta_m0 = None
        self.set_H_d_critical(H_d_crit)

    def set_H_d_critical(self, H_d_crit):
        """
        Set the critical displacement shape factor for separation.

        Since Head's method does not predict when the skin friction will be
        zero, another mechanism needs to be employed to determine if/when
        separation will occur. This value is used as the threshold for the
        displacement shape factor to indicate separation has occurred.

        Parameters
        ----------
        H_d_crit : float
            New value for the .displacement shape factor to be used to indicate
            that the boundary layer has separated.
        """
        self._set_kill_event(_HeadSeparationEvent(H_d_crit))

    def set_solution_parameters(self, x0, x_end, delta_m0, H_d0, nu):
        """
        Set the parameters needed for the solver to propagate.

        Parameters
        ----------
        x0: float
            Location to start integration.
        x_end: float
            Location to end integration.
        delta_m0: float
            Momentum thickness at start location.
        H_d0: float
            Displacement shape factor at start location.
        nu: float
            Kinematic viscosity.

        Raises
        ------
        ValueError
            When negative viscosity provided, or invalid initial conditions
        """
        # pylint: disable=too-many-arguments
        if nu < 0:
            raise ValueError("Viscosity must be positive")

        self._nu = nu
        if delta_m0 < 0:
            raise ValueError("Initial momentum thickness must be positive")

        self._delta_m0 = delta_m0
        if H_d0 <= 1:
            raise ValueError("Initial displacement shape factor must be "
                             "greater than one")

        self._H_d0 = H_d0
        self._set_x_range(x0, x_end)

    def nu(self):
        """
        Return kinematic viscosity used for the solution.

        Returns
        -------
        float
            Kinematic viscosity.
        """
        return self._nu

    def solve(self, term_event=None):
        """
        Solve the ODEs associated with Head's method.

        Parameters
        ----------
        term_event : List based on :class:`IBLTermEventBase`, optional
            User events that can terminate the integration process before the
            end location of the integration is reached. The default is `None`.

        Returns
        -------
        :class:`IBLResult`
            Information associated with the integration process.
        """
        return self._solve_impl([self._delta_m0, self._H_d0],
                                term_event=term_event)

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
        yp = self._ode_impl(x, self._solution(x))
        H_d = self.H_d(x)
        U_e = self.U_e(x)
        dU_edx = self.dU_edx(x)
        delta_m = self.delta_m(x)
        return dU_edx*H_d*delta_m + U_e*yp[1]*delta_m + U_e*H_d*yp[0]

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
        return self.delta_m(x)*self.H_d(x)

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
        return self._solution(x)[0]

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
        return np.zeros_like(x)

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
        return self._solution(x)[1]

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
        return self.delta_k(x)/self.delta_m(x)

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
        delta_m = self._solution(x)[0]
        H_d = self._solution(x)[1]
        U_e = self.U_e(x)
        U_e[np.abs(U_e) < 0.001] = 0.001
        Re_delta_m = U_e*delta_m/self._nu
        c_f = c_f_fun(Re_delta_m, H_d)
        return 0.5*rho*U_e**2*c_f

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
        return np.zeros_like(x)

    def _ode_impl(self, x, F):
        """
        Right-hand-side of the ODE representing Thwaites method.

        Parameters
        ----------
        x: array-like
            Streamwise location of current step.
        F: array-like
            Current step's solution vector of momentum thickness and
            displacement shape factor.

        Returns
        -------
        array-like same shape as `F`
            The right-hand side of the ODE at the given state.
        """
        Fp = np.zeros_like(F)
        delta_m = F[0]
        H_d = np.asarray(F[1])
        if (H_d < 1.11).any():
            H_d[H_d < 1.11] = 1.11
        U_e = self.U_e(x)
        U_e[np.abs(U_e) < 0.001] = 0.001
        dU_edx = self.dU_edx(x)
        Re_delta_m = U_e*delta_m/self._nu
        c_f = c_f_fun(Re_delta_m, H_d)
        H1 = self._H1(H_d)
        H1p = self._H1p(H_d)
        Fp[0] = 0.5*c_f-delta_m*(2+H_d)*dU_edx/U_e
        Fp[1] = (U_e*self._S(H1) - U_e*Fp[0]*H1
                 - dU_edx*delta_m*H1)/(H1p*U_e*delta_m)
        return Fp

    @staticmethod
    def _H1(H_d):
        H_d = np.asarray(H_d)
        if (H_d <= 1.1).any():
            H_d[H_d <= 1.1] = 1.1001
#            raise ValueError("Cannot pass displacement shape factor less "
#                             "than 1.1: {}".format(np.amin(H_d)))

        def H1_low(H_d):
            a = 0.8234
            b = 1.1
            c = 1.287
            d = 3.3
            return d + a/(H_d - b)**c

        def H1_high(H_d):
            a = 1.5501
            b = 0.6778
            c = 3.064
            d = 3.32254659218600974
            return d + a/(H_d - b)**c

        return np.piecewise(H_d, [H_d <= 1.6, H_d > 1.6], [H1_low, H1_high])

    @staticmethod
    def _H1p(H_d):
        H_d_local = np.asarray(H_d)
        if (H_d_local <= 1.1).any():
            H_d_local[H_d_local <= 1.1] = 1.1001
#            raise ValueError("Cannot pass displacement shape factor less "
#                             "than 1.1: {}".format(np.amin(H_d)))

        def H1_low(H_d):
            a = 0.8234
            b = 1.1
            c = 1.287
            return -a*c/(H_d - b)**(c+1)

        def H1_high(H_d):
            a = 1.5501
            b = 0.6778
            c = 3.064
            return -a*c/(H_d - b)**(c+1)

        return np.piecewise(H_d_local, [H_d_local <= 1.6, H_d_local > 1.6],
                            [H1_low, H1_high])

    @staticmethod
    def _H_d(H1):
        H1_local = np.asarray(H1)
        if (H1_local <= 3.32254659218600974).any():
            H1_local[H1_local <= 3.32254659218600974] = 3.323
#            raise ValueError("Cannot pass entrainment shape factor less "
#                             "than 3.323: {}".format(np.amin(H1)))

        def H_d_low(H1):
            a = 1.5501
            b = 0.6778
            c = 3.064
            d = 3.32254659218600974
            return b + (a/(H1 - d))**(1/c)

        def H_d_high(H1):
            a = 0.8234
            b = 1.1
            c = 1.287
            d = 3.3
            return b + (a/(H1 - d))**(1/c)

        H1_break = HeadMethod._H1(1.6)
        return np.piecewise(H1_local, [H1_local <= H1_break,
                                       H1_local > H1_break],
                            [H_d_low, H_d_high])

    @staticmethod
    def _S(H1):
        H1_local = np.asarray(H1, float)
        if (H1_local <= 3).any():
            H1_local[H1_local <= 3] = 3.001
#            raise ValueError("Cannot pass entrainment shape factor less than "
#                             " 3: {}".format(np.amin(H1)))
        return 0.0306/(H1_local-3)**0.6169


class _HeadSeparationEvent(IBLTermEventBase):
    """
    Detects separation and will terminate integration when it occurs.

    This is a callable object that the ODE integrator will use to determine if
    the integration should terminate before the end location.

    Attributes
    ----------
        H_d_crit: Displacement shape factor value that indicates separation
    """

    # pylint: disable=too-few-public-methods
    def __init__(self, H_d_crit):
        super().__init__()
        self._H_d_crit = H_d_crit

    def _call_impl(self, x, F):
        """
        Determine if Head method integrator should terminate.

        This will terminate once the displacement shape factor becomes greater
        than critical H_d.

        Parameters
        ----------
        x: array-like
            Streamwise location of current step.
        F: array-like
            Current step's solution vector of momentum thickness and
            displacement shape factor.

        Returns
        -------
        float
            Current value of the difference between the critical displacement
            shape factor and the current displacement shape factor.
        """
        return self._H_d_crit - F[1]

    def event_info(self):
        return -1, ""

"""
Implementation of Head's method.

This module contains the necessary classes and data for the implementation of
Head's two equation integral boundary layer method.
"""

from typing import Tuple, cast

import numpy as np
import numpy.typing as np_type

from ibl.ibl_method import IBLMethod
from ibl.ibl_method import TermReason
from ibl.ibl_method import TermEvent
from ibl.skin_friction import ludwieg_tillman as c_f_fun
from ibl.initial_condition import ManualCondition
from ibl.typing import InputParam


class HeadMethod(IBLMethod):
    """
    Models a turbulent bondary layer using Head's Method (1958).

    Solves the system of ODEs from Head's method when provided the edge
    velocity profile and other configuration information.
    """

    def __init__(self, nu: float = 1.0, U_e=None, dU_edx=None, d2U_edx2=None,
                 shape_d_crit: float = 2.4) -> None:
        super().__init__(nu=nu, u_e=U_e, du_e=dU_edx, d2u_e=d2U_edx2,
                         ic=ManualCondition(np.inf, np.inf, 0))

        self.set_shape_d_critical(shape_d_crit)

    @property
    def initial_delta_m(self) -> float:
        """
        Momentum thickness at start of integration.
        Must be greater than zero.
        """
        return self._ic.delta_m()

    @initial_delta_m.setter
    def initial_delta_m(self, delta_m0: float) -> None:
        if delta_m0 <= 0:
            raise ValueError(f"Invalid initial momentum thickness: {delta_m0}")
        cast(ManualCondition, self._ic).del_m = delta_m0

    @property
    def initial_shape_d(self) -> float:
        """
        Dispacement thickness at start of integration.
        Must be greater than zero
        """
        return self._ic.shape_d()

    @initial_shape_d.setter
    def initial_shape_d(self, shape_d: float) -> None:
        if shape_d <= 0:
            raise ValueError(f"Invalid displacement shape factor: {shape_d}")
        cast(ManualCondition, self._ic).del_d = shape_d*self.initial_delta_m

    def set_shape_d_critical(self, shape_d_crit: float) -> None:
        """
        Set the critical displacement shape factor for separation.

        Since Head's method does not predict when the skin friction will be
        zero, another mechanism needs to be employed to determine if/when
        separation will occur. This value is used as the threshold for the
        displacement shape factor to indicate separation has occurred.

        Parameters
        ----------
        shape_d_crit : float
            New value for the displacement shape factor to be used to indicate
            that the boundary layer has separated.
        """
        self._set_kill_event(_HeadSeparationEvent(shape_d_crit))

    def v_e(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the transpiration velocity.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired transpiration velocity at the specified locations.
        """
        if self._solution is None:
            raise ValueError("No valid solution.")

        y_p = self._ode_impl(x, self._solution(x))
        shape_d = self.shape_d(x)
        u_e = self.u_e(x)
        du_e = self.du_e(x)
        delta_m = self.delta_m(x)
        return du_e*shape_d*delta_m + u_e*y_p[1]*delta_m + u_e*shape_d*y_p[0]

    def delta_d(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the displacement thickness.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired displacement thickness at the specified locations.
        """
        return self.delta_m(x)*self.shape_d(x)

    def delta_m(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the momentum thickness.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired momentum thickness at the specified locations.
        """
        if self._solution is None:
            raise ValueError("No valid solution.")

        return self._solution(x)[0]

    def delta_k(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the kinetic energy thickness.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired kinetic energy thickness at the specified locations.
        """
        return np.zeros_like(x)

    def shape_d(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the displacement shape factor.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired displacement shape factor at the specified locations.
        """
        if self._solution is None:
            raise ValueError("No valid solution.")

        return self._solution(x)[1]

    def shape_k(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the kinetic energy shape factor.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired kinetic energy shape factor at the specified locations.
        """
        return self.delta_k(x)/self.delta_m(x)

    def tau_w(self, x: InputParam, rho: float) -> np_type.NDArray:
        """
        Calculate the wall shear stress.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.
        rho: float
            Freestream density.

        Returns
        -------
        numpy.ndarray
            Desired wall shear stress at the specified locations.
        """
        if self._solution is None:
            raise ValueError("No valid solution.")

        delta_m = self._solution(x)[0]
        shape_d = self._solution(x)[1]
        u_e = self.u_e(x)
        u_e[np.abs(u_e) < 0.001] = 0.001
        re_delta_m = u_e*delta_m/self._nu
        c_f = c_f_fun(re_delta_m, shape_d)
        return 0.5*rho*u_e**2*c_f

    def dissipation(self, x: InputParam, rho: float) -> np_type.NDArray:
        """
        Calculate the dissipation integral.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.
        rho: float
            Freestream density.

        Returns
        -------
        numpy.ndarray
            Desired dissipation integral at the specified locations.
        """
        return np.zeros_like(x)

    def _ode_setup(self) -> Tuple[np_type.NDArray, float, float]:
        """
        Set the solver specific parameters.

        Returns
        -------
        3-Tuple
            IBL initialization array
            Relative tolerance for ODE solver
            Absolute tolerance for ODE solver
        """
        return np.array([self._ic.delta_m(), self._ic.shape_d()]), 1e-8, 1e-11

    def _ode_impl(self, x: np_type.NDArray,
                  f: np_type.NDArray) -> np_type.NDArray:
        """
        Right-hand-side of the ODE representing Thwaites method.

        Parameters
        ----------
        x: numpy.ndarray
            Streamwise location of current step.
        f: numpy.ndarray
            Current step's solution vector of momentum thickness and
            displacement shape factor.

        Returns
        -------
        numpy.ndarray
            The right-hand side of the ODE at the given state.
        """
        f_p = np.zeros_like(f)
        delta_m = f[0]
        shape_d = np.asarray(f[1])
        if (shape_d < 1.11).any():
            shape_d[shape_d < 1.11] = 1.11
        u_e = self.u_e(x)
        u_e[np.abs(u_e) < 0.001] = 0.001
        du_e = self.du_e(x)
        re_delta_m = u_e*delta_m/self._nu
        c_f = c_f_fun(re_delta_m, shape_d)
        shape_entrainment = self.shape_entrainment(shape_d)
        shape_entrainment_p = self._shape_entrainment_p(shape_d)
        f_p[0] = 0.5*c_f-delta_m*(2+shape_d)*du_e/u_e
        f_p[1] = (u_e*self._entrainment_velocity(shape_entrainment)
                  - u_e*f_p[0]*shape_entrainment
                  - du_e*delta_m*shape_entrainment)/(shape_entrainment_p
                                                     * u_e*delta_m)
        return f_p

    @staticmethod
    def shape_entrainment(shape_d: InputParam) -> np_type.NDArray:
        """
        Calculate the entrainment shape factor from displacement shape factor.

        Parameters
        ----------
        shape_d : InputParam
            Displacement shape factor.

        Returns
        -------
        numpy.ndarray
            Entrainment shape factor.
        """
        shape_d = np.asarray(shape_d)
        if (shape_d <= 1.1).any():
            shape_d[shape_d <= 1.1] = 1.1001
#            raise ValueError("Cannot pass displacement shape factor less "
#                             "than 1.1: {}".format(np.amin(H_d)))

        def shape_entrainment_low(shape_d: InputParam) -> InputParam:
            a = 0.8234
            b = 1.1
            c = 1.287
            d = 3.3
            return d + a/(shape_d - b)**c

        def shape_entrainment_high(shape_d: InputParam) -> InputParam:
            a = 1.5501
            b = 0.6778
            c = 3.064
            d = 3.32254659218600974
            return d + a/(shape_d - b)**c

        return np.piecewise(shape_d, [shape_d <= 1.6, shape_d > 1.6],
                            [shape_entrainment_low, shape_entrainment_high])

    @staticmethod
    def _shape_entrainment_p(shape_d: InputParam) -> np_type.NDArray:
        """
        Calculate the derivative of the shape entrainment factor.

        Parameters
        ----------
        shape_d : InputParam
            Displacement shape factor.

        Returns
        -------
        numpy.ndarray
            Entrainment shape factor.
        """
        shape_d = np.asarray(shape_d)
        if (shape_d <= 1.1).any():
            shape_d[shape_d <= 1.1] = 1.1001
#            raise ValueError("Cannot pass displacement shape factor less "
#                             "than 1.1: {}".format(np.amin(H_d)))

        def shape_entrainment_low(shape_d: InputParam) -> InputParam:
            a = 0.8234
            b = 1.1
            c = 1.287
            return -a*c/(shape_d - b)**(c+1)

        def shape_entrainment_high(shape_d: InputParam) -> InputParam:
            a = 1.5501
            b = 0.6778
            c = 3.064
            return -a*c/(shape_d - b)**(c+1)

        return np.piecewise(shape_d, [shape_d <= 1.6, shape_d > 1.6],
                            [shape_entrainment_low, shape_entrainment_high])

    @staticmethod
    def _shape_d(shape_entrainment: InputParam) -> np_type.NDArray:
        """
        Calculate the displacement shape factor from entrainment shape factor.

        Parameters
        ----------
        shape_entrainment : InputParam
            Entrainment shape factor.

        Returns
        -------
        numpy.ndarray
            Displacement shape factor.
        """
        shape_entrainment = np.asarray(shape_entrainment)
        if (shape_entrainment <= 3.32254659218600974).any():
            shape_entrainment[shape_entrainment <= 3.32254659218600974] = 3.323
#            raise ValueError("Cannot pass entrainment shape factor less "
#                             "than 3.323: {}".format(np.amin(H1)))

        def shape_d_low(shape_entrainment: InputParam) -> InputParam:
            a = 1.5501
            b = 0.6778
            c = 3.064
            d = 3.32254659218600974
            return b + (a/(shape_entrainment - d))**(1/c)

        def shape_d_high(shape_entrainment: InputParam) -> InputParam:
            a = 0.8234
            b = 1.1
            c = 1.287
            d = 3.3
            return b + (a/(shape_entrainment - d))**(1/c)

        shape_entrainment_break = HeadMethod.shape_entrainment(1.6)
        return np.piecewise(shape_entrainment,
                            [shape_entrainment <= shape_entrainment_break,
                             shape_entrainment > shape_entrainment_break],
                            [shape_d_low, shape_d_high])

    @staticmethod
    def _entrainment_velocity(shape_entr: InputParam) -> np_type.NDArray:
        shape_entr = np.asarray(shape_entr, float)
        if (shape_entr <= 3).any():
            shape_entr[shape_entr <= 3] = 3.001
#            raise ValueError("Cannot pass entrainment shape factor less than "
#                             " 3: {}".format(np.amin(H1)))
        return 0.0306/(shape_entr-3)**0.6169


class _HeadSeparationEvent(TermEvent):
    """
    Detects separation and will terminate integration when it occurs.

    This is a callable object that the ODE integrator will use to determine if
    the integration should terminate before the end location.

    Attributes
    ----------
        H_d_crit: Displacement shape factor value that indicates separation
    """

    def __init__(self, shape_d_crit: float) -> None:
        """
        Initialize separation criteria for Head's method.

        Parameters
        ----------
        shape_d_crit : float
            Critical displacement shape factor for separatation.
        """
        super().__init__()
        self._shape_d_crit = shape_d_crit

    def _call_impl(self, x: float, f: np_type.NDArray) -> float:
        """
        Determine if Head method integrator should terminate.

        This will terminate once the displacement shape factor becomes greater
        than critical H_d.

        Parameters
        ----------
        x : float
            Streamwise location of current step.
        f : numpy.ndarray
            Current step's solution vector of momentum thickness and
            displacement shape factor.

        Returns
        -------
        float
            Current value of the difference between the critical displacement
            shape factor and the current displacement shape factor.
        """
        return self._shape_d_crit - f[1]

    def event_info(self) -> Tuple[TermReason, str]:
        return TermReason.SEPARATED, ""

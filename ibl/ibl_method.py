"""
All base classes and data for integral boundary layer method classes.

There are a lot of common characterstics associated with the various integral
boundary layer methods. This module provides as much of that common code as
possible.

All integral boundary layer method classes should inherit from
:class:`IBLMethod`.

All integral boundary layer method classes return an instance of
:class:`IBLResult` when the solver has completed.
"""

from abc import ABC, abstractmethod
from enum import IntEnum, auto

from typing import Union, Tuple, Callable, List, Optional

import numpy as np
import numpy.typing as np_type

from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.misc import derivative as fd

from ibl.initial_condition import InitialCondition
from ibl.initial_condition import FalknerSkanStagCondition
from ibl.typing import InputParam


class TermReason(IntEnum):
    """
    Reasons for the integration process to complete.
    """

    def __str__(self) -> str:
        text = {TermReason.REACHED_END: "Completed",
                TermReason.SEPARATED: "Separated",
                TermReason.TRANSITIONED: "Transition",
                TermReason.UNKNOWN: "Unknown Event"}
        return_text = text.get(TermReason(self.value))
        if return_text is None:
            return_text = "Invalid Termination Code"
        return return_text

    #: Integration completed at given end point.
    REACHED_END = 0
    #: Integration ended early because boundary layer separated.
    SEPARATED = auto()
    #: Integration ended early because laminar boundary layer transitioned.
    TRANSITIONED = auto()
    #: Integration ended early for unknown reason.
    UNKNOWN = auto()


class IntegrationResult:
    """
    Bunch object representing the results of the IBL integration.

    The integrator within the :class:`IBLMethod` is `solve_ivp` from the
    integrate package from SciPy. To provide as much information as possible
    after the integration has completed this class is returned to provide
    detailed information about the integration process. The most important
    attributes are `success`, `status`, and `message`.
    """

    def __init__(self, x_end: float = np.inf, f_end: InputParam = np.inf,
                 status: TermReason = TermReason.UNKNOWN,
                 message: str = "Not Set",
                 success: bool = False):
        """
        Initialize class.

        Parameters
        ----------
        x_end : float, optional
            Termination location of integration, by default np.inf.
        f_end : InputParam, optional
            State value(s) at end of integration, by default np.inf.
        status : TermReason, optional
            Reason for termination, by default TermReason.UNKNOWN.
        message : str, optional
            Long description for termination reason, by default "Not Set".
        success : bool, optional
            Flag indicating whether integration was success, by default False.
        """
        self._x_end = x_end
        self._f_end = f_end
        self._status = status
        self._message = message
        self._success = success

    @property
    def x_end(self) -> float:
        """
        Terminating location of integration.
        """
        return self._x_end

    @property
    def f_end(self) -> InputParam:
        """
        State value(s) at end of integration.
        """
        return self._f_end

    @property
    def status(self) -> TermReason:
        """
        Enumeration for reason integration terminated.
        """
        return self._status

    @property
    def message(self) -> str:
        """
        Longer description of reason for termination.
        """
        return self._message

    @property
    def success(self) -> bool:
        """
        Flag indicating whether solver successfully completed.
        """
        return self._success

    def __str__(self) -> str:
        """
        Return a readable presentation of instance.

        Returns
        -------
        string
            Readable string representation of instance.
        """
        strout = f"{self.__class__.__name__}:\n"
        strout += f"    x_end: {self.x_end}\n"
        strout += f"    f_end: {self.f_end}\n"
        strout += f"    status: {self.status}\n"
        strout += f"    message: {self.message}\n"
        strout += f"    success: {self.success}"

        return strout


class TermEvent(ABC):
    """
    Base class for a termination event for IBL solver.

    The two abstract methods that have to be implemented are `event_info` and
    `_call_impl` for a valid termination event for use in an :class:`IBLMethod`
    derived class. Classes derived from this class can either be used within
    an IBL implementation (i.e., an implementation specific reason why the
    integration should terminate) or as a parameter into the solve method
    (i.e., the user has a particular reason for the integration to terminate).

    This class needs to return a float when called that changes sign when the
    integration should terminate. See the `solve_ivp` and related documentation
    for details.

    Attributes
    ----------
    terminal : bool
        True if event should terminate integration
    """

    def __init__(self) -> None:
        # This is needed by ivp_solver to know that the event means that the
        # integration should terminate
        self.terminal = True

    def __call__(self, x: float, f: np_type.NDArray) -> float:
        """
        Determine if integration should terminate.

        Parameters
        ----------
        x : float
            Current x-location of the integration.
        f : numpy.ndarray
            Current state value(s).

        Returns
        -------
        float
            Value that is zero when the solver should stop.
        """
        return self._call_impl(x, np.asarray(f))

    @abstractmethod
    def event_info(self) -> Tuple[TermReason, str]:
        """
        Return information about the purpose of this event.

        This is used to provide feedback as to what caused the integration to
        terminate and any other helpful information.

        Returns
        -------
        TermReason
            Value indicating reason this event would terminate integration
        str
            Extra information associated with reason for termination.

        Notes
        -----
        The value `TermReason.UNKNOWN` should be used for reasons that do
        not have an existing reason. For these termination situations the
        message can provide more details.
        """

    @abstractmethod
    def _call_impl(self, x: float, f: np_type.NDArray) -> float:
        """
        Information used to determine if IBL integrator should terminate.

        The return value is used in a root finder to find what `(x, F)` will
        result in the termination of the integrator. The function should
        return zero when the integrator should terminate, and change signs
        around the termination state.

        Parameters
        ----------
        x : float
            Current x-location of the integration.
        f : numpy.ndarray
            Current state value(s).

        Returns
        -------
        float
            The current value of the criteria being used to determine if the
            ODE solver should terminate.
        """


class IBLMethod(ABC):
    """
    Base class for integral boundary layer classes.

    This encapsulates the common features and needed parameters for all IBL
    methods. At the very least it provides the inteface that is expected for
    all IBL classes.

    **Integration**

    The intengral boundary layer method is based on solving one or more
    ordinary differential equations in the streamwise direction. These ODEs
    are solved explicitly with the `solve_ivp` class from `SciPy`. This
    class stores the resulting solution as interpolating polynomials of the
    same degree as the solver (known as a dense output). This allows the
    querying of the solution at any point between the start of the boundary
    layer and the end of the boundary layer to return a uniformly accurate
    result.

    **Edge Velocity**

    In order to solve these differential equations, the edge velocity
    variation is needed. There are a number of different ways to specify the
    edge velocity (`u_e`), the first derivative of the edge velocity
    (`du_e`), and the second derivative of the edge velocity (`d2u_e`):

        - `u_e` can be a 2-tuple of xpoints and velocity values.

          In this case a
          monotonic cubic spline will be created and the derivative functions
          will be taken from the cubic spline.

        - `u_e` can be a scalar and `dU_edx` is a 2-tuple of xpoints and rates
          of change of velocity values.

          In this case a monotonic cubic spline
          will be created for `du_e`. `u_e` will be found from the
          antiderivative and the scalar passed in as `u_e` will be used as the
          initial velocity. The other derivative(s) will be taken from the
          cubic spline.

        - `u_e` and the derivatives can be callable objects.

            + If the first derivative object is provided but not the second
              derivative object, then if the first derivative object has a
              method called `derivative` then that method will be used to
              generate the second derivative object. Otherwise the second
              derivative will be approximated by finite differences of the
              first derivative.

            + If neither derivative objects are provided, then if `u_e` has a
              method called `derivative` (like the classes from the
              `interpolate` module of `SciPy`) then that method will be used
              to generate both derivative objects. Otherwise the derivative
              objects will be created from finite difference approximations.

    **Initial Conditions**

    The initial conditions needed to start the integration may depend on
    the specific method being implemented. By default the flow is assumed to
    start at a laminar stagnation point. If other initial conditions are
    needed, then an :class:`InitialCondition` based class can be provided.

    Raises
    ------
    ValueError
        When configuration parameter is invalid (see message).
    """

    # Attributes
    # ----------
    # _U_e : Callable
    #     Function representing the edge velocity profile.
    # _dU_edx : Callable
    #     Function representing the streamwise derivative of the edge velocity.
    # _d2U_edx2 : Callable
    #     Function representing the streamwise second derivative of the edge
    #     velocity.
    # _ic : Initial condition generator
    # _kill_events : List of classes based on :class:`IBLTermEvent`
    #     Events that should be passed into ODE solver that might cause the
    #     integration to terminate early.
    # _solution : vector of callables
    #     Piecewise polynomials representing the state variables from the ODE
    #     solution.
    def __init__(self, nu: float, u_e=None, du_e=None, d2u_e=None,
                 ic: Optional[InitialCondition] = None):
        self._nu = 1e-5
        self._ic: InitialCondition = FalknerSkanStagCondition(0, nu)

        # set the velocity terms
        if u_e is None:
            if du_e is not None:
                raise ValueError("Must specify u_e if specifying du_e")
            if d2u_e is not None:
                raise ValueError("Must specify u_e if specifying d2u_e")
            self._u_e: Optional[Callable] = None
            self._du_e: Optional[Callable] = None
            self._d2u_e: Optional[Callable] = None
        else:
            self.set_velocity(u_e, du_e, d2u_e)

        # initialize other parameters
        if ic is None:
            self._ic = FalknerSkanStagCondition(0, nu)
        else:
            self.set_initial_condition(ic)

        self._kill_events: List[TermEvent] = []
        self._solution = None

    @property
    def nu(self) -> float:
        """
        Kinematic viscosity used for the solution.
        Must be greater than zero.
        """
        return self._nu

    @nu.setter
    def nu(self, nu: float) -> None:
        if nu <= 0:
            raise ValueError(f"Invalid kinematic viscosity: {nu}")
        self._nu = nu

    def set_initial_condition(self, ic: InitialCondition) -> None:
        """
        Set the initial conditions for solver.

        Parameters
        ----------
        ic : InitialCondition
            Desired initial condition.
        """
        self._ic = ic

    def set_velocity(self, u_e, du_e=None, d2u_e=None) -> None:
        """
        Set the edge velocity relations.

        There are a number of different ways to set the velocity relation and
        its derivatives. See class definition for details.

        Parameters
        ----------
        u_e : 2-tuple of array-like, scalar, or function-like
            Representation of the edge velocity to be used in analysis
        du_e : None, 2-tuple of array-like, or function-like, optional
            Representation of the first derivative of the edge velocity to be
            used in analysis. The default is `None`.
        d2u_e : None or function-like, optional
            Representationa of the second derivative of the edge velocity to
            be used in analysis. The default is `None`.

        Raises
        ------
        ValueError
            When configuration parameter is invalid (see message).
        """
        # check if U_e is callable
        if callable(u_e):
            self._u_e = u_e

            # if dU_edx not provided then use finite differences
            if du_e is None:
                if d2u_e is not None:
                    raise ValueError("Can only pass second derivative if "
                                     "first derivative was specified")

                # if U_e has derivative method then use it
                if (hasattr(u_e, "derivative")
                        and callable(getattr(u_e, "derivative"))):
                    self._du_e = u_e.derivative()
                    self._d2u_e = u_e.derivative(2)
                else:
                    # FIX: This is depricated in scipy
                    self._du_e = lambda x: fd(self._u_e, x, 1e-4, n=1,
                                              order=3)
                    self._d2u_e = lambda x: fd(self._u_e, x, 1e-4, n=2,
                                               order=3)
            else:
                if not callable(du_e):
                    raise ValueError("Must pass in callable object for first "
                                     "derivative if callable u_e given")
                self._du_e = du_e

                # if d2U_edx2 not provied then use finite difference
                if d2u_e is None:
                    # if dU_edx has derivative method then use it
                    if (hasattr(du_e, "derivative")
                            and callable(getattr(du_e, "derivative"))):
                        self._d2u_e = du_e.derivative()
                    else:
                        # FIX: This is depricated in scipy
                        self._d2u_e = lambda x: fd(self._du_e, x, 1e-5,
                                                   n=1, order=3)
                else:
                    if not callable(du_e):
                        raise ValueError("Must pass in callable object for "
                                         "first derivative if callable u_e "
                                         "given")

                    self._d2u_e = d2u_e
        elif isinstance(u_e, (int, float)):
            # if is 2-tuple then assume x, dU_edx pairs to build spline
            if len(du_e) == 2:
                x_pts = np.asarray(du_e[0])
                du_e_pts = np.asarray(du_e[1])
                self._du_e = PchipInterpolator(x_pts, du_e_pts)
                self._u_e = self._du_e.antiderivative()
                self._u_e.c[-1, :] = self._u_e.c[-1, :] + u_e
                self._d2u_e = self._du_e.derivative()
            else:
                # otherwise unknown velocity input
                raise ValueError(f"Don't know how to use {du_e} to "
                                 "initialize velocity derivative")
        else:
            # if is 2-tuple then assume x, U_e pairs to build spline
            if len(u_e) == 2:
                x_pts = np.asarray(u_e[0])
                u_e_pts = np.asarray(u_e[1])
                npts = x_pts.shape[0]
                # check to make sure have two vectors of same length suitable
                #   for building splines
                if x_pts.ndim != 1:
                    raise ValueError("First element of u_e 2-tuple must be 1D "
                                     "vector of distances")
                if u_e_pts.ndim != 1:
                    raise ValueError("Second element of u_e 2-tuple must be "
                                     "1D vector of Velocities")
                if npts != u_e_pts.shape[0]:
                    raise ValueError("Vectors in u_e 2-tuple must be of same "
                                     "length")
                if npts < 2:
                    raise ValueError("Must pass at least two points for edge "
                                     "velocity")

                u_e_spline = PchipInterpolator(x_pts, u_e_pts)
                self.set_velocity(u_e_spline)
            else:
                # otherwise unknown velocity input
                raise ValueError(f"Don't know how to use {u_e} to initialize "
                                 "velocity")

    def solve(self, x0: float, x_end: float,
              term_event: Optional[Union[TermEvent, List[TermEvent]]] = None
              ) -> IntegrationResult:
        """
        Solve the ODE associated with particular IBL method.

        This sets up the ODE solver using specific information from the child
        class and then runs the ODE solver to completion or termination
        because a termination event was triggered.

        Parameters
        ----------
        x0 : float
            Location to start integration.
        x_end : float
            Location to end integration.
        term_event : List based on :class:`IBLTermEvent`, optional
            User events that can terminate the integration process before the
            end location of the integration is reached. The default is `None`.

        Returns
        -------
        IBLResult
            Information associated with the integration process.

        Raises
        ------
        TypeError
            When solution parameters have not been set.
        """
        # setup the initial conditions
        self._ic.nu = self._nu
        self._ic.du_e = float(self.du_e(x0))
        y0, rtol_set, atol_set = self._ode_setup()
        if rtol_set is None:
            rtol = 1e-5
        else:
            rtol = rtol_set
        if atol_set is None:
            atol = 1e-8
        else:
            atol = atol_set

        # setup the ODE solver
        kill_events = self._kill_events

        if isinstance(term_event, list):
            kill_events += term_event
        elif term_event is not None:
            kill_events.append(term_event)

        rtn = solve_ivp(fun=self._ode_impl, t_span=[x0, x_end], y0=y0,
                        method="RK45", dense_output=True, events=kill_events,
                        rtol=rtol, atol=atol)

        # if completed gather info
        self._solution = None
        x_end = x0
        f_end = y0
        status = TermReason.UNKNOWN
        message = rtn.message
        if rtn.success:
            self._solution = rtn.sol

            # if terminated on time or early figure out why
            if rtn.status == 0:
                x_end = rtn.t[-1]
                f_end = rtn.sol(x_end)
                status = TermReason.REACHED_END
                message = ""
            elif rtn.status == 1:
                message = "Event not found."
                for i, xe in enumerate(rtn.t_events):
                    if xe.shape[0] > 0:
                        x_end = xe[0]
                        f_end = rtn.sol(x_end)
                        status, message = kill_events[i].event_info()
                        break

        if len(message) > 0:
            message = f"{str(status)}: {message}"
        else:
            message = str(status)
        return IntegrationResult(x_end=x_end, f_end=f_end, status=status,
                                 message=message, success=rtn.success)

    def u_e(self, x: InputParam) -> np_type.NDArray:
        """
        Return the inviscid edge velocity at specified location(s).

        Parameters
        ----------
        x : InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Inviscid edge velocity.

        Raises
        ------
        TypeError
            When velocity parameters have not been set.
        """
        if self._u_e is None:
            raise ValueError("u_e was not set")
        return self._u_e(x)

    def du_e(self, x: InputParam) -> np_type.NDArray:
        """
        Streamwise derivative of inviscid edge velocity at location(s).

        Parameters
        ----------
        x : InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Derivative of inviscid edge velocity.

        Raises
        ------
        TypeError
            When velocity parameters have not been set.
        """
        if self._du_e is None:
            raise ValueError("du_e was not set")
        return self._du_e(x)

    def d2u_e(self, x: InputParam) -> np_type.NDArray:
        """
        Streamwise second derivative of inviscid edge velocity at location(s).

        Parameters
        ----------
        x : InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Second derivative of inviscid edge velocity.

        Raises
        ------
        TypeError
            When velocity parameters have not been set.
        """
        if self._d2u_e is None:
            raise ValueError("d2U_edx2 was not set")
        return self._d2u_e(x)

    def _add_kill_event(self, ke: Union[TermEvent, List[TermEvent]]) -> None:
        """
        Add kill event to the ODE solver.

        Parameters
        ----------
        ke : TermEvent | List[TermEvent]
            Way of child classes to automatically add kill events to the ODE
            solver.
        """
        if self._kill_events is None:
            self._set_kill_event(ke)
        else:
            if isinstance(ke, TermEvent):
                self._kill_events.append(ke)
            else:
                self._kill_events += ke

    def _set_kill_event(self, ke: Union[TermEvent, List[TermEvent]]) -> None:
        """
        Set kill events for the ODE solver.

        Parameters
        ----------
        ke : TermEvent | List[TermEvent]
            Way of setting the kill events to the ODE solver and removing all
            existing events.
        """
        if isinstance(ke, TermEvent):
            self._kill_events = [ke]
        else:
            self._kill_events = ke

    @abstractmethod
    def v_e(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the transpiration velocity.

        Parameters
        ----------
        x : InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired transpiration velocity at the specified locations.
        """

    @abstractmethod
    def delta_d(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the displacement thickness.

        Parameters
        ----------
        x : InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired displacement thickness at the specified locations.
        """

    @abstractmethod
    def delta_m(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the momentum thickness.

        Parameters
        ----------
        x : InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired momentum thickness at the specified locations.
        """

    @abstractmethod
    def delta_k(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the kinetic energy thickness.

        Parameters
        ----------
        x : InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired kinetic energy thickness at the specified locations.
        """

    @abstractmethod
    def shape_d(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the displacement shape factor.

        Parameters
        ----------
        x : InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired displacement shape factor at the specified locations.
        """

    @abstractmethod
    def shape_k(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the kinetic energy shape factor.

        Parameters
        ----------
        x : InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired kinetic energy shape factor at the specified locations.
        """

    @abstractmethod
    def tau_w(self, x: InputParam, rho: float) -> np_type.NDArray:
        """
        Calculate the wall shear stress.

        Parameters
        ----------
        x : InputParam
            Streamwise loations to calculate this property.
        rho : float
            Freestream density.

        Returns
        -------
        numpy.ndarray
            Desired wall shear stress at the specified locations.
        """

    @abstractmethod
    def dissipation(self, x: InputParam, rho: float) -> np_type.NDArray:
        """
        Calculate the dissipation integral.

        Parameters
        ----------
        x : InputParam
            Streamwise loations to calculate this property.
        rho : float
            Freestream density.

        Returns
        -------
        numpy.ndarray
            Desired dissipation integral at the specified locations.
        """

    @abstractmethod
    def _ode_setup(self) -> Tuple[np_type.NDArray, Optional[float],
                                  Optional[float]]:
        """
        Set the solver specific parameters.

        Returns
        -------
        np_type.NDArray
            IBL initialization array.
        Optional[float]
            Relative tolerance for ODE solver.
        Optional[float]
            Absolute tolerance for ODE solver.
        """

    @abstractmethod
    def _ode_impl(self, x: np_type.NDArray,
                  f: np_type.NDArray) -> np_type.NDArray:
        """
        Right-hand-side of the ODE representing Thwaites method.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of current step.
        f : numpy.ndarray
                current step state value(s).

        Returns
        -------
        numpy.ndarray
            The right-hand side of the ODE at the given state.
        """

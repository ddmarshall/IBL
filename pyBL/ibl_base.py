#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All base classes and data for integral boundary layer method classes.

There are a lot of common characterstics associated with the various integral
boundary layer methods. This module provides as much of that common code as
possible.

All integral boundary layer method classes should inherit from
:class:`IBLBase`.

All integral boundary layer method classes return an instance of
:class:`IBLResult` when the solver has completed.
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.misc import derivative as fd


# def _stagnation_y0(iblsimdata,x0):
#     #From Moran
#       return .075*iblsimdata.nu/iblsimdata.du_edx(x0)


class IBLBase(ABC):
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
    edge velocity (`U_e`), the first derivative of the edge velocity
    (`dU_edx`), and the sedond derivative of the edge velocity (`d2U_edx2`):

        - `U_e` can be a 2-tuple of xpoints and velocity values.

          In this case a
          monotonic cubic spline will be created and the derivative functions
          will be taken from the cubic spline.

        - `U_e` can be a scalar and `dU_edx` is a 2-tuple of xpoints and rates
          of change of velocity values.

          In this case a monotonic cubic spline
          will be created for `dU_edx`. `U_e` will be found from the
          antiderivative and the scalar passed in as `U_e` will be used as the
          initial velocity. The other derivative(s) will be taken from the
          cubic spline.

        - `U_e` and the derivatives can be callable objects.

            + If the first derivative object is provided but not the second
              derivative object, then if the first derivative object has a
              method called `derivative` then that method will be used to
              generate the second derivative object. Otherwise the second
              derivative will be approximated by finite differences of the
              first derivative.

            + If neither derivative objects are provided, then if `U_e` has a
              method called `derivative` (like the classes from the
              `interpolate` module of `SciPy`) then that method will be used
              to generate both derivative objects. Otherwise the derivative
              objects will be created from finite difference approximations.

    **Initial Conditions**

    The initial conditions needed to start the integration will depend on
    the specific method being implemented.

    Raises
    ------
    ValueError
        When configuration parameter is invalid (see message).
    """

    # Attributes
    # ----------
    # _U_e: Callable
    #     Function representing the edge velocity profile.
    # _dU_edx: Callable
    #     Function representing the streamwise derivative of the edge velocity.
    # _d2U_edx2: Callable
    #     Function representing the streamwise second derivative of the edge
    #     velocity.
    # _x_range: 2-tuple
    #     Start and end location for integration.
    # _kill_events: List of classes based on :class:`IBLTermEventBase`
    #     Events that should be passed into ODE solver that might cause the
    #     integration to terminate early.
    # _solution: vector of callables
    #     Piecewise polynomials representing the state variables from the ODE
    #     solution.
    def __init__(self, U_e=None, dU_edx=None, d2U_edx2=None):
        # set the velocity terms
        if U_e is None:
            if dU_edx is not None:
                raise ValueError("Must specify U_e if specifying dU_edx")
            if d2U_edx2 is not None:
                raise ValueError("Must specify U_e if specifying d2U_edx2")
            self._U_e = None
            self._dU_edx = None
            self._d2U_edx2 = None
        else:
            self.set_velocity(U_e, dU_edx, d2U_edx2)

        # initialize other parameters
        self._x_range = None
        self._kill_events = None
        self._solution = None

    def set_velocity(self, U_e, dU_edx=None, d2U_edx2=None):
        """
        Set the edge velocity relations.

        There are a number of different ways to set the velocity relation and
        its derivatives. See class definition for details.

        Parameters
        ----------
        U_e : 2-tuple of array-like, scalar, or function-like
            Representation of the edge velocity to be used in analysis
        dU_edx : None, 2-tuple of array-like, or function-like, optional
            Representation of the first derivative of the edge velocity to be
            used in analysis. The default is `None`.
        d2U_edx2 : None or function-like, optional
            Representationa of the second derivative of the edge velocity to
            be used in analysis. The default is `None`.

        Raises
        ------
        ValueError
            When configuration parameter is invalid (see message).
        """
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        # check if U_e is callable
        if callable(U_e):
            self._U_e = U_e

            # if dU_edx not provided then use finite differences
            if dU_edx is None:
                if d2U_edx2 is not None:
                    raise ValueError("Can only pass second derivative if "
                                     "first derivative was specified")

                # if U_e has derivative method then use it
                if (hasattr(U_e, "derivative")
                        and callable(getattr(U_e, "derivative"))):
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
                    if (hasattr(dU_edx, "derivative")
                            and callable(getattr(dU_edx, "derivative"))):
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
        elif isinstance(U_e, (int, float)):
            # if is 2-tuple then assume x, dU_edx pairs to build spline
            if len(dU_edx) == 2:
                x_pts = np.asarray(dU_edx[0])
                dU_edx_pts = np.asarray(dU_edx[1])
                self._dU_edx = PchipInterpolator(x_pts, dU_edx_pts)
                self._U_e = self._dU_edx.antiderivative()
                self._U_e.c[-1, :] = self._U_e.c[-1, :] + U_e
                self._d2U_edx2 = self._dU_edx.derivative()
            else:
                # otherwise unknown velocity input
                raise ValueError(f"Don't know how to use {dU_edx} to "
                                 "initialize velocity derivative")
        else:
            # if is 2-tuple then assume x, U_e pairs to build spline
            if len(U_e) == 2:
                x_pts = np.asarray(U_e[0])
                U_e_pts = np.asarray(U_e[1])
                npts = x_pts.shape[0]
                # check to make sure have two vectors of same length suitable
                #   for building splines
                if x_pts.ndim != 1:
                    raise ValueError("First element of U_e 2-tuple must be 1D "
                                     "vector of distances")
                if U_e_pts.ndim != 1:
                    raise ValueError("Second element of U_e 2-tuple must be "
                                     "1D vector of Velocities")
                if npts != U_e_pts.shape[0]:
                    raise ValueError("Vectors in U_e 2-tuple must be of same "
                                     "length")
                if npts < 2:
                    raise ValueError("Must pass at least two points for edge "
                                     "velocity")

                U_e_spline = PchipInterpolator(x_pts, U_e_pts)
                self.set_velocity(U_e_spline)
            else:
                # otherwise unknown velocity input
                raise ValueError(f"Don't know how to use {U_e} to initialize "
                                 "velocity")

    def _solve_impl(self, y0i, rtol=1e-8, atol=1e-11, term_event=None):
        """
        Solve the ODEs to determine the boundary layer properties.

        Parameters
        ----------
        y0i: scalar or array-like
            Initial condition of the state vector for integration

            The specific type will depend on the details of the differential
            equations that the child class needs solved.
        rtol: float, optional
            Relative tolerance for integration scheme. The default is 1e-8.
        atol: float, optional
            Absolute tolerance for integration scheme. The default is 1e-11.
        term_event: List based on :class:`IBLTermEventBase`, optional
            Additional termination events. The default is `None`.

        Notes
        -----
        These events will be used in addition to any internal ones to determine
        if/when the integration should terminate before the end location. These
        should mostly be for transition to turbulent boundary layer or
        separation.

        Returns
        -------
        Bunch object: :class:`IBLResult`
            Information about the solution process and termination.

        Raises
        ------
        TypeError
            When solution parameters have not been set.
        """
        # pylint: disable=too-many-branches
        # setup the ODE solver
        if self._x_range is None:
            raise TypeError("Derived class needs to set the x-range")

        y0 = np.asarray(y0i)
        if y0.ndim == 0:
            y0 = [y0i]

        kill_events = []
        if self._kill_events is not None:
            kill_events = kill_events + self._kill_events

        if term_event is None:
            if self._kill_events is None:
                kill_events = None
        else:
            if isinstance(term_event, list):
                kill_events = kill_events + term_event
            else:
                kill_events.append(term_event)

        rtn = solve_ivp(fun=self._ode_impl, t_span=self._x_range, y0=y0,
                        method="RK45", dense_output=True, events=kill_events,
                        rtol=rtol, atol=atol)

        # if completed gather info
        self._solution = None
        x_end = self._x_range[0]
        F_end = y0
        status = -99
        message = rtn.message
        if rtn.success:
            self._solution = rtn.sol

            # if terminated on time or early figure out why
            if rtn.status == 0:
                x_end = rtn.t[-1]
                F_end = rtn.sol(x_end)
                status = 0
                message = ""
            elif rtn.status == 1:
                message = "Event not found."
                for i, xe in enumerate(rtn.t_events):
                    if xe.shape[0] > 0:
                        x_end = xe[0]
                        F_end = rtn.sol(x_end)
                        status, message = kill_events[i].event_info()
                        break
            else:
                status = -99

        if len(message) > 0:
            message = f"{TERMINATION_MESSAGES.get(status)}: {message}"
        else:
            message = TERMINATION_MESSAGES.get(status)
        return IBLResult(x_end=x_end, F_end=F_end, status=status,
                         message=message, success=rtn.success)

    def _set_x_range(self, x0, x1):
        """
        Set the start and end location for analysis.

        Parameters
        ----------
        x0 : float
            Starting location along surface for integration.
        x1 : float
            Ending location along surface for integration.
        """
        self._x_range = [x0, x1]

    def U_e(self, x):
        """
        Return the inviscid edge velocity at specified location(s).

        Parameters
        ----------
        x: array-like
            Streamwise loations to calculate this property.

        Returns
        -------
        array-like same shape as `x`
            Inviscid edge velocity.

        Raises
        ------
        TypeError
            When velocity parameters have not been set.
        """
        if self._U_e is None:
            raise ValueError("U_e was not set")
        return self._U_e(x)

    def dU_edx(self, x):
        """
        Streamwise derivative of inviscid edge velocity at location(s).

        Parameters
        ----------
        x: array-like
            Streamwise loations to calculate this property.

        Returns
        -------
        array-like same shape as `x`
            Derivative of inviscid edge velocity.

        Raises
        ------
        TypeError
            When velocity parameters have not been set.
        """
        if self._dU_edx is None:
            raise ValueError("dU_edx was not set")
        return self._dU_edx(x)

    def d2U_edx2(self, x):
        """
        Streamwise second derivative of inviscid edge velocity at location(s).

        Parameters
        ----------
        x: array-like
            Streamwise loations to calculate this property.

        Returns
        -------
        array-like same shape as `x`
            Second derivative of inviscid edge velocity.

        Raises
        ------
        TypeError
            When velocity parameters have not been set.
        """
        if self._d2U_edx2 is None:
            raise ValueError("d2U_edx2 was not set")
        return self._d2U_edx2(x)

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    def _add_kill_event(self, ke):
        """
        Add kill event to the ODE solver.

        Parameters
        ----------
        ke: List of classes based on :class:`IBLTermEventBase`
            Way of child classes to automatically add kill events to the ODE
            solver.
        """
        if self._kill_events is None:
            self._set_kill_event(ke)
        else:
            self._kill_events.append(ke)

    def _set_kill_event(self, ke):
        """
        Set kill events for the ODE solver.

        Parameters
        ----------
        ke: List of classes based on :class:`IBLTermEventBase`
            Way of setting the kill events to the ODE solver and removing all
            existing events.
        """
        self._kill_events = [ke]

    @abstractmethod
    def _ode_impl(self, x, F):
        """
        Right-hand-side of the ODE representing Thwaites method.

        Parameters
        ----------
        x: array-like
            Streamwise location of current step.
        F: array-like
            Current step's square of momentum thickness divided by the
            kinematic viscosity.

        Returns
        -------
        array-like same shape as `F`
            The right-hand side of the ODE at the given state.
        """


TERMINATION_MESSAGES = {0: "Completed",
                        -1: "Separated",
                        1: "Transition",
                        -99: "Unknown Event"}


class IBLResult:
    """
    Bunch object representing the results of the IBL integration.

    The integrator within the :class:`IBLBase` is `solve_ivp` from the
    integrate package from SciPy. To provide as much information as possible
    after the integration has completed this class is returned to provide
    detailed information about the integration process. The most important
    attributes are `success`, `status`, and `message`.

    Attributes
    ----------
    x_end: float
        x-location of end of integration.
    F_end: np.array
        State value(s) at end of integration.
    status: int
        Reason integration terminated:
            **0** Reached final distance

            **-1** Separation occured at x_end

            **1** Transition occured at x_end

            **Other values** Specified by implementations
    message: string
        Description of termination reason.
    success: Boolean
        True if solver successfully completed.
    """

    # pylint: disable=too-few-public-methods
    def __init__(self, x_end=np.inf, F_end=np.inf, status=-99,
                 message="Not Set", success=False):
        # pylint: disable=too-many-arguments
        self.x_end = x_end
        self.F_end = F_end
        self.status = status
        self.message = message
        self.success = success

    def __str__(self):
        """
        Return a readable presentation of instance.

        Returns
        -------
        string
            Readable string representation of instance.
        """
        strout = f"{self.__class__.__name__}:\n"
        strout += f"    x_end: {self.x_end}\n"
        strout += f"    F_end: {self.F_end}\n"
        strout += f"    status: {self.status}\n"
        strout += f"    message: {self.message}\n"
        strout += f"    success: {self.success}"

        return strout


class IBLTermEventBase(ABC):
    """
    Base class for a termination event for IBL solver.

    The two abstract methods that have to be implemented are `event_info` and
    `_call_impl` for a valid termination event for use in an :class:`IBLBase`
    derived class. Classes derived from this class can either be used within
    an IBL implementation (i.e., an implementation specific reason why the
    integration should terminate) or as a parameter into the solve method
    (i.e., the user has a particular reason for the integration to terminate).

    This class needs to return a float when called that changes sign when the
    integration should terminate. See the `solve_ivp` and related documentation
    for details.
    """

    def __init__(self):
        # This is needed by ivp_solver to know that the event means that the
        # integration should terminate
        self.terminal = True

    def __call__(self, x, F):
        """
        Determine if integration should terminate.

        Parameters
        ----------
        x: float
            Current x-location of the integration.
        F: np.array
            Current state value(s).

        Returns
        -------
        float
            Value that is zero when the solver should stop.
        """
        return self._call_impl(x, np.asarray(F))

    @abstractmethod
    def event_info(self):
        """
        Return information about the purpose of this event.

        This is used to provide feedback as to what caused the integration to
        terminate and any other helpful information.

        Returns
        -------
        Event index
            Value indicating reason this event would terminate integration
        Description
            Extra information associated with reason for termination.

        Notes
        -----
        The event index should be -1 for separation and 1 for transition.
        Other values are considered general termination situations and may not
        be cleanly handled by this package.
        """

    @abstractmethod
    def _call_impl(self, x, F):
        """
        Information used to determine if IBL integrator should terminate.

        The return value is used in a root finder to find what `(x, F)` will
        result in the termination of the integrator. The function should
        return zero when the integrator should terminate, and change signs
        around the termination state.

        Parameters
        ----------
        x: float
            Current x-location of the integration.
        F: array-like
            Current state value(s).

        Returns
        -------
        array-like
            The current value of the criteria being used to determine if the
            ODE solver should terminate.
        """

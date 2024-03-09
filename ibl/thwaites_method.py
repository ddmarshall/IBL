"""
Implementations of Thwaites' method.

This module contains the necessary classes and data for the implementation of
Thwaites' one equation integral boundary layer method. There are two concrete
implementations: :class:`ThwaitesMethodLinear` that is based on the traditional
assumption that the ODE to be solved fits a linear relationship, and
:class:`ThwaitesMethodNonlinear` that removes the linear relationship
assumption and provides slightly better results in all cases tested.
"""

from abc import abstractmethod
from typing import Tuple, cast, Union, Callable

import numpy as np
import numpy.typing as np_type

from scipy.interpolate import CubicSpline
from scipy.misc import derivative as fd

from ibl.ibl_method import IBLMethod
from ibl.ibl_method import TermEvent
from ibl.ibl_method import TermReason
from ibl.initial_condition import ManualCondition
from ibl.typing import InputParam


_DataFits = Union[str, Tuple[Callable, Callable], Tuple[Callable, Callable,
                                                        Callable]]


class ThwaitesMethod(IBLMethod):
    """
    Base class for Thwaites' Method.

    This class models a laminar boundary layer using Thwaites' Method from,
    “Approximate Calculation of the Laminar Boundary Layer.” **The Aeronautical
    Journal**, Vol. 1, No. 3, 1949, pp. 245–280. It is the base class for the
    linear (:class:`ThwaitesMethodLinear`)and nonlinear
    (:class:`ThwaitesMethodNonlinear`) versions of Thwaites method.

    In addition to the :class:`IBLMethod` configuration information, the
    initial momentum thickness is needed along with the kinematic viscosity.
    Thwaites' original algorithm relied upon tabulated data for the analysis,
    and there are few different ways of modeling that data in this class.
    """

    # Attributes
    #    _model: Collection of functions for S, H, and H'
    def __init__(self, nu: float = 1.0, U_e=None, dU_edx=None, d2U_edx2=None,
                 data_fits:_DataFits = "Spline") -> None:
        super().__init__(nu=nu, u_e=U_e, du_e=dU_edx, d2u_e=d2U_edx2,
                         ic=ManualCondition(delta_d=0, delta_m=np.inf,
                                            delta_k=0))
        self._model: _ThwaitesFunctions

        self.set_data_fits(data_fits)

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

    def set_data_fits(self, data_fits: _DataFits) -> None:
        """
        Set the data fit functions.

        This method sets the functions used for the data fits of the shear
        function, shape function, and the slope of the shape function.

        Parameters
        ----------
            data_fits: 2-tuple, 3-tuple, or string
                The data fits can be set via one of the following methods:
                    - 3-tuple of callable objects taking one parameter that
                      represent the shear function, the shape function, and
                      the derivative of the shape function;
                    - 2-tuple of callable objects taking one parameter that
                      represent the shear function and the shape function.
                      The derivative of the shear function is then
                      approximated using finite differences; or
                    - String for representing one of the three internal
                      implementations:

                         - "Spline" for spline fits of Thwaites original
                           data (Edland 2022)
                         - "White" for the curve fits from White (2011)
                         - "Cebeci-Bradshaw" for curve fits from
                           Cebeci-Bradshaw (1977)

        Raises
        ------
        ValueError
            When an invalid fit name or unusable 2-tuple or 3-tuple provided
        """
        # data_fits can either be string or 2-tuple of callables
        if isinstance(data_fits, str):
            if data_fits == "Spline":
                self._model = _ThwaitesFunctionsSpline()
            elif data_fits == "White":
                self._model = _ThwaitesFunctionsWhite()
            elif data_fits == "Cebeci-Bradshaw":
                self._model = _ThwaitesFunctionsCebeciBradshaw()
            else:
                raise ValueError("Unknown fitting function name: ", data_fits)
        else:
            # check to make sure have two callables
            if isinstance(data_fits, tuple):
                if len(data_fits) == 3:
                    if callable(data_fits[0]) and callable(data_fits[1]) \
                            and callable(data_fits[2]):  # type: ignore [misc]
                        self._model = _ThwaitesFunctions(
                            "Custom", data_fits[0],
                            data_fits[1], data_fits[2],  # type: ignore [misc]
                            -np.inf, np.inf)
                    else:
                        raise ValueError("Need to pass callable objects for "
                                         "fit functions")
                elif len(data_fits) == 2:
                    if callable(data_fits[0]) and callable(data_fits[1]):
                        def shape_p_fun(lam: InputParam) -> np_type.NDArray:
                            return fd(self._model.shape, lam, 1e-5, n=1,
                                      order=3)
                        self._model = _ThwaitesFunctions("Custom",
                                                         data_fits[0],
                                                         data_fits[1],
                                                         shape_p_fun,
                                                         -np.inf, np.inf)
                    else:
                        raise ValueError("Need to pass callable objects for "
                                         "fit functions")
                else:
                    raise ValueError("Need to pass two or three callable "
                                     "objects for fit functions")
            else:
                raise ValueError("Need to pass a 2-tuple for fit functions")

        self._set_kill_event(_ThwaitesSeparationEvent(self._calc_lambda,
                                                      self._model.shear))

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
        if self._solution is None:
            raise ValueError("Solution has not been set.")

        u_e = self.u_e(x)
        du_e = self.du_e(x)
        delta_m2_on_nu = self._solution(x)[0]
        term1 = du_e*self.delta_d(x)
        term2 = np.sqrt(self._nu/delta_m2_on_nu)
        dsol_dx = self._ode_impl(x, delta_m2_on_nu)
        term3 = 0.5*u_e*self.shape_d(x)*dsol_dx
        term4a = self._model.shape_p(self._calc_lambda(x, delta_m2_on_nu))
        term4 = u_e*delta_m2_on_nu*term4a
        term5 = du_e*dsol_dx+self.d2u_e(x)*delta_m2_on_nu
        return term1 + term2*(term3+term4*term5)

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
        return self.delta_m(x)*self.shape_d(x)

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
        if self._solution is None:
            raise ValueError("Solution has not been set.")

        return np.sqrt(self._solution(x)[0]*self._nu)

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
        return np.zeros_like(x)

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
        if self._solution is None:
            raise ValueError("Solution has not been set.")

        lam = self._calc_lambda(x, self._solution(x)[0])
        return self._model.shape(lam)

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
        return self.delta_k(x)/self.delta_m(x)

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
        if self._solution is None:
            raise ValueError("Solution has not been set.")

        lam = self._calc_lambda(x, self._solution(x)[0])
        return rho*self._nu*self.u_e(x)*self._model.shear(lam)/self.delta_m(x)

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
        return np.array([self._ic.delta_m()**2/self._nu]), 1e-8, 1e-11

    def _ode_impl(self, x: np_type.NDArray,
                  f: np_type.NDArray) -> np_type.NDArray:
        """
        Right-hand-side of the ODE representing Thwaites method.

        Parameters
        ----------
        x: numpy.ndarray
            Streamwise location of current step.
        f: numpy.ndarray
            Current step's square of momentum thickness divided by the
            kinematic viscosity.

        Returns
        -------
        numpy.ndarray
            The right-hand side of the ODE at the given state.
        """
        return self._calc_f(x, f)/(1e-3 + self.u_e(x))

    def _calc_lambda(self, x: InputParam,
                     delta_m2_on_nu: np_type.NDArray) -> np_type.NDArray:
        r"""
        Calculate the :math:`\lambda` term needed in Thwaites' method.

        Parameters
        ----------
        x: numpy.ndarray
            Streamwise location of current step.
        delta_m2_on_nu : numpy.ndarray
            Dependent variable in the ODE solver.

        Returns
        -------
        numpy.ndarray
            The :math:`\lambda` parameter that corresponds to the given state.
        """
        return delta_m2_on_nu*self.du_e(x)

    @abstractmethod
    def _calc_f(self, x: np_type.NDArray,
                delta_m2_on_nu: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the :math:`F` term in the ODE.

        The F term captures the interaction between the shear function and the
        shape function and can be modeled as a linear expression (as the
        standard Thwaites' method does) or can be calculated directly using
        the data fit relations for the shear function and the shape function.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of current step.
        delta_m2_on_nu : numpy.ndarray
            Dependent variable in the ODE solver.

        Returns
        -------
        numpy.ndarray
            The calculated value of :math:`F`
        """


class ThwaitesMethodLinear(ThwaitesMethod):
    r"""
    Laminar boundary layer model using Thwaites Method linear approximation.

    Solves the original approximate ODE from Thwaites' method when provided
    the edge velocity profile. There are a few different ways of modeling the
    tabular data from Thwaites original work that can be set.

    This class solves the following differential equation using the linear
    approximation from Thwaites' original paper

    .. math::
        \frac{d}{dx}\left(\frac{\delta_m^2}{\nu}\right)
            =\frac{1}{U_e}\left(0.45-6\lambda\right)

    using the :class:`IBLMethod` ODE solver.
    """

    def _calc_f(self, x: np_type.NDArray,
                delta_m2_on_nu: np_type.NDArray) -> np_type.NDArray:
        r"""
        Calculate the :math:`F` term in the ODE using the linear approximation.

        The F term captures the interaction between the shear function and the
        shape function and is modeled as the following linear expression (as
        the standard Thwaites' method does)

        .. math:: F\left(\lambda\right)=0.45-6\lambda

        Parameters
        ----------
        x : array-like
            Streamwise location of current step.
        delta_m2_on_nu : array-like
            Dependent variable in the ODE solver.

        Returns
        -------
        array-like same shape as `x`
            The calculated value of :math:`F`
        """
        lam = self._calc_lambda(x, delta_m2_on_nu)
        a = 0.45
        b = 6
        return a - b*lam


class ThwaitesMethodNonlinear(ThwaitesMethod):
    r"""
    Laminar boundary layer model using Thwaites' Method using exact ODE.

    Solves the original ODE from Thwaites' Method (1949) without the linear
    approximation when provided the edge velocity profile. There are a few
    different ways of modeling the tabular data from Thwaites original work
    that can be set.

    This class solves the following differential equation using the data fits
    for the shear function, :math:`S`, and the shape function, :math:`H`, to
    capture a more accurate representation of the laminar boundary layer flow

    .. math::
        \frac{d}{dx}\left(\frac{\delta_m^2}{\nu}\right)
            =\frac{2}{U_e}\left[S-\lambda\left(H+2\right)\right]

    using the :class:`IBLMethod` ODE solver.
    """

    def _calc_f(self, x: np_type.NDArray,
                delta_m2_on_nu: np_type.NDArray) -> np_type.NDArray:
        r"""
        Calculate the :math:`F` term in the ODE using the actual relationship.

        The F term captures the interaction between the shear function and the
        shape function and is modeled as the original ODE expression from
        Thwaites' paper as

        .. math:: F\left(\lambda\right)=2\left[S-\lambda\left(H+2\right)\right]

        Parameters
        ----------
        x : array-like
            Streamwise location of current step.
        delta_m2_on_nu : array-like
            Dependent variable in the ODE solver.

        Returns
        -------
        array-like same shape as `x`
            The calculated value of :math:`F`
        """
        lam = self._calc_lambda(x, delta_m2_on_nu)
        return self._model.f(lam)


class _ThwaitesFunctions:
    """Base class for curve fits for Thwaites data."""

    def __init__(self, name: str, shear_fun: Callable, shape_fun: Callable,
                 shape_p_fun: Callable, lambda_min: float,
                 lambda_max: float) -> None:
        self._range = [lambda_min, lambda_max]
        self._name = name
        self._shape_fun = shape_fun
        self._shape_p_fun = shape_p_fun
        self._shear_fun = shear_fun

    def range(self) -> Tuple[float, float]:
        """Return a 2-tuple for the start and end of range."""
        return self._range[0], self._range[1]

    def shape(self, lam: InputParam) -> np_type.NDArray:
        """Return the shape factor term."""
        return self._shape_fun(self._check_range(lam))

    def shape_p(self, lam: InputParam) -> np_type.NDArray:
        """Return the derivative of the shape factor term."""
        return self._shape_p_fun(self._check_range(lam))

    def shear(self, lam: InputParam) -> np_type.NDArray:
        """Return the shear term."""
        return self._shear_fun(self._check_range(lam))

    def f(self, lam: InputParam) -> np_type.NDArray:
        """Return the F term."""
        return 2*(self.shear(lam) - lam*(self.shape(lam)+2))

    def get_name(self) -> str:
        """Return name of function set."""
        return self._name

    def _check_range(self, lam: InputParam) -> InputParam:
        lam_min, lam_max = self.range()
        lam_local = np.array(lam)

        if (lam_local < lam_min).any():
            lam_local[lam_local < lam_min] = lam_min
#            raise ValueError("Cannot pass value less than {} into this "
#                             "function: {}".format(lam_min, lam))
        elif (lam_local > lam_max).any():
            lam_local[lam_local > lam_max] = lam_max
#            raise ValueError("Cannot pass value greater than {} into this "
#                             "function: {}".format(lam_max, lam))
        return lam_local


class _ThwaitesFunctionsWhite(_ThwaitesFunctions):
    """Returns White's calculation of Thwaites functions."""

    def __init__(self) -> None:
        def shear(lam: InputParam) -> InputParam:
            return pow(lam + 0.09, 0.62)

        def shape(lam: InputParam) -> InputParam:
            z = 0.25 - lam
            return 2 + z*(4.14 + z*(-83.5 + z*(854 + z*(-3337 + z*4576))))

        def shape_p(lam: InputParam) -> InputParam:
            z = 0.25 - lam
            return -(4.14 + z*(-2*83.5 + z*(3*854 + z*(-4*3337 + z*5*4576))))

        super().__init__("White", shear, shape, shape_p, -0.09, np.inf)


class _ThwaitesFunctionsCebeciBradshaw(_ThwaitesFunctions):
    """Returns Cebeci and Bradshaw's calculation of Thwaites functions."""

    def __init__(self) -> None:
        def shear(lam: InputParam) -> InputParam:
            return np.piecewise(lam, [lam < 0, lam >= 0],
                                [lambda lam: (0.22 + 1.402*lam
                                              + 0.018*lam/(0.107 + lam)),
                                 lambda lam: 0.22 + 1.57*lam - 1.8*lam**2])

        def shape(lam: InputParam) -> InputParam:
            # NOTE: C&B's H function is not continuous at lam=0,
            #       so using second interval
            return np.piecewise(lam, [lam < 0, lam >= 0],
                                [lambda lam: 2.088 + 0.0731/(0.14 + lam),
                                 lambda lam: 2.61 - 3.75*lam + 5.24*lam**2])

        def shape_p(lam: InputParam) -> InputParam:
            # NOTE: C&B's H function is not continuous at lam=0,
            #       so using second interval
            return np.piecewise(lam, [lam < 0, lam >= 0],
                                [lambda lam: -0.0731/(0.14 + lam)**2,
                                 lambda lam: -3.75 + 2*5.24*lam])

        super().__init__("Cebeci and Bradshaw", shear, shape, shape_p,
                         -0.1, 0.1)


class _ThwaitesFunctionsDrela(_ThwaitesFunctions):
    """Returns Drela's calculation of Thwaites functions."""

    def __init__(self) -> None:
        def shear(lam: InputParam) -> InputParam:
            return 0.220 + 1.52*lam - 5*lam**3 - 0.072*lam**2/(lam+0.18)**2

        def shape(lam: InputParam) -> InputParam:
            return 2.61 - 4.1*lam + 14*lam**3 + 0.56*lam**2/(lam+0.18)**2

        def shape_p(lam: InputParam) -> InputParam:
            return -4.1 + 42*lam**2 + 0.2016*lam/(lam+0.18)**3

        super().__init__("Drela", shear, shape, shape_p, -0.09, np.inf)


class _ThwaitesFunctionsSpline(_ThwaitesFunctions):
    """Returns cubic splines of Thwaites tables based on Edland 2021."""

    def __init__(self) -> None:
        # Spline fits to Thwaites original data Edland
        shear = CubicSpline(self._tab_lambda, self._tab_shear)
        shape = CubicSpline(self._tab_lambda, self._tab_shape)
        shape_p = shape.derivative()

        super().__init__("Thwaites Splines", shear, shape, shape_p,
                         np.min(self._tab_lambda), np.max(self._tab_lambda))

    # Tabular data section
    _tab_f = np.array([0.938, 0.953, 0.956, 0.962, 0.967, 0.969, 0.971, 0.970,
                       0.963, 0.952, 0.936, 0.919, 0.902, 0.886, 0.854, 0.825,
                       0.797, 0.770, 0.744, 0.691, 0.640, 0.590, 0.539, 0.490,
                       0.440, 0.342, 0.249, 0.156, 0.064,-0.028,-0.138,-0.251,
                      -0.362, -0.702, -1.000])
    _tab_shear = np.array([0.000, 0.011, 0.016, 0.024, 0.030, 0.035, 0.039,
                           0.049, 0.055, 0.067, 0.076, 0.083, 0.089, 0.094,
                           0.104, 0.113, 0.122, 0.130, 0.138, 0.153, 0.168,
                           0.182, 0.195, 0.208, 0.220, 0.244, 0.268, 0.291,
                           0.313, 0.333, 0.359, 0.382,0.404, 0.463, 0.500])
    _tab_shape = np.array([3.70, 3.69, 3.66, 3.63, 3.61, 3.59, 3.58, 3.52,
                           3.47, 3.38, 3.30, 3.23, 3.17, 3.13, 3.05, 2.99,
                           2.94, 2.90, 2.87, 2.81, 2.75, 2.71, 2.67, 2.64,
                           2.61, 2.55, 2.49, 2.44, 2.39, 2.34, 2.28, 2.23,
                           2.18, 2.07, 2.00])
    _tab_lambda = np.array([-0.082,-0.0818,-0.0816,-0.0812,-0.0808,-0.0804,
                            -0.080,-0.079, -0.078, -0.076, -0.074, -0.072,
                            -0.070,-0.068, -0.064, -0.060, -0.056, -0.052,
                            -0.048,-0.040, -0.032, -0.024, -0.016, -0.008,
                            +0.000, 0.016,  0.032,  0.048,  0.064,  0.080,
                            +0.10,  0.12,   0.14,   0.20,   0.25])


class _ThwaitesSeparationEvent(TermEvent):
    """
    Detects separation and will terminate integration when it occurs.

    This is a callable object that the ODE integrator will use to determine if
    the integration should terminate before the end location.
    """

    # Attributes
    # ----------
    #    _calc_lam: Callable that can calculate lambda.
    #    _S_fun: Callable that can calculate the shear function.
    def __init__(self, calc_lam: Callable, shear_fun: Callable) -> None:
        super().__init__()
        self._calc_lam = calc_lam
        self._shear_fun = shear_fun

    def _call_impl(self, x: float, f: np_type.NDArray) -> float:
        """
        Help determine if Thwaites method integrator should terminate.

        This will terminate once the shear function goes negative.

        Parameters
        ----------
        x : float
            Current x-location of the integration.
        f : numpy.ndarray
            Current step square of momentum thickness divided by the
            kinematic viscosity.

        Returns
        -------
        float
            Current value of the shear function.
        """
        return self._shear_fun(self._calc_lam(x, f))

    def event_info(self) -> Tuple[TermReason, str]:
        return TermReason.SEPARATED, ""

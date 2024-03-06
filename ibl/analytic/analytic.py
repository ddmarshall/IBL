"""
Analytic solutions to the boundary layer equations.

This module provides analytic solutions to the boundary layer equations.
These solutions can then be used to obtain a variety of properties
associated with the boundary layer flow. Typically, conditions at the edge
of the boundary layer (and perhaps conditions on the wall) will need to be
provided in order to obtain a solution.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.integrate import quadrature
from scipy.optimize import root_scalar

from ibl.typing import InputParam, SolutionFunc


class Blasius:
    """
    Solution to Blasius equation.

    This class represents the solution to the Blasius equation. While it can
    be initialized with user defined parameters needed for the solution, the
    default parameters are sufficient to obtain an accurate solution.

    Once the solution is obtained, the dense output from the ODE integrator is
    used to report back parameters associated with the boundary layer. Both
    integrated and point properties can be obtained from the similarity
    coordinate or from the corresponding Cartesian coordinates.

    Raises
    ------
    ValueError
        If properties are being set outside of the valid range.
    """

    def __init__(self, u_ref: float, nu_ref: float,
                 f_pp0: Optional[float] = None,
                 eta_inf: Optional[float] = None) -> None:
        """
        Initialize class.

        Parameters
        ----------
        u_ref : float
            Initial value of reference velocity. Must be positive.
        nu_ref : float
            Initial value of reference kinematic viscosity. Must be positive.
        fpp0 : float, optional
            Initial condition for PDE solution, by default 0.469.... Must be
            positive.
        eta_inf : float, optional
            Maximum similarity coordinate, by default 10.0. Must be positive.
        """
        # need to get the class in default state
        self._f: Optional[SolutionFunc] = None
        self._eta_inf = 0.0

        self.u_ref = u_ref
        self.nu_ref = nu_ref
        self.f_pp0 = f_pp0  # type: ignore [assignment]
        self.eta_inf = eta_inf  # type: ignore [assignment]

    @property
    def u_ref(self) -> float:
        """
        Reference velocity used in non-dimensionalization.
        Must be positive.
        """
        return self._u_ref

    @u_ref.setter
    def u_ref(self, u_ref: float) -> None:
        if u_ref <= 0:
            raise ValueError(f"Invalid edge velocity value: {u_ref}")

        self._u_ref = u_ref

    @property
    def nu_ref(self) -> float:
        """
        Reference kinematic viscosity used in non-dimensionalization.
        Must be positive.
        """
        return self._nu_ref

    @nu_ref.setter
    def nu_ref(self, nu_ref: float) -> None:
        if nu_ref <= 0:
            raise ValueError(f"Invalid kinematic viscosity value: {nu_ref}")

        self._nu_ref = nu_ref

    @property
    def f_pp0(self) -> float:
        """
        Initial condition used for PDE solution. Default is approximately
        0.469.... Must be positive.
        """
        if self._f is None:
            return np.inf
        return float(self.f_pp(0.0))

    @f_pp0.setter
    def f_pp0(self, f_pp0: Optional[float]) -> None:
        if (f_pp0 is not None) and (f_pp0 <= 0):
            raise ValueError("Invalid solution initial condition value: "
                             f"{f_pp0}")

        self._calculate_solution(f_pp0=f_pp0, eta_inf=self._eta_inf,
                                 force_resolve=False)

    @property
    def eta_inf(self) -> float:
        """
        Maximum similarity coordinate. Default is 10.0.
        Must be positive.
        """
        return self._eta_inf

    @eta_inf.setter
    def eta_inf(self, eta_inf: Optional[float]) -> None:
        if (eta_inf is not None) and (eta_inf <= 0):
            raise ValueError("Invalid maximum similarity parameter: "
                             f"{eta_inf}")

        force_resolve = False
        if self._f is None:
            f_pp0 = None
            force_resolve = True
        else:
            f_pp0 = float(self.f_pp0)

        if eta_inf is None:
            eta_inf = 10.0
            force_resolve = True

        self._calculate_solution(f_pp0=f_pp0, eta_inf=eta_inf,
                                 force_resolve=force_resolve)

    def f(self, eta: InputParam) -> InputParam:
        """
        Return the non-dimensional stream function from the solution.

        Parameters
        ----------
        eta : numpy.ndarray
            Similarity coordinates to calculate the property.

        Returns
        -------
        numpy.ndarray
            Non-dimensional stream function values.
        """
        if self._f is None:
            raise ValueError("ODE solution not set.")

        return self._f(eta)[0]

    def f_p(self, eta: InputParam) -> InputParam:
        """
        Return the non-dimensional velocity from the solution.

        Parameters
        ----------
        eta : numpy.ndarray
            Similarity coordinates to calculate the property.

        Returns
        -------
        numpy.ndarray
            Non-dimensional velocity values.
        """
        if self._f is None:
            raise ValueError("ODE solution not set.")

        return self._f(eta)[1]

    def f_pp(self, eta: InputParam) -> InputParam:
        """
        Return the derivative of the non-dimensional velocity from solution.

        Parameters
        ----------
        eta : numpy.ndarray
            Similarity coordinates to calculate the property.

        Returns
        -------
        numpy.ndarray
            Derivative of the non-dimensional velocity values.
        """
        if self._f is None:
            raise ValueError("ODE solution not set.")

        return self._f(eta)[2]

    def eta_d(self) -> float:
        """
        Return the displacement thickness in similarity coordinates.

        Returns
        -------
        float
            Displacement thickness in similarity coordinates.
        """
        return float(self.eta_inf-self.f(self.eta_inf))

    def eta_m(self) -> float:
        """
        Return the momentum thickness in similarity coordinates.

        Returns
        -------
        float
            Momentum thickness in similarity coordinates.
        """
        beta = self._get_beta()
        return float(self.f_pp0-beta*self.eta_d())/(1+beta)

    def eta_k(self) -> float:
        """
        Return the kinetic energy thickness in similarity coordinates.

        Returns
        -------
        float
            Kinetic energy thickness in similarity coordinates.
        """

        def fun(eta: float) -> float:
            if self._f is None:
                raise ValueError("ODE solution not set.")
            return 2*np.prod(self._f(eta), axis=0)

        val = quadrature(fun, 0, self.eta_inf)
        return val[0]

    def eta_s(self) -> float:
        """
        Return the shear thickness in similarity coordinates.

        Returns
        -------
        float
            Shear thickness in similarity coordinates.
        """
        def fun(eta: float) -> float:
            return float(0.99-self.f_p(eta))

        sol = root_scalar(fun, method="bisect", bracket=[0, 10])
        if not sol.converged:
            raise ValueError("Could not find shear thickness with error: "
                             + sol.flag)
        return sol.root

    def u_e(self, x: InputParam) -> InputParam:
        """
        Return the inviscid edge velocity at specified locations.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.

        Returns
        -------
        numpy.ndarray
            Edge streamwise velocity at specified locations.
        """
        # x = np.asarray(x)
        beta = self._get_beta()
        return self.u_ref*x**(beta/(2-beta))

    def v_e(self, x: InputParam) -> InputParam:
        """
        Calculate the transpiration velocity.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.

        Returns
        -------
        numpy.ndarray
            Transpiration velocity at specified locations.
        """
        return self.nu_ref*self._g(x)*self.eta_d()

    def eta(self, x: InputParam, y: InputParam) -> InputParam:
        """
        Return the similarity coordinate corresponding to coordinates.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.
        x : numpy.ndarray
            Location normal to the streamwise direction of points of interest.

        Returns
        -------
        numpy.ndarray
            Similarity coordinate at the Cartesian coordinates.

        Notes
        -----
        Both `x` and `y` must be the same shape.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        return y*self._g(x)

    def u(self, x: InputParam, y: InputParam) -> InputParam:
        """
        Return the x-velocity that corresponds to the Cartesian coordinates.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.
        x : numpy.ndarray
            Location normal to the streamwise direction of points of interest.

        Returns
        -------
        numpy.ndarray
            Velocity component in the x-direction at the coordinates.

        Notes
        -----
        Both `x` and `y` must be the same shape.
        """
        return self.u_e(x)*self.f_p(self.eta(x, y))

    def v(self, x: InputParam, y: InputParam) -> InputParam:
        """
        Return the y-velocity that corresponds to the Cartesian coordinates.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.
        x : numpy.ndarray
            Location normal to the streamwise direction of points of interest.

        Returns
        -------
        numpy.ndarray
            Velocity component in the y-direction at the coordinates.

        Notes
        -----
        Both `x` and `y` must be the same shape.
        """
        eta = self.eta(x, y)
        return -(self.nu_ref*self._g(x)
                 * (self.f(eta) + (self._get_beta()-1)*eta*self.f_p(eta)))

    def delta_d(self, x: InputParam) -> InputParam:
        """
        Calculate the displacement thickness.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.

        Returns
        -------
        numpy.ndarray
            Displacement thickness at the specified locations.
        """
        return self.eta_d()/self._g(x)

    def delta_m(self, x: InputParam) -> InputParam:
        """
        Calculate the momentum thickness.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.

        Returns
        -------
        numpy.ndarray
            Momentum thickness at the specified locations.
        """
        return self.eta_m()/self._g(x)

    def delta_k(self, x: InputParam) -> InputParam:
        """
        Calculate the kinetic energy thickness.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.

        Returns
        -------
        numpy.ndarray
            Kinetic energy thickness at the specified locations.
        """
        return self.eta_k()/self._g(x)

    def delta_s(self, x: InputParam) -> InputParam:
        """
        Calculate the shear thickness.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.

        Returns
        -------
        numpy.ndarray
            Shear thickness at the specified locations.
        """
        return self.eta_s()/self._g(x)

    def shape_d(self, x: InputParam) -> InputParam:
        """
        Calculate the displacement shape factor.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.

        Returns
        -------
        numpy.ndarray
            Displacement shape factor at the specified locations.
        """
        x = np.asarray(x)
        return (self.eta_d()/self.eta_m())*np.ones_like(x)

    def shape_k(self, x: InputParam) -> InputParam:
        """
        Calculate the kinetic energy shape factor.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.

        Returns
        -------
        numpy.ndarray
            Kinetic energy shape factor at the specified locations.
        """
        x = np.asarray(x)
        return (self.eta_k()/self.eta_m())*np.ones_like(x)

    def tau_w(self, x: InputParam, rho_ref: float) -> InputParam:
        """
        Calculate the wall shear stress.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.
        rho_ref: float
            Reference density.

        Returns
        -------
        numpy.ndarray
            Wall shear stress at the specified locations.
        """
        return rho_ref*self.nu_ref*self.u_e(x)*self._g(x)*self.f_pp0

    def dissipation(self, x: InputParam, rho_ref: float) -> InputParam:
        """
        Calculate the dissipation integral.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.
        rho_ref: float
            Reference density.

        Returns
        -------
        numpy.ndarray
            Dissipation integral at the specified locations.
        """
        diss_term = 0.5*(1+2*self._get_beta())*self.eta_k()
        return rho_ref*self.nu_ref*self._g(x)*self.u_e(x)**2*diss_term

    def _calculate_solution(self, f_pp0: Optional[float],
                            eta_inf: float, force_resolve: bool) -> None:
        """
        Set the initial condition for the ODE being solved.

        Parameters
        ----------
        f_pp0 : float | None
            Second derivative condition for the ODE.
        eta_inf : float | None
            Maximum similarity coordinate.
        force_resolve : bool
            Flag whether should resolve even if the values have not changed

        Raises
        ------
        ValueError
            If ODE solver does not result in converged solution.
        """
        resolve = force_resolve

        # check f_pp0 input to see what value to use and if it is changing
        if f_pp0 is None:
            f_pp0 = 0.46959998713136886
            resolve = True
        elif self._f is None:
            resolve = True
        elif f_pp0 != self.f_pp0:
            resolve = True

        # check eta_inf input to see what value to use and if it is changing
        if self._eta_inf != eta_inf:
            resolve = True

        # only resolve if needed
        if resolve:
            f0 = [0, 0, f_pp0]
            rtn = solve_ivp(fun=self._ode_fun, t_span=[0, eta_inf],
                            y0=f0, method="RK45", dense_output=True,
                            events=None, rtol=1e-8, atol=1e-11)

            self._f = None
            if rtn.success:
                self._f = rtn.sol
                self._eta_inf = eta_inf
            else:
                raise ValueError("Initial conditions for solver, "
                                 f"f\'\'(0)={f_pp0:.6f} and "
                                 f"eta_inf={eta_inf:.6f}, did not produce "
                                 "converged solution.")

    def _get_beta(self) -> float:
        """
        Return the wall angle term in PDE.

        For the Blasius solution the wall angle is always 0.0.

        Returns
        -------
        float
            Wall angle term.
        """
        return 0.0

    def _ode_fun(self, eta: npt.NDArray, f: npt.NDArray) -> npt.NDArray:
        """
        ODE function to be solved.

        Parameters
        ----------
        eta : numpy.ndarray
            Current similarity coordinate for solution.
        f : numpy.ndarray
            Current solution vector of ODE.

        Returns
        -------
        mypy.ndarray
            ODE derivatives.
        """
        _ = eta  # To avoid pylint unused-argument message
        f_p = np.zeros_like(f)

        f_p[0] = f[1]
        f_p[1] = f[2]
        f_p[2] = -f[0]*f[2]-self._get_beta()*(1-f[1]**2)

        return f_p

    def _g(self, x: InputParam) -> InputParam:
        """
        Calculates the transformation parameter.

        Parameters
        ----------
        x : numpy.ndarray
            Streamwise location of points of interest.

        Returns
        -------
        numpy.ndarray
            Transformation parameter at points of interest.
        """
        return np.sqrt(self.u_e(x)/((2-self._get_beta())*self.nu_ref*x))


class FalknerSkan(Blasius):
    """
    Solution to Falkner-Skan equation.

    This class represents the solution to the Falkner-Skan equation. It needs
    to perform a search for the appropriate initial condition during the
    initialization.

    Once the solution is obtained, the dense output from the ODE integrator is
    used to report back a wide variety of parameters associated with the
    boundary layer. Both integrated and point properties can be obtained from
    the similarity coordinate or from the corresponding Cartesian coordinates.

    Raises
    ------
    ValueError
        If properties are being set outside of the valid range.
    """

    def __init__(self, beta: float, u_ref: float, nu_ref: float,
                 eta_inf: Optional[float] = None) -> None:
        """
        Initialize class.

        Parameters
        ----------
        beta : float
            Inviscid wedge angle parameter. Must be in range [-0.19884, 2].
        u_ref : float
            Initial value of reference velocity. Must be positive.
        nu_ref : float
            Initial value of reference kinematic viscosity. Must be positive.
        fpp0 : float, optional
            Initial condition for PDE solution, by default 0.469.... Must be
            positive.
        eta_inf : float, optional
            Maximum similarity coordinate, by default 10.0. Must be positive.
        """
        # need to get the class in default state
        if (beta < -0.19884) or (beta > 2):
            raise ValueError(f"Invalid inviscid wedge angle parameter: {beta}")
        self._beta = beta

        # find the correct ODE intial condition
        if eta_inf is None:
            eta_inf_in = 10.0
        else:
            eta_inf_in = eta_inf
        f_pp0 = self._find_fpp0(eta_inf_in)

        # initialize base class
        super().__init__(u_ref=u_ref, nu_ref=nu_ref, f_pp0=f_pp0,
                         eta_inf=eta_inf)

    @property
    def beta(self) -> float:
        """
        Inviscid wedge angle parameter. Must be in range [-0.19884, 2].
        """
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if (beta < -0.19884) or (beta > 2):
            raise ValueError(f"Invalid inviscid wedge angle parameter: {beta}")

        if beta != self._beta:
            # find the correct ODE intial condition
            self._beta = beta
            f_pp0 = self._find_fpp0(self.eta_inf)
            self._calculate_solution(f_pp0=f_pp0, eta_inf=self.eta_inf,
                                     force_resolve=True)

    @property
    def m(self) -> float:
        """
        Edge velocity profile parameter.
        """
        beta = self.beta
        if beta == 2.0:
            return np.inf
        return beta/(2-beta)

    @m.setter
    def m(self, m: float) -> None:
        if m == np.inf:
            beta = 2.0
        else:
            beta = 2*m/(1+m)
        self.beta = beta

    def _get_beta(self) -> float:
        """
        Return the wall angle term in PDE.

        For the Blasius solution the wall angle is always 0.0.

        Returns
        -------
        float
            Wall angle term.
        """
        return self.beta

    def _find_fpp0(self, eta_inf: float) -> float:
        """
        Find the appropriate initial condition for ODE.

        Parameters
        ----------
        eta_inf : Maximum similarity coordinate. Must be positive.

        Returns
        -------
        float
            Appropriate initial condition for ODE.
        """
        def fun(fpp0: float) -> float:
            class BCEvent:
                """Bounday condition event to terminate ODE solver."""

                def __init__(self) -> None:
                    self.terminal = True

                def __call__(self, x: float, f: npt.NDArray) -> float:
                    return f[1] - 1.01

            f0 = [0, 0, fpp0]
            rtn = solve_ivp(fun=self._ode_fun,
                            t_span=[0, eta_inf], y0=f0,
                            method="RK45", dense_output=False,
                            events=BCEvent(), rtol=1e-8, atol=1e-11)
            if not rtn.success:
                raise ValueError("Could not find boundary condition")

            val = 1-rtn.y[1, -1]
            # # hack to get beta at separation to work
            # if (m < 0) and (-2e-6 < val < 0):
            #     val = 0
            return val

        # This Pade approximation is based on fitting values from White (2011)
        def calc_fpp0(beta: float) -> float:
            a = np.array([0.469600, 3.817635, 7.570524, 1.249101])
            b = np.array([5.430058, 4.203534])
            num = a[0] + a[1]*beta + a[2]*beta**2 + a[3]*beta**3
            den = 1 + b[0]*beta + b[1]*beta**2
            return num/den

        if self.beta < 0:
            x0 = calc_fpp0(self.beta)
            x1 = calc_fpp0(self.beta + 1e-3)
        else:
            x0 = calc_fpp0(self.beta - 1e-3)
            x1 = calc_fpp0(self.beta)
        sol = root_scalar(fun, x0=x0, x1=x1)
        if not sol.converged:
            raise ValueError("Root finded could not find boundary condition")
        return sol.root

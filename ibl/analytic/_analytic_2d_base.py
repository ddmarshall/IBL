"""
Provide the base classes for 2D analytic solutions.

This module provides the core functionality for the 2D analytic solutions to
the boundary layer equations.
"""


from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from scipy.optimize import root_scalar

from ibl.typing import InputParam, SolutionFunc


class Analytic2dSimilarityIncompressible(ABC):
    """
    Base implementation to the 2d similarity solutions.

    This class represents a set of generic similarity solutions to the
    incompressible boundary layer equations. The class can be initialized with 
    use defined parameters needed for the solution, or default parameters can
    be used/found.

    Once the solution is obtained, the dense output from the ODE integrator is
    used to report back parameters associated with the boundary layer. Both
    integrated and point properties can be obtained from the similarity
    coordinate or from the corresponding Cartesian coordinates.

    Attributes
    ----------
    u_ref : float
        Reference velocity.
    nu_ref : float
        Reference kinematic viscosity.
    fw_pp : float
        Initial condition for PDE solution.
    eta_inf : float
        Value for the end of the PDE integration.

    Raises
    ------
    ValueError
        If properties are being set outside of the valid range.
    """

    def __init__(self, u_ref: float, nu_ref: float, alpha: float, beta: float,
                 gamma: float, fw_pp: Optional[float] = None,
                 eta_inf: Optional[float] = None) -> None:
        # initialize input terms
        self.u_ref = u_ref
        self.nu_ref = nu_ref
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

        # initialize terms found after initialization
        self._f: Optional[SolutionFunc] = None
        self._eta_inf = -1.0
        self._eta_d = -1.0
        self._eta_m = -1.0
        self._eta_k = -1.0
        self._eta_s = -1.0

        # find the solution
        self.set_solution_parameters(eta_inf, fw_pp)

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
    def fw_pp(self) -> float:
        """
        Initial condition used for ODE solution.
        """
        if self._f is None:
            return np.inf
        return float(self.f_pp(0.0))

    @property
    def eta_inf(self) -> float:
        """
        Maximum similarity coordinate. Default value is found as part of the
        ODE solution process.
        """
        return self._eta_inf

    @property
    def eta_d(self) -> float:
        """
        The displacement thickness in similarity coordinates.
        """
        return self._eta_d

    @property
    def eta_m(self) -> float:
        """
        The momentum thickness in similarity coordinates.
        """
        return self._eta_m

    @property
    def eta_k(self) -> float:
        """
        The kinetic energy thickness in similarity coordinates.
        """
        return self._eta_k

    @property
    def eta_s(self) -> float:
        """
        The shear thickness in similarity coordinates.
        """
        return self._eta_s

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
        alpha = self._alpha
        beta = self._beta
        return self.u_ref*x**(beta/(2*alpha-beta))

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
        return self.nu_ref*self._g(x)*self.eta_d

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
        return -(self._alpha*self.nu_ref*self._g(x)
                 * (self.f(eta) + (self._beta/self._alpha
                                   - 1)*eta*self.f_p(eta)))

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
        return self.eta_d/self._g(x)

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
        return self.eta_m/self._g(x)

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
        return self.eta_k/self._g(x)

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
        return self.eta_s/self._g(x)

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
        return (self.eta_d/self.eta_m)*np.ones_like(x)

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
        return (self.eta_k/self.eta_m)*np.ones_like(x)

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
        return rho_ref*self.nu_ref*self.u_e(x)*self._g(x)*self.fw_pp

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
        diss_term = 0.5*(self._alpha+2*self._beta)*self.eta_k
        return rho_ref*self.nu_ref*self._g(x)*self.u_e(x)**2*diss_term

    def set_solution_parameters(self, eta_inf: Optional[float] = None,
                                fw_pp: Optional[float] = None) -> None:
        """
        Set the solver parameters to override the default values.

        If None is passed in to either parameter then that parameter is solved
        for, otherwise the value passed in will be used as is. This can cause
        instability and should only be used for circumstances that require the
        use of specific values.

        Parameters
        ----------
        eta_inf : Optional[float]
            Maximum similarity coordinate. Must be positive.
        fw_pp : Optional[float]
            Initial condition used for ODE solution. Default value is found as
            part of the ODE solution process. Must be positive.

        Raises
        ------
        ValueError
            If invalid value is passed in.
        """
        # Error checking
        if (eta_inf is not None) and (eta_inf <= 0):
            raise ValueError("Invalid maximum similarity parameter: "
                             + f"{eta_inf}")
        if (fw_pp is not None) and (fw_pp <= 0):
            raise ValueError("Invalid solution initial condition value: "
                             + f"{fw_pp}")

        self._calculate_solution(fw_pp=fw_pp, eta_inf=eta_inf)

        # calculate the invariant thickness now that have solution
        self._eta_d = float(self.eta_inf-self.f(self.eta_inf))
        self._eta_m = (self.fw_pp
                       - self._beta*self.eta_d)/(self._alpha+self._beta)

        def k_fun(eta: float) -> float:
            return float(2*self.f(eta)*self.f_p(eta)*self.f_pp(eta))

        self._eta_k = quad(k_fun, 0, self.eta_inf)[0]

        def s_fun(eta: float) -> float:
            return float(0.99-self.f_p(eta))

        sol = root_scalar(s_fun, method="bisect", bracket=[0, self.eta_inf])
        if not sol.converged:
            raise ValueError("Could not find shear thickness with error: "
                             + sol.flag)
        self._eta_s = sol.root

    def _calculate_solution(self, fw_pp: Optional[float],
                            eta_inf: Optional[float]) -> None:
        """
        Set the initial condition for the ODE being solved.

        This solution process follows the algorithm of Asaithambi (1997) to
        find both fw_pp and eta_inf using standard ODE solver. Then finds the
        resulting solution functions with a dense output ODE solver run.

        Parameters
        ----------
        fw_pp : float | None
            Second derivative condition for the ODE.
        eta_inf : float | None
            Maximum similarity coordinate.

        Raises
        ------
        ValueError
            If ODE solver does not result in converged solution.
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        alpha = self._alpha
        beta = self._beta
        gamma = self._gamma

        if (eta_inf is None) or (fw_pp is None):
            # setup eta_inf solution process
            if eta_inf is None:
                eta_inf_prev = 1.0
            else:
                eta_inf_prev = eta_inf

            # setup fw_pp solution process
            if fw_pp is None:
                fw_pp_prev = 1.0
            else:
                fw_pp_prev = fw_pp

            j_max = 100
            i_max = 100
            eta_inf_tol = 1e-8
            fw_pp_tol = 1e-8

            def _solve(eta_inf: float, fw_pp: float) -> Tuple[float, float]:
                """
                Calculate the ODE to solve for eta_inf and fw_pp

                Parameters
                ----------
                eta_inf : float
                    Current guess of eta_inf
                fw_pp : float
                    Current guess of fw_pp

                Returns
                -------
                float
                    F1 value at the end of the integration
                float
                    F2 value at the end of the integration

                Raises
                ------
                ValueError
                    If solver could not converge
                """

                def _ode_fun(eta: npt.NDArray, f: npt.NDArray) -> npt.NDArray:
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

                    f_p[0] = eta_inf*f[1]
                    f_p[1] = eta_inf*f[2]
                    f_p[2] = -eta_inf*(alpha*f[0]*f[2]+beta*(gamma-f[1]**2))

                    return f_p


                rtn = solve_ivp(fun=_ode_fun, t_span=[0.0, 1.0],
                                y0=[0.0, 0.0, fw_pp], method="RK45",
                                dense_output=False, events=None, rtol=1e-8,
                                atol=1e-11)
                if rtn.success:
                    return rtn.y[1,-1], rtn.y[2,-1]

                raise ValueError("Initial conditions for solver, "
                                 + f"f\'\'(0)={fw_pp:.6f} and "
                                 + f"eta_inf={eta_inf:.6f}, did not "
                                 + "produce converged solution.")


            # initialize the solution process
            j=0
            f1_prev, f2_prev = _solve(eta_inf_prev, fw_pp_prev)
            eta_inf_converged = False
            eta_inf_curr = eta_inf_prev
            if eta_inf is None:
                eta_inf_curr = 1.01*eta_inf_prev

            fw_pp_curr = 0.0
            while not eta_inf_converged and (j < j_max):
                # find fw_pp for this eta_inf
                i=0
                fw_pp_converged = False
                fw_pp_curr = fw_pp_prev
                f2_curr = 0.0
                if fw_pp is None:
                    fw_pp_curr = 1.01*fw_pp_prev
                while not fw_pp_converged and (i < i_max):
                    f1_curr, f2_curr = _solve(eta_inf_curr, fw_pp_curr)
                    fw_pp_converged = ((fw_pp_prev == fw_pp_curr)
                                    or (np.abs(1.0 - f1_curr) < fw_pp_tol))
                    if not fw_pp_converged:
                        delta_fw_pp = ((1.0 - f1_curr)*(fw_pp_curr-fw_pp_prev)
                                    /(f1_curr - f1_prev))
                        f1_prev = f1_curr
                        fw_pp_prev = fw_pp_curr
                        fw_pp_curr = fw_pp_prev + delta_fw_pp
                    i = i + 1
                eta_inf_converged = ((eta_inf_curr == eta_inf_prev)
                                    or (np.abs(f2_curr) < eta_inf_tol))
                if not eta_inf_converged:
                    delta_eta_inf = (-f2_curr*(eta_inf_curr - eta_inf_prev)
                                    /(f2_curr - f2_prev))
                    f2_prev = f2_curr
                    eta_inf_prev = eta_inf_curr
                    eta_inf_curr = eta_inf_prev + delta_eta_inf
                j = j + 1
        else:
            eta_inf_curr = eta_inf
            fw_pp_curr = fw_pp

        # must have found desired parameters
        def _ode_fun(eta: npt.NDArray, f: npt.NDArray) -> npt.NDArray:
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
            f_p[2] = -(alpha*f[0]*f[2]+beta*(gamma-f[1]**2))

            return f_p

        self._eta_inf = eta_inf_curr
        rtn = solve_ivp(fun=_ode_fun, t_span=[0.0, self.eta_inf],
                        y0=[0.0, 0.0, fw_pp_curr], method="RK45",
                        dense_output=True, events=None, rtol=1e-8,
                        atol=1e-11)
        if rtn.success:
            self._f = rtn.sol
        else:
            self._f = None
            raise ValueError("Initial conditions for solver, "
                             + f"f\'\'(0)={fw_pp_curr:.6f} and "
                             + f"eta_inf={eta_inf_curr:.6f}, did not produce "
                             + "converged solution.")

    @abstractmethod
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

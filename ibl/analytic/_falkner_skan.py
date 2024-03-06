"""
Provide the incompressible Falkner-Skan solution.

This module provides the Falkner-Skan solution to the incompressible boundary
layer equations.
"""


from typing import Optional

import numpy as np

from ._blasius import Blasius



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

    Attributes
    ----------
    u_ref : float
        Reference velocity.
    nu_ref : float
        Reference kinematic viscosity.
    beta : float
        Inviscid wedge angle parameter. Must be in range [-0.19884, 2].
    fpp0 : float
        Initial condition for PDE solution.
    eta_inf : float
        Value for the end of the PDE integration.

    Raises
    ------
    ValueError
        If properties are being set outside of the valid range.
    """

    def __init__(self, beta: float, u_ref: float, nu_ref: float,
                 f_pp0: Optional[float] = None,
                 eta_inf: Optional[float] = None) -> None:
        # need to get the class in default state
        if (beta < -0.19884) or (beta > 2):
            raise ValueError(f"Invalid inviscid wedge angle parameter: {beta}")
        self._beta = beta

        # initialize base class
        super().__init__(u_ref=u_ref, nu_ref=nu_ref, f_pp0=f_pp0,
                         eta_inf=eta_inf)

    @property
    def beta(self) -> float:
        """
        Inviscid wedge angle parameter. Must be in range [-0.19884, 2].
        """
        return self._beta

    @property
    def m(self) -> float:
        """
        Edge velocity profile parameter.
        """
        alpha = self._get_alpha()
        beta = self.beta
        if beta == 2.0:
            return np.inf
        return beta/(2*alpha-beta)

    def reset_beta(self, beta: float, eta_inf: Optional[float],
                   f_pp0: Optional[float]) -> None:
        """
        Set beta and the solver parameters to override the default values.

        If None is passed in to either parameter then that parameter is solved
        for, otherwise the value passed in will be used as is. This can cause
        instability and should only be used for circumstances that require the
        use of specific values.

        Parameters
        ----------
        beta : float
            Inviscid wedge angle parameter. Must be in range [-0.19884, 2).
        eta_inf : Optional[float]
            Maximum similarity coordinate. Must be positive.
        f_pp0 : Optional[float]
            Initial condition used for ODE solution. Default value is found as
            part of the ODE solution process. Must be positive.

        Raises
        ------
        ValueError
            If invalid value is passed in.
        """
        # Error checking
        if (beta < -0.19884) or (beta >= 2):
            raise ValueError("Invalid wedge angle parameter: "
                             f"{beta}")

        self.set_solution_parameters(f_pp0=f_pp0, eta_inf=eta_inf)

    def reset_edge_velocity_parameter(self, m: float,
                                      eta_inf: Optional[float],
                                      f_pp0: Optional[float]) -> None:
        """
        Set edge velocity param and solver params to override default values.

        If None is passed in to either parameter then that parameter is solved
        for, otherwise the value passed in will be used as is. This can cause
        instability and should only be used for circumstances that require the
        use of specific values.

        Parameters
        ----------
        m : float
            Inviscid wedge angle parameter. Must be in range [-0.19884, 2).
        eta_inf : Optional[float]
            Maximum similarity coordinate. Must be positive.
        f_pp0 : Optional[float]
            Initial condition used for ODE solution. Default value is found as
            part of the ODE solution process. Must be positive.

        Raises
        ------
        ValueError
            If invalid value is passed in.
        """
        if m == np.inf:
            beta = 2.0
        else:
            beta = 2*m/(1+m)

        self.reset_beta(beta=beta, eta_inf=eta_inf, f_pp0=f_pp0)

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

    # def _find_fpp0(self, eta_inf: float) -> float:
    #     """
    #     Find the appropriate initial condition for ODE.

    #     Parameters
    #     ----------
    #     eta_inf : Maximum similarity coordinate. Must be positive.

    #     Returns
    #     -------
    #     float
    #         Appropriate initial condition for ODE.
    #     """
    #     def fun(fpp0: float) -> float:
    #         class BCEvent:
    #             """Bounday condition event to terminate ODE solver."""

    #             def __init__(self) -> None:
    #                 self.terminal = True

    #             def __call__(self, x: float, f: npt.NDArray) -> float:
    #                 return f[1] - 1.01

    #         f0 = [0, 0, fpp0]
    #         rtn = solve_ivp(fun=self._ode_fun,
    #                         t_span=[0, eta_inf], y0=f0,
    #                         method="RK45", dense_output=False,
    #                         events=BCEvent(), rtol=1e-8, atol=1e-11)
    #         if not rtn.success:
    #             raise ValueError("Could not find boundary condition")

    #         val = 1-rtn.y[1, -1]
    #         # # hack to get beta at separation to work
    #         # if (m < 0) and (-2e-6 < val < 0):
    #         #     val = 0
    #         return val

    #     # This Pade approximation is based on fitting values from White (2011)
    #     def calc_fpp0(beta: float) -> float:
    #         a = np.array([0.469600, 3.817635, 7.570524, 1.249101])
    #         b = np.array([5.430058, 4.203534])
    #         num = a[0] + a[1]*beta + a[2]*beta**2 + a[3]*beta**3
    #         den = 1 + b[0]*beta + b[1]*beta**2
    #         return num/den

    #     if self.beta < 0:
    #         x0 = calc_fpp0(self.beta)
    #         x1 = calc_fpp0(self.beta + 1e-3)
    #     else:
    #         x0 = calc_fpp0(self.beta - 1e-3)
    #         x1 = calc_fpp0(self.beta)
    #     sol = root_scalar(fun, x0=x0, x1=x1)
    #     if not sol.converged:
    #         raise ValueError("Root finded could not find boundary condition")
    #     return sol.root

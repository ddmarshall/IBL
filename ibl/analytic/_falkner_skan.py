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

    def reset_beta(self, beta: float, eta_inf: Optional[float] = None,
                   f_pp0: Optional[float] = None) -> None:
        """
        Set beta and the solver parameters to override the default values.

        If None is passed in to either parameter then that parameter is solved
        for, otherwise the value passed in will be used as is. This can cause
        instability and should only be used for circumstances that require the
        use of specific values.

        Parameters
        ----------
        beta : float
            Inviscid wedge angle parameter. Must be in range [-0.19884, 2].
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
        if (beta < -0.19884) or (beta > 2):
            raise ValueError("Invalid wedge angle parameter: "
                             f"{beta}")
        self._beta = beta

        self.set_solution_parameters(f_pp0=f_pp0, eta_inf=eta_inf)

    def reset_m(self, m: float, eta_inf: Optional[float] = None,
                f_pp0: Optional[float] = None) -> None:
        """
        Set m and solver parameters to override default values.

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
        alpha = self._get_alpha()
        if m == np.inf:
            beta = 2.0*alpha
        else:
            beta = 2*alpha*m/(1+m)

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

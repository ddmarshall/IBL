"""
Provide the incompressible Falkner-Skan solution.

This module provides the Falkner-Skan solution to the incompressible boundary
layer equations.
"""


from typing import Optional

import numpy as np

from ibl.typing import InputParam

from ._analytic_2d_base import Analytic2dSimilarityIncompressible


class FalknerSkan(Analytic2dSimilarityIncompressible):
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
    beta : float
        Inviscid wedge angle parameter. Must be in range [-0.19884, 2].
    m : float
        Inviscid edge velocity parameter. Must be greater than 

    Raises
    ------
    ValueError
        If properties are being set outside of the valid range.
    """

    def __init__(self, beta: float, u_ref: float, nu_ref: float,
                 fw_pp: Optional[float] = None,
                 eta_inf: Optional[float] = None) -> None:
        # need to get the class in default state
        if (beta < -0.19884) or (beta > 2):
            raise ValueError(f"Invalid inviscid wedge angle parameter: {beta}")

        # initialize base class
        super().__init__(u_ref=u_ref, nu_ref=nu_ref, alpha=1.0, beta=beta,
                         gamma=1.0, fw_pp=fw_pp, eta_inf=eta_inf)

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
        if self.beta == 2.0:
            return np.inf
        return self.beta/(2 - self.beta)

    def reset_beta(self, beta: float, eta_inf: Optional[float] = None,
                   fw_pp: Optional[float] = None) -> None:
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
        fw_pp : Optional[float]
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

        self.set_solution_parameters(fw_pp=fw_pp, eta_inf=eta_inf)

    def reset_m(self, m: float, eta_inf: Optional[float] = None,
                fw_pp: Optional[float] = None) -> None:
        """
        Set m and solver parameters to override default values.

        If None is passed in to either parameter then that parameter is solved
        for, otherwise the value passed in will be used as is. This can cause
        instability and should only be used for circumstances that require the
        use of specific values.

        Parameters
        ----------
        m : float
            Inviscid wedge angle parameter. Must be greater than approximately
            -0.0904295
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
        if m == np.inf:
            beta = 2.0
        else:
            beta = 2*m/(1+m)

        self.reset_beta(beta=beta, eta_inf=eta_inf, fw_pp=fw_pp)

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
        return np.sqrt(self.u_e(x)/((2 - self.beta)*self.nu_ref*x))

"""
Provide the incompressible Blasius solution.

This module provides the Blasius solution to the incompressible boundary
layer equations.
"""


from typing import Optional

import numpy as np

from ibl.analytic import Analytic2dSimilarityIncompressible
from ibl.typing import InputParam


class Blasius(Analytic2dSimilarityIncompressible):
    """
    Solution to Blasius equation.

    This class represents the solution to the Blasius equation. While it can
    be initialized with user defined parameters needed for the solution, the
    default parameters are sufficient to obtain an accurate solution.

    Once the solution is obtained, the dense output from the ODE integrator is
    used to report back parameters associated with the boundary layer. Both
    integrated and point properties can be obtained from the similarity
    coordinate or from the corresponding Cartesian coordinates.

    Attributes
    ----------
    eta_inf_default : float
        Default value for eta_inf if no value is provided.
    fw_pp_default : float
        Default value for f\'\'(0) condition for PDE solution.

    Raises
    ------
    ValueError
        If properties are being set outside of the valid range.
    """

    def __init__(self, u_ref: float, nu_ref: float,
                 fw_pp: Optional[float] = None,
                 eta_inf: Optional[float] = None) -> None:

        # special case if no values are provided then use the hard-coded values
        # that the solver would converge to.
        self._fw_pp_default = 0.4695999888844369
        self._eta_inf_default = 7.14513231841832
        if (fw_pp is None) and (eta_inf is None):
            eta_inf = self.eta_inf_default
            fw_pp = self.fw_pp_default

        super().__init__(u_ref=u_ref, nu_ref=nu_ref, alpha=1.0, beta=0.0,
                         gamma=1.0, fw_pp=fw_pp, eta_inf=eta_inf)

    @property
    def fw_pp_default(self) -> float:
        """Default value for the f_pp at the wall."""
        return self._fw_pp_default

    @property
    def eta_inf_default(self) -> float:
        """Default value for the eta_inf."""
        return self._eta_inf_default

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
        return np.sqrt(self.u_e(x)/(2*self.nu_ref*x))

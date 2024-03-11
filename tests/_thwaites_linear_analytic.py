"""Module stores the analytic solution to Thwaites method for testing."""

from typing import Callable

import numpy as np
import numpy.typing as npt

from ibl.typing import InputParam

def fd_1f(fun: Callable, xo: InputParam, dx: float) -> InputParam:
    """Use finite differences to approximate the derivative."""
    return ((fun(xo-2*dx) - fun(xo+2*dx))/12
            - 2*(fun(xo-dx) - fun(xo+dx))/3)/dx


class ThwaitesLinearAnalytic:
    """Analytic result for power-law, linear Thwaites method."""

    def __init__(self, u_ref: float, m: float, nu: float, shape_fun: Callable,
                 shear_fun: Callable) -> None:

        self.m = m
        self.u_ref = u_ref
        self.nu = nu
        self.shape_fun = shape_fun
        self.shear_fun = shear_fun

    def v_e(self, x: InputParam) -> npt.NDArray:
        """Return the transpiration velocity."""
        ddelta_ddx = fd_1f(self.delta_d, x, 1e-5)
        return self.u_ref*x**self.m*(self.m*self.delta_d(x)/x+ddelta_ddx)

    def delta_d(self, x: InputParam) -> npt.NDArray:
        """Return the displacment thickness."""
        return self.delta_m(x)*self.shape_d(x)

    def delta_m(self, x: InputParam) -> npt.NDArray:
        """Return the momentum thickness."""
        k = np.sqrt(0.45/(5*self.m+1))
        rex_sqrt = np.sqrt(self.u_ref*x**(self.m+1)/self.nu)
        return x*k/rex_sqrt

    def shape_d(self, x: InputParam) -> npt.NDArray:
        """Return the displacement shape factor."""
        if self.m == 0:
            lam = np.zeros_like(x)
        else:
            k = np.sqrt(0.45/(5*self.m+1))
            lam = self.m*k**2*np.ones_like(x)
        return self.shape_fun(lam)

    def tau_w(self, x: InputParam, rho: float) -> npt.NDArray:
        """Return the wall shear stress."""
        k = np.sqrt(0.45/(5*self.m+1))
        rex_sqrt = np.sqrt(self.u_ref*x**(self.m+1)/self.nu)

        if self.m == 0:
            lam = np.zeros_like(x)
        else:
            lam = self.m*k**2*np.ones_like(x)
        return rho*(self.u_ref*x**self.m)**2*self.shear_fun(lam)/(k*rex_sqrt)

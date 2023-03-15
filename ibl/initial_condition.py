"""
Classes and data for set initial conditions of IBL methods.

These classes abstract out the setting of initial conditions for integral
boundary layer solution methods. These classes provide a common interface to
the parameters needed to start the integration method.

Custom initial conditions can be created, but they need to inheric from
:class:`InitialCondition` in order to be used in the :class:`IBLMethod` based
classes.
"""

from abc import ABC, abstractmethod

import numpy as np


class InitialCondition(ABC):
    """
    Determines the initial conditions for integral boundary layer solutions.

    This class provides the interface for the initial conditions of the
    integral boundary layer solutions. This class is intended to provide
    """

    def __init__(self, du_e: float, nu: float) -> None:
        self._du_e = np.inf
        self._nu = np.inf

        self.du_e = du_e
        self.nu = nu

    @property
    def du_e(self) -> float:
        """
        Rate of change of edge velocity profile at initial condition.
        """
        return self._du_e

    @du_e.setter
    def du_e(self, du_e: float) -> None:
        self._du_e = du_e

    @property
    def nu(self) -> float:
        """
        Kinematic viscosity at initial condtion.
        Must be greater than zero.
        """
        return self._nu

    @nu.setter
    def nu(self, nu: float) -> None:
        if nu <= 0:
            raise ValueError(f"Invaid kinematic viscosity: {nu}")
        self._nu = nu

    @abstractmethod
    def shape_d(self) -> float:
        """
        Return the displacement shape factor for this initial condition.

        Returns
        -------
        float
            Displacement shape factor.
        """

    @abstractmethod
    def shape_k(self) -> float:
        """
        Return the kinetic energy shape factor for this initial condition.

        Returns
        -------
        float
            Kinetic energy shape factor.
        """

    @abstractmethod
    def delta_d(self) -> float:
        """
        Return the displacement thickness for this initial condition.

        Returns
        -------
        float
            Displacement thickness.
        """

    @abstractmethod
    def delta_m(self) -> float:
        """
        Return the momentum thickness for this initial condition.

        Returns
        -------
        float
            Momentum thickness.
        """

    @abstractmethod
    def delta_k(self) -> float:
        """
        Return the kinetic energy thickness for this initial condition.

        Returns
        -------
        float
            Kinetic energy thickness.
        """


class FalknerSkanStagCondition(InitialCondition):
    """
    Returns the stagnation conditions based on the Falkner-Skan solution.

    This class returns the stagnation conditions obtained from the Falkner-Skan
    solution to the stagnation point flow.
    """

    def __init__(self, du_e: float, nu: float):
        super().__init__(du_e, nu)
        self._fpp0 = 1.23259
        self._shape_d = 2.2162
        self._shape_k = 1.6257
        self._eta_m = 0.29235

    def shape_d(self) -> float:
        """
        Return the displacement shape factor for this initial condition.

        Returns
        -------
        float
            Displacement shape factor.
        """
        return self._shape_d

    def shape_k(self) -> float:
        """
        Return the kinetic energy shape factor for this initial condition.

        Returns
        -------
        float
            Kinetic energy shape factor.
        """
        return self._shape_k

    def delta_d(self) -> float:
        """
        Return the displacement thickness for this initial condition.

        Returns
        -------
        float
            Displacement thickness.
        """
        return self.delta_m()*self.shape_d()

    def delta_m(self) -> float:
        """
        Return the momentum thickness for this initial condition.

        Returns
        -------
        float
            Momentum thickness.
        """
        return np.sqrt(self.nu*self._eta_m*self._fpp0
                       / ((self._shape_d+2)*self.du_e))

    def delta_k(self) -> float:
        """
        Return the kinetic energy thickness for this initial condition.

        Returns
        -------
        float
            Kinetic energy thickness.
        """
        return self.delta_m()*self.shape_k()


class ManualCondition(InitialCondition):
    """
    Returns the stagnation conditions from manually set conditions.

    This class returns the stagnation conditions obtained from the parameters
    provided.

    Attributes
    ----------
    del_d: float
        Displacement thickness.
    del_m: float
        Momentum thickness.
    del_k: float
        Kinetic energy thickness.
    """

    def __init__(self, delta_d: float, delta_m: float, delta_k: float):
        super().__init__(du_e=0.0, nu=1e-5)
        self.del_d = delta_d
        self.del_m = delta_m
        self.del_k = delta_k

    def shape_d(self) -> float:
        """
        Return the displacement shape factor for this initial condition.

        Returns
        -------
        float
            Displacement shape factor.
        """
        return self.del_d/self.del_m

    def shape_k(self) -> float:
        """
        Return the kinetic energy shape factor for this initial condition.

        Returns
        -------
        float
            Kinetic energy shape factor.
        """
        return self.del_k/self.del_m

    def delta_d(self) -> float:
        """
        Return the displacement thickness for this initial condition.

        Returns
        -------
        float
            Displacement thickness.
        """
        return self.del_d

    def delta_m(self) -> float:
        """
        Return the momentum thickness for this initial condition.

        Returns
        -------
        float
            Momentum thickness.
        """
        return self.del_m

    def delta_k(self) -> float:
        """
        Return the kinetic energy thickness for this initial condition.

        Returns
        -------
        float
            Kinetic energy thickness.
        """
        return self.del_k

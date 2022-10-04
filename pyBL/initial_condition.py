#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

    @abstractmethod
    def H_d(self) -> float:
        """
        Return the displacement shape factor for this initial condition.

        Returns
        -------
        float
            Displacement shape factor.
        """

    @abstractmethod
    def H_k(self) -> float:
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


class FalknerSkanStagnationCondition(InitialCondition):
    """
    Returns the stagnation conditions based on the Falkner-Skan solution.

    This class returns the stagnation conditions obtained from the Falkner-Skan
    solution to the stagnation point flow.

    Attributes
    ----------
    dU_edx: float
        Rate of change of the inviscid edge velocity at stagnation point.
    nu: float
        Kinematic viscosity.
    """

    def __init__(self, dU_edx: float, nu: float):
        self.dU_edx = dU_edx
        self.nu = nu
        self._fpp0 = 1.23259
        self._H_d = 2.2162
        self._H_k = 1.6257
        self._eta_m = 0.29235

    def H_d(self) -> float:
        """
        Return the displacement shape factor for this initial condition.

        Returns
        -------
        float
            Displacement shape factor.
        """
        return self._H_d

    def H_k(self) -> float:
        """
        Return the kinetic energy shape factor for this initial condition.

        Returns
        -------
        float
            Kinetic energy shape factor.
        """
        return self._H_k

    def delta_d(self) -> float:
        """
        Return the displacement thickness for this initial condition.

        Returns
        -------
        float
            Displacement thickness.
        """
        return self.delta_m()*self.H_d()

    def delta_m(self) -> float:
        """
        Return the momentum thickness for this initial condition.

        Returns
        -------
        float
            Momentum thickness.
        """
        return np.sqrt(self.nu*self._eta_m*self._fpp0
                       / ((self._H_d+2)*self.dU_edx))

    def delta_k(self) -> float:
        """
        Return the kinetic energy thickness for this initial condition.

        Returns
        -------
        float
            Kinetic energy thickness.
        """
        return self.delta_m()*self.H_k()


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
        self.del_d = delta_d
        self.del_m = delta_m
        self.del_k = delta_k

    def H_d(self) -> float:
        """
        Return the displacement shape factor for this initial condition.

        Returns
        -------
        float
            Displacement shape factor.
        """
        return self.del_d/self.del_m

    def H_k(self) -> float:
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

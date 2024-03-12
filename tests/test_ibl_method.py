#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:26:49 2022

@author: ddmarshall
"""


import unittest
from typing import Tuple, Optional

import numpy as np
import numpy.testing as npt
import numpy.typing as np_type

from scipy.interpolate import PchipInterpolator

from ibl.ibl_method import IBLMethod
from ibl.ibl_method import TermEvent
from ibl.ibl_method import TermReason
from ibl.ibl_method import IntegrationResult
from ibl.typing import InputParam


class TestIntegrationResult(unittest.TestCase):
    """Class to test IntegrationResult."""

    def test_string(self) -> None:
        """Test string representation."""
        iblr = IntegrationResult(x_end=2.1, f_end=np.array([1.4, -2.3]),
                                 status=TermReason.REACHED_END,
                                 message="Success", success=True)
        str_ref = ("IntegrationResult:\n"
                   "    x_end: 2.1\n"
                   "    f_end: [ 1.4 -2.3]\n"
                   "    status: Completed\n"
                   "    message: Success\n"
                   "    success: True")
        self.assertEqual(str_ref, str(iblr))


class _TestTermEvent(TermEvent):
    """
    Sample class to test the termination capabilities of the base class.

    This is a callable object that the ODE integrator will use to determine if
    the integration should terminate before the end location.

    Attributes
    ----------
        _x_kill: x-location that the integrator should stop.
    """

    def __init__(self, x_kill: float) -> None:
        self._x_kill = x_kill
        super().__init__()

    def _call_impl(self, x: float, f: InputParam) -> float:
        """
        Information used to determine if IBL test integrator should terminate.

        This will terminate once x passed specified value (x_kill) and will be
        negative before then (positive afterwards).

        Parameters
        ----------
            x : float
                Current x-location of the integration
            f : InputParam
                Current state value(s)

        Returns
        -------
            Negative value when the integration should continue, positive when
            the integration has passed the termination condition, and zero at
            the state when the integrator should terminate.
        """
        _ = f
        return x - self._x_kill

    def event_info(self) -> Tuple[TermReason, str]:
        """
        Return information about the purpose of this event.

        This is used to provide feedback as to what caused the integration to
        terminate and any other helpful information.

        Returns
        -------
        TermReason
            Value indicating reason this event would terminate integration
        str
            Extra information associated with reason for termination.

        Notes
        -----
        The value `TermReason.UNKNOWN` should be used for reasons that do
        not have an existing reason. For these termination situations the
        message can provide more details.
        """
        return TermReason.SEPARATED, ""


class IBLMethodTest(IBLMethod):
    """Generic class to test the concrete methods in IBLMethod"""

    def __init__(self, u_e=None, du_e=None, d2u_e=None,
                 x_kill: Optional[float] = None) -> None:
        # setup base class
        super().__init__(nu=1, u_e=u_e, du_e=du_e, d2u_e=d2u_e)
        self.y0 = np.array([0])

        # set up this class
        if x_kill is not None:
            self._set_kill_event(_TestTermEvent(x_kill))

    def _ode_setup(self) -> Tuple[np_type.NDArray, Optional[float],
                                  Optional[float]]:
        """
        Set the solver specific parameters.

        Returns
        -------
        np_type.NDArray
            IBL initialization array.
        Optional[float]
            Relative tolerance for ODE solver.
        Optional[float]
            Absolute tolerance for ODE solver.
        """
        return self.y0, None, None

    def _ode_impl(self, x: np_type.NDArray,
                  f: np_type.NDArray) -> np_type.NDArray:
        """
        This is the derivatives of the ODEs that are to be solved

        Parameters
        ----------
            x : numpy.ndarray
                x-location of current step
            f : numpy.ndarray
                current step state value(s)

        Returns
        -------
        numpy.ndarray
            The right-hand side of the ODE at the given state.
        """
        _ = f
        return x

    def v_e(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the transpiration velocity.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired transpiration velocity at the specified locations.
        """
        return np.zeros_like(x)

    def delta_d(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the displacement thickness.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired displacement thickness at the specified locations.
        """
        return np.zeros_like(x)

    def delta_m(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the momentum thickness.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired momentum thickness at the specified locations.
        """
        return np.zeros_like(x)

    def delta_k(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the kinetic energy thickness.

        Parameters
        ----------
        x: InputParam
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
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired displacement shape factor at the specified locations.
        """
        return np.zeros_like(x)

    def shape_k(self, x: InputParam) -> np_type.NDArray:
        """
        Calculate the kinetic energy shape factor.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.

        Returns
        -------
        numpy.ndarray
            Desired kinetic energy shape factor at the specified locations.
        """
        return np.zeros_like(x)

    def tau_w(self, x: InputParam, rho: float) -> np_type.NDArray:
        """
        Calculate the wall shear stress.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.
        rho: float
            Freestream density.

        Returns
        -------
        numpy.ndarray
            Desired wall shear stress at the specified locations.
        """
        _ = rho
        return np.zeros_like(x)

    def dissipation(self, x: InputParam, rho: float) -> np_type.NDArray:
        """
        Calculate the dissipation integral.

        Parameters
        ----------
        x: InputParam
            Streamwise loations to calculate this property.
        rho: float
            Freestream density.

        Returns
        -------
        numpy.ndarray
            Desired dissipation integral at the specified locations.
        """
        _ = rho
        return np.zeros_like(x)


class IBLMethodTestTransition(TermEvent):
    """Generic class to pass termination events during the solve method."""

    def __init__(self, f_kill: float) -> None:
        self._f_kill = f_kill
        super().__init__()

    def _call_impl(self, x: float, f: np_type.NDArray) -> float:
        """
        Information used to determine if IBL test integrator should terminate.

        This will terminate once f passed specified value (f_kill) and will be
        negative before then (positive afterwards).

        Parameters
        ----------
            x : float
                Current x-location of the integration
            f : numpy.ndarray
                Current state value(s)

        Returns
        -------
        float
            Negative value when the integration should continue, positive when
            the integration has passed the termination condition, and zero at
            the state when the integrator should terminate.
        """
        _ = x
        return f[0] - self._f_kill

    def event_info(self) -> Tuple[TermReason, str]:
        """
        Return information about the purpose of this event.

        This is used to provide feedback as to what caused the integration to
        terminate and any other helpful information.

        Returns
        -------
        TermReason
            Value indicating reason this event would terminate integration
        str
            Extra information associated with reason for termination.

        Notes
        -----
        The value `TermReason.UNKNOWN` should be used for reasons that do
        not have an existing reason. For these termination situations the
        message can provide more details.
        """
        return TermReason.TRANSITIONED, ""


class TestEdgeVelocity(unittest.TestCase):
    """Class to test various functions and curve fits for Thwaites method"""

    # define the edge velocity functions
    @staticmethod
    def u_e_fun(x: InputParam, u_ref: float, m: float) -> np_type.NDArray:
        """
        Return edge velocity.

        Parameters
        ----------
        x : InputParam
            Streamwise location.
        u_ref : float
            Scale for edge velocity model.
        m : float
            Exponent for edge velocity model.

        Returns
        -------
        numpy.ndarray
            Edge velocity.
        """
        x = np.asarray(x)
        if m == 0:
            return u_ref*np.ones_like(x)
        return u_ref*x**m

    @staticmethod
    def du_e_fun(x: InputParam, u_ref: float, m: float) -> np_type.NDArray:
        """
        Return the streamwise derivative of edge velocity.

        Parameters
        ----------
        x : InputParam
            Streamwise location.
        u_ref : float
            Scale for edge velocity model.
        m : float
            Exponent for edge velocity model.

        Returns
        -------
        numpy.ndarray
            First derivative of edge velocity.
        """
        x = np.asarray(x)
        if m == 0:
            return np.zeros_like(x)
        if m == 1:
            return u_ref*np.ones_like(x)
        return m*u_ref*x**(m-1)

    @staticmethod
    def d2u_e_fun(x: InputParam, u_ref: float, m: float) -> np_type.NDArray:
        """
        Return the streamwise second derivative of edge velocity.

        Parameters
        ----------
        x : InputParam
            Streamwise location.
        u_ref : float
            Scale for edge velocity model.
        m : float
            Exponent for edge velocity model.

        Returns
        -------
        numpy.ndarray
            Second derivative of edge velocity.
        """
        x = np.asarray(x)
        if m in (0, 1):
            return np.zeros_like(x)
        if m == 2:
            return m*u_ref*np.ones_like(x)
        return m*(m-1)*u_ref*x**(m-2)

    @staticmethod
    def d3u_e_fun(x: InputParam, u_ref: float, m: float) -> np_type.NDArray:
        """
        Return the streamwise third derivative of edge velocity.

        Parameters
        ----------
        x : InputParam
            Streamwise location.
        u_ref : float
            Scale for edge velocity model.
        m : float
            Exponent for edge velocity model.

        Returns
        -------
        numpy.ndarray
            Third derivative of edge velocity.
        """
        x = np.asarray(x)
        if m in (0, 1, 2):
            return np.zeros_like(x)
        if m == 3:
            return m*(m-1)*u_ref*np.ones_like(x)
        return m*(m-1)*(m-2)*u_ref*x**(m-3)

    def test_setting_velocity_functions(self) -> None:
        """Test setting the velocity functions."""
        # create test class with all three functions
        u_inf = 10
        m = 0.75
        iblb = IBLMethodTest(u_e=lambda x: self.u_e_fun(x, u_inf, m),
                             du_e=lambda x: self.du_e_fun(x, u_inf, m),
                             d2u_e=lambda x: self.d2u_e_fun(x, u_inf, m))

        x = np.linspace(0.1, 5, 21)
        u_e_ref = self.u_e_fun(x, u_inf, m)
        du_e_ref = self.du_e_fun(x, u_inf, m)
        d2u_e_ref = self.d2u_e_fun(x, u_inf, m)
        self.assertIsNone(npt.assert_allclose(iblb.u_e(x), u_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.du_e(x), du_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2u_e(x), d2u_e_ref))

        # create test class with two functions
        u_inf = 10
        m = 0.75
        iblb = IBLMethodTest(u_e=lambda x: self.u_e_fun(x, u_inf, m),
                             du_e=lambda x: self.du_e_fun(x, u_inf, m))

        x = np.linspace(0.1, 5, 21)
        u_e_ref = self.u_e_fun(x, u_inf, m)
        du_e_ref = self.du_e_fun(x, u_inf, m)
        d2u_e_ref = self.d2u_e_fun(x, u_inf, m)
        self.assertIsNone(npt.assert_allclose(iblb.u_e(x), u_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.du_e(x), du_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2u_e(x), d2u_e_ref))

        # create test class with one function
        u_inf = 10
        m = 0.75
        iblb = IBLMethodTest(u_e=lambda x: self.u_e_fun(x, u_inf, m))

        x = np.linspace(0.1, 5, 21)
        u_e_ref = self.u_e_fun(x, u_inf, m)
        du_e_ref = self.du_e_fun(x, u_inf, m)
        d2u_e_ref = self.d2u_e_fun(x, u_inf, m)
        self.assertIsNone(npt.assert_allclose(iblb.u_e(x), u_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.du_e(x), du_e_ref))
        # NOTE: second derivative has slightly larger errors
        self.assertIsNone(npt.assert_allclose(iblb.d2u_e(x), d2u_e_ref,
                                              rtol=1e-5, atol=0))

    def test_setting_velocity_splines(self) -> None:
        """Test setting the velocity with splines."""
        # set the edge velocity spline
        x_sample = np.linspace(0.1, 5, 8)
        u_inf = 10
        m = 1.25
        u_e = PchipInterpolator(x_sample, self.u_e_fun(x_sample, u_inf, m))
        du_e = u_e.derivative()
        d2u_e = du_e.derivative()
        iblb = IBLMethodTest(u_e=u_e)

        x = np.linspace(0.1, 5, 21)
        u_e_ref = u_e(x)
        du_e_ref = du_e(x)
        d2u_e_ref = d2u_e(x)
        self.assertIsNone(npt.assert_allclose(iblb.u_e(x), u_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.du_e(x), du_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2u_e(x), d2u_e_ref))

        # set the edge velocity derivative spline
        x_sample = np.linspace(0.1, 5, 8)
        u_inf = 10
        m = 1.25
        du_e = PchipInterpolator(x_sample, self.du_e_fun(x_sample, u_inf, m))
        u_e = du_e.antiderivative()
        u_e.c[-1,:] = u_e.c[-1,:]+self.u_e_fun(x_sample[0], u_inf, m)
        d2u_e = du_e.derivative()
        iblb = IBLMethodTest(u_e=u_e, du_e=du_e)

        x = np.linspace(0.1, 5, 21)
        u_e_ref = u_e(x)
        du_e_ref = du_e(x)
        d2u_e_ref = d2u_e(x)
        self.assertIsNone(npt.assert_allclose(iblb.u_e(x), u_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.du_e(x), du_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2u_e(x), d2u_e_ref))

        # set the edge velocity second derivative spline
        x_sample = np.linspace(0.1, 5, 8)
        u_inf = 10
        m = 1.25
        d2u_e = PchipInterpolator(x_sample, self.d2u_e_fun(x_sample, u_inf, m))
        du_e = d2u_e.antiderivative()
        du_e.c[-1,:] = du_e.c[-1,:]+self.du_e_fun(x_sample[0], u_inf, m)
        u_e = du_e.antiderivative()
        u_e.c[-1,:] = u_e.c[-1,:]+self.u_e_fun(x_sample[0], u_inf, m)
        iblb = IBLMethodTest(u_e=u_e, du_e=du_e, d2u_e=d2u_e)

        x = np.linspace(0.1, 5, 21)
        u_e_ref = u_e(x)
        du_e_ref = du_e(x)
        d2u_e_ref = d2u_e(x)
        self.assertIsNone(npt.assert_allclose(iblb.u_e(x), u_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.du_e(x), du_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2u_e(x), d2u_e_ref))

    def test_setting_velocity_points(self) -> None:
        """Test setting velocity from points."""
        # set the edge velocity values
        x_sample = np.linspace(0.1, 5, 8)
        u_inf = 10
        m = 1.25
        u_e = [x_sample, self.u_e_fun(x_sample, u_inf, m)]
        u_e_spline = PchipInterpolator(x_sample, self.u_e_fun(x_sample,
                                                              u_inf, m))
        du_e_spline = u_e_spline.derivative()
        d2u_e_spline = du_e_spline.derivative()
        iblb = IBLMethodTest(u_e=u_e)

        x = np.linspace(0.1, 5, 21)
        u_e_ref = u_e_spline(x)
        du_e_ref = du_e_spline(x)
        d2u_e_ref = d2u_e_spline(x)
        self.assertIsNone(npt.assert_allclose(iblb.u_e(x), u_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.du_e(x), du_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2u_e(x), d2u_e_ref))

    def test_setting_velocity_derivative_points(self) -> None:
        """Test setting velocity derivative from points."""
        # set the edge velocity derivative points
        x_sample = np.linspace(0.1, 5, 8)
        u_inf = 10
        m = 1.25
        u_e = self.u_e_fun(x_sample[0], u_inf, m)
        du_e = [x_sample, self.du_e_fun(x_sample, u_inf, m)]
        du_e_spline = PchipInterpolator(x_sample, self.du_e_fun(x_sample,
                                                                u_inf, m))
        u_e_spline = du_e_spline.antiderivative()
        u_e_spline.c[-1,:] = (u_e_spline.c[-1,:]
                              + self.u_e_fun(x_sample[0], u_inf, m))
        d2u_e_spline = du_e_spline.derivative()
        iblb = IBLMethodTest(u_e=u_e, du_e=du_e)

        x = np.linspace(0.1, 5, 21)
        u_e_ref = u_e_spline(x)
        du_e_ref = du_e_spline(x)
        d2u_e_ref = d2u_e_spline(x)
        self.assertIsNone(npt.assert_allclose(iblb.u_e(x), u_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.du_e(x), du_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2u_e(x), d2u_e_ref))

    def test_delay_setting_velocity(self) -> None:
        """Test setting the velocity after class creation."""
        # create test class with all three functions
        u_inf = 10
        m = 0.75
        iblb = IBLMethodTest()
        x = np.linspace(0.1, 5, 21)
        u_e_ref = self.u_e_fun(x, u_inf, m)
        du_e_ref = self.du_e_fun(x, u_inf, m)
        d2u_e_ref = self.d2u_e_fun(x, u_inf, m)

        with self.assertRaises(ValueError):
            iblb.u_e(x)
        with self.assertRaises(ValueError):
            iblb.du_e(x)
        with self.assertRaises(ValueError):
            iblb.d2u_e(x)

        iblb.set_velocity(u_e=lambda x: self.u_e_fun(x, u_inf, m),
                          du_e=lambda x: self.du_e_fun(x, u_inf, m),
                          d2u_e=lambda x: self.d2u_e_fun(x, u_inf, m))
        self.assertIsNone(npt.assert_allclose(iblb.u_e(x), u_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.du_e(x), du_e_ref))
        self.assertIsNone(npt.assert_allclose(iblb.d2u_e(x), d2u_e_ref))

    def test_terminating_solver(self) -> None:
        """Test early termination of solver."""
        u_inf = 10
        m = 1
        x_kill = 3
        iblb = IBLMethodTest(u_e=lambda x: self.u_e_fun(x, u_inf, m),
                             x_kill=x_kill)

        # test setting viscosity
        iblb.nu = 1e-5
        self.assertAlmostEqual(iblb.nu, 1e-5)
        with self.assertRaises(ValueError):
            iblb.nu = 0.0

        # go through the entire xrange
        #
        # NOTE: No need to set the velocity terms because they are not used in
        #       this basic implementation.
        # NOTE: This solves the simple differential equation y'=x
        def ref_fun(x: InputParam) -> np_type.NDArray:
            return np.array([0.5*x**2+1])

        x_start = 1
        x_end = 2
        iblb.y0 = ref_fun(x_start)
        rtn = iblb.solve(x_start, x_end)
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, 0)
        self.assertEqual(rtn.message, "Completed")
        self.assertEqual(rtn.x_end, x_end)
        self.assertIsNone(npt.assert_allclose(rtn.f_end, ref_fun(x_end)))

        # stop because solver terminated early
        x_start = 1
        x_end = x_kill + 1
        iblb.y0 = ref_fun(x_start)
        rtn = iblb.solve(x_start, x_end)
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, TermReason.SEPARATED)
        self.assertEqual(rtn.message, "Separated")
        self.assertEqual(rtn.x_end, x_kill)
        self.assertIsNone(npt.assert_allclose(rtn.f_end, ref_fun(x_kill)))

        # stop because external trigger
        x_start = 1
        x_end = x_kill + 1
        iblb.y0 = ref_fun(x_start)
        y_trans = 0.5*(iblb.y0+ref_fun(x_kill))[0]
        x_trans = np.sqrt(2*(y_trans-1))
        rtn = iblb.solve(x_start, x_end,
                         term_event=IBLMethodTestTransition(y_trans))
        self.assertTrue(rtn.success)
        self.assertEqual(rtn.status, TermReason.TRANSITIONED)
        self.assertEqual(rtn.message, "Transition")
        self.assertEqual(rtn.x_end, x_trans)
        self.assertIsNone(npt.assert_allclose(rtn.f_end, y_trans))


if __name__ == "__main__":
    unittest.main(verbosity=1)

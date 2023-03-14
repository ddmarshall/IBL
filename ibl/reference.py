"""
Reference results for boundary layer results.

This module provides classes that can provide comparison data from external
resources. Currently a small number of experimental cases from the 1968
Stanford Olympics are included, and there is a class that can read a XFOIL dump
file and report boundary layer parameters from it.

**Stanford Olympics**

The 1968 Stanford Turbulence Olympics is the informal name for the Computation
of Turbulent Boundary Layers AFOSR-IFP-Stanford Conference that met in 1968.
The proceedings are a two volume set with volume I dedicated to the numerical
methods used to predict the turbulent boundary layer behavior. Volume II is
a currated collection of high quality experimental data of a wide variety of
turbulent boundary layer flows.

**XFOIL Dump Files**

XFoil is a tightly coupled panel code and integral boundary layer solver
written by Mark Drela at MIT, (https://web.mit.edu/drela/Public/web/xfoil/).
The dump files contain all of the information needed to characterize its
solution of the integral boundary layer equation results from a solution. See
XFOIL's `documentation
file <https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt>`__ for more
information.
"""

import copy
from typing import Optional, List, Tuple
from importlib.resources import files

import numpy as np
import numpy.typing as npt

# Unit conversions
_FOOT_PER_METER = 0.3048
_METER_PER_FOOT = 1/_FOOT_PER_METER
_METER_PER_INCH = 12*_METER_PER_FOOT
_INCH_PER_METER = 1/_METER_PER_INCH


class StanfordOlympics1968StationData:
    """
    Data from Standford Olympics 1968 reported for each station in flow.

    This class is initialized with the row data associated with the
    summary table and a flag indicating whether the case is in SI units.

    Notes
    -----
    See page ix of proceedings for precise definition of each term.

    Raises
    ------
    ValueError
        If invalid data is used.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, si_unit: bool, summ_data: str, stat_summ: str,
                 stat_data: List[str]) -> None:
        """
        Initialize class.

        Parameters
        ----------
        si_unit : bool
            Flag indicating whether station data is SI or English units.
        summ_data : str
            String containing the summary data from case.
        stat_summ : str
            String containing the station specific data.
        stat_data : List[str]
            Data at y-locations in the flow.
        """

        self._x = np.inf
        self._u_e = np.inf
        self._du_e = np.inf
        self._delta_m = np.inf
        self._shape_d = np.inf
        self._shape_k = np.inf
        self._shape_eq = np.inf
        self._c_f = np.inf
        self._c_f_lt = np.inf
        self._c_f_exp = np.inf
        self._beta_eq = np.inf

        self._u_star = np.inf
        self._nu = np.inf
        self._delta_d = np.inf
        self._delta_k = np.inf
        self._delta_c = np.inf
        self._re_delta_m = np.inf
        self._re_delta_d = np.inf

        self._y_plus = np.empty(0)
        self._u_plus = np.empty(0)
        self._y = np.empty(0)
        self._u = np.empty(0)
        self._y_c = np.empty(0)
        self._u_defect = np.empty(0)

        self.reset(si_unit=si_unit, summ_data=summ_data, stat_summ=stat_summ,
                   stat_data=stat_data)

    @property
    def x(self) -> float:
        """
        Streamwise coordinate in [m] of this station.
        Original symbol: `X`.
        """
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        self._x = x

    @property
    def u_e(self) -> float:
        """
        Edge velocity in [m/s] at this station.
        Must be greater than zero.
        Original symbol: `UI`.
        """
        return self._u_e

    @u_e.setter
    def u_e(self, u_e: float) -> None:
        if u_e <= 0:
            raise ValueError(f"Invalid edge velocity: {u_e}")
        self._u_e = u_e

    @property
    def du_e(self) -> float:
        """
        Rate of change in edge velocity in [1/s] at this station.
        Original symbol: `DUI`.
        """
        return self._du_e

    @du_e.setter
    def du_e(self, du_e: float) -> None:
        self._du_e = du_e

    @property
    def delta_d(self) -> float:
        """
        Displacement thickness in [m] at this station.
        Must be greater than zero.
        Original symbol: `DELS`.
        """
        return self._delta_d

    @delta_d.setter
    def delta_d(self, delta_d: float) -> None:
        if delta_d <= 0:
            raise ValueError(f"Invalid momentum thickness: {delta_d}")
        self._delta_d = delta_d

    @property
    def delta_m(self) -> float:
        """
        Momentum thickness in [m] at this station.
        Must be greater than zero.
        Original symbol: `THETA`.
        """
        return self._delta_m

    @delta_m.setter
    def delta_m(self, delta_m: float) -> None:
        if delta_m <= 0:
            raise ValueError(f"Invalid momentum thickness: {delta_m}")
        self._delta_m = delta_m

    @property
    def delta_k(self) -> float:
        """
        Kinetic energy thickness in [m] at this station.
        Must be greater than zero.
        Original symbol: `EN TH`.
        """
        return self._delta_k

    @delta_k.setter
    def delta_k(self, delta_k: float) -> None:
        if delta_k <= 0:
            raise ValueError(f"Invalid kinetic energy thickness: {delta_k}")
        self._delta_k = delta_k

    @property
    def delta_c(self) -> float:
        """
        Clauser thickness in [m] at this station.
        Must be greater than zero.
        Original symbol: `CL TH` or 1CD`.
        """
        return self._delta_c

    @delta_c.setter
    def delta_c(self, delta_c: float) -> None:
        if delta_c <= 0:
            raise ValueError(f"Invalid Clauser thickness: {delta_c}")
        self._delta_c = delta_c

    @property
    def shape_d(self) -> float:
        """
        Displacement shape factor at this station.
        Must be greater than zero.
        Original symbol: `H`.
        """
        return self._shape_d

    @shape_d.setter
    def shape_d(self, shape_d: float) -> None:
        if shape_d <= 0:
            raise ValueError(f"Invalid displacement shape factor: {shape_d}")
        self._shape_d = shape_d

    @property
    def shape_k(self) -> float:
        """
        Kinetic energy shape factor at this station.
        Must be greater than zero.
        Original symbol `HS`.
        """
        return self._shape_k

    @shape_k.setter
    def shape_k(self, shape_k: float) -> None:
        if shape_k <= 0:
            raise ValueError(f"Invalid kinetic energy shape factor: {shape_k}")
        self._shape_k = shape_k

    @property
    def shape_eq(self) -> float:
        """
        Equilibrium shape factor at this station.
        Must be greater than zero.
        Original symbol: `G`.
        """
        return self._shape_eq

    @shape_eq.setter
    def shape_eq(self, shape_eq: float) -> None:
        if shape_eq <= 0:
            raise ValueError("Invalid kinetic energy shape factor: "
                             f"{shape_eq}")
        self._shape_eq = shape_eq

    @property
    def c_f(self) -> float:
        """
        Skin friction coefficient at this station.
        Original symbol: `CF`.
        """
        return self._c_f

    @c_f.setter
    def c_f(self, c_f: float) -> None:
        self._c_f = c_f

    @property
    def c_f_lt(self) -> float:
        """
        Skin friction coefficient from Ludwieg-Tillman formula.
        Original symbol: `CFLT`.
        """
        return self._c_f_lt

    @c_f_lt.setter
    def c_f_lt(self, c_f_lt: float) -> None:
        self._c_f_lt = c_f_lt

    @property
    def c_f_exp(self) -> float:
        """
        Skin friction coefficient reported by data originator.
        Original symbol: `CFE`.
        """
        return self._c_f_exp

    @c_f_exp.setter
    def c_f_exp(self, c_f_exp: float) -> None:
        self._c_f_exp = c_f_exp

    @property
    def beta_eq(self) -> float:
        """
        Equilibrium parameter at this station.
        Original symbol: `BETA`.
        """
        return self._beta_eq

    @beta_eq.setter
    def beta_eq(self, beta_eq: float) -> None:
        self._beta_eq = beta_eq

    @property
    def u_star(self) -> float:
        """
        Wall shear velocity in [m/s] at this station.
        Must be greater than zero.
        Original symbol: `US`.
        """
        return self._u_star

    @u_star.setter
    def u_star(self, u_star: float) -> None:
        if u_star <= 0:
            raise ValueError(f"Invalid wall shear velocity: {u_star}")
        self._u_star = u_star

    @property
    def nu(self) -> float:
        """
        Kinematic viscosity in [m^2/s] at this station.
        Must be greater than zero.
        Original symbol: V
        """
        return self._nu

    @nu.setter
    def nu(self, nu: float) -> None:
        if nu <= 0:
            raise ValueError(f"Invalid kinematic viscosity: {nu}")
        self._nu = nu

    @property
    def re_delta_d(self) -> float:
        """
        Displacement thickness Reynolds number at this station.
        Must be greater than zero.
        Original symbol: `RDELS`.
        """
        return self._re_delta_d

    @re_delta_d.setter
    def re_delta_d(self, re_delta_d: float) -> None:
        if re_delta_d <= 0:
            raise ValueError("Invalid displacement thickness Reynolds number: "
                             f"{re_delta_d}")
        self._re_delta_d = re_delta_d

    @property
    def re_delta_m(self) -> float:
        """
        Momentum thickness Reynolds number at this station.
        Must be greater than zero.
        Original symbol: `RTHETA`.
        """
        return self._re_delta_m

    @re_delta_m.setter
    def re_delta_m(self, re_delta_m: float) -> None:
        if re_delta_m <= 0:
            raise ValueError("Invalid momentum thickness Reynolds number: "
                             f"{re_delta_m}")
        self._re_delta_m = re_delta_m

    @property
    def y(self) -> npt.NDArray:
        """
        Distances from surface in [m] at this station.
        Must be greater than or equal to zero.
        Original symbol: `Y`.
        """
        return self._y

    @y.setter
    def y(self, y: npt.NDArray) -> None:
        if (y < 0).any():
            raise ValueError(f"Invalid y: {y}")
        self._y = y

    @property
    def y_plus(self) -> npt.NDArray:
        """
        Non-dimensionalized turbulence distances from surface at this station.
        Must be greater than or equal to zero.
        Original symbol: `Y+`.
        """
        return self._y_plus

    @y_plus.setter
    def y_plus(self, y_plus: npt.NDArray) -> None:
        if (y_plus < 0).any():
            raise ValueError(f"Invalid y+: {y_plus}")
        self._y_plus = y_plus

    @property
    def y_c(self) -> npt.NDArray:
        """
        Non-dimensionalized (by Clauser) distance from surface at this station.
        Must be greater than or equal to zero.
        Original symbol: `Y/CD`.
        """
        return self._y_c

    @y_c.setter
    def y_c(self, y_c: npt.NDArray) -> None:
        if (y_c < 0).any():
            raise ValueError(f"Invalid Clauser relative distance: {y_c}")
        self._y_c = y_c

    @property
    def u(self) -> npt.NDArray:
        """
        Local velocities in [m/s] at this station.
        Must be greater than or equal to zero.
        Original symbol: `U/UI`.
        """
        return self._u

    @u.setter
    def u(self, u: npt.NDArray) -> None:
        if (u < 0).any():
            raise ValueError(f"Invalid relative velocity: {u}")
        self._u = u

    @property
    def u_plus(self) -> npt.NDArray:
        """
        Non-dimensionalized local turbulence velocities at this station.
        Must be greater than or equal to zero.
        Original symbol: `U+`.
        """
        return self._u_plus

    @u_plus.setter
    def u_plus(self, u_plus: npt.NDArray) -> None:
        if (u_plus < 0).any():
            raise ValueError(f"Invalid u+: {u_plus}")
        self._u_plus = u_plus

    @property
    def u_defect(self) -> npt.NDArray:
        """
        Non-dimensionalized velocity defect at this station.
        Must be greater than or equal to zero.
        Original symbol: `UDEF`.
        """
        return self._u_defect

    @u_defect.setter
    def u_defect(self, u_defect: npt.NDArray) -> None:
        if (u_defect < 0).any():
            raise ValueError(f"Invalid defect velocity: {u_defect}")
        self._u_defect = u_defect

    def reset(self, si_unit: bool, summ_data: str, stat_summ: str,
              stat_data: List[str]) -> None:
        """
        Reset all data for this station.

        Parameters
        ----------
        si_unit : bool
            Flag indicating whether station data is SI or English units.
        summ_data : str
            String containing the summary data from case.
        stat_summ : str
            String containing the station specific data.
        stat_data : List[str]
            Data at y-locations in the flow.

        Raises
        ------
        ValueError
            If invalid data is used.
        """
        self._reset_summary_data(si_unit=si_unit, summary_data=summ_data)
        self._reset_station_summary(si_unit=si_unit, station_summary=stat_summ)
        self._reset_station_data(si_unit=si_unit, station_data=stat_data)

    def sample_count(self) -> int:
        """
        Return number of boundary layer samples at this station.

        Returns
        -------
        int
            Number of samples
        """
        return self._y.size

    def _reset_summary_data(self, si_unit: bool, summary_data: str) -> None:
        """
        Reset the summary data for this station.

        Parameters
        ----------
        si_unit : bool
            Flag indicating whether station data is SI or English units.
        summary_data : str
            String containing the summary data from case.

        Raises
        ------
        ValueError
            If invalid data is used.
        """
        # Reset values
        self._x = np.inf
        self._u_e = np.inf
        self._du_e = np.inf
        self._delta_m = np.inf
        self._shape_d = np.inf
        self._shape_k = np.inf
        self._shape_eq = np.inf
        self._c_f = np.inf
        self._c_f_lt = np.inf
        self._c_f_exp = np.inf
        self._beta_eq = np.inf

        # unpack values from string
        try:
            (x, ui, dui, theta, h, hs, g, cf, cflt,
             cfe, beta) = [float(x) for x in summary_data.split()]
        except ValueError:
            raise ValueError("Invalid number of columns in summary "
                             f"data: {summary_data}") from None

        # perform unit conversions
        if si_unit:
            theta = theta*1e-2
        else:
            x = x*_FOOT_PER_METER
            ui = ui*_FOOT_PER_METER
            theta = theta*_INCH_PER_METER

        # set values
        self.x = x
        self.u_e = ui
        self.du_e = dui
        self.delta_m = theta
        self.shape_d = h
        self.shape_k = hs
        self.shape_eq = g
        self.c_f = cf
        self.c_f_lt = cflt
        self.c_f_exp = cfe
        self.beta_eq = beta

        # calculate the derived parameters
        self.delta_d = self.shape_d*self.delta_m
        self.delta_k = self.shape_k*self.delta_m
        self.u_star = self.u_e*np.sqrt(0.5*self.c_f)

    def _reset_station_summary(self, si_unit: bool,
                               station_summary: str) -> None:
        """
        Reset the station summary data for this station.

        Parameters
        ----------
        si_unit : bool
            Flag indicating whether station data is SI or English units.
        station_summary : str
            String containing the station specific data.

        Raises
        ------
        ValueError
            If invalid data is used.
        """
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements

        # reset values
        self._nu = np.inf
        self._delta_c = np.inf
        self._re_delta_m = np.inf
        self._re_delta_d = np.inf
        if self.x == np.inf:
            self._u_star = np.inf
            self._delta_d = np.inf
            self._delta_k = np.inf
        else:
            self.delta_d = self.shape_d*self.delta_m
            self.delta_k = self.shape_k*self.delta_m
            self.u_star = self.u_e*np.sqrt(0.5*self.c_f)

        if station_summary == "":
            return

        # unpack values from string
        try:
            (x, us, ui, v, theta, dels, enth, clth, h, g, hs, rtheta,
             rdels) = [float(x) for x in station_summary.split()]
        except ValueError:
            raise ValueError("Invalid number of columns in summary "
                             f"data: {station_summary}") from None

        # perform unit conversions
        if si_unit:
            v = v*1e-4
            theta = theta*1e-2
            dels = dels*1e-2
            enth = enth*1e-2
            clth = clth*1e-2
        else:
            v = v*_FOOT_PER_METER**2
            x = x*_FOOT_PER_METER
            us = us*_FOOT_PER_METER
            ui = ui*_FOOT_PER_METER
            theta = theta*_INCH_PER_METER
            dels = dels*_INCH_PER_METER
            enth = enth*_INCH_PER_METER
            clth = clth*_INCH_PER_METER

        # check duplicate values
        if np.abs(x-self.x) >= 2e-3:
            raise ValueError(f"station summary data ({x}) does not match "
                             f"Summary data ({self.x}) for streamwise "
                             "coordinate")
        if np.abs(ui-self.u_e) >= 6e-2:
            raise ValueError(f"Station summary data ({ui}) does not match "
                             f"summary data ({self.u_e}) for edge velocity")
        if np.abs(theta-self.delta_m) >= 2e-3:
            raise ValueError(f"Station summary data ({theta}) does not match "
                             f"summary data ({self.delta_m}) for momentum "
                             "thickness")
        if np.abs(h-self.shape_d) >= 2e-3:
            raise ValueError(f"Station summary data ({h}) does not match "
                             f"summary data ({self.shape_d}) for displacement "
                             "shape factor")
        if np.abs(hs-self.shape_k) >= 2e-3:
            raise ValueError(f"Station summary data ({hs}) does not match "
                             f"summary data ({self.shape_k}) for displacement "
                             "shape factor")
        if np.abs(g-self.shape_eq) >= 2e-3:
            raise ValueError(f"Station summary data ({g}) does not match "
                             f"summary data ({self.shape_eq}) for equilibrium "
                             "shape factor")

        # check derived values
        if np.abs(dels - self.shape_d*self.delta_m) >= 5e-3:
            raise ValueError(f"Station summary data ({dels}) does not match "
                             f"summary data ({self.shape_d*self.delta_m}) for "
                             "displacement thickness")
        if np.abs(enth - self.shape_k*self.delta_m) >= 5e-3:
            raise ValueError(f"Station summary data ({enth}) does not match "
                             f"summary data ({self.shape_k*self.delta_m}) for "
                             "kinetic energy thickness")
        if np.abs(rdels - self.u_e*dels/v) >= 5e0:
            raise ValueError(f"Station summary data ({rdels}) does not match "
                             f"summary data ({self.u_e*dels/v}) for "
                             "displacement thickness Reynolds number")
        if np.abs(rtheta - self.u_e*theta/v) >= 5e0:
            raise ValueError(f"Station summary data ({rtheta}) does not match "
                             f"summary data ({self.u_e*theta/v}) for "
                             "momentum thickness Reynolds number")
        if np.abs(us - self.u_e*np.sqrt(0.5*self.c_f)) >= 5e-3:
            raise ValueError(f"Station summary data ({us}) does not match "
                             "summary data "
                             f"({self.u_e*np.sqrt(0.5*self.c_f)}) for wall "
                             "shear velocity")

        # set values
        self.x = x
        self.u_star = us
        self.u_e = ui
        self.nu = v
        self.delta_m = theta
        self.delta_d = dels
        self.delta_k = enth
        self.delta_c = clth
        self.shape_d = h
        self.shape_eq = g
        self.shape_k = hs
        self.re_delta_m = rtheta
        self.re_delta_d = rdels

    def _reset_station_data(self, si_unit: bool,
                            station_data: List[str]) -> None:
        """
        Reset the station data for this station.

        Parameters
        ----------
        si_unit : bool
            Flag indicating whether station data is SI or English units.
        station_data : List[str]
            Data at y-locations in the flow.

        Raises
        ------
        ValueError
            If invalid data is used.
        """
        # reset values
        n_station = len(station_data)
        self._y_plus = np.empty(n_station)
        self._u_plus = np.empty(n_station)
        self._y = np.empty(n_station)
        self._u = np.empty(n_station)
        self._y_c = np.empty(n_station)
        self._u_defect = np.empty(n_station)

        # unpack values from string
        for i, st in enumerate(station_data):
            try:
                (yplus, uplus, y, uoui, ycd,
                 udef) = [float(x) for x in st.split()]
                u = self.u_e*uoui
            except ValueError:
                raise ValueError(f"Invalid number of columns in station {i} "
                                 f"data: {st}") from None

            # perform unit conversions
            if si_unit:
                y = y*1e-2
            else:
                y = y*_INCH_PER_METER

            # check derived values
            if (np.abs(yplus - y*self.u_star/self.nu) > 1e0).any():
                raise ValueError(f"Station data ({yplus}) does not match "
                                 f"summary data ({y*self.u_star/self.nu}) "
                                 f"for y+ sample {i}")
            if (np.abs(ycd - y/self.delta_c) > 5e-1).any():
                raise ValueError(f"Station data ({ycd}) does not match "
                                 f"summary data ({y/self.delta_c}) "
                                 f"for Clauser distance sample {i}")
            if (np.abs(uplus - u/self.u_star) > 5e-1).any():
                raise ValueError(f"Station data ({uplus}) does not match "
                                 f"summary data ({u/self.u_star}) "
                                 f"for U+ sample {i}")
            if (np.abs(udef - (self.u_e - u)/self.u_star) > 5e-1).any():
                raise ValueError(f"Station data ({udef}) does not match "
                                 f"summary data ({(self.u_e-u)/self.u_star}) "
                                 f"for defect velocity sample {i}")

            # set values
            self.y_plus[i] = yplus
            self.u_plus[i] = uplus
            self.y[i] = y
            self.u[i] = u
            self.y_c[i] = ycd
            self.u_defect[i] = udef


class StanfordOlympics1968SmoothVel:
    """
    Smoothed velocity data information from 1968 Stanford Olympics data.

    This data is obtained by interpolating points from the smoothed velocity
    profile and velocity profile derivative curves.

    Notes
    -----
    See page ix of proceedings for precise definition of each term.
    """

    def __init__(self, si_unit: bool, data: List[str]) -> None:
        self._x = np.array([])
        self._u_e = np.array([])
        self._du_e = np.array([])

        self.reset(si_unit=si_unit, data=data)

    @property
    def x(self) -> npt.NDArray:
        """Streamwise coordinate in [m] of sampled points."""
        return self._x

    @property
    def u_e(self) -> npt.NDArray:
        """Edge velocity in [m/s] at streamwise points."""
        return self._u_e

    @property
    def du_e(self) -> npt.NDArray:
        """Rate of change of edge velocity in [1/s] at streamwise points."""
        return self._du_e

    def size(self) -> int:
        """
        Return the number of streamwise samples.

        Returns
        -------
        int
            Number of streamwise samples.
        """
        return self.x.size

    def reset(self, si_unit: bool, data: List[str]) -> None:
        """
        Reset the smoothed velocity data.

        Parameters
        ----------
        si_unit : bool
            Flag indicating whether station data is SI or Enlish units.
        data : List[str]
            Smoothed velocity data samples at streamwise location.

        Raises
        ------
        ValueError
            If invalid data is used.
        """
        # reset values
        n_vals = len(data)
        self._x = np.empty(n_vals)
        self._u_e = np.empty(n_vals)
        self._du_e = np.empty(n_vals)

        # unpack values from string
        for i, st in enumerate(data):
            try:
                x, ui, dui = [float(x) for x in st.split()]
            except ValueError:
                raise ValueError("Invalid number of columns in smoothed "
                                 f"velocity row {i} data: {st}") from None

            # perform unit conversion
            if not si_unit:
                x = x*_FOOT_PER_METER
                ui = ui*_FOOT_PER_METER

            # set values
            self.x[i] = x
            self.u_e[i] = ui
            self.du_e[i] = dui


class StanfordOlympics1968:
    """
    Interface to 1968 Stanford Olympics data.

    This class is an interface to the data from the Proceedings of the
    Computation of Turbulent Boundary Layers AFOSR-IFP-Stanford Conference in
    1968, also referred to the 1968 Stanford Olympics. The data comes from
    volume II and is a compilation of a variety of cases that were used as
    reference data.

    The experimental data consists of a number of measurements from the
    boundary layer at variety of streamwise stations along the flow. These
    measurements are both surface properties, boundary layer edge properties,
    and measurements throughout the boundary layer between the surface and
    the boundary layer edge.

    While the original reference data is a mixture of English and SI units,
    this class provides all data in SI units.

    Notes
    -----
    See page ix of proceedings for precise definition of each term.
    """
    # pylint: disable=too-many-public-methods

    def __init__(self, case: Optional[str] = None) -> None:
        """
        Initialize class.

        Parameters
        ----------
        case : Optional[str], optional
            String representation of four digit case number, by default None
        """
        self._nu_ref = np.inf
        self._smooth_vel: Optional[StanfordOlympics1968SmoothVel] = None
        self._station: List[StanfordOlympics1968StationData] = []
        if case is None:
            self.case = ""
        else:
            self.case = case

    @property
    def nu_ref(self) -> float:
        """Kinematic viscosity in [m^2/s] for case."""
        return self._nu_ref

    @property
    def case(self) -> str:
        """Four digit number corresponding to the case data."""
        return self._case

    @case.setter
    def case(self, case: str) -> None:
        """
        Change the case data that this class stores.

        This method will either reset the case data to no data if the `case`
        string is empty, or it will load the specified case from the
        collection of case files in the library.

        Parameters
        ----------
        case : str
            String representation of four digit case number

        Raises
        ------
        ValueError
            When case file to be read contains invalid data.
        """
        # reset everything
        self._nu_ref = np.inf
        self._smooth_vel = None
        self._station = []

        if case == "":
            self._case = ""
            return

        case_data = self._get_case_data(case_id=case)

        # extract case info
        case_parts = [x.rstrip().lstrip() for x in case_data[0][0].split("=")]

        if (case_parts[0] != "IDENT") and (case_parts[0] != "ident"):
            raise ValueError("Invalid case identifier line: "
                             f"{case_data[0][0]}")
        if case_parts[1] != case:
            raise ValueError(f"Case {case} does not match data file case "
                             f"{case_parts[1]}")
        self._case = case
        si_unit = case[0] == "1"

        nu_parts = [x.rstrip().lstrip() for x in case_data[0][1].split("=")]
        if (nu_parts[0] != "V") and (case_parts[0] != "v"):
            raise ValueError(f"Invalid viscosity line: {case_data[0][1]}")
        if si_unit:
            self._nu_ref = float(nu_parts[1])*1e-4
        else:
            self._nu_ref = float(nu_parts[1])*_FOOT_PER_METER**2

        # extract smoothed velocity info
        self._smooth_vel = StanfordOlympics1968SmoothVel(si_unit=si_unit,
                                                         data=case_data[2])

        # extract station data
        if len(case_data) < 4:
            for sum_d in case_data[1]:
                sod = StanfordOlympics1968StationData(si_unit=si_unit,
                                                      summ_data=sum_d,
                                                      stat_summ="",
                                                      stat_data=[])
                self._station.append(sod)
        elif len(case_data) < 5:
            for sum_d, stat_s in zip(case_data[1], case_data[3]):
                sod = StanfordOlympics1968StationData(si_unit=si_unit,
                                                      summ_data=sum_d,
                                                      stat_summ=stat_s,
                                                      stat_data=[])
                self._station.append(sod)
        else:
            for sum_d, stat_s, stat_d in zip(case_data[1], case_data[3],
                                             case_data[4:]):
                sod = StanfordOlympics1968StationData(si_unit=si_unit,
                                                      summ_data=sum_d,
                                                      stat_summ=stat_s,
                                                      stat_data=stat_d)
                self._station.append(sod)

    def station_count(self) -> int:
        """
        Return the number of stations for this case.

        Returns
        -------
        int
            Number of stations for this case.
        """
        return len(self._station)

    def station(self, i: int) -> StanfordOlympics1968StationData:
        """
        Return the station class for the specified station.

        Parameters
        ----------
        i : int
            Particular station that the information is wanted.

        Returns
        -------
        StationData
            The entire data for this desired station.

        """
        return self._station[i]

    def x_smooth(self) -> npt.NDArray:
        """
        Return streamwise locations for smoothed velocity samples in [m].

        Returns
        -------
        numpy.ndarray
            Streamwise location of samples.
        """
        if self._smooth_vel is None:
            return np.array([])
        return self._smooth_vel.x

    def u_e_smooth(self) -> npt.NDArray:
        """
        Return smoothed edge velocity in [m/s] at samples.

        Returns
        -------
        numpy.ndarray
            Smoothed edge velocity at samples.
        """
        if self._smooth_vel is None:
            return np.array([])
        return self._smooth_vel.u_e

    def du_e_smooth(self) -> npt.NDArray:
        """
        Return rate of change of smooth edge velocity in [1/s] at samples.

        Returns
        -------
        numpy.ndarray
            Rate of change of smooth edge velocity at stations.
        """
        if self._smooth_vel is None:
            return np.array([])
        return self._smooth_vel.du_e

    def x(self) -> npt.NDArray:
        """
        Return streamwise location of stations in [m].

        Returns
        -------
        numpy.ndarray
            Streamwise location of stations.
        """
        x = np.fromiter((sod.x for sod in self._station), float)

        return x

    def u_e(self) -> npt.NDArray:
        """
        Return edge velocity in [m/s] at stations.

        Returns
        -------
        numpy.ndarray
            Edge velocity at stations.
        """
        u_e = np.fromiter((sod.u_e for sod in self._station), float)

        return u_e

    def du_e(self) -> npt.NDArray:
        """
        Return rate of change of edge velocity in [1/s] at stations.

        Returns
        -------
        numpy.ndarray
            Rate of change of edge velocity at stations.
        """
        du_e = np.fromiter((sod.du_e for sod in self._station), float)

        return du_e

    def delta_d(self) -> npt.NDArray:
        """
        Return displacement thickness in [m] at stations.

        Returns
        -------
        numpy.ndarray
            Displacement thickness at stations.
        """
        delta_d = np.fromiter((sod.delta_d for sod in self._station), float)

        return delta_d

    def delta_m(self) -> npt.NDArray:
        """
        Return momentum thickness in [m] at stations.

        Returns
        -------
        numpy.ndarray
            Momentum thickness at stations.
        """
        delta_m = np.fromiter((sod.delta_m for sod in self._station), float)

        return delta_m

    def delta_k(self) -> npt.NDArray:
        """
        Return kinetic energy thickness in [m] at stations.

        Returns
        -------
        numpy.ndarray
            Kinetic energy thickness at stations.
        """
        delta_k = np.fromiter((sod.delta_k for sod in self._station), float)

        return delta_k

    def delta_c(self) -> npt.NDArray:
        """
        Return Clauser thickness in [m] at stations.

        Returns
        -------
        numpy.ndarray
            Clauser thickness at stations.
        """
        delta_c = np.fromiter((sod.delta_c for sod in self._station), float)

        return delta_c

    def shape_d(self) -> npt.NDArray:
        """
        Return displacement shape factor at stations.

        Returns
        -------
        numpy.ndarray
            Displacement shape factor at stations.
        """
        shape_d = np.fromiter((sod.shape_d for sod in self._station), float)

        return shape_d

    def shape_k(self) -> npt.NDArray:
        """
        Return kinetic energy shape factor at stations.

        Returns
        -------
        numpy.ndarray
            Kinetic energy shape factor at stations.
        """
        shape_k = np.fromiter((sod.shape_k for sod in self._station), float)

        return shape_k

    def shape_eq(self) -> npt.NDArray:
        """
        Return equilibrium shape factor at stations.

        Returns
        -------
        numpy.ndarray
            Equilibrium shape factor at stations.
        """
        shape_eq = np.fromiter((sod.shape_eq for sod in self._station), float)

        return shape_eq

    def c_f(self) -> npt.NDArray:
        """
        Return skin friction coefficient at stations.

        Returns
        -------
        numpy.ndarray
            Skin friction coefficient at stations.
        """
        c_f = np.fromiter((sod.c_f for sod in self._station), float)

        return c_f

    def c_f_lt(self) -> npt.NDArray:
        """
        Return Ludwieg-Tillman skin friction coefficient at stations.

        Returns
        -------
        numpy.ndarray
            Ludwieg-Tillman skin friction coefficient at stations.
        """
        c_f_lt = np.fromiter((sod.c_f_lt for sod in self._station), float)

        return c_f_lt

    def c_f_exp(self) -> npt.NDArray:
        """
        Return originally reported skin friction coefficient at stations.

        Returns
        -------
        numpy.ndarray
            Originally reported skin friction coefficient at stations.
        """
        c_f_exp = np.fromiter((sod.c_f_exp for sod in self._station), float)

        return c_f_exp

    def beta_eq(self) -> npt.NDArray:
        """
        Return equilibrium parameter at stations.

        Returns
        -------
        numpy.ndarray
            Equilibrium parameter at stations.
        """
        beta_eq = np.fromiter((sod.beta_eq for sod in self._station), float)

        return beta_eq

    def u_star(self) -> npt.NDArray:
        """
        Return wall shear stress velocity in [m/s] at stations.

        Returns
        -------
        numpy.ndarray
            Wall shear stress velocity at stations.
        """
        u_star = np.fromiter((sod.u_star for sod in self._station), float)

        return u_star

    def nu(self) -> npt.NDArray:
        """
        Return kinematic viscosity in [m^2/s] at stations.

        Returns
        -------
        numpy.ndarray
            Kinematic viscosity at stations.
        """
        nu = np.fromiter((sod.nu for sod in self._station), float)

        return nu

    def re_delta_d(self) -> npt.NDArray:
        """
        Return displacement thickness Reynolds number at stations.

        Returns
        -------
        numpy.ndarray
            Displacement thickness Reynolds number at stations.
        """
        re_delta_d = np.fromiter((sod.re_delta_d for sod in self._station),
                                 float)

        return re_delta_d

    def re_delta_m(self) -> npt.NDArray:
        """
        Return momentum thickness Reynolds number at stations.

        Returns
        -------
        numpy.ndarray
            Momentum thickness Reynolds number at stations.
        """
        re_delta_m = np.fromiter((sod.re_delta_m for sod in self._station),
                                 float)

        return re_delta_m

    def y(self, idx: int) -> npt.NDArray:
        """
        Return distance from surface in [m] at given station.

        Parameters
        ----------
        idx: float
            Index of station.

        Returns
        -------
        numpy.ndarray
            Distance from surface.
        """
        return self._station[idx].y

    def y_plus(self, idx: int) -> npt.NDArray:
        """
        Return non-dimensionalized turbulence distances at given station.

        Parameters
        ----------
        idx: float
            Index of station.

        Returns
        -------
        numpy.ndarray
            Non-dimensionalized turbulence distances from surface.
        """
        return self._station[idx].y_plus

    def y_c(self, idx: int) -> npt.NDArray:
        """
        Return Clauser non-dimensionalized turbulence distances at station.

        Parameters
        ----------
        idx: float
            Index of station.

        Returns
        -------
        numpy.ndarray
            Clauser non-dimensionalized turbulence distances from surface.
        """
        return self._station[idx].y_c

    def u(self, idx: int) -> npt.NDArray:
        """
        Return local velocities in [m/s] at given station.

        Parameters
        ----------
        idx: float
            Index of station.

        Returns
        -------
        numpy.ndarray
            Local velocities.
        """
        return self._station[idx].u

    def u_plus(self, idx: int) -> npt.NDArray:
        """
        Return non-dimensionalized turbulence velocities at given station.

        Parameters
        ----------
        idx: float
            Index of station.

        Returns
        -------
        numpy.ndarray
            Non-dimensionalized local turbulence velocities.
        """
        return self._station[idx].u_plus

    def u_defect(self, idx: int) -> npt.NDArray:
        """
        Return non-dimensionalized velocity defects at given station.

        Parameters
        ----------
        idx: float
            Index of station.

        Returns
        -------
        numpy.ndarray
            Non-dimensionalized velocity defects.
        """
        return self._station[idx].u_defect

    @staticmethod
    def _get_case_data(case_id: str) -> List[List[str]]:
        """
        Return the case data from file.

        Parameters
        ----------
        case_id : str
            4-digit string representing the case to be loaded

        Returns
        -------
        List[List[str]]
            List containing lists for each block of data from file.
        """

        # get the file name to import
        rel_path = f"data/stanford_olympics/1968/case {case_id}.txt"
        filename = files('ibl').joinpath(rel_path)

        # check if file exists and get contents
        contents: List[List[str]] = []
        if filename.is_file():
            # with open(filename, "r", encoding="utf8") as f:
            with filename.open("r", encoding="utf8") as f:
                buff = f.readlines()

            # split into chunks for different sections
            # buff = "".join(buff)
            chunk_idx = [idx for idx, s in enumerate(buff) if "# " in s]
            chunk_idx.append(len(buff))
            idx_prev = chunk_idx[0]
            for cidx in chunk_idx[1:]:
                if cidx > idx_prev:
                    block = buff[idx_prev+1:cidx]  # extract lines
                    block = "".join(block).split("\n")  # split into rows
                    block = list(filter(None, block))  # remove empty rows
                    contents.append(block)
                    idx_prev = cidx + 1
                else:
                    idx_prev = cidx

        return contents


class XFoilAirfoilData:
    """
    Data reported for each station on airfoil.

    This class is initialized with the row data from the dump file from XFOIL.

    Notes
    -----
    See the XFOIL documentation for details on these terms.

    Raises
    ------
    ValueError
        If invalid data is used.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, data: str) -> None:
        """
        Initialize class.

        Parameters
        ----------
        data : str
            String containing the data at this point.
        """
        self._s = np.inf
        self._x = np.inf
        self._y = np.inf
        self._u_e_rel = np.inf
        self._delta_d = np.inf
        self._delta_m = np.inf
        self._c_f = np.inf
        self._shape_d = np.inf
        self._shape_k = np.inf
        self._mom_defect = np.inf
        self._mass_defect = np.inf
        self._ke_defect = np.inf

        self.reset(data=data)

    @property
    def s(self) -> float:
        """
        Arclength distance from stagnation point along surface.
        Must be greater than or equal to zero.
        Original symbol: `s`.
        """
        return self._s

    @s.setter
    def s(self, s: float) -> None:
        if s < 0:
            raise ValueError("Invalid arclength distance from leading edge: "
                             f"{s}")
        self._s = s

    @property
    def x(self) -> float:
        """
        Chord location of point along surface.
        Original symbol: `x`.
        """
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        self._x = x

    @property
    def y(self) -> float:
        """
        Normal coordinate of point along surface.
        Original symbol: `y`.
        """
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        self._y = y

    @property
    def u_e_rel(self) -> float:
        """
        Nondimensionalized edge velocity at point along surface.
        Original symbol: `Ue/Vinf`.
        """
        return self._u_e_rel

    @u_e_rel.setter
    def u_e_rel(self, u_e_rel: float) -> None:
        self._u_e_rel = u_e_rel

    @property
    def delta_d(self) -> float:
        """
        Displacement thickness at point along surface.
        Must be greater than or equal to zero.
        Original symbol: `Dstar`.
        """
        return self._delta_d

    @delta_d.setter
    def delta_d(self, delta_d: float) -> None:
        if delta_d < 0:
            raise ValueError(f"Invalid displacement thickness: {delta_d}")
        self._delta_d = delta_d

    @property
    def delta_m(self) -> float:
        """
        Momentum thickness at point along surface.
        Must be greater than or equal to zero.
        Original symbol: `Theta`.
        """
        return self._delta_m

    @delta_m.setter
    def delta_m(self, delta_m: float) -> None:
        if delta_m < 0:
            raise ValueError(f"Invalid momentum thickness: {delta_m}")
        self._delta_m = delta_m

    @property
    def c_f(self) -> float:
        """
        Skin friction coefficient at point along surface.
        Must be greater than or equal to zero.
        Original symbol: `Cf`.
        """
        return self._c_f

    @c_f.setter
    def c_f(self, c_f: float) -> None:
        if c_f < 0:
            raise ValueError(f"Invalid skin friction coefficient: {c_f}")
        self._c_f = c_f

    @property
    def shape_d(self) -> float:
        """
        Displacement shape factor at point along surface.
        Must be greater than zero.
        Original symbol: `H`.
        """
        return self._shape_d

    @shape_d.setter
    def shape_d(self, shape_d: float) -> None:
        if shape_d <= 0:
            raise ValueError(f"Invalid displacement shape factor: {shape_d}")
        self._shape_d = shape_d

    @property
    def shape_k(self) -> float:
        """
        Kinetic energy shape factor at point along surface.
        Must be greater than zero.
        Original symbol: `H*`.
        """
        return self._shape_k

    @shape_k.setter
    def shape_k(self, shape_k: float) -> None:
        if shape_k <= 0:
            raise ValueError(f"Invalid kinetic energy shape factor: {shape_k}")
        self._shape_k = shape_k

    @property
    def mass_defect(self) -> float:
        """
        Mass defect at point along surface.
        Original symbol: `m`.
        """
        return self._mass_defect

    @mass_defect.setter
    def mass_defect(self, mass_defect: float) -> None:
        self._mass_defect = mass_defect

    @property
    def mom_defect(self) -> float:
        """
        Momentum defect at point along surface.
        Original symbol: `P`.
        """
        return self._mom_defect

    @mom_defect.setter
    def mom_defect(self, mom_defect: float) -> None:
        self._mom_defect = mom_defect

    @property
    def ke_defect(self) -> float:
        """
        Kinetic energy defect at point along surface.
        Original symbol: `K`.
        """
        return self._ke_defect

    @ke_defect.setter
    def ke_defect(self, ke_defect: float) -> None:
        self._ke_defect = ke_defect

    def reset(self, data: str) -> None:
        """
        Reset the data to new values.

        Parameters
        ----------
        data : str
            String containing new data at point, empty string resets values.

        Raises
        ------
        ValueError
            If invalid data is used.
        """
        # Reset values
        self.s = np.inf
        self.x = np.inf
        self.y = np.inf
        self.u_e_rel = np.inf
        self.delta_d = np.inf
        self.delta_m = np.inf
        self.c_f = np.inf
        self.shape_d = np.inf
        self.shape_k = np.inf
        self.mom_defect = np.inf
        self.mass_defect = np.inf
        self.ke_defect = np.inf
        if data == "":
            return

        # unpack values from string
        try:
            (s, x, y, u_e_rel, delta_d, delta_m,
             c_f, shape_d, shape_k, mom_defect,
             mass_defect, ke_defect) = [float(x) for x in data.split()]
        except ValueError:
            raise ValueError("Invalid number of columns in airfoil "
                             f"data: {data}") from None
        self.s = s
        self.x = x
        self.y = y
        self.u_e_rel = u_e_rel
        self.delta_d = delta_d
        self.delta_m = delta_m
        self.c_f = c_f
        self.shape_d = shape_d
        self.shape_k = shape_k
        self.mom_defect = mom_defect
        self.mass_defect = mass_defect
        self.ke_defect = ke_defect


class XFoilWakeData:
    """
    Data reported for each station in wake.

    This class is initialized with the row data from the dump file from XFOIL.
    """

    def __init__(self, data: str) -> None:
        """
        Initialize class.

        Parameters
        ----------
        data : str
            String containing the data at this point.
        """
        self._s = np.inf
        self._x = np.inf
        self._y = np.inf
        self._u_e_rel = np.inf
        self._delta_d = np.inf
        self._delta_m = np.inf
        self._shape_d = np.inf

        self.reset(data=data)

    @property
    def s(self) -> float:
        """
        Arclength distance from stagnation point along surface.
        Must be greater than or equal to zero.
        Original symbol: `s`.
        """
        return self._s

    @s.setter
    def s(self, s: float) -> None:
        if s < 0:
            raise ValueError("Invalid arclength distance from leading edge: "
                             f"{s}")
        self._s = s

    @property
    def x(self) -> float:
        """
        Chord location of point along surface.
        Original symbol: `x`.
        """
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        self._x = x

    @property
    def y(self) -> float:
        """
        Normal coordinate of point along surface.
        Original symbol: `y`.
        """
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        self._y = y

    @property
    def u_e_rel(self) -> float:
        """
        Nondimensionalized edge velocity at point along surface.
        Original symbol: `Ue/Vinf`.
        """
        return self._u_e_rel

    @u_e_rel.setter
    def u_e_rel(self, u_e_rel: float) -> None:
        self._u_e_rel = u_e_rel

    @property
    def delta_d(self) -> float:
        """
        Displacement thickness at point along surface.
        Must be greater than or equal to zero.
        Original symbol: `Dstar`.
        """
        return self._delta_d

    @delta_d.setter
    def delta_d(self, delta_d: float) -> None:
        if delta_d < 0:
            raise ValueError(f"Invalid displacement thickness: {delta_d}")
        self._delta_d = delta_d

    @property
    def delta_m(self) -> float:
        """
        Momentum thickness at point along surface.
        Must be greater than or equal to zero.
        Original symbol: `Theta`.
        """
        return self._delta_m

    @delta_m.setter
    def delta_m(self, delta_m: float) -> None:
        if delta_m < 0:
            raise ValueError(f"Invalid momentum thickness: {delta_m}")
        self._delta_m = delta_m

    @property
    def shape_d(self) -> float:
        """
        Displacement shape factor at point along surface.
        Must be greater than zero.
        Original symbol: `H`.
        """
        return self._shape_d

    @shape_d.setter
    def shape_d(self, shape_d: float) -> None:
        if shape_d <= 0:
            raise ValueError(f"Invalid displacement shape factor: {shape_d}")
        self._shape_d = shape_d

    def reset(self, data: str) -> None:
        """
        Reset the data to new values.

        Parameters
        ----------
        data : str
            String containing new data at point, empty string resets values.

        Raises
        ------
        ValueError
            If invalid data is used.
        """
        # Reset values
        self.s = np.inf
        self.x = np.inf
        self.y = np.inf
        self.u_e_rel = np.inf
        self.delta_d = np.inf
        self.delta_m = np.inf
        self.shape_d = np.inf
        if data == "":
            return

        # unpack values from string
        try:
            (s, x, y, u_e_rel, delta_d, delta_m, _,
             shape_d) = [float(x) for x in data.split()]
        except ValueError:
            raise ValueError("Invalid number of columns in wake "
                             f"data: {data}") from None
        self.s = s
        self.x = x
        self.y = y
        self.u_e_rel = u_e_rel
        self.delta_d = delta_d
        self.delta_m = delta_m
        self.shape_d = shape_d


class XFoilReader:
    """
    XFoil dump file reader.

    This class is an interface to the dump file from XFoil. The data for the
    airfoil and wake are read in and processed into upper flow and lower flow
    at the stagnation point (not leading edge) and the wake. Each portion is
    stored separately and the parameters are obtained separately.
    """
    # pylint: disable=too-many-public-methods

    # Attributes
    # ----------
    # _filename: File name of dump
    # _upper: Data at each upper station
    # _lower: Data at each lower station
    # _wake: Data at each wake station

    def __init__(self, filename: str = "") -> None:
        """
        Initialize class with filename.

        Parameters
        ----------
        filename : str, optional
            Name of XFoil dump file to parse, by default ""
        """
        self._filename = ""
        self._name = ""
        self._alpha = np.inf
        self._u_e = 1.0
        self._c = 1.0
        self._re = 0.0
        self._x_trans = [np.inf, np.inf]
        self._n_trans = np.inf
        self._upper: List[XFoilAirfoilData] = []
        self._lower: List[XFoilAirfoilData] = []
        self._wake: List[XFoilWakeData] = []
        self.filename = filename

    @property
    def name(self) -> str:
        """
        Name airfoil being analyzed.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def alpha(self) -> float:
        """
        Angle of attack, in radians, for case.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        self._alpha = alpha

    @property
    def c(self) -> float:
        """
        Chord length in [m] of airfoil.
        Must be greater than zero.
        """
        return self._c

    @c.setter
    def c(self, c: float) -> None:
        if c <= 0:
            raise ValueError(f"Invalid chord length: {c}")
        self._c = c

    @property
    def reynolds(self) -> float:
        """
        Reynolds number based on airfoil chord length for case.
        Must be greater than zero. If zero then the solution is assumed
        inviscid.
        """
        return self._re

    @reynolds.setter
    def reynolds(self, re: float) -> None:
        if re < 0:
            raise ValueError(f"Invalid Reynolds number: {re}")
        self._re = re

    @property
    def n_trans(self) -> float:
        """
        Amplification factor used in transition model.
        Must be greater than zero.
        """
        return self._n_trans

    @n_trans.setter
    def n_trans(self, n_trans: float) -> None:
        if n_trans <= 0:
            raise ValueError(f"Invalid amplification factor: {n_trans}")
        self._n_trans = n_trans

    @property
    def x_trans_upper(self) -> float:
        """
        Chord location of boundary layer transition on upper surface.
        """
        return self._x_trans[0]

    @x_trans_upper.setter
    def x_trans_upper(self, x_tu: float) -> None:
        self._x_trans[0] = x_tu

    @property
    def x_trans_lower(self) -> float:
        """
        Chord location of boundary layer transition on lower surface.
        """
        return self._x_trans[1]

    @x_trans_lower.setter
    def x_trans_lower(self, x_tl: float) -> None:
        self._x_trans[1] = x_tl

    @property
    def filename(self) -> str:
        """
        Name of file containing XFoil dump data.
        """
        return self._filename

    @filename.setter
    def filename(self, filename: str) -> None:
        # reset everything
        self._filename = ""
        self._name = ""
        self._alpha = np.inf
        self._u_e = 1.0
        self._c = 1.0
        self._re = 0.0
        self._x_trans = [np.inf, np.inf]
        self._n_trans = np.inf
        self._upper = []
        self._lower = []
        self._wake = []
        if filename == "":
            return

        # get rows from file
        airfoil_data, wake_data = self._get_dump_data(filename)

        # extract the upper airfoil data
        idx_lower = -1
        for idx, ad in enumerate(airfoil_data):
            new_pt = XFoilAirfoilData(ad)
            if new_pt.u_e_rel < 0:
                idx_lower = idx
                break
            self._upper.insert(0, new_pt)

        # extract the lower aifoil data
        self._lower = [XFoilAirfoilData(x) for x in airfoil_data[idx_lower:]]
        for a in self._lower:
            a.u_e_rel *= -1
            a.mass_defect *= -1
            a.ke_defect *= -1

        # find stagnation point
        if self._upper[0].u_e_rel == 0:
            stag_s = self._upper[0].s
            self._lower.insert(0, self._upper[0])
        else:
            # interpolate properties
            up = self._upper[0]
            lo = self._lower[0]
            du = up.u_e_rel + lo.u_e_rel
            frac = up.u_e_rel/du
            stag_pt = self._interpolate_aifoil_data(up, lo, frac)

            stag_s = stag_pt.s
            self._upper.insert(0, stag_pt)
            self._lower.insert(0, copy.copy(stag_pt))

        # adjust arc-length parameter to be based on leading edge
        for a in self._upper:
            a.s = stag_s - a.s
        for a in self._lower:
            a.s -= stag_s

        # extract the wake data adjust for wake to be from trailing edge
        if wake_data:
            self._wake = [XFoilWakeData(x) for x in wake_data]
            s_offset = self._wake[0].s
            for w in self._wake:
                w.s -= s_offset

    def upper_count(self) -> int:
        """
        Return number of points on the upper surface of airofil.

        Returns
        -------
        int
            Number of points on the upper surface of airfoil.
        """
        return len(self._upper)

    def lower_count(self) -> int:
        """
        Return number of points on the lower surface of airofil.

        Returns
        -------
        int
            Number of points on the lower surface of airfoil.
        """
        return len(self._lower)

    def wake_count(self) -> int:
        """
        Return number of points in the airofil wake.

        Returns
        -------
        int
            Number of points in the airfoil wake.
        """
        return len(self._wake)

    def upper(self, i: int) -> XFoilAirfoilData:
        """
        Return the specified data on the upper surface of airfoil.

        Parameters
        ----------
        i : int
            Index of point.

        Returns
        -------
        AirfoilData
            Upper airfoil surface data.
        """
        return self._upper[i]

    def lower(self, i: int) -> XFoilAirfoilData:
        """
        Return the specified data on the lower surface of airfoil.

        Parameters
        ----------
        i : int
            Index of point.

        Returns
        -------
        AirfoilData
            Lower airfoil surface data.
        """
        return self._lower[i]

    def wake(self, i: int) -> XFoilWakeData:
        """
        Return the specified airfoil wake data.

        Parameters
        ----------
        i : int
            Index of point.

        Returns
        -------
        AirfoilData
            Airfoil wake data.
        """
        return self._wake[i]

    def s_upper(self) -> npt.NDArray:
        """
        Return arc-length distances from stagnation point for upper surface.

        Returns
        -------
        numpy.ndarray
            Arc-length distances from the stagnation point for the upper
            surface.
        """
        s = []
        for sd in self._upper:
            s.append(sd.s)
        return np.array(s)

    def s_lower(self) -> npt.NDArray:
        """
        Return arc-length distances from stagnation point for lower surface.

        Returns
        -------
        numpy.ndarray
            Arc-length distances from the stagnation point for the lower
            surface.
        """
        s = []
        for sd in self._lower:
            s.append(sd.s)
        return np.array(s)

    def s_wake(self) -> npt.NDArray:
        """
        Return arc-length distances from airfoil trailing edge.

        Returns
        -------
        numpy.ndarray
            Arc-length distances from the airfoil trailing edge.
        """
        s = []
        for sd in self._wake:
            s.append(sd.s)
        return np.array(s)

    def x_upper(self) -> npt.NDArray:
        """
        Return the chord locations for upper surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Chord locations for upper surface of airfoil.
        """
        x = []
        for sd in self._upper:
            x.append(sd.x)
        return np.array(x)

    def x_lower(self) -> npt.NDArray:
        """
        Return the chord locations for lower surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Chord locations for lower surface of airfoil.
        """
        x = []
        for sd in self._lower:
            x.append(sd.x)
        return np.array(x)

    def x_wake(self) -> npt.NDArray:
        """
        Return the chord locations for airfoil wake.

        Returns
        -------
        numpy.ndarray
            Chord locations for airfoil wake.
        """
        x = []
        for sd in self._wake:
            x.append(sd.x)
        return np.array(x)

    def y_upper(self) -> npt.NDArray:
        """
        Return the normal locations for upper surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Normal locations for upper surface of airfoil.
        """
        y = []
        for sd in self._upper:
            y.append(sd.y)
        return np.array(y)

    def y_lower(self) -> npt.NDArray:
        """
        Return the normal locations for lower surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Normal locations for lower surface of airfoil.
        """
        y = []
        for sd in self._lower:
            y.append(sd.y)
        return np.array(y)

    def y_wake(self) -> npt.NDArray:
        """
        Return the normal locations for airfoil wake.

        Returns
        -------
        numpy.ndarray
            Normal locations for airfoil wake.
        """
        y = []
        for sd in self._wake:
            y.append(sd.y)
        return np.array(y)

    def u_e_rel_upper(self) -> npt.NDArray:
        """
        Return the nondimensionalized velocities for upper surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Nondimensionalized velocities for upper surface of airfoil.
        """
        u_e_rel = []
        for sd in self._upper:
            u_e_rel.append(sd.u_e_rel)
        return np.array(u_e_rel)

    def u_e_rel_lower(self) -> npt.NDArray:
        """
        Return the nondimensionalized velocities for lower surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Nondimensionalized velocities for lower surface of airfoil.
        """
        u_e_rel = []
        for sd in self._lower:
            u_e_rel.append(sd.u_e_rel)
        return np.array(u_e_rel)

    def u_e_rel_wake(self) -> npt.NDArray:
        """
        Return the nondimensionalized velocities for airfoil wake.

        Returns
        -------
        numpy.ndarray
            Nondimensionalized velocities for airfoil wake.
        """
        u_e_rel = []
        for sd in self._wake:
            u_e_rel.append(sd.u_e_rel)
        return np.array(u_e_rel)

    def delta_d_upper(self) -> npt.NDArray:
        """
        Return the displacement thicknesses for upper surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Displacement thicknesses for upper surface of airfoil.
        """
        delta_d = []
        for sd in self._upper:
            delta_d.append(sd.delta_d)
        return np.array(delta_d)

    def delta_d_lower(self) -> npt.NDArray:
        """
        Return the displacement thicknesses for lower surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Displacement thicknesses for lower surface of airfoil.
        """
        delta_d = []
        for sd in self._lower:
            delta_d.append(sd.delta_d)
        return np.array(delta_d)

    def delta_d_wake(self) -> npt.NDArray:
        """
        Return the displacement thicknesses for airfoil wake.

        Returns
        -------
        numpy.ndarray
            Displacement thicknesses for airfoil wake.
        """
        delta_d = []
        for sd in self._wake:
            delta_d.append(sd.delta_d)
        return np.array(delta_d)

    def delta_m_upper(self) -> npt.NDArray:
        """
        Return the momentum thicknesses for upper surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Momentum thicknesses for upper surface of airfoil.
        """
        delta_m = []
        for sd in self._upper:
            delta_m.append(sd.delta_m)
        return np.array(delta_m)

    def delta_m_lower(self) -> npt.NDArray:
        """
        Return the momentum thicknesses for lower surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Momentum thicknesses for lower surface of airfoil.
        """
        delta_m = []
        for sd in self._lower:
            delta_m.append(sd.delta_m)
        return np.array(delta_m)

    def delta_m_wake(self) -> npt.NDArray:
        """
        Return the momentum thicknesses for airfoil wake.

        Returns
        -------
        numpy.ndarray
            Momentum thicknesses for airfoil wake.
        """
        delta_m = []
        for sd in self._wake:
            delta_m.append(sd.delta_m)
        return np.array(delta_m)

    def delta_k_upper(self) -> npt.NDArray:
        """
        Return the kinetic energy thicknesses for upper surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Kinetic energy thicknesses for upper surface of airfoil.
        """
        delta_k = []
        for sd in self._upper:
            delta_k.append(sd.shape_k*sd.delta_m)
        return np.array(delta_k)

    def delta_k_lower(self) -> npt.NDArray:
        """
        Return the kinetic energy thicknesses for lower surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Kinetic energy thicknesses for lower surface of airfoil.
        """
        delta_k = []
        for sd in self._lower:
            delta_k.append(sd.shape_k*sd.delta_m)
        return np.array(delta_k)

    def shape_d_upper(self) -> npt.NDArray:
        """
        Return the displacement shape factor for upper surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Displacement shape factor for upper surface of airfoil.
        """
        shape_d = []
        for sd in self._upper:
            shape_d.append(sd.shape_d)
        return np.array(shape_d)

    def shape_d_lower(self) -> npt.NDArray:
        """
        Return the displacement shape factor for lower surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Displacement shape factor for lower surface of airfoil.
        """
        shape_d = []
        for sd in self._lower:
            shape_d.append(sd.shape_d)
        return np.array(shape_d)

    def shape_d_wake(self) -> npt.NDArray:
        """
        Return the displacement shape factor for airfoil wake.

        Returns
        -------
        numpy.ndarray
            Displacement shape factor for airfoil wake.
        """
        shape_d = []
        for sd in self._wake:
            shape_d.append(sd.shape_d)
        return np.array(shape_d)

    def shape_k_upper(self) -> npt.NDArray:
        """
        Return the kinetic energy shape factor for upper surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Kinetic energy shape factor for upper surface of airfoil.
        """
        shape_k = []
        for sd in self._upper:
            shape_k.append(sd.shape_k)
        return np.array(shape_k)

    def shape_k_lower(self) -> npt.NDArray:
        """
        Return the kinetic energy shape factor for lower surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Kinetic energy shape factor for lower surface of airfoil.
        """
        shape_k = []
        for sd in self._lower:
            shape_k.append(sd.shape_k)
        return np.array(shape_k)

    def c_f_upper(self) -> npt.NDArray:
        """
        Return the skin friction coefficient for upper surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Skin friction coefficient for upper surface of airfoil.
        """
        c_f = []
        for sd in self._upper:
            c_f.append(sd.c_f)
        return np.array(c_f)

    def c_f_lower(self) -> npt.NDArray:
        """
        Return the skin friction coefficient for lower surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Skin friction coefficient for lower surface of airfoil.
        """
        c_f = []
        for sd in self._lower:
            c_f.append(sd.c_f)
        return np.array(c_f)

    def mass_defect_upper(self) -> npt.NDArray:
        """
        Return the mass defect for upper surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Mass defect for upper surface of airfoil.
        """
        mass_defect = []
        for sd in self._upper:
            mass_defect.append(sd.mass_defect)
        return np.array(mass_defect)

    def mass_defect_lower(self) -> npt.NDArray:
        """
        Return the mass defect for lower surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Mass defect for lower surface of airfoil.
        """
        mass_defect = []
        for sd in self._lower:
            mass_defect.append(sd.mass_defect)
        return np.array(mass_defect)

    def mom_defect_upper(self) -> npt.NDArray:
        """
        Return the momentum defect for upper surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Momentum defect for upper surface of airfoil.
        """
        mom_defect = []
        for sd in self._upper:
            mom_defect.append(sd.mom_defect)
        return np.array(mom_defect)

    def mom_defect_lower(self) -> npt.NDArray:
        """
        Return the momentum defect for lower surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Momentum defect for lower surface of airfoil.
        """
        mom_defect = []
        for sd in self._lower:
            mom_defect.append(sd.mom_defect)
        return np.array(mom_defect)

    def ke_defect_upper(self) -> npt.NDArray:
        """
        Return the kinetic energy defect for upper surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Kinetic energy defect for upper surface of airfoil.
        """
        ke_defect = []
        for sd in self._upper:
            ke_defect.append(sd.ke_defect)
        return np.array(ke_defect)

    def ke_defect_lower(self) -> npt.NDArray:
        """
        Return the kinetic energy defect for lower surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Kinetic energy defect for lower surface of airfoil.
        """
        ke_defect = []
        for sd in self._lower:
            ke_defect.append(sd.ke_defect)
        return np.array(ke_defect)

    def _get_dump_data(self, filename: str) -> Tuple[List[str], List[str]]:
        """
        Return the dump file data.

        Parameters
        ----------
        filename : str
            Name of file containing data.

        Returns
        -------
        List[str]
            List containing rows of airfoil data
        List[str]
            List containing rows of wake data
        """

        # get data from file
        airfoil: List[str] = []
        wake: List[str] = []
        with open(filename, "r", encoding="utf8") as f:
            buff = f.readlines()

        # find the wake index (if exists)
        wake_idx = -1
        for idx, line in enumerate(buff):
            if len(line) < 80:
                wake_idx = idx
                break
        if buff[0][0] == "#":
            airfoil_start = 1
        else:
            airfoil_start = 0

        if wake_idx > 0:
            airfoil = buff[airfoil_start:wake_idx]
            wake = buff[wake_idx:]
        else:
            airfoil = buff[airfoil_start:]

        return airfoil, wake

    @staticmethod
    def _interpolate_aifoil_data(up: XFoilAirfoilData, lo: XFoilAirfoilData,
                                 frac: float) -> XFoilAirfoilData:
        stag_pt = XFoilAirfoilData(data="")

        stag_pt.u_e_rel = 0.0
        stag_pt.x = np.abs(-frac*lo.x + (1-frac)*up.x)
        stag_pt.u_e_rel = np.abs(-frac*lo.u_e_rel + (1-frac)*up.u_e_rel)
        stag_pt.s = frac*lo.s + (1-frac)*up.s
        stag_pt.y = frac*lo.y + (1-frac)*up.y
        stag_pt.delta_d = frac*lo.delta_d + (1-frac)*up.delta_d
        stag_pt.delta_m = frac*lo.delta_m + (1-frac)*up.delta_m
        stag_pt.c_f = frac*lo.c_f + (1-frac)*up.c_f
        stag_pt.shape_d = frac*lo.shape_d + (1-frac)*up.shape_d
        stag_pt.shape_k = frac*lo.shape_k + (1-frac)*up.shape_k
        stag_pt.mom_defect = frac*lo.mom_defect + (1-frac)*up.mom_defect
        stag_pt.mass_defect = frac*lo.mass_defect + (1-frac)*up.mass_defect
        stag_pt.ke_defect = frac*lo.ke_defect + (1-frac)*up.ke_defect
        return stag_pt

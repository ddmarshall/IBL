"""
Provide the class for the 1968 Stanford Olympics station data.

This module provides the class needed to represent the station data from the
experiments from the 1968 Stanford Olympics.
"""

from typing import List

import numpy as np
import numpy.typing as npt

# Unit conversions
FOOT_PER_METER = 0.3048
METER_PER_FOOT = 1/FOOT_PER_METER
METER_PER_INCH = 12*METER_PER_FOOT
INCH_PER_METER = 1/METER_PER_INCH


class StanfordOlympics1968StationData:
    """
    Data from Stanford Olympics 1968 reported for each station in flow.

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
                             + f"{shape_eq}")
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
                             + f"{re_delta_d}")
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
                             + f"{re_delta_m}")
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
                             + f"data: {summary_data}") from None

        # perform unit conversions
        if si_unit:
            theta = theta*1e-2
        else:
            x = x*FOOT_PER_METER
            ui = ui*FOOT_PER_METER
            theta = theta*INCH_PER_METER

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
                             + f"data: {station_summary}") from None

        # perform unit conversions
        if si_unit:
            v = v*1e-4
            theta = theta*1e-2
            dels = dels*1e-2
            enth = enth*1e-2
            clth = clth*1e-2
        else:
            v = v*FOOT_PER_METER**2
            x = x*FOOT_PER_METER
            us = us*FOOT_PER_METER
            ui = ui*FOOT_PER_METER
            theta = theta*INCH_PER_METER
            dels = dels*INCH_PER_METER
            enth = enth*INCH_PER_METER
            clth = clth*INCH_PER_METER

        # check duplicate values
        if np.abs(x-self.x) >= 2e-3:
            raise ValueError(f"station summary data ({x}) does not match "
                             + f"Summary data ({self.x}) for streamwise "
                             + "coordinate")
        if np.abs(ui-self.u_e) >= 6e-2:
            raise ValueError(f"Station summary data ({ui}) does not match "
                             + f"summary data ({self.u_e}) for edge velocity")
        if np.abs(theta-self.delta_m) >= 2e-3:
            raise ValueError(f"Station summary data ({theta}) does not match "
                             + f"summary data ({self.delta_m}) for momentum "
                             + "thickness")
        if np.abs(h-self.shape_d) >= 2e-3:
            raise ValueError(f"Station summary data ({h}) does not match "
                             + f"summary data ({self.shape_d}) for "
                             + "displacement shape factor")
        if np.abs(hs-self.shape_k) >= 2e-3:
            raise ValueError(f"Station summary data ({hs}) does not match "
                             + f"summary data ({self.shape_k}) for "
                             + "displacement shape factor")
        if np.abs(g-self.shape_eq) >= 2e-3:
            raise ValueError(f"Station summary data ({g}) does not match "
                             + f"summary data ({self.shape_eq}) for "
                             + "equilibrium shape factor")

        # check derived values
        if np.abs(dels - self.shape_d*self.delta_m) >= 5e-3:
            raise ValueError(f"Station summary data ({dels}) does not match "
                             + f"summary data ({self.shape_d*self.delta_m}) "
                             + "for displacement thickness")
        if np.abs(enth - self.shape_k*self.delta_m) >= 5e-3:
            raise ValueError(f"Station summary data ({enth}) does not match "
                             + f"summary data ({self.shape_k*self.delta_m}) "
                             + "for kinetic energy thickness")
        if np.abs(rdels - self.u_e*dels/v) >= 5e0:
            raise ValueError(f"Station summary data ({rdels}) does not match "
                             + f"summary data ({self.u_e*dels/v}) for "
                             + "displacement thickness Reynolds number")
        if np.abs(rtheta - self.u_e*theta/v) >= 5e0:
            raise ValueError(f"Station summary data ({rtheta}) does not match "
                             + f"summary data ({self.u_e*theta/v}) for "
                             + "momentum thickness Reynolds number")
        if np.abs(us - self.u_e*np.sqrt(0.5*self.c_f)) >= 5e-3:
            raise ValueError(f"Station summary data ({us}) does not match "
                             + "summary data "
                             + f"({self.u_e*np.sqrt(0.5*self.c_f)}) for wall "
                             + "shear velocity")

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
                                 + f"data: {st}") from None

            # perform unit conversions
            if si_unit:
                y = y*1e-2
            else:
                y = y*INCH_PER_METER

            # check derived values
            if (np.abs(yplus - y*self.u_star/self.nu) > 1e0).any():
                raise ValueError(f"Station data ({yplus}) does not match "
                                 + f"summary data ({y*self.u_star/self.nu}) "
                                 + f"for y+ sample {i}")
            if (np.abs(ycd - y/self.delta_c) > 5e-1).any():
                raise ValueError(f"Station data ({ycd}) does not match "
                                 + f"summary data ({y/self.delta_c}) "
                                 + f"for Clauser distance sample {i}")
            if (np.abs(uplus - u/self.u_star) > 5e-1).any():
                raise ValueError(f"Station data ({uplus}) does not match "
                                 + f"summary data ({u/self.u_star}) "
                                 + f"for U+ sample {i}")
            if (np.abs(udef - (self.u_e - u)/self.u_star) > 5e-1).any():
                raise ValueError(f"Station data ({udef}) does not match "
                                 + "summary data "
                                 + f"({(self.u_e-u)/self.u_star}) "
                                 + f"for defect velocity sample {i}")

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
                                 + f"velocity row {i} data: {st}") from None

            # perform unit conversion
            if not si_unit:
                x = x*FOOT_PER_METER
                ui = ui*FOOT_PER_METER

            # set values
            self.x[i] = x
            self.u_e[i] = ui
            self.du_e[i] = dui

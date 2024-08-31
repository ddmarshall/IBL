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

from typing import Optional, List
from importlib.resources import files

import numpy as np
import numpy.typing as npt

from ibl.reference.src.stanford_1968_station import (
    StanfordOlympics1968StationData, StanfordOlympics1968SmoothVel,
    FOOT_PER_METER
)


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
            self._case = ""
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
                             + f"{case_data[0][0]}")
        if case_parts[1] != case:
            raise ValueError(f"Case {case} does not match data file case "
                             + f"{case_parts[1]}")
        self._case = case
        si_unit = case[0] == "1"

        nu_parts = [x.rstrip().lstrip() for x in case_data[0][1].split("=")]
        if (nu_parts[0] != "V") and (nu_parts[0] != "v"):
            raise ValueError(f"Invalid viscosity line: {case_data[0][1]}")
        if si_unit:
            self._nu_ref = float(nu_parts[1])*1e-4
        else:
            self._nu_ref = float(nu_parts[1])*FOOT_PER_METER**2

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

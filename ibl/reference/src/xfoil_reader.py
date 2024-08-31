"""
Provide the classes for reading XFoil dump files.

This module provides the capability of reading in XFoil dump files.
"""

import copy
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from ibl.reference.src.xfoil_data import XFoilAirfoilData, XFoilWakeData

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
        self._u_ref = 1.0
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
    def u_ref(self) -> float:
        """
        Reference velocity in [m/s] for case.
        Must be greater than zero.
        """
        return self._u_ref

    @u_ref.setter
    def u_ref(self, u_ref: float) -> None:
        if u_ref <= 0:
            raise ValueError(f"Invalid reference velocity: {u_ref}")
        self._u_ref = u_ref

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
        self._u_ref = 1.0
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

    def u_e_upper(self) -> npt.NDArray:
        """
        Return the velocities for upper surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Nondimensionalized velocities for upper surface of airfoil.
        """
        u_e_rel = []
        for sd in self._upper:
            u_e_rel.append(sd.u_e_rel)
        return self.u_ref*np.array(u_e_rel)

    def u_e_lower(self) -> npt.NDArray:
        """
        Return the velocities for lower surface of airfoil.

        Returns
        -------
        numpy.ndarray
            Nondimensionalized velocities for lower surface of airfoil.
        """
        u_e_rel = []
        for sd in self._lower:
            u_e_rel.append(sd.u_e_rel)
        return self.u_ref*np.array(u_e_rel)

    def u_e_wake(self) -> npt.NDArray:
        """
        Return the velocities for airfoil wake.

        Returns
        -------
        numpy.ndarray
            Nondimensionalized velocities for airfoil wake.
        """
        u_e_rel = []
        for sd in self._wake:
            u_e_rel.append(sd.u_e_rel)
        return self.u_ref*np.array(u_e_rel)

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

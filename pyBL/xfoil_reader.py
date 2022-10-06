#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes needed to parse XFoil dump files.

XFoil dump files contain all of the information needed to characterize its
solution of the integral boundary layer equations. This class takes a filename
of the dump file and other information needed about the XFoil case that
generated the dump file and stores the information for analysis/comparison.
"""

import copy
import numpy as np


class XFoilReader:
    """
    XFoil dump file reader.

    This class is an interface to the dump file from XFoil. The data for the
    airfoil and wake are read in and processed into upper flow and lower flow
    at the stagnation point (not leading edge) and the wake. Each portion is
    stored separately and the parameters are obtained separately.

    Attributes
    ----------
    airfoil: string
        Name of airfoil analyzed.
    alpha: float
        Angle of attack for this case.
    x_trans: 2-tuple of floats
        Chord location of transition for upper and lower surface.
    n_trans: float
        Amplification factor used for transition model.
    c: float
        Airfoil chord length used for this case.
    Re: float
        Freestream Reynolds number based on the airfoil chord length.
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods
    # Attributes
    # ----------
    # _filename: File name of dump
    # _upper: Data at each upper station
    # _lower: Data at each lower station
    # _wake: Data at each wake station
    class AirfoilData:
        """
        Data reported for each station on airfoil.

        Attributes
        ----------
        s: float
            Arclength distance from stagnation point.
        x: float
            Chord location of point.
        y: float
            Normal coordinate of point.
        U_e: float
            Nondimensionalized edge velocity at point.
        delta_d: float
            Displacement thickness at point.
        delta_m: float
            Momentum thickness at point.
        c_f: float
            Skin friction coefficient at point.
        H_d: float
            Displacement shape factor at point.
        H_k: float
            Kinetic energy shape factor at point.
        p: float
            Momentum defect at point.
        m: float
            Mass defect at point.
        K: float
            Kinetic energy defect at point.
        """

        # pylint: disable=too-few-public-methods
        def __init__(self, row):
            if row == "":
                self.s = np.inf
                self.x = np.inf
                self.y = np.inf
                self.U_e = np.inf
                self.delta_d = np.inf
                self.delta_m = np.inf
                self.c_f = np.inf
                self.H_d = np.inf
                self.H_k = np.inf
                self.P = np.inf
                self.m = np.inf
                self.K = np.inf
            else:
                col = row.split()
                if len(col) != 12:
                    raise Exception("Invalid number of columns in airfoil "
                                    f"data: {row}")
                self.s = float(col[0])
                self.x = float(col[1])
                self.y = float(col[2])
                self.U_e = float(col[3])
                self.delta_d = float(col[4])
                self.delta_m = float(col[5])
                self.c_f = float(col[6])
                self.H_d = float(col[7])
                self.H_k = float(col[8])
                self.P = float(col[9])
                self.m = float(col[10])
                self.K = float(col[11])

        def __str__(self):
            """
            Return a readable presentation of instance.

            Returns
            -------
            string
                Readable string representation of instance.
            """
            strout = f"{self.__class__.__name__}:\n"
            strout += f"    s: {self.s}\n"
            strout += f"    x: {self.x}\n"
            strout += f"    y: {self.y}\n"
            strout += f"    U_e: {self.U_e}\n"
            strout += f"    delta_d: {self.delta_d}\n"
            strout += f"    delta_m: {self.delta_m}\n"
            strout += f"    c_f: {self.c_f}\n"
            strout += f"    H_d: {self.H_d}\n"
            strout += f"    H_k: {self.H_k}\n"
            strout += f"    P: {self.P}\n"
            strout += f"    m: {self.m}\n"
            strout += f"    K: {self.K}\n"

            return strout

    class WakeData:
        """
        Data reported for each station in wake.

        Attributes
        ----------
        s: float
            Arclength distance from stagnation point.
        x: float
            Chord location of point.
        y: float
            Normal coordinate of point.
        U_e: float
            Nondimensionalized edge velocity at point.
        delta_d: float
            Displacement thickness at point.
        delta_m: float
            Momentum thickness at point.
        H_d: float
            Displacement shape factor at point.
        """

        # pylint: disable=too-few-public-methods
        def __init__(self, row):
            if row == "":
                self.s = np.inf
                self.x = np.inf
                self.y = np.inf
                self.U_e = np.inf
                self.delta_d = np.inf
                self.delta_m = np.inf
                self.H_d = np.inf
            else:
                col = row.split()
                if len(col) != 8:
                    raise Exception("Invalid number of columns in wake "
                                    f"data: {row}")
                self.s = float(col[0])
                self.x = float(col[1])
                self.y = float(col[2])
                self.U_e = float(col[3])
                self.delta_d = float(col[4])
                self.delta_m = float(col[5])
                self.H_d = float(col[7])

        def __str__(self):
            """
            Return a readable presentation of instance.

            Returns
            -------
            string
                Readable string representation of instance.
            """
            strout = f"{self.__class__.__name__}:\n"
            strout += f"    s: {self.s}\n"
            strout += f"    x: {self.x}\n"
            strout += f"    y: {self.y}\n"
            strout += f"    U_e: {self.U_e}\n"
            strout += f"    delta_d: {self.delta_d}\n"
            strout += f"    delta_m: {self.delta_m}\n"
            strout += f"    H_d: {self.H_d}\n"

            return strout

    def __init__(self, filename="", airfoil="", alpha=np.inf, c=1, Re=None,
                 x_trans=None, n_trans=None):
        # pylint: disable=too-many-arguments
        self.change_case_data(filename, airfoil, alpha, c, Re, x_trans,
                              n_trans)

    def change_case_data(self, filename, airfoil="", alpha=np.inf, c=1,
                         Re=None, x_trans=None, n_trans=None):
        """
        Reset the case data to new case.

        Parameters
        ----------
        filename : string
            Name of file containing dump data.
        airfoil : string, optional
            Name of airfoil. The default is "".
        alpha : float, optional
            Angle of attack for this case. The default is np.inf.
        c : float, optional
            Chord length of airfoil for this case. The default is 1.
        Re : float, optional
            Reynolds based on the airfoil chord for this case. The default
            is `None`.
        x_trans : float or 2-tuple, optional
            Chord location specified for the boundary layer to transition from
            laminar to turbulent. The default is `None`.
        n_trans : float, optional
            Amplification factor used for the transition model. The default
            is `None`.

        Raises
        ------
        Exception
            When dump file is not correctly formatted.
        """
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-arguments
        # Reset everything
        self._filename = filename
        self.aifoil = airfoil
        self.alpha = alpha
        if isinstance(x_trans, (int, float)):
            self.x_trans = [x_trans, x_trans]
        else:
            self.x_trans = x_trans
        self.n_trans = n_trans
        self.c = c
        self. Re = Re

        self._upper = []
        self._lower = []
        self._wake = []

        if filename == "":
            return

        # define function to each comments and empty lines
        def next_chunk(f):
            line = f.readline()
            if line == "":
                return ""
            while (line == "\n") or (line[0] == "#"):
                line = f.readline()
                if line == "":
                    return ""

            chunk = [line]
            line = f.readline()
            if line == "":
                return ""
            while (line != "") and (line != "\n") and (line[0] != "#"):
                chunk.append(line)
                line = f.readline()
            return chunk

        with open(filename, "r", encoding="utf8") as dump_file:
            # read case information
            chunk = next_chunk(dump_file)

            # collect raw airfoil and wake info
            found_stag_pt = False
            found_wake = False
            stag_s = 0
            te_s = 0
            for row in chunk:
                col = row.split()
                if len(col) == 12:
                    if found_wake:
                        raise Exception("Cannot have airfoil data after waike"
                                        "data")
                    info = self.AirfoilData(row)
                    if found_stag_pt:
                        te_s = info.s
                        info.s = info.s - stag_s
                        info.U_e = -info.U_e
                        info.m = -info.m
                        info.K = -info.K
                        self._lower.append(info)
                    else:
                        if info.U_e < 0:
                            # append the first lower element after corrections
                            found_stag_pt = True
                            info.U_e = -info.U_e
                            info.m = -info.m
                            info.K = -info.K

                            # interpolate actual stagnation point
                            stag_info = self.AirfoilData("")
                            u = self._upper[-1]
                            dU = info.U_e + u.U_e
                            frac = info.U_e/dU

                            # standard interpolation for sign changes
                            stag_info.U_e = 0.0
                            stag_info.x = np.abs(-frac*u.x + (1-frac)*info.x)
                            stag_info.s = frac*u.s + (1-frac)*info.s
                            stag_info.y = frac*u.y + (1-frac)*info.y
                            # invert sign for one term for rest
                            stag_info.delta_d = (frac*u.delta_d
                                                 + (1-frac)*info.delta_d)
                            stag_info.delta_m = (frac*u.delta_m
                                                 + (1-frac)*info.delta_m)
                            stag_info.c_f = frac*u.c_f + (1-frac)*info.c_f
                            stag_info.H_d = frac*u.H_d + (1-frac)*info.H_d
                            stag_info.H_k = frac*u.H_k + (1-frac)*info.H_k
                            stag_info.P = frac*u.P + (1-frac)*info.P
                            stag_info.m = frac*u.m + (1-frac)*info.m
                            stag_info.K = frac*u.K + (1-frac)*info.K

                            # append stag_info to upper
                            stag_s = stag_info.s
                            self._upper.append(copy.copy(stag_info))

                            # correct arc lengths for two lower terms then add
                            stag_info.s = 0
                            info.s = info.s - stag_s
                            self._lower.append(stag_info)
                            self._lower.append(info)

                            # correct upper with the stag_info.s
                            for af in self._upper:
                                af.s = stag_s - af.s

                            # Reverse upper surface so that it goes from
                            # stagnation point to trailing edge
                            self._upper.reverse()
                        elif info.U_e == 0:
                            found_stag_pt = True
                            stag_s = info.s
                            self._upper.append(info)
                            for af in self._upper:
                                af.s = af.s - stag_s
                            self._lower.append(info)
                        else:
                            self._upper.append(info)
                elif len(col) == 8:
                    found_wake = True
                    info = self.WakeData(row)
                    info.s = info.s - te_s
                    self._wake.append(info)
                else:
                    raise Exception("Invalid data in XFoil dump file: "
                                    f"{col}")

    def num_points_upper(self):
        """
        Return number of points on the upper surface of airofil.

        Returns
        -------
        int
            Number of points on the upper surface of airfoil.
        """
        return len(self._upper)

    def num_points_lower(self):
        """
        Return number of points on the lower surface of airofil.

        Returns
        -------
        int
            Number of points on the lower surface of airfoil.
        """
        return len(self._lower)

    def num_points_wake(self):
        """
        Return number of points in the airofil wake.

        Returns
        -------
        int
            Number of points in the airfoil wake.
        """
        return len(self._wake)

    def point_upper(self, i):
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

    def point_lower(self, i):
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

    def point_wake(self, i):
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

    def s_upper(self):
        """
        Return arc-length distances from stagnation point for upper surface.

        Returns
        -------
        array-like
            Arc-length distances from the stagnation point for the upper
            surface.
        """
        s = []
        for sd in self._upper:
            s.append(sd.s)
        return s

    def s_lower(self):
        """
        Return arc-length distances from stagnation point for lower surface.

        Returns
        -------
        array-like
            Arc-length distances from the stagnation point for the lower
            surface.
        """
        s = []
        for sd in self._lower:
            s.append(sd.s)
        return s

    def s_wake(self):
        """
        Return arc-length distances from airfoil trailing edge.

        Returns
        -------
        array-like
            Arc-length distances from the airfoil trailing edge.
        """
        s = []
        for sd in self._wake:
            s.append(sd.s)
        return s

    def x_upper(self):
        """
        Return the chord locations for upper surface of airfoil.

        Returns
        -------
        array-like
            Chord locations for upper surface of airfoil.
        """
        x = []
        for sd in self._upper:
            x.append(sd.x)
        return x

    def x_lower(self):
        """
        Return the chord locations for lower surface of airfoil.

        Returns
        -------
        array-like
            Chord locations for lower surface of airfoil.
        """
        x = []
        for sd in self._lower:
            x.append(sd.x)
        return x

    def x_wake(self):
        """
        Return the chord locations for airfoil wake.

        Returns
        -------
        array-like
            Chord locations for airfoil wake.
        """
        x = []
        for sd in self._wake:
            x.append(sd.x)
        return x

    def y_upper(self):
        """
        Return the normal locations for upper surface of airfoil.

        Returns
        -------
        array-like
            Normal locations for upper surface of airfoil.
        """
        y = []
        for sd in self._upper:
            y.append(sd.y)
        return y

    def y_lower(self):
        """
        Return the normal locations for lower surface of airfoil.

        Returns
        -------
        array-like
            Normal locations for lower surface of airfoil.
        """
        y = []
        for sd in self._lower:
            y.append(sd.y)
        return y

    def y_wake(self):
        """
        Return the normal locations for airfoil wake.

        Returns
        -------
        array-like
            Normal locations for airfoil wake.
        """
        y = []
        for sd in self._wake:
            y.append(sd.y)
        return y

    def U_e_upper(self):
        """
        Return the nondimensionalized velocities for upper surface of airfoil.

        Returns
        -------
        array-like
            Nondimensionalized velocities for upper surface of airfoil.
        """
        U_e = []
        for sd in self._upper:
            U_e.append(sd.U_e)
        return U_e

    def U_e_lower(self):
        """
        Return the nondimensionalized velocities for lower surface of airfoil.

        Returns
        -------
        array-like
            Nondimensionalized velocities for lower surface of airfoil.
        """
        U_e = []
        for sd in self._lower:
            U_e.append(sd.U_e)
        return U_e

    def U_e_wake(self):
        """
        Return the nondimensionalized velocities for airfoil wake.

        Returns
        -------
        array-like
            Nondimensionalized velocities for airfoil wake.
        """
        U_e = []
        for sd in self._wake:
            U_e.append(sd.U_e)
        return U_e

    def delta_d_upper(self):
        """
        Return the displacement thicknesses for upper surface of airfoil.

        Returns
        -------
        array-like
            Displacement thicknesses for upper surface of airfoil.
        """
        delta_d = []
        for sd in self._upper:
            delta_d.append(sd.delta_d)
        return delta_d

    def delta_d_lower(self):
        """
        Return the displacement thicknesses for lower surface of airfoil.

        Returns
        -------
        array-like
            Displacement thicknesses for lower surface of airfoil.
        """
        delta_d = []
        for sd in self._lower:
            delta_d.append(sd.delta_d)
        return delta_d

    def delta_d_wake(self):
        """
        Return the displacement thicknesses for airfoil wake.

        Returns
        -------
        array-like
            Displacement thicknesses for airfoil wake.
        """
        delta_d = []
        for sd in self._wake:
            delta_d.append(sd.delta_d)
        return delta_d

    def delta_m_upper(self):
        """
        Return the momentum thicknesses for upper surface of airfoil.

        Returns
        -------
        array-like
            Momentum thicknesses for upper surface of airfoil.
        """
        delta_m = []
        for sd in self._upper:
            delta_m.append(sd.delta_m)
        return delta_m

    def delta_m_lower(self):
        """
        Return the momentum thicknesses for lower surface of airfoil.

        Returns
        -------
        rray-like
            Momentum thicknesses for lower surface of airfoil.
        """
        delta_m = []
        for sd in self._lower:
            delta_m.append(sd.delta_m)
        return delta_m

    def delta_m_wake(self):
        """
        Return the momentum thicknesses for airfoil wake.

        Returns
        -------
        array-like
            Momentum thicknesses for airfoil wake.
        """
        delta_m = []
        for sd in self._wake:
            delta_m.append(sd.delta_m)
        return delta_m

    def delta_k_upper(self):
        """
        Return the kinetic energy thicknesses for upper surface of airfoil.

        Returns
        -------
        array-like
            Kinetic energy thicknesses for upper surface of airfoil.
        """
        delta_k = []
        for sd in self._upper:
            delta_k.append(sd.H_k*sd.delta_m)
        return delta_k

    def delta_k_lower(self):
        """
        Return the kinetic energy thicknesses for lower surface of airfoil.

        Returns
        -------
        array-like
            Kinetic energy thicknesses for lower surface of airfoil.
        """
        delta_k = []
        for sd in self._lower:
            delta_k.append(sd.H_k*sd.delta_m)
        return delta_k

    def H_d_upper(self):
        """
        Return the displacement shape factor for upper surface of airfoil.

        Returns
        -------
        array-like
            Displacement shape factor for upper surface of airfoil.
        """
        H_d = []
        for sd in self._upper:
            H_d.append(sd.H_d)
        return H_d

    def H_d_lower(self):
        """
        Return the displacement shape factor for lower surface of airfoil.

        Returns
        -------
        array-like
            Displacement shape factor for lower surface of airfoil.
        """
        H_d = []
        for sd in self._lower:
            H_d.append(sd.H_d)
        return H_d

    def H_d_wake(self):
        """
        Return the displacement shape factor for airfoil wake.

        Returns
        -------
        array-like
            Displacement shape factor for airfoil wake.
        """
        H_d = []
        for sd in self._wake:
            H_d.append(sd.H_d)
        return H_d

    def H_k_upper(self):
        """
        Return the kinetic energy shape factor for upper surface of airfoil.

        Returns
        -------
        array-like
            Kinetic energy shape factor for upper surface of airfoil.
        """
        H_k = []
        for sd in self._upper:
            H_k.append(sd.H_k)
        return H_k

    def H_k_lower(self):
        """
        Return the kinetic energy shape factor for lower surface of airfoil.

        Returns
        -------
        array-like
            Kinetic energy shape factor for lower surface of airfoil.
        """
        H_k = []
        for sd in self._lower:
            H_k.append(sd.H_k)
        return H_k

    def c_f_upper(self):
        """
        Return the skin friction coefficient for upper surface of airfoil.

        Returns
        -------
        array-like
            Skin friction coefficient for upper surface of airfoil.
        """
        c_f = []
        for sd in self._upper:
            c_f.append(sd.c_f)
        return c_f

    def c_f_lower(self):
        """
        Return the skin friction coefficient for lower surface of airfoil.

        Returns
        -------
        array-like
            Skin friction coefficient for lower surface of airfoil.
        """
        c_f = []
        for sd in self._lower:
            c_f.append(sd.c_f)
        return c_f

    def m_upper(self):
        """
        Return the mass defect for upper surface of airfoil.

        Returns
        -------
        array-like
            Mass defect for upper surface of airfoil.
        """
        m = []
        for sd in self._upper:
            m.append(sd.m)
        return m

    def m_lower(self):
        """
        Return the mass defect for lower surface of airfoil.

        Returns
        -------
        array-like
            Mass defect for lower surface of airfoil.
        """
        m = []
        for sd in self._lower:
            m.append(sd.m)
        return m

    def P_upper(self):
        """
        Return the momentum defect for upper surface of airfoil.

        Returns
        -------
        array-like
            Momentum defect for upper surface of airfoil.
        """
        P = []
        for sd in self._upper:
            P.append(sd.P)
        return P

    def P_lower(self):
        """
        Return the momentum defect for lower surface of airfoil.

        Returns
        -------
        array-like
            Momentum defect for lower surface of airfoil.
        """
        P = []
        for sd in self._lower:
            P.append(sd.P)
        return P

    def K_upper(self):
        """
        Return the kinetic energy defect for upper surface of airfoil.

        Returns
        -------
        array-like
            Kinetic energy defect for upper surface of airfoil.
        """
        K = []
        for sd in self._upper:
            K.append(sd.K)
        return K

    def K_lower(self):
        """
        Return the kinetic energy defect for lower surface of airfoil.

        Returns
        -------
        array-like
            Kinetic energy defect for lower surface of airfoil.
        """
        K = []
        for sd in self._lower:
            K.append(sd.K)
        return K

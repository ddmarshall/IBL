#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 22:48:33 2022

@author: ddmarshall
"""

import numpy as np
import copy


class XFoilReader:
    """
    This class is an interface to the dump file from XFoil. The data for the 
    airfoil and wake are read in and processed into upper flow and lower flow
    at the stagnation point (not leading edge) and the wake.
    
    Attributes
    ----------
        airfoil: Name of airfoil analyzed
        alpha: Angle of attack
        x_trans: Chord location of transition for upper and lower surface
        n_trans: Amplification factor used for transition model
        c: Airfoil chord length
        Re: Freestream Reynolds number
        
        _filename: File name of dump
        _upper: Data at each upper station
        _lower: Data at each lower station
        _wake: Data at each wake station
    """
    class AirfoilData:
        """
        This class is the data that is reported for each station on airfoil.
        
        Attributes
        ----------
            s: Arclength distance from stagnation point
            x: Chord location of point
            y: Normal coordinate of point
            U_e: Nondimensionalized edge velocity
            delta_d: Displacement thickness
            delta_m: Momentum thickness
            c_f: Skin friction coefficient
            H_d: Displacement shape factor
            H_k: Kinetic energy shape factor
            p: Momentum defect
            m: Mass defect
            K: Kinetic energy defect
        """
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
                    raise Exception("Invalid number of columns in airfoil data: {}".format(row))
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
                Readable string representation of instance.
            """
            strout = "%s:\n" % (self.__class__.__name__)
            strout += "    s: %f\n" % (self.s)
            strout += "    x: %f\n" % (self.x)
            strout += "    y: %f\n" % (self.y)
            strout += "    U_e: %f\n" % (self.U_e)
            strout += "    delta_d: %f\n" % (self.delta_d)
            strout += "    delta_m: %f\n" % (self.delta_m)
            strout += "    c_f: %f\n" % (self.c_f)
            strout += "    H_d: %f\n" % (self.H_d)
            strout += "    H_k: %f\n" % (self.H_k)
            strout += "    P: %f\n" % (self.P)
            strout += "    m: %f\n" % (self.m)
            strout += "    K: %f\n" % (self.K)
            
            return strout
    
    
    class WakeData:
        """
        This class is the data that is reported for each station in wake.
        
        Attributes
        ----------
            s: distance from airfoil trailing edge
            x: Chord location of point
            y: Normal coordinate of point
            U_e: Edge velocity
            delta_d: Displacement thickness
            delta_m: Momentum thickness
            H_d: Displacement shape factor
        """
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
                    raise Exception("Invalid number of columns in wake data: {}".format(row))
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
                Readable string representation of instance.
            """
            strout = "%s:\n" % (self.__class__.__name__)
            strout += "    s: %f\n" % (self.s)
            strout += "    x: %f\n" % (self.x)
            strout += "    y: %f\n" % (self.y)
            strout += "    U_e: %f\n" % (self.U_e)
            strout += "    delta_d: %f\n" % (self.delta_d)
            strout += "    delta_m: %f\n" % (self.delta_m)
            strout += "    H_d: %f\n" % (self.H_d)
            
            return strout
    
    
    def __init__(self, filename = "", airfoil = "", alpha = np.inf, c = 1,
                 Re = None, x_trans = None, n_trans = None):
        self.changeCaseData(filename, airfoil, alpha, c, Re, x_trans, n_trans)
    
    def changeCaseData(self, filename, airfoil = "", alpha = np.inf, c = 1,
                       Re = None, x_trans = None, n_trans = None):
        ## Reset everything
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
        
        ## define function to each comments and empty lines
        def next_chunk(f):
            line = f.readline()
            if (line == ""):
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
        
        with open(filename, 'r') as dump_file:
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
                    raise Exception("Invalid data in XFoil dump file: {}".format(col))
    
    def num_upper_points(self):
        return len(self._upper)
    
    def num_lower_points(self):
        return len(self._lower)
    
    def num_wake_points(self):
        return len(self._wake)
    
    def get_upper_point(self, i):
        return self._upper[i]
    
    def get_lower_point(self, i):
        return self._lower[i]
    
    def get_wake_point(self, i):
        return self._wake[i]
    
    def get_upper_s(self):
        s = []
        for sd in self._upper:
            s.append(sd.s)
        return s
    
    def get_lower_s(self):
        s = []
        for sd in self._lower:
            s.append(sd.s)
        return s
    
    def get_wake_s(self):
        s = []
        for sd in self._wake:
            s.append(sd.s)
        return s
    
    def get_upper_x(self):
        x = []
        for sd in self._upper:
            x.append(sd.x)
        return x
    
    def get_lower_x(self):
        x = []
        for sd in self._lower:
            x.append(sd.x)
        return x
    
    def get_wake_x(self):
        x = []
        for sd in self._wake:
            x.append(sd.x)
        return x
    
    def get_upper_y(self):
        y = []
        for sd in self._upper:
            y.append(sd.y)
        return y
    
    def get_lower_y(self):
        y = []
        for sd in self._lower:
            y.append(sd.y)
        return y
    
    def get_wake_y(self):
        y = []
        for sd in self._wake:
            y.append(sd.y)
        return y
    
    def get_upper_U_e(self):
        U_e = []
        for sd in self._upper:
            U_e.append(sd.U_e)
        return U_e
    
    def get_lower_U_e(self):
        U_e = []
        for sd in self._lower:
            U_e.append(sd.U_e)
        return U_e
    
    def get_wake_U_e(self):
        U_e = []
        for sd in self._wake:
            U_e.append(sd.U_e)
        return U_e
    
    def get_upper_delta_d(self):
        delta_d = []
        for sd in self._upper:
            delta_d.append(sd.delta_d)
        return delta_d
    
    def get_lower_delta_d(self):
        delta_d = []
        for sd in self._lower:
            delta_d.append(sd.delta_d)
        return delta_d
    
    def get_wake_delta_d(self):
        delta_d = []
        for sd in self._wake:
            delta_d.append(sd.delta_d)
        return delta_d
    
    def get_upper_delta_m(self):
        delta_m = []
        for sd in self._upper:
            delta_m.append(sd.delta_m)
        return delta_m
    
    def get_lower_delta_m(self):
        delta_m = []
        for sd in self._lower:
            delta_m.append(sd.delta_m)
        return delta_m
    
    def get_wake_delta_m(self):
        delta_m = []
        for sd in self._wake:
            delta_m.append(sd.delta_m)
        return delta_m
    
    def get_upper_delta_k(self):
        delta_k = []
        for sd in self._upper:
            delta_k.append(sd.H_k*sd.delta_m)
        return delta_k
    
    def get_lower_delta_k(self):
        delta_k = []
        for sd in self._lower:
            delta_k.append(sd.H_k*sd.delta_m)
        return delta_k
    
    def get_upper_H_d(self):
        H_d = []
        for sd in self._upper:
            H_d.append(sd.H_d)
        return H_d
    
    def get_lower_H_d(self):
        H_d = []
        for sd in self._lower:
            H_d.append(sd.H_d)
        return H_d
    
    def get_wake_H_d(self):
        H_d = []
        for sd in self._wake:
            H_d.append(sd.H_d)
        return H_d
    
    def get_upper_H_k(self):
        H_k = []
        for sd in self._upper:
            H_k.append(sd.H_k)
        return H_k
    
    def get_lower_H_k(self):
        H_k = []
        for sd in self._lower:
            H_k.append(sd.H_k)
        return H_k
    
    def get_upper_c_f(self):
        c_f = []
        for sd in self._upper:
            c_f.append(sd.c_f)
        return c_f
    
    def get_lower_c_f(self):
        c_f = []
        for sd in self._lower:
            c_f.append(sd.c_f)
        return c_f
    
    def get_upper_m(self):
        m = []
        for sd in self._upper:
            m.append(sd.m)
        return m
    
    def get_lower_m(self):
        m = []
        for sd in self._lower:
            m.append(sd.m)
        return m
    
    def get_upper_P(self):
        P = []
        for sd in self._upper:
            P.append(sd.P)
        return P
    
    def get_lower_P(self):
        P = []
        for sd in self._lower:
            P.append(sd.P)
        return P
    
    def get_upper_K(self):
        K = []
        for sd in self._upper:
            K.append(sd.K)
        return K
    
    def get_lower_K(self):
        K = []
        for sd in self._lower:
            K.append(sd.K)
        return K

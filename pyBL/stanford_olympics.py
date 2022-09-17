#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 16:45:17 2022

@author: ddmarshall
"""

import pyBL


# Unit conversions
_METER_TO_FOOT = 0.3048
_FOOT_TO_METER = 1/_METER_TO_FOOT
_METER_TO_INCH = _METER_TO_FOOT/12
_INCH_TO_METER = 1/_METER_TO_INCH


class StanfordOlympics1968:
    """
    This class is an interface to the data from the Proceedings of the
    Computation of Turbulent Boundary Layers AFOSR-IFP-Stanford Conference in
    1968, also referred to the 1968 Stanford Olympics. The data comes from 
    volume II and comes from a variety of cases that were used as reference
    data.
    
    Attributes
    ----------
        case: Case number
        nu: Kinematic viscosity
        _station: Data at each station
        
        _x_sm: Smoothed plot x-location
        _U_e_sm: Smoothed edge velocity
        _dU_edx: Smoothed change in edge velocity
        
        See page ix of proceedings for precise definition of each term
    """
    class StationData:
        """
        This class is the data that is reported for each station in flow.
        
        Attributes
        ----------
            U_e: Edge velocity
            dU_edx: Change in edge velocity
            delta_m: Momentum thickness
            H_d: Displacement shape factor
            H_k: Kinetic energy shape factor
            G: Equilibrium shape factor
            c_f: Skin friction coefficient
            c_f_LT: Skin friction coefficient from Ludwieg-Tillman formula
            c_f_E: Skin friction coefficient reported by data originator
            beta: Equilibrium parameter
        """
        def __init__(self, row, si_unit):
            col = row.split()
            if len(col) != 11:
                raise Exception("Invalid number of columns in summary "
                                "data: {}".format(row))
            
            temp0 = float(col[0])
            temp1 = float(col[1])
            temp2 = float(col[2])
            temp3 = float(col[3])
            temp4 = float(col[4])
            temp5 = float(col[5])
            temp6 = float(col[6])
            temp7 = float(col[7])
            temp8 = float(col[8])
            temp9 = float(col[9])
            temp10 = float(col[10])
            if si_unit:
                temp3 = temp3*1e-2
            else:
                temp0 = temp0*_FOOT_TO_METER
                temp1 = temp1*_FOOT_TO_METER
                temp3 = temp1*_INCH_TO_METER
            self.x = temp0
            self.U_e = temp1
            self.dU_edx = temp2
            self.delta_m = temp3
            self.H_d = temp4
            self.H_k = temp5
            self.G = temp6
            self.c_f = temp7
            self.c_f_LT = temp8
            self.c_f_E = temp9
            self.beta = temp10
            
            self.u_star = []
            self.delta_d = []
            self.delta_k = []
            self.delta_c = []
            self.Re_delta_m = []
            self.Re_delta_d = []
            
            self.y_plus = []
            self.u_plus = []
            self.y = []
            self.u_on_Ue = []
            self.y_on_delta_c = []
            self.udef = []
        
        def __str__(self):
            """
            Return a readable presentation of instance.
    
            Returns
            -------
                Readable string representation of instance.
            """
            strout = "%s:\n" % (self.__class__.__name__)
            strout += "    x: %f\n" % (self.x)
            strout += "    U_e: %f\n" % (self.U_e)
            strout += "    dU_edx: %f\n" % (self.dU_edx)
            strout += "    delta_m: %f\n" % (self.delta_m)
            strout += "    H_d: %f\n" % (self.H_d)
            strout += "    H_k: %f\n" % (self.H_k)
            strout += "    G: %f\n" % (self.G)
            strout += "    c_f: %f\n" % (self.c_f)
            strout += "    c_f (Ludwieg-Tillman): %f\n" % (self.c_f_LT)
            strout += "    c_f (Reported): %f\n" % (self.c_f_E)
            strout += "    beta: %f\n" % (self.beta)
            
            return strout
    
    
    def __init__(self, case = None):
        self.change_case_data(case)
    
    def change_case_data(self, case):
        ## Reset everything
        self.case = ""
        self.nu = 0
        
        self._station = []
        
        self._x_sm = []
        self._U_e_sm = []
        self._dU_edx_sm = []
        
        if case is None:
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
        
        case_filename = ("{}/../stanford_olympics/".format(pyBL.__path__[0])
                        + "1968/case {}.txt".format(case))
        with open(case_filename, 'r') as case_file:
            # read case information
            chunk = next_chunk(case_file)
            if chunk[0].startswith("IDENT = "):
                self.case = chunk[0].split()[2]
            else:
                raise Exception("Expected IDENT line but have: "
                                "{}".format(chunk[0]))
            si_unit = self.case[0] == '1'
            if chunk[1].startswith("V = "):
                temp0 = float(chunk[1].split()[2])
                if si_unit:
                    temp0 = temp0*1e-4
                else:
                    temp0 = temp0*_FOOT_TO_METER**2
                self.nu = temp0
            else:
                raise Exception("Expected V line but have: "
                                "{}".format(chunk[1]))
            
            # read the summary data
            chunk = next_chunk(case_file)
            for row in chunk:
                self._station.append(self.StationData(row, si_unit))
            
            # read the edge velocity info
            chunk = next_chunk(case_file)
            for row in chunk:
                col = row.split()
                if len(col) != 3:
                    raise Exception("Invalid number of columns in summary "
                                    "data: {}".format(row))
                tmp0 = float(col[0])
                tmp1 = float(col[1])
                tmp2 = float(col[2])
                if not si_unit:
                    tmp0 = tmp0*_FOOT_TO_METER
                    tmp1 = tmp1*_FOOT_TO_METER
                
                self._x_sm.append(tmp0)
                self._U_e_sm.append(tmp1)
                self._dU_edx_sm.append(tmp2)
            
            # read the x-station data
            chunk = next_chunk(case_file)
            
            # Note: Use unit conversion flag
    
    def num_stations(self):
        return len(self._station)
    
    def station(self, i):
        return self._station[i]
    
    def velocity(self):
        return self.x(), self.U_e(), self.dU_edx()
    
    def velocity_smooth(self):
        return self._x_sm, self._U_e_sm, self._dU_edx_sm
    
    def x(self):
        x = []
        for s in self._station:
            x.append(s.x)
        return x
    
    def U_e(self):
        U_e = []
        for s in self._station:
            U_e.append(s.U_e)
        return U_e
        
    def dU_edx(self):
        dU_edx = []
        for s in self._station:
            dU_edx.append(s.dU_edx)
        return dU_edx
        
    def delta_d(self):
        delta_d = []
        for s in self._station:
            delta_d.append(s.H_d*s.delta_m)
        return delta_d
    
    def delta_m(self):
        delta_m = []
        for s in self._station:
            delta_m.append(s.delta_m)
        return delta_m
    
    def delta_k(self):
        delta_k = []
        for s in self._station:
            delta_k.append(s.H_k*s.delta_m)
        return delta_k
    
    def H_d(self):
        H_d = []
        for s in self._station:
            H_d.append(s.H_d)
        return H_d
    
    def H_k(self):
        H_k = []
        for s in self._station:
            H_k.append(s.H_k)
        return H_k
    
    def G(self):
        G = []
        for s in self._station:
            G.append(s.G)
        return G
    
    def c_f(self):
        c_f = []
        for s in self._station:
            c_f.append(s.c_f)
        return c_f
    
    def c_f_LT(self):
        c_f_LT = []
        for s in self._station:
            c_f_LT.append(s.c_f_LT)
        return c_f_LT
    
    def c_f_E(self):
        c_f_E = []
        for s in self._station:
            c_f_E.append(s.c_f_E)
        return c_f_E
    
    def beta(self):
        beta = []
        for s in self._station:
            beta.append(s.beta)
        return beta

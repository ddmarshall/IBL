"""
Provides reference data for boundary layer flows comparisons.

This module provides a variety of classes that provide interfaces to
experimental data or computational solutions for boundary layer
flows.

Currently, this module provides classes that can provide comparison data from
external resources. Currently a small number of experimental cases from the
1968 Stanford Olympics are included, and there is a class that can read a XFOIL
dump file and report boundary layer parameters from it.

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
__all__ = ['XFoilReader', 'XFoilAirfoilData', 'XFoilWakeData',
           'StanfordOlympics1968', 'StanfordOlympics1968StationData',
           'StanfordOlympics1968SmoothVel']

# XFoil reader classes
from ibl.reference.src.xfoil_data import (
    XFoilAirfoilData as XFoilAirfoilData,
    XFoilWakeData as XFoilWakeData
)
from ibl.reference.src.xfoil_reader import XFoilReader as XFoilReader

# Stanford Olympics classes
from ibl.reference.src.stanford_1968_station import (
    StanfordOlympics1968StationData as StanfordOlympics1968StationData,
    StanfordOlympics1968SmoothVel as StanfordOlympics1968SmoothVel
)
from ibl.reference.src.stanford_1968 import (
    StanfordOlympics1968 as StanfordOlympics1968
)

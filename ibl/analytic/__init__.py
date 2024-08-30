"""
Provides analytic solutions to boundary layer flows.

This module provides a variety of analytic solutions to the boundary layer
equations.
"""
__all__ = ['Blasius', 'FalknerSkan', 'Analytic2dSimilarityIncompressible']

# 2D incompressible boundary layers
from ._analytic_2d_base import (
    Analytic2dSimilarityIncompressible as Analytic2dSimilarityIncompressible
)
from ._blasius import Blasius as Blasius
from ._falkner_skan import FalknerSkan as FalknerSkan

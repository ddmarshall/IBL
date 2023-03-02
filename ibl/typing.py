"""
Typing information neeeded for use with this library.
"""

from typing import Callable, Union

import numpy.typing as npt


SolutionFunc = Callable
"""
Type used for the solution to the ODE system of equations.
"""

InputParam = Union[float, npt.NDArray]
"""
Type used for cases when either float or numpy arrays can be passed in.
"""

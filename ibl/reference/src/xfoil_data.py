"""
Provide the classes for the various datasets from XFoil solutions.

This module provides the classes needed to represent the datasets from XFoil
solutions associated with the airfoil and the wake regions.
"""

import numpy as np


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
                             + f"{s}")
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
                             + f"data: {data}") from None
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
                             + f"{s}")
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
                             + f"data: {data}") from None
        self.s = s
        self.x = x
        self.y = y
        self.u_e_rel = u_e_rel
        self.delta_d = delta_d
        self.delta_m = delta_m
        self.shape_d = shape_d

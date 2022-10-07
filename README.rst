Summary
=======

This project provides a python library to model the viscous effects for thin boundary layers using the integral boundary layer method.

Example Usage
-------------

Currently there are a few concrete models to use. One is Thwaites' method [Thwaites1949]_, which is a single equation model for laminar boundary layers.
Given an edge velocity distribution, ``U_e``, points along the boundary layer edge, ``s``, and initial momentum thickness, ``delta_m0``, the class will calculate the boundary layer properties.
These properties can then be queried at any point along the body.
For example:

.. code-block:: python

    from pyBL.thwaites_method import ThwaitesMethodNonlinear

    # Configure edge information
    U_e = ...
    s = ...
    rho_inf = ...
    nu_inf = ...

    # Set the initial coditions
    delta_m0 = ...

    tm = ThwaitesMethodNonlinear(U_e=U_e)
    rtn = tm.solve(x0=s[0], x_end=s[-1])
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
    else
        tau_wall = tm.tau_w(s, rho_inf)

Similarly for a turbulent boundary layer, Head's method [Head1958]_ can be used to calculate the properties for a turbulent boundary layer.
In addition to the initial momentum thickness, the initial displacement shape factor, ``H_d0`` is needed to initialize the model.
Otherwise, the interface is the same as for Thwaites' method:

.. code-block:: python

    from pyBL.head_method import HeadMethod

    # Configure edge information
    U_e = ...
    s = ...
    rho_inf = ...
    nu_inf = ...

    # Set the initial coditions
    delta_m0 = ...
    H_d0 = ...

    hm = ThwaitesMethodNonlinear(U_e=U_e)
    rtn = hm.solve(x0=s[0], x_end=s[-1])
    if not rtn.success:
        print("Could not get solution for Thwaites method: " + rtn.message)
    else
        tau_wall = hm.tau_w(s, rho_inf)

.. [Thwaites1949] Thwaites, B. “Approximate Calculation of the Laminar Boundary Layer.” **The Aeronautical Journal**, Vol. 1, No. 3, 1949, pp. 245–280.
.. [Head1958] Head, M. R. Entrainment in the Turbulent Boundary Layer. Publication 3152. Ministry of Aviation, Aeronautical Research Council, 1958.


Contributors
------------

The main contributors to this project are:

- David D. Marshall
- Malachi Edland (original implementation of Thwaites’ Method, Head’s
  Method, and Michel transition criteria).

Version History
---------------

* 0.5.3 - Fixed documentation to display on GitHub
* 0.5.0 - Updated interface to IBL methods to simplify configuration and provided more features that can be obtained from the IBL methods. Added documentation and cleaned up the code.
* 0.0.3 - Last release directly supporting Malachi’s thesis
* 0.0.2 - Code is mostly working as from Malachi’s thesis
* 0.0.1 - Initial Release

License
-------

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

You should have received a copy of the `GNU General Public License <license.md>`__ along with this program. If not, see http://www.gnu.org/licenses/.

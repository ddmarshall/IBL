# falkner_skan.py
# By Peter Sharpe

import casadi as cas


def falkner_skan(
        m,
        eta_edge=7,
        n_points=100,
        max_iter=100
):
    """
    Solves the Falkner-Skan equation for a given value of m.
    See Wikipedia for reference: https://en.wikipedia.org/wiki/Falkner–Skan_boundary_layer
    :param m: power-law exponent of the edge velocity (i.e. u_e(x) = U_inf * x ^ m)
    :return: eta, f0, f1, and f2 as a tuple of 1-dimensional ndarrays.
    Governing equation:
    f''' + f*f'' + beta*( 1 - (f')^2 ) = 0, where:
    beta = 2 * m / (m+1)
    f(0) = f'(0) = 0
    f'(inf) = 1
    Syntax:
    f0 is f
    f1 is f'
    f2 is f''
    f3 is f'''
    """

    # Assign beta
    beta = 2 * m / (m + 1)

    opti = cas.Opti()

    eta = cas.linspace(0, eta_edge, n_points)

    def trapz(x):
        out = (x[:-1] + x[1:]) / 2
        # out[0] += x[0] / 2
        # out[-1] += x[-1] / 2
        return out

    # Vars
    f0 = opti.variable(n_points)
    f1 = opti.variable(n_points)
    f2 = opti.variable(n_points)

    # Guess (guess a quadratic velocity profile, integrate and differentiate accordingly)
    opti.set_initial(f0,
                     -eta ** 2 * (eta - 3 * eta_edge) / (3 * eta_edge ** 2)
                     )
    opti.set_initial(f1,
                     1 - (1 - eta / eta_edge) ** 2
                     )
    opti.set_initial(f2,
                     2 * (eta_edge - eta) / eta_edge ** 2
                     )

    # BCs
    opti.subject_to([
        f0[0] == 0,
        f1[0] == 0,
        f1[-1] == 1
    ])

    # ODE
    f3 = -f0 * f2 - beta * (1 - f1 ** 2)

    # Derivative definitions (midpoint-method)
    df0 = cas.diff(f0)
    df1 = cas.diff(f1)
    df2 = cas.diff(f2)
    deta = cas.diff(eta)
    opti.subject_to([
        df0 == trapz(f1) * deta,
        df1 == trapz(f2) * deta,
        df2 == trapz(f3) * deta
    ])

    # Require unseparated solutions
    opti.subject_to([
        f2[0] > 0
    ])

    p_opts = {}
    p_opts["print_time"] = 0 # turn off solution diagnostics printing
    s_opts = {}
    s_opts["max_iter"] = max_iter  # If you need to interrupt, just use ctrl+c
    s_opts["print_level"] = 0 # turn off run diagnostics printing
    opti.solver('ipopt', p_opts, s_opts)

    try:
        sol = opti.solve()
    except:
        raise Exception("Solver failed for m = %f!" % m)

    return (
        sol.value(eta),
        sol.value(f0),
        sol.value(f1),
        sol.value(f2)
    )


if __name__ == "__main__":
    # Run through a few tests to ensure that these functions are working correctly.
    # Includes all examples in Table 4.1 of Drela's Flight Vehicle Aerodynamics textbook, along with a few others.
    # Then plots all their velocity profiles.
    import matplotlib.pyplot as plt
    import matplotlib.style as style

    style.use("seaborn")

    import numpy as np

    from time import time

    start = time()

    m_tests = [-0.0904, -0.08, -0.05, 0, 0.1, 0.3, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(m_tests)))[::-1]
    for i in range(len(m_tests)):
        m_val = m_tests[i]
        eta, f0, f1, f2 = falkner_skan(m=m_val)
        plt.plot(f1, eta, "-", label=r"$m = %.2f$" % m_val, color=colors[i], zorder=3 + len(m_tests) - i)
    plt.xlabel(r"$f'$ ($u/u_e$)")
    plt.ylabel(r"$\eta$ (Nondim. wall distance)")
    plt.title("Falkner-Skan Velocity Profiles")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("falkner_skan.svg")
    plt.show()

    print("Time Elapsed: %f sec" % (time() - start))

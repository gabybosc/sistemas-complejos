import numpy as np

"""
functions for dubinin2004
"""

"""
equations
"""


def B_field(upt_phi, params):
    """
    Returns B**2 / (2*M**2)
    """
    alpha, Ub, M = params
    upt, phi = upt_phi

    f = M ** 2 / 2 * upt ** 2 * (-alpha * Ub + 1 + np.sqrt(-alpha * Ub) * np.real(np.exp(1j * phi)))

    return f


def eq_upx(ubx, upt_phi, params):
    """Returns upx"""
    alpha, Ub, M = params

    upx = 1 - alpha * Ub * (ubx - Ub) - B_field(upt_phi, params)

    return upx


def implicit_ubx(ubx, upt_phi, params):
    """Implicit equation for ubx"""
    alpha, Ub, M = params
    f = (1 - alpha * Ub * (ubx - Ub) - B_field(upt_phi, params)) ** 2 - 1 + alpha * Ub * (ubx ** 2 - Ub ** 2)

    return f


def eq_15(ux, upt_phi, params):
    """
    Complete system of diff. equations for upt and phi
    To be solved by a RK method
    """
    alpha, Ub, M = params
    upx, ubx = ux
    upt, phi = upt_phi

    dupt_dx = np.sqrt(-alpha * Ub) * M ** 2 * np.sin(phi) * upt
    dphi_dx = -M ** 2 * (1 - alpha * Ub) + 2 * M ** 2 * np.sqrt(-alpha * Ub) * np.cos(phi) + 1 / upx - 1 / ubx

    return np.array([dupt_dx, dphi_dx])


def eqs_13(u_plus, ux, params):
    """
    System of diff. eqs. for up_plus and ub_plus
    To be solved by a RK method
    """

    alpha, Ub, M = params
    up_plus, ub_plus = u_plus
    upx, ubx = ux

    dup_plus_dx = 1j * (M ** 2 * (alpha * Ub * ub_plus + up_plus) - up_plus / upx)
    dub_plus_dx = 1j * (M ** 2 * (alpha * Ub * ub_plus + up_plus) - ub_plus / ubx)

    return np.array([dup_plus_dx, dub_plus_dx])


def B_field_transverse(uy, uz, params):
    """
    :return: By, Bz
    """

    alpha, Ub, M = params
    upy, uby = uy
    upz, ubz = uz
    Bz = M ** 2 * (alpha * Ub * ubz + upz)
    By = M ** 2 * (alpha * Ub * uby + upy)

    return By, Bz


def eqs_24(upt_phi, params):
    """
    System of diff. equations for linearized upt and phi
    """
    a, b, delta, d = params
    upt, phi = upt_phi

    dupt_dx = a * upt * np.sin(phi)
    dphi_dx = 2 * delta + b * upt ** 2 + 2 * a * np.cos(phi) + d * upt ** 2 * np.cos(phi)

    return np.array([dupt_dx, dphi_dx])


def eqs_31_lin(u_plus, params):
    alpha, Ub, M = params
    up_plus, ub_plus = u_plus

    dup_plus_dx = 1j * (M ** 2 * (alpha * Ub * ub_plus + up_plus) - up_plus)
    dub_plus_dx = 1j * (M ** 2 * (alpha * Ub * ub_plus + up_plus) - ub_plus / Ub)

    return np.array([dup_plus_dx, dub_plus_dx])


"""
methods
"""


def RK4_modfied(f, ubx0, ubx1, upt_phi, params, h, steps):
    """
    Runge Kutta order 4 plus secant method
    ONLY solves system 15 + equations for ubx
    """
    # initial guess
    ubx = ubx0
    upx = eq_upx(ubx, upt_phi, params)
    R = [np.array(upt_phi)]
    T = [np.array([upx, ubx])]

    # loop
    for i in range(steps):
        ux = [upx, ubx]
        k1 = f(ux, upt_phi, params)
        k2 = f(ux, upt_phi + h / 2 * k1, params)
        k3 = f(ux, upt_phi + h / 2 * k2, params)
        k4 = f(ux, upt_phi + h * k3, params)
        upt_phi = upt_phi + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        R.append(upt_phi)
        ub_new = secant_method(implicit_ubx, ubx, ubx1, upt_phi, params)
        ubx1 = ubx
        ubx = ub_new
        upx = eq_upx(ubx, upt_phi, params)
        T.append([upx, ubx])

    return np.array(R), np.array(T)


def RK_13(f, u_plus, ux, params, h=0.1, steps=1000):
    """
    Runge Kutta order 4 for solving system 13
    """
    # initial guess
    R = [np.array(u_plus)]

    # loop
    for i in range(steps):
        k1 = f(u_plus, ux[i], params)
        k2 = f(u_plus + h / 2 * k1, ux[i], params)
        k3 = f(u_plus + h / 2 * k2, ux[i], params)
        k4 = f(u_plus + h * k3, ux[i], params)
        u_plus = u_plus + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        R.append(u_plus)

    return np.array(R)


def RK4_for_linearized(f, r0, params, h, pasos):
    """
    RK order 4 for solving the linearized system eqs for ubx and u_plus
    """
    # inicial
    r = r0
    R = [np.array(r)]
    for i in range(pasos):
        k1 = f(r, params)
        k2 = f(r + h / 2 * k1, params)
        k3 = f(r + h / 2 * k2, params)
        k4 = f(r + h * k3, params)
        r = r + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        R.append(r)

    return np.array(R)


def secant_method(f, x0, x1, upt_phi, params, iterations=1000, tolerance=1e-6):
    """
    Return the root calculated using the secant method.
    ONLY for solving implicit_ubx
    """
    for i in range(iterations):
        f0 = f(x0, upt_phi, params)
        f1 = f(x1, upt_phi, params)

        x2 = x1 - f1 * (x1 - x0) / float(f1 - f0)
        x0, x1 = x1, x2

        if abs(f1) < tolerance:
            return x2

    return x2

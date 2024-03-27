import numpy as np
from findiff import FinDiff
from scipy.integrate import ode


def RK4(f, t0, r0, params, h, pasos):
    """RK de orden 4"""
    # inicial
    t = t0
    r = r0
    R = [np.array(r)]
    for i in range(pasos):
        k1 = f(t, r, params)
        k2 = f(t + h / 2, r + h / 2 * k1, params)
        k3 = f(t + h / 2, r + h / 2 * k2, params)
        k4 = f(t + h, r + h * k3, params)
        t = t + h
        r = r + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        R.append(r)

    return np.array(R)


def RK2(f, t0, r0, params, h):
    """RK de orden 4"""
    # inicial
    t = t0
    r = r0

    k1 = f(t, r, params)
    k4 = f(t + h, r + h * k1, params)

    r = r + h / 6 * (k1 + k4)

    return np.array(r)


def balance(Energy, dEdt, uv, dt, params):
    """
    Esta función recibe u (x), v (dx/dt), parametros extra y
    las funciones de la energía y su derivada. Devuelve un vector
    de balance que tiene la diferencia de la energía del sistema
    en cada tiempo con respecto al valor teórico normalizado por
    el valor inicial de la energía.
    """
    d_dt = FinDiff(0, dt, acc=6)
    E = Energy(uv, params)
    dE = d_dt(E)
    dEt = dEdt(uv, params)
    return dt * (dE - dEt) / np.min(E)


def lorenz(t, r, params):
    x, y, z = r
    sigma, rho, beta = params
    xdot = sigma * (y - x)
    ydot = rho * x - y - x * z
    zdot = x * y - beta * z

    return np.array([xdot, ydot, zdot])


def correr(f, r0, tf, dt, params):
    """Resuelve la ode usando scipy, devuelve un array"""

    solver = ode(f).set_integrator("dopri5")
    solver.set_initial_value(r0, t=0.0).set_f_params(params)

    i = 0
    evolution = []
    while solver.successful() and solver.t < tf:
        i += 1
        solver.integrate(solver.t + dt)
        evolution.append(solver.y)

    return np.array(evolution)

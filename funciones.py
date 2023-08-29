import numpy as np
from findiff import FinDiff


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

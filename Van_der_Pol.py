from findiff import FinDiff
import matplotlib.pyplot as plt
from funciones import RK4
import numpy as np
from scipy.integrate import odeint


def VdP(uv, t, alpha):
    u, v = uv
    vdot = -alpha * (u**2 - 1) * v - u
    udot = v

    return udot, vdot


def Energy(uv, params):
    u, v = uv
    E = (v**2 + u**2) / 2

    return E


def dEdt(uv, alpha):
    u, v = uv
    dE = -alpha * v**2 * (u**2 - 1)

    return dE


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


def RK4(f, t0, u0, v0, alpha, h, pasos):
    # inicial
    u = u0
    v = v0
    t = t0
    U, V = [u], [v]

    for i in range(pasos):
        k1 = f([u, v], t, alpha)
        k2 = f([u + h / 2 * k1[0], v + h / 2 * k1[1]], t + h / 2, alpha)
        k3 = f([u + h / 2 * k2[0], v + h / 2 * k2[1]], t + h / 2, alpha)
        k4 = f([u + h * k3[0], v + h * k3[1]], t + h, alpha)
        t = t + h
        u = u + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v = v + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        U.append(u)
        V.append(v)

    return (np.array(U), np.array(V))


h = 0.01
pasos = int(5 / h)
t = np.linspace(0, 5, pasos + 1)
u0 = 1
v0 = 1

"""
e) Utilice la herramienta odeint de scipy para integrar 
tanto en las condiciones del punto c. Analice la elección de intervalos temporales.
"""

for alpha in [0.5, 1.5, 3]:
    u, v = RK4(VdP, 0, u0, v0, alpha, h, pasos)
    bal = balance(Energy, dEdt, [u, v], h, alpha)
    sol = odeint(VdP, [u0, v0], t, args=(alpha,))
    if max(bal) < 10 - 3:
        plt.figure()
        plt.grid()
        plt.plot(u, v, label="RK4")
        plt.plot(sol[:, 0], sol[:, 1], label="odeint", linestyle="-.")
        plt.ylabel("v")
        plt.xlabel("u")
        plt.title(f"Diagrama de Fases, alpha ={alpha}")
        plt.legend()
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(t, u, label="RK4")
        ax2.plot(t, v, label="RK4")
        ax1.plot(t, sol[:, 0], label="odeint", linestyle="-.")
        ax2.plot(t, sol[:, 1], label="odeint", linestyle="-.")
        ax1.set_ylabel("u")
        ax2.set_ylabel("v")
        ax2.set_xlabel("t")
        ax1.set_title(f"Trayectoria, alpha ={alpha}")
        ax1.grid()
        ax2.grid()
        ax1.legend()
        ax2.legend()
        plt.show()

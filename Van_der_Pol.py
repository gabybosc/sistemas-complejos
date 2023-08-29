import matplotlib.pyplot as plt
from funciones import RK4, balance
import numpy as np
from scipy.integrate import odeint


def VdP(t, uv, alpha):
    u, v = uv
    vdot = -alpha * (u**2 - 1) * v - u
    udot = v

    return np.array([udot, vdot])


def Energy(uv, params):
    u, v = uv
    E = (v**2 + u**2) / 2

    return E


def dEdt(uv, alpha):
    u, v = uv
    dE = -alpha * v**2 * (u**2 - 1)

    return dE


h = 0.01
tf = 10
pasos = int(tf / h)
t = np.linspace(0, tf, pasos + 1)
CI = np.array([1, 10])  # cond ini [u0, v0]
alpha = 0.5

"""
e) Utilice la herramienta odeint de scipy para integrar 
tanto en las condiciones del punto c. Analice la elección de intervalos temporales.
"""

for alpha in [0.5, 1.5, 3]:
    uv = RK4(VdP, 0, CI, alpha, h, pasos)
    u, v = uv[:, 0], uv[:, 1]
    bal = balance(Energy, dEdt, [u, v], h, alpha)
    sol = odeint(VdP, CI, t, args=(alpha,), tfirst=True)  # va primero t y después uv
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

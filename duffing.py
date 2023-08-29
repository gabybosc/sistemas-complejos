import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from funciones import RK4, balance


"""
Usando α=1, β=1, γ=0 y δ=0, integre la ecuación por un tiempo T=20 para varias condiciones iniciales.
Use la conservación de la energía para elegir un paso temporal tal que esta se conserve.
"""


def duffing(t, uv, params):
    u, v = uv
    omega, gamma, delta, beta, alpha = params
    udot = v
    vdot = gamma * np.cos(omega * t) - delta * v - beta * u - alpha * u**3
    return np.array([udot, vdot])


def Energy_duffing(uv, params):
    u, v = uv
    omega, gamma, delta, beta, alpha = params
    E = -(v**2 + beta * u**2 + alpha * u**4 / 2) / 2

    return E


def dEdt_duffing(uv, params):
    u, v = uv
    omega, gamma, delta, beta, alpha = params
    dE = -delta * v**2

    return dE


h = 0.01
pasos = int(20 / h)
t = np.linspace(0, 20, pasos + 1)
u0 = [1, 1]  # agregar un loop en CIs
params = [1, 0.3, 0.22, -1, 1]  # omega, gamma, delta, beta, alpha

uv = RK4(duffing, 0, u0, params, h, pasos)
u, v = uv[:, 0], uv[:, 1]
bal = balance(Energy_duffing, dEdt_duffing, [u, v], h, params)
if max(bal) < 10e-3:
    plt.figure()
    plt.grid()
    plt.plot(u, v)
    plt.scatter(u[0], v[0], label="inicio")
    plt.ylabel("v")
    plt.xlabel("u")
    plt.title(f"Diagrama de Fases")
    plt.legend()
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(t, u)
    ax2.plot(t, v)
    ax1.set_ylabel("u")
    ax2.set_ylabel("v")
    ax2.set_xlabel("t")
    ax1.set_title(f"Trayectoria")
    ax1.grid()
    ax2.grid()
    plt.show()

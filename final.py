import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from funciones import correr


alpha = 0.005
U = 1.31
M = 1.3
upx = 4
ubx = 5
params = [alpha * U, M**2, U, upx, ubx]
par = [alpha * U, M**2]


def ec_13(up, ub, params):
    alphaU, M2 = params
    dupdx = 1j * (M2 * (alphaU * ub + up) - up / upx)


def sistema(u, phi, params):
    alphaU, M2, U, upx, ubx = params
    dudx = np.sqrt(alphaU) * M2 * np.sin(phi) * u
    dpdx = (
        -M2 * (1 + alphaU) + 2 * M2 * np.sqrt(alphaU) * np.cos(phi) + 1 / upx + 1 / ubx
    )
    return dudx, dpdx


evo = correr(sistema, [0.1, 1], 50, 1e-4, params)
fig = plt.figure()
plt.plot(
    evo[:, 0],
    evo[:, 1],
)
plt.show()


"""Haz débil"""


def haz_debil(u, phi, params):
    alphaU, M2, U, upx, ubx = params

    a = M2 * np.sqrt(alphaU)
    delta = (1 + 1 / U - M2 * (1 + alphaU)) / 2
    b = (M2 * (1 + alphaU) * (1 + alphaU * U**3)) / (2 * U**2 * (1 + U) * alphaU)
    d = -M2 * (1 + alphaU * U**3) / (U**2 * (1 + U) * np.sqrt(alphaU))

    dudx = a * u * np.sin(phi)
    dpdx = 2 * delta + b * u**2 + 2 * a * np.cos(phi) + d * u**2 * np.cos(phi)
    return dudx, dpdx

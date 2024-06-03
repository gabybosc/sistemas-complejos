import numpy as np
import matplotlib.pyplot as plt
from dubinin_functions import eq_15, RK4_modfied, abdelta

alpha = 0.005
# Ub = -1.31
# U = -Ub
# M = 1.3

steps = 2000



"""
como defino la amplitud del osciliton?
no es max-min
no es max - mean
para upx es 1 - min(upx)
"""


def upt_max(a, delta, b):
    return 2 * np.sqrt((a - delta) / b)


def upx_max(ubx, params):
    alpha, U, M = params
    return 1 - alpha * U * (U ** 2 - ubx ** 2) / 2


def ubx_max(upt, params):
    alpha, U, M = params
    return 1 - (1 + U) * np.sqrt(
        1 - (M ** 2 * upt ** 2 / (alpha * U) * ((1 + alpha * U) - 2 * np.sqrt(alpha * U))) / (1 + U) ** 2)


def figure_4(x, amp_upt, upx, ubx, list_max_upt, list_max_upx, list_max_ubx, params):
    alpha, U, M = params
    a, b, delta = abdelta(alpha, U, M)
    if np.abs(delta / a) <= 1:
        parameters = [alpha, -U, M]
        upt_phi_0 = [0.001, np.arccos(-delta / a)]
        upt_phi, ux = RK4_modfied(eq_15, -1.5, -1, upt_phi_0, parameters, 0.1, steps)
        # plt.plot(ux)
        # plt.title(f"U={U}, M={M}")
        # plt.show()
        max_upt = upt_max(a, delta, b)
        max_ubx = ubx_max(max_upt, params)
        max_upx = upx_max(max_ubx, params)

        list_max_upt.append(max_upt)
        list_max_upx.append(max_upx)
        list_max_ubx.append(max_ubx)
        amp_upt.append(max(upt_phi[1:, 0]))
        ubx.append(max(ux[1:, 1]))
        upx.append(min(ux[1:, 0]))
        # upx.append(np.abs(max(ux[1:, 0]) - min(ux[1:, 0])))
        # ubx.append(np.abs(max(ux[1:, 1]) - min(ux[1:, 1])))
        x.append(M)
    return np.array(x), np.array(amp_upt), np.array(upx), np.array(ubx), np.array(list_max_upt), np.array(list_max_upx), np.array(list_max_ubx)


def lists():
    list_upt = []
    list_x = []
    list_upx = []
    list_ubx = []
    list_function_upt = []
    list_function_upx = []
    list_function_ubx = []
    return list_x, list_upt, list_upx, list_ubx, list_function_upt, list_function_upx, list_function_ubx,


def plots(x, ux, u_max, color):
    ubx, upx, upt = ux
    max_ubx, max_upx, max_upt = u_max
    ax1.plot(x, ubx, ".", label=f"U={U}")
    ax2.plot(x, upx, ".", label=f"U={U}")
    ax3.plot(x, upt, ".", label=f"U={U}")
    ax1.plot(x, max_ubx, c=color)
    ax2.plot(x, max_upx, c=color)
    ax3.plot(x, max_upt, c=color)


fig = plt.figure(1)
fig.subplots_adjust(
    top=0.95, bottom=0.1, left=0.05, right=0.95, hspace=0.005, wspace=0.15
)
ax1 = plt.subplot2grid((3, 1), (0, 0))
ax2 = plt.subplot2grid((3, 1), (1, 0), sharex=ax1)
ax3 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)

U = 2
list_x, list_upt, list_upx, list_ubx, list_function_upt, list_function_upx, list_function_ubx = lists()

for M in np.linspace(1.1, 1.2, 10):
    params = [alpha, U, M]
    a, b, delta = abdelta(alpha, U, M)
    X, amplitude_upt, upx, ubx, max_upt, max_upx, max_ubx = figure_4(list_x, list_upt, list_upx, list_ubx, list_function_upt, list_function_upx, list_function_ubx, params)

plots(X, [ubx, upx, amplitude_upt], [max_ubx, max_upx, max_upt], "C0")

U = 5
list_x, list_upt, list_upx, list_ubx, list_function_upt, list_function_upx, list_function_ubx = lists()

for M in np.linspace(0.95, 1.1, 10):
    params = [alpha, U, M]
    a, b, delta = abdelta(alpha, U, M)
    X, amplitude_upt, upx, ubx, max_upt, max_upx, max_ubx = figure_4(list_x, list_upt, list_upx, list_ubx, list_function_upt, list_function_upx, list_function_ubx, params)

plots(X, [ubx, upx, amplitude_upt], [max_ubx, max_upx, max_upt], "C1")

U = 10
list_x, list_upt, list_upx, list_ubx, list_function_upt, list_function_upx, list_function_ubx = lists()

for M in np.linspace(0.87, 1.2, 10):
    params = [alpha, U, M]
    a, b, delta = abdelta(alpha, U, M)
    X, amplitude_upt, upx, ubx, max_upt, max_upx, max_ubx = figure_4(list_x, list_upt, list_upx, list_ubx, list_function_upt, list_function_upx, list_function_ubx, params)

plots(X, [ubx, upx, amplitude_upt], [max_ubx, max_upx, max_upt], "C2")


for ax in [ax1, ax2, ax3]:
    ax.legend()
    ax.grid(True)
    ax.set_xlim([0.5, 1.5])

ax3.set_xlabel(r"$M_A$")

ax1.set_ylim([-15, 0.1])
ax2.set_ylim([0, 1.5])
ax3.set_ylim([-0.2, 1.5])

ax1.set_ylabel(r"$u_{bx}$ max")
ax2.set_ylabel(r"$u_{px}$ max")
ax3.set_ylabel(r"$u_{pt}$ max")
plt.show()


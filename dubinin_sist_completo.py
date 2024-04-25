import numpy as np
import matplotlib.pyplot as plt

alpha = 0.005
Ub = -1.31
U = -Ub
M = 1.3
# Bx = 1
params = [alpha, Ub, M]
a = M ** 2 * np.sqrt(alpha * U)
b = M ** 2 * (1 + alpha * U) * (1 + alpha * U ** 4) / (2 * U ** 2 * (1 + U) * alpha * U)
delta = (1 + 1 / U - M ** 2 * (1 + alpha * U)) / 2


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
    """System of diff. equations for upt and phi"""
    alpha, Ub, M = params
    upx, ubx = ux
    upt, phi = upt_phi

    dupt_dx = np.sqrt(-alpha * Ub) * M ** 2 * np.sin(phi) * upt
    dphi_dx = -M ** 2 * (1 - alpha * Ub) + 2 * M ** 2 * np.sqrt(-alpha * Ub) * np.cos(phi) + 1 / upx - 1 / ubx

    return np.array([dupt_dx, dphi_dx])


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


# upt_phi_0 = [0.001, np.pi / 2]
# upt_phi_0 = [np.sqrt(2 * (a - delta) / b), np.pi]
# phi0 = np.arccos(-(1 + 1/U - M**2 * (1 + alpha*U)) / (2*M**2 * np.sqrt(alpha*U)))
upt_phi_0 = [0.001, np.arccos(-delta/a)]
steps = 2000
upt_phi, ux = RK4_modfied(eq_15, -1.5, -1, upt_phi_0, params, 0.1, steps)
x = np.linspace(0, 100, steps+1)

"""
Con CI upt_phi_0 = [0.001, np.pi/2] funciona perfecto
Recupero dos solitones en vez de uno solo, pero los valores son los correctos
"""

"""
Como tengo upx, ubx puedo integrar el sistema de ecs 13 para despejar u_pm
"""


def eqs_13(u_plus, ux, params):
    alpha, Ub, M = params
    up_plus, ub_plus = u_plus
    upx, ubx = ux

    dup_plus_dx = 1j * (M ** 2 * (alpha * Ub * ub_plus + up_plus) - up_plus / upx)
    dub_plus_dx = 1j * (M ** 2 * (alpha * Ub * ub_plus + up_plus) - ub_plus / ubx)

    return np.array([dup_plus_dx, dub_plus_dx])


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


u_plus = RK_13(eqs_13, [0.01, 0.1], ux, params, steps=steps)

upy = np.real(u_plus[:, 0])
upz = np.imag(u_plus[:, 0])
uby = np.real(u_plus[:, 1])
ubz = np.imag(u_plus[:, 1])

"""
Magnetic field
"""


def magnetic_field(uy, uz, params):
    alpha, Ub, M = params
    upy, uby = uy
    upz, ubz = uz
    Bz = M ** 2 * (alpha * Ub * ubz + upz)
    By = M ** 2 * (alpha * Ub * uby + upy)

    return By, Bz


By, Bz = magnetic_field([upy, uby], [upz, ubz], params)

fig = plt.figure()
fig.subplots_adjust(
    top=0.95, bottom=0.1, left=0.05, right=0.95, hspace=0.005, wspace=0.15
)
# plt.title("")
ax1 = plt.subplot2grid((5, 1), (0, 0))
ax2 = plt.subplot2grid((5, 1), (1, 0), sharex=ax1)
ax3 = plt.subplot2grid((5, 1), (2, 0), sharex=ax1)
ax4 = plt.subplot2grid((5, 1), (3, 0), sharex=ax1)
ax5 = plt.subplot2grid((5, 1), (4, 0), sharex=ax1)

# ax1.set_title(r"$\phi = \pi$, $u_{pt}=$"+f"{np.sqrt(2 * (a - delta) / b):.3g}")  # estacionario
ax1.set_title(r"$\phi=$"+f"{np.arccos(-delta/a):.3g}" + r"$u_{pt} = 10^{-3}$")  # soliton
ax1.plot(ux[:, 0])
ax1.set_ylabel("upx")

ax2.plot(ux[:, 1], c="C1")
ax2.set_ylabel("ubx")

ax3.plot(upy, c="C0", label="upy")
ax3.plot(upz, c="C0", linestyle="--", label="upz")
ax3.plot(uby, c="C1", label="uby")
ax3.plot(ubz, c="C1", linestyle="--", label="ubz")
ax3.set_ylabel("upy, uby")

ax4.plot(By, c="C0", label="By")
ax4.plot(Bz, c="C0", linestyle="--", label="Bz")
ax4.set_ylabel("By, Bz")

ax5.plot(np.linalg.norm([By, Bz], axis=0))
ax5.set_ylabel(r"$|B_\perp$|")

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_xlim([100, 600])
    ax.grid()

for ax in [ax1, ax2, ax3, ax4]:
    plt.setp(ax.get_xticklabels(), visible=False)

for ax in [ax3, ax4]:
    ax.legend(loc="right")
plt.show()

# for phi in [0, np.pi/2, np.pi]:
#     if phi == np.pi:
#         c = "C0"
#     else:
#         c = "C1"
#     for upt in [0.025, 0.05, 0.075]:
#         upt_phi_0 = [upt, phi]
#         upt_phi, ux = RK4_modfied(eq_15, -1.5, -1, upt_phi_0, params, 0.1, 1000)
#         plt.plot(upt_phi[:,0], upt_phi[:, 1], c=c)
# plt.ylim([0, 2*np.pi])
# plt.xlabel(r"$u_{pt}$")
# plt.ylabel(r"$\phi$")
# plt.show()
#

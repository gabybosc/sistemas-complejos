import numpy as np
import matplotlib.pyplot as plt

alpha = 0.005
Ub = -1.31
U = -Ub
M = 1.3
# Bx = 1
a = M ** 2 * np.sqrt(alpha * U)
b = M ** 2 * (1 + alpha * U) * (1 + alpha * U ** 4) / (2 * U ** 2 * (1 + U) * alpha * U)
delta = (1 + 1 / U - M ** 2 * (1 + alpha * U)) / 2
d = - M ** 2 * (1 + alpha * U ** 4) / (U ** 2 * (1 + U) * np.sqrt(alpha * U))

params = [alpha, U, M]
params_lin = [a, b, delta, d]


def eqs_24(upt_phi, params):
    """System of diff. equations for linearized upt and phi"""
    a, b, delta, d = params
    upt, phi = upt_phi

    dupt_dx = a * upt * np.sin(phi)
    dphi_dx = 2 * delta + b * upt ** 2 + 2 * a * np.cos(phi) + d * upt ** 2 * np.cos(phi)

    return np.array([dupt_dx, dphi_dx])


def eqs_other_vel(upt_phi, params):
    alpha, U, M = params
    upt, phi = upt_phi[0], upt_phi[1]

    ubt = upt / np.sqrt(alpha * U)
    ubx_inv = -1 / U - M ** 2 * ((1 + alpha * U) - 2 * np.sqrt(alpha * U) * np.cos(phi)) * ubt ** 2 / (
            2 * U ** 2 * (1 + U))
    ubx = 1 / ubx_inv

    upx_inv = 1 + alpha * U ** 2 / (2 * (1 + alpha * U)) * M ** 2 * (
                (1 + alpha * U) - 2 * np.sqrt(alpha * U) * np.cos(phi)) * ubt ** 2
    upx = 1 / upx_inv

    return ubx, upx


def RK4(f, r0, params, h, pasos):
    """RK order 4"""
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


def B_field(upt_phi, params):
    """
    Returns B**2 / (2*M**2)
    """
    alpha, U, M = params
    Ub = -U
    upt, phi = upt_phi[0], upt_phi[1]

    f = M ** 2 / 2 * upt ** 2 * (-alpha * Ub + 1 + np.sqrt(-alpha * Ub) * np.real(np.exp(1j * phi)))

    return f


def eq_upx(ubx, upt_phi, params):
    """Returns upx"""
    alpha, U, M = params
    Ub = -U
    upx = 1 - alpha * Ub * (ubx - Ub) - B_field(upt_phi, params)

    return upx


def implicit_ubx(ubx, upt_phi, params):
    """Implicit equation for ubx"""
    alpha, U, M = params
    Ub = -U
    f = (1 - alpha * Ub * (ubx - Ub) - B_field(upt_phi, params)) ** 2 - 1 + alpha * Ub * (ubx ** 2 - Ub ** 2)

    return f

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


steps = 1000
upt_phi_0 = [0.001, np.pi/4]
upt_phi = RK4(eqs_24, upt_phi_0, params_lin, 0.1, steps)

ubx = []
upx = []
for i in range(len(upt_phi)):
    ubx.append(secant_method(implicit_ubx, -1.5, -1, upt_phi[i], params))
    upx.append(eq_upx(ubx[-1], upt_phi[i], params))

fig = plt.figure()
fig.subplots_adjust(
    top=0.95, bottom=0.1, left=0.05, right=0.95, hspace=0.005, wspace=0.15
)
# plt.title("")
ax1 = plt.subplot2grid((2, 1), (0, 0))
ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)
# ax3 = plt.subplot2grid((5, 1), (2, 0), sharex=ax1)
# ax4 = plt.subplot2grid((5, 1), (3, 0), sharex=ax1)
# ax5 = plt.subplot2grid((5, 1), (4, 0), sharex=ax1)

# ax1.set_title(r"$\phi = \pi$, $u_{pt}=$"+f"{np.sqrt(2 * (a - delta) / b):.3g}")  # estacionario
ax1.set_title(r"$\phi=$"+f"{np.arccos(-delta/a):.3g}" + r"$u_{pt} = 10^{-3}$")  # soliton
ax1.plot(upx)
ax1.set_ylabel("upx")

ax2.plot(ubx, c="C1")
ax2.set_ylabel("ubx")
plt.show()
# plt.figure()
# for phi in [0, np.pi / 2, np.pi]:
#     if phi == np.pi:
#         c = "C3"
#     else:
#         c = "C1"
#     for upt in [0.025, 0.05, 0.075]:
#         upt_phi_0 = [upt, phi]
#         upt_phi = RK4(eqs_24, upt_phi_0, params_lin, 0.1, steps)
#         plt.plot(upt_phi[:, 0], upt_phi[:, 1], c=c)
#
# upt_phi_0 = [np.sqrt(2 * (a - delta) / b), np.pi]
# upt_phi = RK4(eqs_24, upt_phi_0, params_lin, 0.1, steps)
# plt.plot(upt_phi[:, 0], upt_phi[:, 1], c="C2",
#          label=r"$\phi = \pi $" + "\t" + r"$u_{pt}=$" + f"{np.sqrt(2 * (a - delta) / b):.3g}")
#
# upt_phi_0 = [0.001, np.arccos(-delta / a)]
# upt_phi = RK4(eqs_24, upt_phi_0, params_lin, 0.1, steps)
# plt.plot(upt_phi[:, 0], upt_phi[:, 1], c="C0", label=r"$\phi =$" + f"{np.arccos(-delta / a):.3g}\t" + r"$u_{pt}=0.001$")
#
# plt.xlim([0, 0.15])
# plt.ylim([0, 2 * np.pi])
# plt.xlabel(r"$u_{pt}$")
# plt.ylabel(r"$\phi$")
# plt.legend()
# # plt.title(r"")
# plt.show()

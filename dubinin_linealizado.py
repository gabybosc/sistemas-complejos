import numpy as np
import matplotlib.pyplot as plt
from dubinin_functions import eqs_24, implicit_ubx, eq_upx, eqs_31_lin, B_field_transverse, RK4_for_linearized, secant_method

alpha = 0.005
U = 1.31
Ub = -U
M = 1.3
# Bx = 1
a = M ** 2 * np.sqrt(alpha * U)
b = M ** 2 * (1 + alpha * U) * (1 + alpha * U ** 4) / (2 * U ** 2 * (1 + U) * alpha * U)
delta = (1 + 1 / U - M ** 2 * (1 + alpha * U)) / 2
d = - M ** 2 * (1 + alpha * U ** 4) / (U ** 2 * (1 + U) * np.sqrt(alpha * U))

params = [alpha, Ub, M]
params_lin = [a, b, delta, d]

steps = 1000
upt_phi_0 = [0.001, np.pi/4]
upt_phi = RK4_for_linearized(eqs_24, upt_phi_0, params_lin, 0.1, steps)

ubx = []
upx = []
for i in range(len(upt_phi)):
    ubx.append(secant_method(implicit_ubx, -1.5, -1, upt_phi[i], params))
    upx.append(eq_upx(ubx[-1], upt_phi[i], params))


# u_plus = RK4_for_linearized(eqs_31_lin, np.array([0.01, 0.1]), params, 0.1, steps)
# upy = np.real(u_plus[:, 0])
# upz = np.imag(u_plus[:, 0])
# uby = np.real(u_plus[:, 1])
# ubz = np.imag(u_plus[:, 1])
# By, Bz = B_field_transverse([upy, uby], [upz, ubz], params)


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
ax1.set_ylabel(r"$u_{px}$")

ax2.plot(ubx, c="C1")
ax2.set_ylabel(r"$u_{bx}$")

# ax3.plot(upy, c="C0", label="upy")
# ax3.plot(upz, c="C0", linestyle="--", label="upz")
# ax3.plot(uby, c="C1", label="uby")
# ax3.plot(ubz, c="C1", linestyle="--", label="ubz")
# ax3.set_ylabel("upy, uby")
#
# ax4.plot(By, c="C0", label="By")
# ax4.plot(Bz, c="C0", linestyle="--", label="Bz")
# ax4.set_ylabel("By, Bz")
#
# ax5.plot(np.linalg.norm([By, Bz], axis=0))
# ax5.set_ylabel(r"$|B_\perp$|")

for ax in [ax1, ax2]:
    ax.set_xlim([100, 600])
    ax.grid()

# for ax in [ax1, ax2, ax3, ax4]:
#     plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)
# for ax in [ax3, ax4]:
#     ax.legend(loc="right")
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


def amplitude(u):
    return max(u)- min(u)


amp_upt = []
alpha = 0.005
U = 1.31
Ub = -U
upt_phi_0 = [0.001, np.pi / 4]

for M in np.linspace(0.8, 1.4, 10):
    a = M ** 2 * np.sqrt(alpha * U)
    b = M ** 2 * (1 + alpha * U) * (1 + alpha * U ** 4) / (2 * U ** 2 * (1 + U) * alpha * U)
    delta = (1 + 1 / U - M ** 2 * (1 + alpha * U)) / 2
    d = - M ** 2 * (1 + alpha * U ** 4) / (U ** 2 * (1 + U) * np.sqrt(alpha * U))

    params = [alpha, Ub, M]
    params_lin = [a, b, delta, d]

    upt_phi = RK4_for_linearized(eqs_24, upt_phi_0, params_lin, 0.1, steps)

    ubx = []
    upx = []
    for i in range(len(upt_phi)):
        ubx.append(secant_method(implicit_ubx, -1.5, -1, upt_phi[i], params))
        upx.append(eq_upx(ubx[-1], upt_phi[i], params))

    amp_upt.append(amplitude(upt_phi[:, 0]))

fig = plt.figure()
plt.plot(amp_upt, ".")
plt.show()
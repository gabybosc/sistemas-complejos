import numpy as np
import matplotlib.pyplot as plt
from dubinin_functions import B_field_transverse, eq_15, RK4_modfied, RK_13, eqs_13

alpha = 0.005
Ub = -1.31
U = -Ub
M = 1.3
# Bx = 1
params = [alpha, Ub, M]
a = M ** 2 * np.sqrt(alpha * U)
b = M ** 2 * (1 + alpha * U) * (1 + alpha * U ** 4) / (2 * U ** 2 * (1 + U) * alpha * U)
delta = (1 + 1 / U - M ** 2 * (1 + alpha * U)) / 2


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


u_plus = RK_13(eqs_13, [0.01, 0.1], ux, params, steps=steps)

upy = np.real(u_plus[:, 0])
upz = np.imag(u_plus[:, 0])
uby = np.real(u_plus[:, 1])
ubz = np.imag(u_plus[:, 1])

"""
Magnetic field
"""


By, Bz = B_field_transverse([upy, uby], [upz, ubz], params)

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

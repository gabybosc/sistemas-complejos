from scipy.integrate import ode
import matplotlib.pyplot as plt
import numpy as np
from funciones import RK4


def lorenz(t, r, params):
    x, y, z = r
    sigma, rho, beta = params
    xdot = sigma * (y - x)
    ydot = rho * x - y - x * z
    zdot = x * y - beta * z

    return np.array([xdot, ydot, zdot])


def correr(f, r0, tf, dt, params):
    """Resuelve la ode usando scipy, devuelve un array"""

    solver = ode(f).set_integrator("dopri5")
    solver.set_initial_value(r0, t=0.0).set_f_params(params)

    i = 0
    evolution = []
    while solver.successful() and solver.t < tf:
        i += 1
        solver.integrate(solver.t + dt)
        evolution.append(solver.y)

    return np.array(evolution)


r0 = [0, 0.5, 0.5]
sigmarhobeta = (10, 25, 8 / 3)
tf = 50
h = 1e-4
evo = correr(lorenz, r0, tf, h, sigmarhobeta)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.set_xlabel("X", fontsize=14)
ax.set_ylabel("Y", fontsize=14)
ax.set_zlabel("Z", fontsize=14)
ax.plot(
    evo[:, 0],
    evo[:, 1],
    evo[:, 2],
)
plt.show()

"""
b) Integrando como en c y usando rho=30 compare la evolución temporal de y
para las condiciones iniciales (x0,y0,z0)=(0,0.5,0.5) y (x′0,y′0,z′0)=(0,0.5,0.50001). ¿Qué observa?
# """
r0 = [0, 0.5, 0.5]
r0_2 = [0, 0.5, 0.50001]
sigmarhobeta = (10, 30, 8 / 3)

evo = correr(lorenz, r0, tf, h, sigmarhobeta)
evo2 = correr(lorenz, r0_2, tf, h, sigmarhobeta)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.set_xlabel("X", fontsize=14)
ax.set_ylabel("Y", fontsize=14)
ax.set_zlabel("Z", fontsize=14)
ax.plot(
    evo[:, 0],
    evo[:, 1],
    evo[:, 2],
    label="CI (0, 0.5, 0.5)",
)
ax.plot(
    evo2[:, 0],
    evo2[:, 1],
    evo2[:, 2],
    label="CI (0, 0.5, 0.50001)",
)
plt.legend()
plt.show()


"""
c) Resuelva numéricamente las ecuaciones de Lorenz con un método
de Runge–Kutta de orden 4 (RK4) con paso fijo.
Integre las ecuaciones con los mismos parámetros del inciso anterior
usando la condición inicial (x0,y0,z0) = (0,0.5,0.5) con
(i) el método de RK4 con pasofijo con δt=0.005
(ii) con el método utilizado en el primer inciso. Compare las dos soluciones.
Grafique la diferencia absoluta entre las dos soluciones en función del tiempo. ¿Qué ocurre?
"""

h = 0.005
pasos = int(50 / h)

sol = RK4(lorenz, 0, r0, sigmarhobeta, h, pasos)
evo = correr(lorenz, r0, tf, h, sigmarhobeta)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.set_xlabel("X", fontsize=14)
ax.set_ylabel("Y", fontsize=14)
ax.set_zlabel("Z", fontsize=14)
ax.plot(
    sol[:, 0],
    sol[:, 1],
    sol[:, 2],
    label="RK4",
)
ax.plot(
    evo[:, 0],
    evo[:, 1],
    evo[:, 2],
    label="scipy ode",
)
plt.legend()


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.set_xlabel("X", fontsize=14)
ax.set_ylabel("Y", fontsize=14)
ax.set_zlabel("Z", fontsize=14)
ax.plot(
    np.abs(sol[1:, 0] - evo[:, 0]),
    np.abs(sol[1:, 1] - evo[:, 0]),
    np.abs(sol[1:, 2] - evo[:, 0]),
)
plt.title("diferencia entre ambos")
plt.show()

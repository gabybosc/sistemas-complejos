"""Consideremos el sistema de Lorenz ()

x˙= σ(y−x)
y˙= ρx−y−xz
z˙= xy−βz

donde x, y y z son funciones que dependen del tiempo.

a) Para valores de β=8/3, σ=10 y ρ=25 integre utilizando odeint por un tiempo T=50 usando dt=10−4 
y con condiciones iniciales (x0,y0,z0)=(0,0.5,0.5). 
Grafique z(t), y(t) y z(y). Pruebe cambiar valores de dt y compare las trayectorias de cada una de las variables.

Integre usando el siguiente bloque
solver = scipy.integrate.ode(f).set_integrator('dopri5')
solver.set_initial_value(x0, t=0.0)

i = 0
while solver.successful() and solver.t < tf:
    i += 1
    solver.integrate(solver.t + dt)
    evolution[i,:] = solver.y

Este es un integrador de Runge-Kutta 4(5) con pasos temporales variables
 a partir del error local. Compare con los resultados anteriores. 
 Grafique el caso anterior en el espacio de fases tridimensional. ¿Qué conclusiones obtiene?"""

from scipy.integrate import ode, odeint
import numpy as np
import matplotlib.pyplot as plt
from funciones import RK4


def lorenz(t, r, params):
    x, y, z = r
    sigma, rho, beta = params
    xdot = sigma * (y - x)
    ydot = rho * x - y - x * z
    zdot = x * y - beta * z

    return [xdot, ydot, zdot]


def correr(f, r0, params):
    solver = ode(f).set_integrator("dopri5")
    solver.set_initial_value(r0, t=0.0).set_f_params(params)

    i = 0
    tf = 50
    dt = 1e-4
    evolution = []
    while solver.successful() and solver.t < tf:
        i += 1
        solver.integrate(solver.t + dt)
        evolution.append(solver.y)

    return np.array(evolution)


r0 = [0, 0.5, 0.5]
sigmarhobeta = (10, 25, 8 / 3)

# plt.plot(evo[:, 0], evo[:, 1])
# plt.show()

"""
b) Integrando como en c y usando rho=30 compare la evolución temporal de y
para las condiciones iniciales (x0,y0,z0)=(0,0.5,0.5) y (x′0,y′0,z′0)=(0,0.5,0.50001). ¿Qué observa?
"""
solver = ode(lorenz).set_integrator("dopri5")
r0 = [0, 0.5, 0.5]
r0_2 = [0, 0.5, 0.50001]
sigmarhobeta = (10, 30, 8 / 3)
# evo = correr(lorenz, r0, sigmarhobeta)
# evo2 = correr(lorenz, r0_2, sigmarhobeta)

# plt.figure()
# plt.plot(evo[:, 0], evo[:, 1])
# plt.figure()
# plt.plot(evo2[:, 0], evo2[:, 1])
# plt.show()


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


def RK4_lorenz(f, t0, r0, params, h, pasos):
    # inicial
    x, y, z = r0
    t = t0
    X, Y, Z = [x], [y], [z]

    for i in range(pasos):
        k1 = f(t, [x, y, z], params)
        k2 = f(
            t + h / 2, [x + h / 2 * k1[0], y + h / 2 * k1[1], z + h / 2 * k1[2]], params
        )
        k3 = f(
            t + h / 2, [x + h / 2 * k2[0], y + h / 2 * k2[1], z + h / 2 * k2[2]], params
        )
        k4 = f(t + h, [x + h * k3[0], y + h * k3[1], z + h * k3[2]], params)
        t = t + h
        x = x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y = y + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z = z + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        X.append(x)
        Y.append(y)
        Z.append(z)

    return (np.array(X), np.array(Y), np.array(Z))


sol = RK4_lorenz(lorenz, 0, r0, sigmarhobeta, h, pasos)

plt.plot(sol[0], sol[1])
plt.show()

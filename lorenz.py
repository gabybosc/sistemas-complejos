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

    return np.array([xdot, ydot, zdot])


def correr(f, r0, params):
    """Resuelve la ode usando scipy, devuelve un array"""

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

sol_ode = correr(lorenz, r0, sigmarhobeta)
sol = RK4(lorenz, 0, r0, sigmarhobeta, h, pasos)

plt.plot(sol[:, 0], sol[:, 1])
plt.plot(sol_ode[:, 0], sol_ode[:, 1])
plt.show()

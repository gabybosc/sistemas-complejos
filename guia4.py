import numpy as np
import matplotlib.pyplot as plt


def compute_advection(f, k):
    """Calcula v*vx usando el método pseudoespectral y devuelve el resultado en el espacio Fourier"""
    fx = 1j * k * f  # derivamos en el espacio de Fourier
    v = np.fft.irfft(f)  # vuelvo al espacio real para poder multiplicar!
    vx = np.fft.irfft(fx)  # íd.
    v = v * vx  # multiplico en el espacio real
    return v


def rip(t, u, params):
    k, beta = params

    utilde = np.fft.rfft(u)
    u3tilde = -1j * k**3 * utilde
    u3 = np.fft.irfft(u3tilde)
    f = -compute_advection(utilde, k) - beta * u3
    return f


# def evol(f, t, u_k, Nt, params, dt):
#     # u = np.zeros((len(u_k), len(u_k[0,:])))
#     for i in range(Nt - 1):
#         u_k[i + 1, :] = u_k[i, :] + dt * f(t, u_k[i, :], params)

#     u = np.fft.irfft(u_k)
#     return u


def RK2(f, t0, r0, params, h):
    """RK de orden 2"""
    # inicial
    t = t0
    r = r0

    k1 = f(t, r, params)
    k4 = f(t + h, r + h * k1, params)

    r = r + h / 6 * (k1 + k4)

    return np.array(r)


def evol(f, u, Nt, params, dt):
    # Evoluciona en el tiempo la PDE usando el método pseudoespectral y Runge-Kutta de segundo orden
    t0 = 0
    for i in range(Nt - 1):
        u[i + 1, :] = RK2(f, t0, u[i, :], params, dt)
    # utilde[int(N / 3) :] = 0  # Dealiasing (eliminemos modos espurios!)

    # out = np.fft.irfft(utilde)  t# Vuelve al espacio real

    return u


beta = 0.022
nx = 128
dx = 2 * np.pi / nx
dt = 5e-5
t = np.arange(0, 1, dt)
nt = len(t)
x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
k = np.arange(0, nx / 2 + 1)

v = 1

u_ini = 3 * v * np.cosh(np.sqrt(v / 4 / beta) * (x - np.pi)) ** -2
u = np.zeros((nt, nx))
u[0, :] = u_ini

sol = evol(rip, u, nt, [k, beta], dt)
plt.imshow(sol, aspect="auto")
plt.show()

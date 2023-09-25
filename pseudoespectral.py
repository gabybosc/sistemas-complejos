import numpy as np
from funciones import RK2
import matplotlib.pyplot as plt

# Defino mis puntos en el espacio real y los modos para el espacio de Fourier

dt = 1e-3
step = 10000
N = 512
nu = 1e-2

x = np.linspace(0, 2 * np.pi, N, endpoint=False)  # Coordenada espacial en [0,2*pi)
t = np.arange(step) * dt  # Tiempo

k = np.arange(0, N / 2 + 1)  # Números de onda ordenados como en la FFT

u = np.sin(x)  # condición inicial


def rip(t, u, params):
    k, nu = params
    f = -(k**2) * nu * u - compute_advection(u, k)
    return f


def RK2(f, t0, r0, params, h):
    """RK de orden 2"""
    # inicial
    t = t0
    r = r0

    k1 = f(t, r, params)
    k4 = f(t + h, r + h * k1, params)

    r = r + h / 6 * (k1 + k4)

    return np.array(r)


def evol(f, u, k, params, dt):
    # Evoluciona en el tiempo la PDE usando el método pseudoespectral y Runge-Kutta de segundo orden
    utilde = np.fft.rfft(u)  # vaya al espacio de Fourier
    nu, N = params
    t0 = 0
    utilde = RK2(f, t0, utilde, [k, nu], dt)  # escriba aquí su integrador temporal
    utilde[int(N / 3) :] = 0  # Dealiasing (eliminemos modos espurios!)

    out = np.fft.irfft(utilde)  # Vuelva del espacio de Fourier

    return out


def compute_advection(f, k):
    """Calcula v*vx usando el método pseudoespectral y devuelve el resultado en el espacio Fourier"""

    fx = 1j * k * f  # derivamos en el espacio de Fourier
    v = np.fft.irfft(f)  # vuelvo al espacio real para poder multiplicar!
    vx = np.fft.irfft(fx)  # íd.
    v = v * vx  # multiplico en el espacio real
    out = np.fft.rfft(v)  # vuelvo al espacio de Fourier

    return out


# sol = evol(rip, u, k, [nu, N], dt)

sol = np.zeros((N, step))
sol[:, 0] = u
for i in np.arange(step - 1):  # Evolución temporal
    sol[:, i + 1] = evol(rip, sol[:, i], k, [nu, N], dt)


for i in [0, 1900, 3900, 5900, 7900, 9900]:
    plt.plot(x, sol[:, i], label=f"iteración {i}")
plt.legend()
plt.title("Formación del choque")
plt.show()

import numpy as np
from funciones import RK2
import matplotlib.pyplot as plt

# Defino mis puntos en el espacio real y los modos para el espacio de Fourier

dt = 1e-3
step = 10000
N = 512
nu = 1e-3

x = np.linspace(0, 2 * np.pi, N, endpoint=False)  # Coordenada espacial en [0,2*pi)
t = np.arange(step) * dt  # Tiempo

k = np.arange(0, N / 2 + 1)  # Números de onda ordenados como en la FFT

u = 0  # -np.cos(x)  # condición inicial


def kpz(t, u, params):
    k, nu, N = params
    eta = compute_random(N)
    f = -(k**2) * nu * u - 0.5 * compute_vv(u, k) + eta
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
    utilde = RK2(f, t0, utilde, [k, nu, N], dt)  # escriba aquí su integrador temporal
    utilde[int(N / 3) :] = 0  # Dealiasing (eliminemos modos espurios!)

    out = np.fft.irfft(utilde)  # Vuelva del espacio de Fourier

    return out


def compute_vv(f, k):
    # Calcula v^2 usando el método pseudoespectral y devuelve el resultado en el espacio Fourier
    dx = 1j * k * f
    v = np.fft.irfft(dx)
    out = np.fft.rfft(v * v)
    return out


def compute_random(N):
    # Genera ruido Gaussiano con media nula y varianza unitaria en el espacio Fourier
    phase = 2 * np.pi * np.random.rand(int(N / 2 + 1))
    ampl = np.random.randn(int(N / 2 + 1))
    out = ampl * np.exp(1j * phase)
    out[0] = 0.0
    return out


# sol = evol(rip, u, k, [nu, N], dt)

sol = np.zeros((N, step))
sol[:, 0] = u
for i in np.arange(step - 1):  # Evolución temporal
    sol[:, i + 1] = evol(kpz, sol[:, i], k, [nu, N], dt)


for i in [0, 1900, 3900, 5900, 7900, 9900]:
    plt.plot(x, sol[:, i], label=f"iteración {i}")
plt.legend()
plt.title("Formación del choque")
plt.show()


def dif_centradas(f, dx, Nt, Nx):
    """
    hace dif centradas 1D
    """
    df = np.zeros((Nt, Nx))
    ddf = np.zeros((Nt, Nx))

    for j in range(Nt - 1):
        for i in range(Nx - 1):
            df[j, i] = (f[j, i + 1] - f[j, i - 1]) / (2 * dx)
            ddf[j, i] = (f[j, i + 1] - 2 * f[j, i] + f[j, i - 1]) / dx**2

    return df, ddf


df, ddf = dif_centradas(np.transpose(sol), np.abs(x[1] - x[0]), len(t), len(x))
for i in [0, 1900, 3900, 5900, 7900, 9900]:
    plt.plot(x, df[i, :], label=f"iteración {i}")


plt.imshow(np.transpose(df))
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Defino mis puntos en el espacio real y los modos para el espacio de Fourier

dt = 1e-3
Nx = 1024
nu = 5e-5
a = 3
c0 = 15
w1 = 2
w2 = 1

x = np.linspace(0, 2 * np.pi, Nx)  # Coordenada espacial en [0,2*pi]
t = np.arange(0, 4 * np.pi, dt)  # Tiempo
Nt = len(t)
k = np.arange(0, Nx / 2 + 1)

N = np.zeros((Nt, Nx))
N[0, :] = 10

C = np.zeros((Nt, Nx))
for i in range(Nt):
    for j in range(Nx):
        C[i, j] = c0 * (1 + 0.8 * np.cos(w1 * t[i]) * np.sin(w2 * x[j]))


def pseudo(N, k):
    fn = np.fft.rfft(N)
    dn = -(k**2) * fn
    out = np.fft.irfft(dn)
    return out


def rip(t, n, params, c):
    k, a, nu = params

    f = a * (1 - n / c) + nu * pseudo(n, k)
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


def evol(f, N, Nt, params, C, dt):
    for i in range(Nt - 1):
        c = C[i, :]
        n = N[i, :]
        N[i + 1, :] = N[i, :] + dt * f(t, n, params, c)

    return N


sol = evol(rip, N, Nt, [k, a, nu], C, dt)

plt.imshow(sol, aspect="auto")
plt.show()

# rip(t, N, [k, a, nu], C[1, :])

# sol = np.zeros((N, step))
# sol[:, 0] = u
# for i in np.arange(step - 1):  # Evolución temporal
#     sol[:, i + 1] = evol(rip, sol[:, i], k, [nu, N], dt)


# for i in [0, 1900, 3900, 5900, 7900, 9900]:
#     plt.plot(x, sol[:, i], label=f"iteración {i}")
# plt.legend()
# plt.title("Formación del choque")
# plt.show()

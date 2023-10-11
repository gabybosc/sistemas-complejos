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

# C = np.zeros((Nt, Nx))
# for i in range(Nt):
#     for j in range(Nx):
#         C[i, j] = c0 * (1 + 0.8 * np.cos(w1 * t[i]) * np.sin(w2 * x[j]))

# np.save("C.npy", C)
C = np.load("C.npy")


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


# sol = evol(rip, N, Nt, [k, a, nu], C, dt)
# np.save("solucion_a.npy", sol)
sol = np.load("solucion_a.npy")
# plt.show()


# calculamos la matrix U'U/Nt
mat = np.dot(np.transpose(sol), sol) / Nt

# calculamos av y av
l, vv = np.linalg.eigh(mat)
v = np.transpose(vv)

# la matriz phi será la matriz de av traspuesta
phi = np.transpose([v[-1], v[-2], v[-3]])  # , v[-4], v[-5]])

# A es el producto de la U con phi
A = np.dot(sol, phi)

# la solución reducida es A * phi
sol_3m = np.dot(A, np.transpose(phi))

fig = plt.figure()
fig.subplots_adjust(
    top=0.95, bottom=0.1, left=0.05, right=0.95, hspace=0.005, wspace=0.15
)
ax1 = plt.subplot2grid((2, 1), (0, 0))
ax2 = plt.subplot2grid((2, 1), (1, 0))

ax1.imshow(sol, aspect="auto")
ax2.imshow(sol_3m, aspect="auto")
plt.show()

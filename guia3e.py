import numpy as np
import matplotlib.pyplot as plt

# Defino mis puntos en el espacio real y los modos para el espacio de Fourier

dt = 1e-3
Nx = 1024
nu = 5e-5
alpha = 3
c0 = 15
w1 = 2
w2 = 1

x = np.linspace(0, 2 * np.pi, Nx)  # Coordenada espacial en [0,2*pi]
t = np.arange(0, 4 * np.pi, dt)  # Tiempo
Nt = len(t)
k = np.arange(0, Nx / 2 + 1)


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


# N5 = np.zeros((Nt, Nx))

# a5 = 10 * np.sum(phi5, axis=0)

# CI5 = np.dot(a5, np.transpose(phi5))

# N5[0, :] = CI5


p = np.zeros((Nt, Nx, 3))
B = []
i = 0
for g in [1, 3, 7]:
    A = np.load(f"A{g}.npy")
    phi = np.load(f"phi{g}.npy")

    N = np.zeros((Nt, Nx))

    a = 10 * np.sum(phi, axis=0)
    CI = np.dot(a, np.transpose(phi))
    N[0, :] = CI
    sol = evol(rip, N, Nt, [k, alpha, nu], C, dt)
    b = np.dot(sol, phi)

    B.append(b)
    p[:, :, i] = sol
    i += 1

fig = plt.figure()
fig.subplots_adjust(
    top=0.95, bottom=0.1, left=0.05, right=0.95, hspace=0.005, wspace=0.15
)
plt.title("Coeficientes del primer modo")
ax1 = plt.subplot2grid((3, 1), (0, 0))
ax2 = plt.subplot2grid((3, 1), (1, 0), sharex=ax1)
ax3 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
ax1.plot(np.load(f"A1.npy"), label="A")
ax1.plot(B[0], "--", label="b")
ax1.set_ylabel("Coeficientes para 1 modo")
ax2.plot(np.load(f"A3.npy")[:, 0], label="A")
ax2.plot(B[1][:, 0], "--", label="b")
ax2.set_ylabel("Coeficientes para 3 modos")
ax3.plot(np.load(f"A7.npy")[:, 0], label="A")
ax3.plot(B[2][:, 0], "--", label="b")
ax3.set_ylabel("Coeficientes para 7 modos")
plt.show()


fig = plt.figure()
fig.subplots_adjust(
    top=0.95, bottom=0.1, left=0.05, right=0.95, hspace=0.005, wspace=0.15
)
plt.title("Coeficientes de los modos 2 y 3")
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (1, 0), sharex=ax1)
ax3 = plt.subplot2grid((2, 2), (0, 1))
ax4 = plt.subplot2grid((2, 2), (1, 1), sharex=ax3)

ax1.plot(np.load(f"A3.npy")[:, 1], label="A")
ax1.plot(B[1][:, 1], "--", label="b")
ax2.plot(np.load(f"A7.npy")[:, 1], label="A")
ax2.plot(B[2][:, 1], "--", label="b")

ax3.plot(np.load(f"A3.npy")[:, 2], label="A")
ax3.plot(B[1][:, 2], "--", label="b")
ax4.plot(np.load(f"A7.npy")[:, 2], label="A")
ax4.plot(B[2][:, 2], "--", label="b")

plt.show()


fig = plt.figure()
fig.subplots_adjust(
    top=0.95, bottom=0.1, left=0.05, right=0.95, hspace=0.005, wspace=0.15
)
ax1 = plt.subplot2grid((2, 1), (0, 0))
ax2 = plt.subplot2grid((2, 1), (1, 0))
ax1.imshow(p[:, :, 0], aspect="auto")
ax2.imshow(p[:, :, 1], aspect="auto")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Defino mis puntos en el espacio real y los modos para el espacio de Fourier

dt = 1e-3
Nx = 1024
nu = 5e-5
c0 = 15
w2 = 1

x = np.linspace(0, 2 * np.pi, Nx)  # Coordenada espacial en [0,2*pi]
t = np.arange(0, 4 * np.pi, dt)  # Tiempo
Nt = len(t)
k = np.arange(0, Nx / 2 + 1)

N = np.zeros((Nt, Nx))
N[0, :] = 10


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


C1 = np.load("C_omega1.npy")
C2 = np.load("C.npy")
for C in [C1, C2]:
    B = []
    lst_A = []
    for alpha in [0.1, 1, 1.5, 3]:
        sol = evol(rip, N, Nt, [k, alpha, nu], C, dt)

        # calculamos la matrix U'U/Nt
        mat = np.dot(np.transpose(sol), sol) / Nt

        # calculamos av y av
        l, vv = np.linalg.eigh(mat)
        v = np.transpose(vv)

        # la matriz phi será la matriz de av traspuesta
        phi = np.transpose([v[-1], v[-2], v[-3]])

        # A es el producto de la U con phi
        A = np.dot(sol, phi)

        # la solución reducida es A * phi
        sol_3m = np.dot(A, np.transpose(phi))

        N = np.zeros((Nt, Nx))

        a = 10 * np.sum(phi, axis=0)
        CI = np.dot(a, np.transpose(phi))
        N[0, :] = CI
        sol_e = evol(rip, N, Nt, [k, alpha, nu], C, dt)
        b = np.dot(sol_e, phi)
        B.append(b)
        lst_A.append(A)

    m = 1

    fig = plt.figure()
    fig.subplots_adjust(
        top=0.95, bottom=0.1, left=0.05, right=0.95, hspace=0.005, wspace=0.15
    )
    plt.title("Coeficientes del primer modo")
    ax1 = plt.subplot2grid((4, 1), (0, 0))
    ax2 = plt.subplot2grid((4, 1), (1, 0), sharex=ax1)
    ax3 = plt.subplot2grid((4, 1), (2, 0), sharex=ax1)
    ax4 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

    ax1.plot(lst_A[0][:, m], label="A")
    ax2.plot(lst_A[1][:, m], label="A")
    ax3.plot(lst_A[2][:, m], label="A")
    ax4.plot(lst_A[3][:, m], label="A")
    ax1.plot(B[0][:, m], "--", label="b")
    ax2.plot(B[1][:, m], "--", label="b")
    ax3.plot(B[2][:, m], "--", label="b")
    ax4.plot(B[3][:, m], "--", label="b")

    ax1.set_ylabel("alpha = 0.1")
    ax2.set_ylabel("alpha = 1")
    ax3.set_ylabel("alpha = 3")
    ax4.set_ylabel("alpha = 10")
    plt.show()

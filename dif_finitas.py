import numpy as np
import matplotlib.pyplot as plt


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


def solver_burgers(u, nu, dx, dt):
    """
    hace dif adelantadas 1D para Burgers con CC periódicas
    """
    Nx = len(u)
    Nt = len(u[0, :])

    for j in range(Nt - 2):
        for i in range(Nx - 3):
            # ojo que a todo le estoy restando uno extra porque empiezo desde cero!
            u[i, j + 1] = (
                u[i, j]
                - dt / dx * (u[i + 1, j] - u[i, j])
                + nu * dt / dx**2 * (u[i, j] - u[i + 1, j] + u[i + 2, j])
            )
        u[Nx - 2, j + 1] = (
            u[-1, j]
            - dt / dx * (u[0, j] - u[-1, j])
            + nu * dt / dx**2 * (u[-1, j] - u[0, j] + u[1, j])
        )
        u[Nx - 1, j + 1] = (
            u[0, j]
            - dt / dx * (u[1, j] - u[0, j])
            + nu * dt / dx**2 * (u[0, j] - u[1, j] + u[2, j])
        )

    return u


nx = 512
nt = 1000
dx = 2 * np.pi / (nx - 1)
dt = 1e-3
nu = 0.01

x = np.linspace(0, 2 * np.pi, nx, endpoint=False)  # Coordenada espacial en [0,2*pi)
t = np.arange(nt) * dt

u = np.zeros([nx, nt])
u[:, 0] = np.sin(x)  # alguna condicion inicial

U = solver_burgers(u, nu, dx, dt)

for t in range(100, 105):
    plt.plot(x[:450], U[:450, t])
plt.show()

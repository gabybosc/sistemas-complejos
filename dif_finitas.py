import numpy as np
import matplotlib.pyplot as plt


nx = 512
nt = 1000
#dx = 2 * np.pi / (nx - 1)
dt = 1e-3
nu = 0.01

x = np.linspace(0, 2 * np.pi, nx, endpoint=False)  # Coordenada espacial en [0,2*pi)
t = np.arange(nt) * dt

dx = x[1]-x[0]

def solver_burgers(u, nu, dx, dt):
    """
    hace dif adelantadas 1D para Burgers con CC periódicas
    """
    Nx = len(u)
    Nt = len(u[0, :])
    for j in range(Nt - 1):
        u[0, j + 1] = (
                u[0, j]
                - dt / dx * 0.5 * u[0, j] * (u[1, j] - u[-1, j])
                + nu * dt / dx**2 * (u[-1, j] - 2*u[0, j] + u[1, j])
            )
        for i in range(1, Nx - 1):
            # ojo que a todo le estoy restando uno extra porque empiezo desde cero!
            u[i, j + 1] = (
                u[i, j]
                - dt / dx * 0.5 * u[i,j] * (u[i + 1, j] - u[i - 1, j])
                + nu * dt / dx**2 * (u[i - 1, j] - 2*u[i, j] + u[i + 1, j])
            )
        u[Nx - 1, j + 1] = (
                u[Nx - 1, j]
                - dt / dx * 0.5 * u[Nx - 1, j] * (u[0, j] - u[Nx - 2, j])
                + nu * dt / dx**2 * (u[Nx - 2, j] - 2*u[Nx - 1, j] + u[0, j])
            )
    return u

u = np.zeros([nx, nt])
u[:, 0] = np.sin(x)  # alguna condicion inicial
#u[:, 0] = np.sin(3 * x) ** 2 + np.cos(2 * x)  # alguna condicion inicial
# func escalón
# u[:250, 0] = 0
# u[250:, 0] = 1e-3

U = solver_burgers(u, nu, dx, dt)

for tt in [100, 500, 999]:
    plt.plot(x, U[:, tt])
plt.ylim([-2, 2])
plt.show()

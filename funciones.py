import numpy as np


def RK4(f, t0, u0, v0, alpha, h, pasos):
    # inicial
    u = u0
    v = v0
    t = t0
    U, V = [u], [v]

    for i in range(pasos):
        k1 = f([u, v], t, alpha)
        k2 = f([u + h / 2 * k1[0], v + h / 2 * k1[1]], t + h / 2, alpha)
        k3 = f([u + h / 2 * k2[0], v + h / 2 * k2[1]], t + h / 2, alpha)
        k4 = f([u + h * k3[0], v + h * k3[1]], t + h, alpha)
        t = t + h
        u = u + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v = v + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        U.append(u)
        V.append(v)

    return (np.array(U), np.array(V))

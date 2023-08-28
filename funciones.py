import numpy as np


def RK4(f, t0, u0, params, h, pasos):
    # inicial
    u, v = u0
    t = t0
    U, V = [u], [v]

    for i in range(pasos):
        k1 = f([u, v], t, params)
        k2 = f([u + h / 2 * k1[0], v + h / 2 * k1[1]], t + h / 2, params)
        k3 = f([u + h / 2 * k2[0], v + h / 2 * k2[1]], t + h / 2, params)
        k4 = f([u + h * k3[0], v + h * k3[1]], t + h, params)
        t = t + h
        u = u + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v = v + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        U.append(u)
        V.append(v)

    return (np.array(U), np.array(V))


def RK4_lorenz(f, t0, r0, params, h, pasos):
    # inicial
    x, y, z = r0
    t = t0
    X, Y, Z = [x], [y], [z]

    for i in range(pasos):
        k1 = f(t, [x, y, z], params)
        k2 = f(
            t + h / 2, [x + h / 2 * k1[0], y + h / 2 * k1[1], z + h / 2 * k1[2]], params
        )
        k3 = f(
            t + h / 2, [x + h / 2 * k2[0], y + h / 2 * k2[1], z + h / 2 * k2[2]], params
        )
        k4 = f(t + h, [x + h * k3[0], y + h * k3[1], z + h * k3[2]], params)
        t = t + h
        x = x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y = y + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z = z + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        X.append(x)
        Y.append(y)
        Z.append(z)

    return (np.array(X), np.array(Y), np.array(Z))


def RK4_gen(f, t0, r0, params, h, pasos):
    # inicial
    t = t0

    R = [r0]
    for i in range(pasos):
        k1 = f(t, r0, params)
        k2 = f(t + h / 2, r0 + h / 2 * k1, params)
        k3 = f(t + h / 2, r0 + h / 2 * k2, params)
        k4 = f(t + h, r0 + h * k3, params)
        t = t + h
        r = r + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        R.append(r)

    return R

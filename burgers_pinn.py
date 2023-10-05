import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(torch.nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, x):
        h = x
        for hidden in self.layers[:-1]:
            h = torch.tanh(hidden(h))
        output = self.layers[-1]
        y = output(h)
        return y


nu = 0

# Condiciones en t = 0

t_data_init = torch.linspace(0, 1, 100).view(-1, 1)
x_data_init = torch.linspace(0, 2 * np.pi, 100).view(-1, 1)
y_data_init = np.sin(x_data_init).view(-1, 1)
# Junta las condiciones iniciales para que tengan la forma correcta:
input_init = torch.cat(
    (x_data_init.flatten().view(-1, 1), t_data_init.flatten().view(-1, 1)), dim=1
)


# condiciones de contorno
t_data_cc = torch.cat(
    (torch.linspace(0, 1, 100), torch.linspace(0, 1, 100)), dim=0
).view(
    -1, 1
)  # 100 puntos temporales dos veces
x_data_cc = (
    2 * torch.pi * torch.cat((torch.zeros(100), torch.ones(100)), dim=0).view(-1, 1)
)  # 100 puntos en 0 y 100 en 1
y_data_cc = torch.cat((torch.zeros(100), torch.zeros(100)), dim=0).view(
    -1, 1
)  # 200 puntos que valen 0
input_cc = torch.cat(
    (x_data_cc.flatten().view(-1, 1), t_data_cc.flatten().view(-1, 1)), dim=1
)  # Los juntamos para que pueda ser input la red


t_physics = torch.linspace(0, 1, 100).requires_grad_(True)
x_physics = torch.linspace(0, 2 * np.pi, 100).requires_grad_(True)
x_grid, t_grid = torch.meshgrid(t_physics, x_physics, indexing="ij")
x_grid = x_grid[:, :, None].requires_grad_(
    True
)  # Agregamos una dimensión al final para que pueda ser input de la red
t_grid = t_grid[:, :, None].requires_grad_(
    True
)  # Agregamos una dimensión al final para que pueda ser input de la red
input_physics = torch.cat((x_grid, t_grid), dim=-1)


pinn = MLP([2] + [20] * 8 + [1])  # la red feedforward
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-4)  # usamos este optimizador
l = 1  # Lambda
iterations = 100000

for epoch in range(iterations):
    optimizer.zero_grad()
    # Condiciones iniciales
    yh_init = pinn(
        input_init.to(torch.float32),
    )
    loss1 = torch.mean((yh_init - y_data_init) ** 2)
    # condiciones de contorno
    yh_cc = pinn(input_cc.to(torch.float32))
    loss2 = torch.mean((yh_cc - y_data_cc) ** 2)

    # Condiciones de la física
    yhp = pinn(input_physics.to(torch.float32))
    dt = torch.autograd.grad(yhp, t_grid, torch.ones_like(yhp), create_graph=True)[
        0
    ]  # computamos u_t
    dx = torch.autograd.grad(yhp, x_grid, torch.ones_like(yhp), create_graph=True)[
        0
    ]  # computamos u_x
    dx2 = torch.autograd.grad(
        dx, x_grid, torch.ones_like(yhp), create_graph=True, allow_unused=True
    )[
        0
    ]  # computamos u_xx
    physics = dt + yhp * dx - nu * dx2
    loss3 = l * torch.mean(physics**2)
    loss = loss1 + loss2 + loss3  # Sumamos todos los errores
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        torch.save(pinn, "pinn_burgers2.pt")
# with torch.autograd.no_grad():
# print(epoch, loss1, loss2, "Traning Loss:", loss.data)
# pinn = torch.load("pinn_burgers.pt")

t0 = 0  # Tiempo inicial
tf = 1  # Tiempo final
N = 100  # Numero de puntos
test_time = torch.linspace(t0, tf, N).view(-1, 1)
test_x = torch.linspace(0, 2 * np.pi, N).view(-1, 1)
test_physics = torch.cat((test_x, test_time), dim=-1)

yhp = pinn(input_physics)  # Evaluación de la red

plt.figure(0)
for i in range(0, 100, 10):
    plt.plot(x_physics.detach().numpy(), yhp.detach().numpy()[i, :])
plt.show()

# plt.plot(y_test[:-1, 0], yhp[:, 0].detach().numpy())
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

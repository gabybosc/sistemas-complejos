# Importamos las librerías necesarias para esto
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(torch.nn.Module):
    """
    Multilayer perceptron (MLP) // Perceptríon Multicapa .

    Esta clase define una red neuronal feedforward con múltiples capas ocultas
    lineales, funciones de activación tangente hiperbólica en  las capas ocultas
    y una salida lineal.

    Args:
        sizes (lista): Lista de enteros que especifica el número de neuronas en
        cada capa. El primer elemento debe coincidir con la dimensión de entrada
        y el último con la dimensión de salida.

    Atributos:
        capas (torch.nn.ModuleList): Lista que contiene las capas lineales del MLP.

    Métodos:
        forward(x): Realiza una pasada hacia adelante a través de la red MLP.

    Ejemplo:
        tamaños = [entrada_dim, oculta1_dim, oculta2_dim, salida_dim]
        mlp = MLP(tamaños)
        tensor_entrada = torch.tensor([...])
        salida = mlp(tensor_entrada)
    """

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


delta = 2
w0 = 18


def oscilador(t, d, w0):
    """
    Solución al oscilador subamortiguado con CI
    x = 1, dxdt = 0
    """
    Omega = np.sqrt(w0**2 - delta**2)

    x = (np.cos(Omega * t) + delta / Omega * np.sin(Omega * t)) * np.exp(-delta * t)

    return x


"""
Defina los puntos que serán los datos "medidos" de nuestro problema.
Vamos a tomar 20 datos equiespaciados entre t=0 y t=0.5
"""

t_data = torch.linspace(0, 0.5, 20).view(-1, 1)  # Tiempos de los datos
y_data = oscilador(t_data, delta, w0).view(-1, 1)  # Valores de los datos 'medidos'
mlp = MLP([1, 32, 32, 32, 1])
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

"""
En la siguiente celda se encuentra el loop de entrenamiento. 
Este consiste en darle a la red los datos, computar el valor 
la función costo que elegimos usando los datos que tenemos, 
calcular los gradientes de todos los parámetros θ y cambiarlos 
de forma conveniente para bajar el error.

Al entrenar, se debe elegir el número de veces que queremos 
realizar este proceso y eso lo seteamos con el parámetro iterations.

"""
iterations = 25000

for epoch in range(iterations):
    optimizer.zero_grad()  # El optimizador tiene guardados los gradientes y esto los resetea para poder volver a calcularlos
    yh = mlp(t_data)  # Aplicamos la red a los datos
    loss = torch.mean((yh - y_data) ** 2)  # Computamos el error cuadrático medio
    loss.backward()  # Calcula los gradientes
    optimizer.step()  # El optimizador evoluciona los parámetros

# with torch.autograd.no_grad():
# 	print(epoch,loss,"Traning Loss:",loss.data) #Imprimimos el error


t0 = 0  # Tiempo inicial
tf = 1  # Tiempo final
N = 1000  # Numero de puntos
test_time = torch.linspace(t0, tf, N).view(
    -1, 1
)  # .requires_grad_(True)  # le saqué el requires grad porque es None
y_test = oscilador(test_time, delta, w0).view(-1, 1)  # Esta es la solución teórica
yhp = mlp(test_time)  # Evaluación de la red

plt.plot(test_time.detach().numpy(), y_test, label="teórica")
plt.plot(test_time.detach().numpy(), yhp.detach().numpy(), label="red")
plt.xlabel("t")
plt.ylabel("x")
plt.legend()
plt.show()


t_physics = torch.linspace(0, 1, 20).view(-1, 1).requires_grad_(True)


iterations = 15000
l = 1e-4  # Lambda
pinn = MLP([1, 32, 32, 32, 1])
for epoch in range(iterations):
    optimizer.zero_grad()
    yh = pinn(t_data)
    loss1 = torch.mean((yh - y_data) ** 2)
    yhp = pinn(t_physics)
    # Computo de derivada
    xdot = torch.autograd.grad(yhp, t_physics, torch.ones_like(yhp), create_graph=True)[
        0
    ]
    xddot = torch.autograd.grad(
        xdot, t_physics, torch.ones_like(xdot), create_graph=True
    )[0]
    physics = xddot + 2 * delta * xdot + w0**2 * yhp  # la ec dif
    loss2 = l * torch.mean(
        physics**2
    )  # Calculo el error cuadrático medio para la física

    loss = loss1 + loss2  # Se suma el error de la física con el de los datos
    loss.backward()
    optimizer.step()

    # with torch.autograd.no_grad():
    #     print(epoch, loss1, loss2, "Traning Loss:", loss.data)


t0 = 0  # Tiempo inicial
tf = 1  # Tiempo final
N = 20  # Numero de puntos
test_time = torch.linspace(t0, tf, N).view(
    -1, 1
)  # .requires_grad_(True)  # le saqué el requires grad porque es None
y_test = oscilador(test_time, delta, w0).view(-1, 1)  # Esta es la solución teórica
yhp = pinn(test_time)  # Evaluación de la red

plt.plot(test_time.detach().numpy(), y_test, label="teórica")
plt.plot(test_time.detach().numpy(), yhp.detach().numpy(), label="red")
plt.xlabel("t")
plt.ylabel("x")
plt.legend()
plt.show()

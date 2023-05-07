from mylibneural.train import train
from mylibneural.neuralnetwork import NeuralNet
from mylibneural.layers import Linear, Tanh, Sigmoid
from mylibneural.optim import SGD
from typing import List

import numpy as np

inputs = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
  [40, 13],   # Aaron
  [-20, -9],  # Selene
])

targets = np.array([
    [1, 0], #M
    [0, 1], #H
    [0, 1], #H
    [1, 0], #M
    [0, 1], #H
    [1, 0], #M
])

net = NeuralNet([
    Linear(input_size=2, output_size=50),
    Sigmoid(),
    Linear(input_size=50, output_size=2)
])

train(
    net,
    inputs,
    targets,
    num_epochs=500,
    optimizer=SGD(0.01)
)

emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches

print(net.forward(emily))
print(net.forward(frank))

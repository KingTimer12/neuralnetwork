"""
O exemplo clássico de uma função que não pode ser
aprendido com um modelo linear simples é XOR
"""

from mylibneural.train import train
from mylibneural.neuralnetwork import NeuralNet
from mylibneural.layers import Linear, Tanh

import numpy as np

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0], #not xor
    [0, 1], # xor
    [0, 1], # xor
    [1, 0] #not xor
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])
train(net, inputs, targets)

for x, y in zip(inputs, targets):
    pred = net.forward(x)
    print(x, pred, y)
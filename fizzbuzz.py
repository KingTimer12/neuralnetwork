from mylibneural.train import train
from mylibneural.neuralnetwork import NeuralNet
from mylibneural.layers import Linear, Tanh, Sigmoid
from mylibneural.optim import SGD
from typing import List

import numpy as np

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]
    
def binary_encode(x: int) -> List[int]:
    return [x >> i & 1 for i in range(10)]

inputs = np.array([
    binary_encode(x) for x in range(101, 1024)
])

targets = np.array([
    fizz_buzz_encode(x) for x in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(
    net,
    inputs,
    targets,
    num_epochs=5000,
    optimizer=SGD(0.001)
)

for x in range(1, 101):
    #IA Resposta
    print(binary_encode(x))
    pred = net.forward(binary_encode(x))
    pred_idx = np.argmax(pred)

    #Resposta correta
    actual_idx = np.argmax(fizz_buzz_encode(x))

    labels = [str(x), 'fizz', 'buzz', 'fizzbuzz']
    print(x, labels[pred_idx], labels[actual_idx])
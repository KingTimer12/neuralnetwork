"""
Uma rede neural é um conjunto de layer.
Ele se comporta muito como uma layer em si, embora não seja.
"""

from typing import Sequence, Iterator, Tuple
from mylibneural.tensor import Tensor
from mylibneural.layers import Layer

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def paramsAndGrads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, params in layer.params.items():
                grad = layer.grads[name]
                yield params, grad
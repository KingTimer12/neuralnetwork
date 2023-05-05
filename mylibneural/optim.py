"""
Usamos um otimizador para ajustar os parâmetros
da nossa rede com base nos gradientes calculados
durante a retropropagação
"""

from mylibneural.neuralnetwork import NeuralNet

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learn_rate: float = 0.01) -> None:
        super().__init__()
        self.learn_rate = learn_rate
    
    def step(self, net: NeuralNet) -> None:
        for param, grad in net.paramsAndGrads():
            param -= self.learn_rate * grad

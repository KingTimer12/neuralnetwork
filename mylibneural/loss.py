"""
Função de perca para saber quão bom está nossas precisões,
podemos usar para ajustar os parâmentros da nossa rede
"""

import numpy as np
from mylibneural.tensor import Tensor

#classe abstrata
class Loss:
    def loss(self, pred: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, pred: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
    
class MSE(Loss):
    """
    MSE é mean squared error(erro quadrático médio), embora vamos
    apenas fazer o erro quadrático total
    """
    def loss(self, pred: Tensor, actual: Tensor) -> float:
        return np.sum((pred - actual) ** 2)
    
    def grad(self, pred: Tensor, actual: Tensor) -> Tensor:
        return 2 * (pred - actual)

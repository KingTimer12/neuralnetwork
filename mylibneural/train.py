"""
Aqui é onde a mágica acontece.
"""

from mylibneural.tensor import Tensor
from mylibneural.data import DataIterator, BatchIterator
from mylibneural.neuralnetwork import NeuralNet
from mylibneural.loss import Loss, MSE
from mylibneural.optim import Optimizer, SGD

def train(net: NeuralNet, inputs: Tensor, targets: Tensor, num_epochs: int = 5000, 
          iterator: DataIterator = BatchIterator(), 
          loss: Loss = MSE(), 
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            pred = net.forward(batch.inputs)
            epoch_loss += loss.loss(pred, batch.targets)
            grad = loss.grad(pred, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print("Epoch %d loss: %.3f" % (epoch, epoch_loss))
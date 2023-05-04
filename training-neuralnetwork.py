import numpy as np

def sigmoid(x,deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#função para calcular perda
#serve para dizer a máquina
#se ela precisa melhorar ou não
#y_true = valor correto
#y_pred = valor que a máquina acha ser correto
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, inputs):
        return sigmoid(self.weights[0] * inputs[0] + self.weights[1] * inputs[1] + self.bias)

class NeuralNetwork:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        #h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h1 = Neuron([self.w1, self.w2], self.b1).feedforward(x)
        h2 = Neuron([self.w3, self.w4], self.b2).feedforward(x)
        o1 = Neuron([self.w5, self.w6], self.b3).feedforward([h1, h2])

        return o1
    
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                h1 = Neuron([self.w1, self.w2], self.b1).feedforward(x)
                h2 = Neuron([self.w3, self.w4], self.b2).feedforward(x)
                o1 = Neuron([self.w5, self.w6], self.b3).feedforward([h1, h2])

                y_pred = o1

                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * sigmoid(o1, True)
                d_ypred_d_w6 = h2 * sigmoid(o1, True)
                d_ypred_d_b3 = sigmoid(o1, True)

                d_ypred_d_h1 = self.w5 * sigmoid(o1, True)
                d_ypred_d_h2 = self.w6 * sigmoid(o1, True)

                # Neuron h1
                d_h1_d_w1 = x[0] * sigmoid(h1, True)
                d_h1_d_w2 = x[1] * sigmoid(h1, True)
                d_h1_d_b1 = sigmoid(h1, True)

                # Neuron h2
                d_h2_d_w3 = x[0] * sigmoid(h2, True)
                d_h2_d_w4 = x[1] * sigmoid(h2, True)
                d_h2_d_b2 = sigmoid(h2, True)

                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                #print("Epoch %d loss: %.3f" % (epoch, loss))
                

  
# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Treinar a rede neural
network = NeuralNetwork()
network.train(data, all_y_trues)

# Testando os pred
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches

print("Emily: " + ("F" if round(network.feedforward(emily)) == 1 else "M")) # 0.951 - F
print("Frank: " + ("F" if round(network.feedforward(frank)) == 1 else "M")) # 0.039 - M
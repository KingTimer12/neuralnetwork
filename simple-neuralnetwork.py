#Lib de álgebra linear
import numpy as np

# função do sigmoid
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# matriz de entrada
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# matriz de saída
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)

# inicializar pesos aleatoriamente com média 0
syn0 = 2*np.random.random((3,4)) - 1 #peso de x
syn1 = 2*np.random.random((4,1)) - 1 #peso de y

for iter in range(60000):

    # forward
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    l2_error = y - l2

    if (iter % 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l2_error))))

    # não mudar muito se tiver certeza sobre a direção do alvo
    l2_delta = l2_error * nonlin(l2,True)

    # quanto cada valor l1 contribuiu para o erro l2 (de acordo com os pesos)?
    l1_error = l2_delta.dot(syn1.T)

    # em que direção está o alvo l1?
    # temos certeza? se assim for, não mude muito.
    l1_delta = l1_error * nonlin(l1,True)

    # atualizar os pesos
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print ("Output After Training:")
print (l1)
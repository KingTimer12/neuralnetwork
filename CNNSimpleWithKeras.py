import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical, normalize
 
mnist = tf.keras.datasets.mnist #28x28 imagens de digitos escritos na mão (0-9)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

model = Sequential([
  Flatten(),
  Dense(128, activation=tf.nn.relu), # neurônios, função de ativação
  Dense(128, activation=tf.nn.relu), # neurônios, função de ativação
  Dense(10, activation=tf.nn.softmax) # neurônios de saída, função de ativação
])

model.compile(
  'adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  x_train,
  y_train,
  epochs=3,
)

model.save('cnnsimple.h5')
new_model = load_model('cnnsimple.h5')
pred = new_model.predict([x_test])

print(np.argmax(pred[0]))
plt.imshow(x_test[0])
plt.show()
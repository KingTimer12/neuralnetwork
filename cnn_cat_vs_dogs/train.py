from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout

import pickle

X = pickle.load(open('X.pickle', 'rb'))
y = pickle.load(open('y.pickle', 'rb'))

X = X /  255.0

model = Sequential([
    Conv2D(64, (3,3), input_shape = X.shape[1:]),
    Activation("relu"),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3)),
    Activation("relu"),
    MaxPooling2D(pool_size=(2,2)),
    
    Flatten(),
    Dense(64),
    Dense(1),
    Activation("sigmoid")
])

model.compile(
  'adam',
  loss='binary_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  X,
  y,
  batch_size=32,
  validation_split=0.1
)
import cv2
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np

DATADIR = './Datasets/PetImages'
CATEGORIES = ['Dog', 'Cat']

IMG_SIZE = 50
#plt.imshow(new_array, cmap='gray')
#plt.show()

training_data = []
def createTrainingData():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
createTrainingData()

random.shuffle(training_data)

x = []
y = []
for features, label in training_data:
    x.append(features)
    y.append(label)
    
x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
import pickle
import numpy as np

with open('train_qa.txt', 'rb') as f:
    train_data = pickle.load(f)
    
with open('test_qa.txt', 'rb') as f:
    test_data = pickle.load(f)
    
print(train_data[10])
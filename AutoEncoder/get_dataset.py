import os
import numpy as np
from os import listdir
from scipy.misc import imread, imresize
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def get_img(data_path):
    # Getting image array from path:
    img_size = 64
    img = imread(data_path)
    img = imresize(img, (img_size, img_size, 3))
    return img

def get_dataset(dataset_path='Data/Train_Data'):
    # Getting all data from data path:
    labels = listdir(dataset_path) # Geting labels
    X = []
    Y = []
    for i, label in enumerate(labels):
        datas_path = dataset_path+'/'+label
        for data in listdir(datas_path):
            img = get_img(datas_path+'/'+data)
            X.append(img)
            Y.append(i)

    # Create dateset:
    X = np.array(X).astype('float32')/255.
    Y = np.array(Y).astype('float32')

    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    return X, X_test, Y, Y_test

'''
use scene-15 dataset to verify AutoEncoder algorithm, check https://qixianbiao.github.io/Scene.htmlsee
'''

import os
import random
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.datasets import fashion_mnist
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_class = []
train_dataset = []
X_train = np.zeros((1500, 256, 256))
y_train = np.zeros((1500,))

# the train data path
path = "data/train/"
dirs = os.listdir(path)
# get the all image class
for file in dirs:
    image_class.append(file)
# get the x and y of train data
for class_index, class_name in enumerate(image_class):
    images_path = path + class_name
    images_collection = os.listdir(images_path)
    for image_index, image_name in enumerate(images_collection):
        img = cv2.imread(images_path + "/" + image_name, 0)
        img = cv2.resize(img, (256, 256))
        train_dataset.append([img, class_index])
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
random.shuffle(train_dataset)
for i, train_data in enumerate(train_dataset):
    X_train[i] = train_data[0]
    y_train[i] = train_data[1]


test_dataset = []
X_test = np.zeros((2985, 256, 256))
y_test = np.zeros((2985,))
# the train data path
path = "data/test/"
dirs = os.listdir(path)
# get the x and y of test data
sum = 0
for class_index, class_name in enumerate(image_class):
    images_path = path + class_name
    images_collection = os.listdir(images_path)
    for image_index, image_name in enumerate(images_collection):
        img = cv2.imread(images_path + "/" + image_name, 0)
        img = cv2.resize(img, (256, 256))
        test_dataset.append([img, class_index])
random.shuffle(test_dataset)
for i, test_data in enumerate(test_dataset):
    X_test[i] = test_data[0]
    y_test[i] = test_data[1]




# input dimension = 256*256 = 65536
input_dim = np.prod(X_train.shape[1:])

# this is the size of our encoded representations
encoding_dim = 2048

# The compression factor is the ratio of the input dimension (65536) to the encoded dimension(2048),which is 32
compression_factor = float(input_dim) / encoding_dim
print("Compression factor: %s" % compression_factor)

# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# Train the model, iterating on the data in batches of 256 samples
history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, shuffle=True,
                validation_data=(X_test, X_test))

#_________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         (None, 65536)             0
# _________________________________________________________________
# dense_1 (Dense)              (None, 2048)              134219776
# _________________________________________________________________
# dense_2 (Dense)              (None, 65536)             134283264
# =================================================================
# Total params: 268,503,040
# Trainable params: 268,503,040
# Non-trainable params: 0
# _________________________________________________________________
print(autoencoder.summary())

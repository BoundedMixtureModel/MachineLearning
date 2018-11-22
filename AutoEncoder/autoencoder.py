'''
use scene-15 dataset to verify AutoEncoder algorithm, check https://qixianbiao.github.io/Scene.htmlsee
using gpu to accelerate
'''

import os
import random
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.client import device_lib
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from get_dataset import *

# configure the gpu
# set log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# model will be trained on GPU 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# output : physical_device_desc: "device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0"
print(device_lib.list_local_devices())


# Getting Dataset:
from get_dataset import get_dataset
X_train, X_test, Y_train, Y_test = get_dataset()
classes = ['cat', 'dog']

# About Dataset:
img_size = X_train.shape[1] # 64
print('Training shape:', X_train.shape)
print(X_train.shape[0], 'sample,',X_train.shape[1] ,'x',X_train.shape[2] ,'size RGB image.\n')
print('Test shape:', X_test.shape)
print(X_test.shape[0], 'sample,',X_test.shape[1] ,'x',X_test.shape[2] ,'size RGB image.\n')

# print('Examples:')
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(1, n+1):
#     # Display some data:
#     ax = plt.subplot(1, n, i)
#     plt.imshow(X_train[i])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# whether show the examples
# plt.show()

X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
# build model
# input dimension = 64*64*3 = 12288
input_dim = np.prod(X_train.shape[1:])
# this is the size of our encoded representations
encoding_dim = 384
# The compression factor is the ratio of the input dimension (12288) to the encoded dimension(384),which is 32
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
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',)

# Train the model, iterating on the data in batches of 256 samples
history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, shuffle=True,
                validation_data=(X_test, X_test))
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         (None, 12288)             0
# _________________________________________________________________
# dense_1 (Dense)              (None, 384)               4718976
# _________________________________________________________________
# dense_2 (Dense)              (None, 12288)             4730880
# =================================================================
# Total params: 9,449,856
# Trainable params: 9,449,856
# Non-trainable params: 0
print(autoencoder.summary())

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()



# use encoded layer to encode the training input which is separate encoder model:
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
print(encoder.summary())
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
print(decoder.summary())
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(X_test)
decoded_imgs = autoencoder.predict(X_test)  #=decoder.predict(encoded_imgs)

# Compare Original images (top row) with reconstructed ones (bottom row)
m = 10  # how many digits we will display
plt.figure(figsize=(9, 3))
for i in range(m):
    # display original
    ax = plt.subplot(2, m, i + 1)
    plt.imshow(X_test[i].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, m, i + 1 + m)
    plt.imshow(decoded_imgs[i].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Using encoding layer with softmax layer to train a classifier
# For a single-input model with 10 classes (categorical classification):
model = Sequential()
model.add(autoencoder)

model.add(Dense(2, activation=tf.nn.softmax))
# model.add(Activation(tf.nn.softmax))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Convert labels to categorical one-hot encoding
Y_train = to_categorical(Y_train, num_classes=2)

# Train the model, iterating on the data in batches of 32 samples
model.fit(X_train, Y_train, epochs=100, batch_size=1500)
print(model.summary())

#predicted labels/classes
Y_pred = model.predict_classes(X_test)

#Precision and recall
# print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
df = pd.DataFrame(cm, classes, classes)
plt.figure()
sns.set(font_scale=1.2)#for label size
#comap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True)
ax = sns.heatmap(cm,annot=True,annot_kws={"size": 16},linewidths=.5,cbar=False,
        xticklabels=classes,yticklabels=classes,square=True, cmap='Blues_r', fmt="d")
# This sets the yticks "upright" with 0, as opposed to sideways with 90.
plt.yticks(rotation=0)
ax.tick_params(labelbottom=False,labeltop=True)
plt.xticks(rotation=90)
plt.show()

#accuracy score
acc = accuracy_score(Y_test, Y_pred)
print('\nAccuracy for the test data: {:.2%}\n'.format(acc))


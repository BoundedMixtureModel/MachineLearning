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
from keras.layers import Input, Flatten, Dense, Conv1D, MaxPooling1D, UpSampling1D, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.optimizers import RMSprop
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.client import device_lib
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from get_dataset import get_dataset

# img = cv2.imread("data/train/Bedroom/image_0001.jpg", 0)
# img = cv2.resize(img, (100, 100))
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# configure the gpu
# set log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# model will be trained on GPU 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# output : physical_device_desc: "device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0"
print(device_lib.list_local_devices())

# Getting Dataset:

X_train, X_test, Y_train, Y_test = get_dataset()
classes = ['cat', 'dog']

# About Dataset:
img_size = X_train.shape[1] # 64
print('Training shape:', X_train.shape)
print(X_train.shape[0], 'sample,',X_train.shape[1] ,'x',X_train.shape[2] ,'size RGB image.\n')
print('Test shape:', X_test.shape)
print(X_test.shape[0], 'sample,',X_test.shape[1] ,'x',X_test.shape[2] ,'size RGB image.\n')


# build model
# input dimension = 64*64*3 = 12288
input_dim = X_train.shape[1:]
# this is the size of our encoded representations
encoding_dim = 384
# The compression factor is the ratio of the input dimension (12288) to the encoded dimension(384),which is 32
compression_factor = float(np.prod(input_dim)) / encoding_dim
print("Compression factor: %s" % compression_factor)


# this is our input placeholder
print(input_dim)
input_img = Input(shape=input_dim)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# this is output part
x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',)
# autoencoder.compile(optimizer='rmsprop', loss='mse')

# Train the model, iterating on the data in batches of 256 samples
history = autoencoder.fit(X_train, X_train, epochs=5, batch_size=500, shuffle=True,
                validation_data=(X_test, X_test))
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         (None, 64, 64, 3)         0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 64, 64, 32)        896
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 32, 32, 32)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 32, 32, 64)        18496
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 16, 16, 64)        0
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 16, 16, 64)        36928
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 8, 8, 64)          0
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 8, 8, 64)          36928
# _________________________________________________________________
# up_sampling2d_1 (UpSampling2 (None, 16, 16, 64)        0
# _________________________________________________________________
# conv2d_5 (Conv2D)            (None, 16, 16, 64)        36928
# _________________________________________________________________
# up_sampling2d_2 (UpSampling2 (None, 32, 32, 64)        0
# _________________________________________________________________
# conv2d_6 (Conv2D)            (None, 32, 32, 32)        18464
# _________________________________________________________________
# up_sampling2d_3 (UpSampling2 (None, 64, 64, 32)        0
# _________________________________________________________________
# conv2d_7 (Conv2D)            (None, 64, 64, 3)         867
# =================================================================
# Total params: 149,507
# Trainable params: 149,507
# Non-trainable params: 0
# _________________________________________________________________
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
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(X_train)
decoded_imgs = autoencoder.predict(X_train)  #=decoder.predict(encoded_imgs)

# Compare Original images (top row) with reconstructed ones (bottom row)
m = 10  # how many digits we will display
plt.figure(figsize=(9, 3))
for i in range(m):
    # display original
    ax = plt.subplot(2, m, i + 1)
    plt.imshow(X_train[i].reshape(64, 64, 3))
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
# For a single-input model with 15 classes (categorical classification):
# model = Sequential()
# # autocoder is convolutional autoencoder
# model.add(autoencoder)

def fc(encode):
    flat = Flatten()(encode)
    den = Dense(950, activation='relu')(flat)
    out = Dense(2, activation=tf.nn.softmax)(den)
    return out

encode = encoder(input_img)
model = Model(input_img, fc(encode))

# Convert labels to categorical one-hot encoding
Y_train = to_categorical(Y_train, num_classes=2)
Y_test = to_categorical(Y_test, num_classes=2)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


classify_train = model.fit(X_train, Y_train, epochs=10, batch_size=200,
                           validation_data=(X_test, Y_test))
print(model.summary())

print(classify_train.history)
accuracy = classify_train.history['acc']
val_accuracy = classify_train.history['val_acc']
loss = classify_train.history['loss']
val_loss = classify_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#predicted labels/classes
Y_pred = model.predict (X_test)

threshold = 0.5
# Precision and recall
print(classification_report(Y_test, Y_pred > threshold))

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

# accuracy score
acc = accuracy_score(Y_test, Y_pred > threshold)
print('\nAccuracy for the test data: {:.2%}\n'.format(acc))

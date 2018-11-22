import os
import cv2
import random
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.utils import to_categorical
from tensorflow.python.client import device_lib
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# configure the gpu
# set log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# model will be trained on GPU 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# output : physical_device_desc: "device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0"
print(device_lib.list_local_devices())

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# data preprocessing
image_class = []
X_train = np.zeros((100, 100, 100, 1))
y_train = np.zeros((100,))

# the train data path
path = "Data/Path/"
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
        img = cv2.resize(img, (100, 100))
        X_train[image_index] = img_to_array(img)
        y_train[image_index] = class_index
    i = 0
    print('generated_data/' + class_name)
    for batch in datagen.flow(X_train, y_train, batch_size=25, save_to_dir='generated_data/' + class_name ,
                              save_prefix='img', save_format='jpg'):
        i += 1
        if i > 80:
            break  # otherwise the generator would loop indefinitely


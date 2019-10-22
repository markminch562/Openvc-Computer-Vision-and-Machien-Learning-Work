# -*- coding: utf-8 -*-
"""
Created on Sun May 26 21:24:48 2019

@author: markm
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
import time
Name = "cat-vs-dog-cnn-64-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))

data_dir = "D:/imagepaths/data/train"
categories = ["dog", "cat"]
def create_trainning_data(IMG_SIZE):
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
img_size = 50

training_data = []
create_trainning_data(img_size)
print(len(training_data))

import random
random.shuffle(training_data)

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, img_size, img_size, 1)

x = x/255.0
gene_train = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, zoom_range=0.3, shear_range=0.3)

shape= (50, 50, 1)


model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())
model.add(Dense(62))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])


model.fit(x, y, batch_size=32, epochs=5, validation_split=0.1, callbacks=[tensorboard])







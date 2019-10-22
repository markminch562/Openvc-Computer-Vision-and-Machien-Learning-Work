# -*- coding: utf-8 -*-
"""
Created on Sat May 25 03:00:09 2019

@author: markm
"""
import os
import shutil
import tensorflow



def copy_files(prefix_str, range_start,  range_end, target_dir):
    image_paths = [os.path.join(work_dir, 'train', prefix_str + '.' + str(i) + '.jpg')
                  for i in range(range_start, range_end)]
    dest_dir = os.path.join(work_dir, 'data', target_dir, prefix_str)
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        for image_path in image_paths:
            shutil.copy(image_path, dest_dir)
    

work_dir = 'D:\imagepaths'

#createfiles(0, 100, 'doggey', start_path, 'data')
copy_files('dog', 0, 1500, 'train')
copy_files('cat', 0, 1500, 'train')
copy_files('dog', 1501, 2001, 'test')
copy_files('cat', 1501, 2001, 'test')

train_dir = os.path.join(work_dir, 'data', 'train')
validation_dir = os.path.join(work_dir, 'data', 'test')
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(2 , activation='softmax'))
model.summary()


from keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=35,
        # NEW...Since we are not using a binary class system class_mod is set to categorical
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=35,
        class_mode='categorical')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


# Let's fit our model to the data using the generator.

history = model.fit_generator(
      train_generator,
      steps_per_epoch=80,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=80)


# It is good practice to always save your models after training:
model.save('too_manny_flowers_1.h5')


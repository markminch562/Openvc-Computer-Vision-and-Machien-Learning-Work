# Mark Minch, Taylor Colon, Charlie Dickson

import keras
import os, shutil

# original data set directory. original_dataset_dir must be set for where it is on your computer.
original_dataset_dir = 'D:/Machen learing/project 5/Project_4 (1)/Project_4/flower_photos'

# This is the directory that will store our new datasets. It'll be the same on every computer.
base_dir = 'c:/flower_photos'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)   # tells program to make directory

# These are our directories for the  Training, Validation and Test groups
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# this block of code makes all of the training directories
train_daisy_dir = os.path.join(train_dir, 'daisy')
if not os.path.exists(train_daisy_dir):
    os.mkdir(train_daisy_dir)

train_dandelion_dir = os.path.join(train_dir, 'dandelion')
if not os.path.exists(train_dandelion_dir):
    os.mkdir(train_dandelion_dir)

train_roses_dir = os.path.join(train_dir, 'roses')
if not os.path.exists(train_roses_dir):
    os.mkdir(train_roses_dir)

train_sunflowers_dir = os.path.join(train_dir, 'sunflowers')
if not os.path.exists(train_sunflowers_dir):
    os.mkdir(train_sunflowers_dir)

train_tulips_dir = os.path.join(train_dir, 'tulips')
if not os.path.exists(train_tulips_dir):
    os.mkdir(train_tulips_dir)

# this block makes all of the validation directories
validation_daisy_dir = os.path.join(validation_dir, 'daisy')
if not os.path.exists(validation_daisy_dir):
    os.mkdir(validation_daisy_dir)

validation_dandelion_dir = os.path.join(validation_dir, 'dandelion')
if not os.path.exists(validation_dandelion_dir):
    os.mkdir(validation_dandelion_dir)

validation_roses_dir = os.path.join(validation_dir, 'roses')
if not os.path.exists(validation_roses_dir):
    os.mkdir(validation_roses_dir)

validation_sunflowers_dir = os.path.join(validation_dir, 'sunflowers')
if not os.path.exists(validation_sunflowers_dir):
    os.mkdir(validation_sunflowers_dir)

validation_tulips_dir = os.path.join(validation_dir, 'tulips')
if not os.path.exists(validation_tulips_dir):
    os.mkdir(validation_tulips_dir)

# same thing as the last block but for the test directories
test_daisy_dir = os.path.join(test_dir, 'daisy')
if not os.path.exists(test_daisy_dir):
    os.mkdir(test_daisy_dir)

test_dandelion_dir = os.path.join(test_dir, 'dandelion')
if not os.path.exists(test_dandelion_dir):
    os.mkdir(test_dandelion_dir)

test_roses_dir = os.path.join(test_dir, 'roses')
if not os.path.exists(test_roses_dir):
    os.mkdir(test_roses_dir)

test_sunflowers_dir = os.path.join(test_dir, 'sunflowers')
if not os.path.exists(test_sunflowers_dir):
    os.mkdir(test_sunflowers_dir)

test_tulips_dir = os.path.join(test_dir, 'tulips')
if not os.path.exists(test_tulips_dir):
    os.mkdir(test_tulips_dir)

# copies the images into the validation directories
fnames = ['Daisy ({}).jpg'.format(i) for i in range(1, 101)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_daisy_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dandelion ({}).jpg'.format(i) for i in range(1, 101)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dandelion_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['roses ({}).jpg'.format(i) for i in range(1, 101)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_roses_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['sunflowers ({}).jpg'.format(i) for i in range(1, 101)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_sunflowers_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['tulips ({}).jpg'.format(i) for i in range(1, 101)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_tulips_dir, fname)
    shutil.copyfile(src, dst)

# copies the images into the test directories

fnames = ['Daisy ({}).jpg'.format(i) for i in range(101, 201)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_daisy_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dandelion ({}).jpg'.format(i) for i in range(101, 201)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dandelion_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['roses ({}).jpg'.format(i) for i in range(101, 201)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_roses_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['sunflowers ({}).jpg'.format(i) for i in range(101, 201)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_sunflowers_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['tulips ({}).jpg'.format(i) for i in range(101, 201)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_tulips_dir, fname)
    shutil.copyfile(src, dst)

# copies the images into the test directories

fnames = ['Daisy ({}).jpg'.format(i) for i in range(201, 633)]  # total number in each set is not the same. might change
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_daisy_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dandelion ({}).jpg'.format(i) for i in range(201, 898)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dandelion_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['roses ({}).jpg'.format(i) for i in range(201, 641)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_roses_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['sunflowers ({}).jpg'.format(i) for i in range(201, 699)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_sunflowers_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['tulips ({}).jpg'.format(i) for i in range(201, 799)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_tulips_dir, fname)
    shutil.copyfile(src, dst)

# prints the amount of pics in each folder. makes sure it works

print('total validation daisy images:', len(os.listdir(validation_daisy_dir)))
print('total validation dandelion images:', len(os.listdir(validation_dandelion_dir)))
print('total validation roses images:', len(os.listdir(validation_roses_dir)))
print('total validation sunflowers images:', len(os.listdir(validation_sunflowers_dir)))
print('total validation tulips images:', len(os.listdir(validation_tulips_dir)))


print('total test daisy images:', len(os.listdir(test_daisy_dir)))
print('total test dandelion images:', len(os.listdir(test_dandelion_dir)))
print('total test roses images:', len(os.listdir(test_roses_dir)))
print('total test sunflowers images:', len(os.listdir(test_sunflowers_dir)))
print('total test tulips images:', len(os.listdir(test_tulips_dir)))

print('total training daisy images:', len(os.listdir(train_daisy_dir)))
print('total training dandelion images:', len(os.listdir(train_dandelion_dir)))
print('total training roses images:', len(os.listdir(train_roses_dir)))
print('total training sunflowers images:', len(os.listdir(train_sunflowers_dir)))
print('total training tulips images:', len(os.listdir(train_tulips_dir)))




# Create our CNN model

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))
model.summary()

#NEW... activation at end dense layer was changer from sigmoid to softmax because we have multiable classes and dense became 5 instead of 1 because we have five classes

from keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

#NEW...loss was changed from binary cross entropy because we have multiple classes

# ## Data preprocessing
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




# Let's plot the loss and accuracy of the model over the training and validation data during training:
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

test_datagen2 = ImageDataGenerator(rescale=1./255)

validation_generator2 = test_datagen2.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=35,
        class_mode='categorical')

history2 = model.fit_generator(
      train_generator,
      steps_per_epoch=2,
      epochs=25,
      validation_data=validation_generator2,
      validation_steps=2)


plt.figure()
plt.ylim(0, 1)
acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Test acc')
plt.title('Training and Test accuracy')
plt.legend()



plt.show()







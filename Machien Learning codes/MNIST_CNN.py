### This Python program using a convolutional neural network (CNN) to classify the grey image in the MNIST dataset.

from keras import layers
from keras import models

### Create the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

### Display the model architecture
model.summary()

### Load the MNIST data set
from keras.datasets import mnist
from keras.utils import to_categorical

### Prepare the training set
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

### Prepare the test set
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

### encode the training labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

### Training the model with the training model
history=model.fit(train_images, train_labels, epochs=5, batch_size=64)

### plot the training history
import matplotlib.pyplot as plt

acc = history.history['acc']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)

plt.figure(1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure(2)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

### Test the model with the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)


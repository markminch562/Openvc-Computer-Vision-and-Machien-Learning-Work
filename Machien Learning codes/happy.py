import os
import shutil
import tensorflow as tf

work_dir = 'D:/imagepaths'
testings= os.path.join(work_dir, 'train')
print(testings)
image_names = sorted(os.listdir(os.path.join(work_dir, 'train')))


def copy_files(prefix_str, range_start,  range_end, target_dir):
    image_paths = [os.path.join(work_dir, 'train', prefix_str + '.' + str(i) + '.jpg')
                  for i in range(range_start, range_end)]
    dest_dir = os.path.join(work_dir, 'data', target_dir, prefix_str)
    os.makedirs(dest_dir)
    for image_path in image_paths:
        shutil.copy(image_path, dest_dir)


def simple_cnn(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size =(3, 3),
            activation='relu',
            input_shape = input_shape
            ))
    model.add(tf.keras.layers.Conv2D(
            filters = 128,
            kernel_size =(3, 3),
            activation='relu',
            input_shape = input_shape
            ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units= 1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(units=no_classes,
                                    activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer = tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


make_files = True


if make_files==True:
    copy_files('dog', 0, 400, 'train')
    copy_files('cat', 0, 400, 'train')
    copy_files('dog', 401, 501, 'test')
    copy_files('cat', 401, 501, 'test')


image_height, image_width = 150, 150
train_dir = os.path.join(work_dir, 'data/train')
test_dir = os.path.join(work_dir, 'data/test')
no_classes = 2
no_validation = 100
no_train = 400
epochs = 2
batch_size = 20
no_test = 100
input_shapes = (image_height, image_width, 3)
epoch_steps = no_train // batch_size
test_steps = no_test // batch_size


simple_cnn_model = simple_cnn(input_shapes)

generator_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
generator_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
print(train_dir)
print(test_dir)

train_images = generator_train.flow_from_directory(
    train_dir,
    batch_size= batch_size,
    target_size=(image_width, image_height)
)
test_images = generator_train.flow_from_directory(
    test_dir,
    batch_size= batch_size,
    target_size=(image_width, image_height)
)

simple_cnn_model.fit_generator(train_images,
                         steps_per_epoch=epoch_steps,
                         epochs=epochs,
                         validation_data=test_images,
                         validation_steps=test_steps)






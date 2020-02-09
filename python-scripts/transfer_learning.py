# https://adventuresinmachinelearning.com/transfer-learning-tensorflow-2/

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pylab as plt
import tensorflowjs as tfjs

def create_model1():

    head = tf.keras.Sequential()

    head.add(layers.Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    head.add(layers.BatchNormalization())
    head.add(layers.Activation('relu'))
    head.add(layers.MaxPooling2D(pool_size=(2, 2)))

    head.add(layers.Conv2D(32, (3, 3)))
    head.add(layers.BatchNormalization())
    head.add(layers.Activation('relu'))
    head.add(layers.MaxPooling2D(pool_size=(2, 2)))

    head.add(layers.Conv2D(64, (3, 3)))
    head.add(layers.BatchNormalization())
    head.add(layers.Activation('relu'))
    head.add(layers.MaxPooling2D(pool_size=(2, 2)))

    average_pool = tf.keras.Sequential()
    average_pool.add(layers.AveragePooling2D())
    average_pool.add(layers.Flatten())
    average_pool.add(layers.Dense(1, activation='sigmoid'))

    standard_model = tf.keras.Sequential([
        head, 
        average_pool
    ])

    standard_model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    return standard_model

def create_model2():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(IMAGE_SIZE,IMAGE_SIZE, 3)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

def create_model3():

    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    res_net = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
    res_net.trainable = False

    global_average_layer = layers.GlobalAveragePooling2D()
    output_layer = layers.Dense(1, activation='sigmoid')
    tl_model = tf.keras.Sequential([
        res_net,
        global_average_layer,
        output_layer
    ])

    tl_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return tl_model


def train_and_save(standard_model, filename, name):

    split = (80, 10, 10)
    splits = tfds.Split.TRAIN.subsplit(weighted=split)

    (cat_train, cat_valid, cat_test), info = tfds.load('cats_vs_dogs', split=list(splits), with_info=True, as_supervised=True)
      
    for image, label in cat_train.take(2):
        plt.figure()
        plt.imshow(image)

    def pre_process_image(image, label):
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return image, label

    cat_train = cat_train.map(pre_process_image).shuffle(1000).repeat().batch(TRAIN_BATCH_SIZE)
    cat_valid = cat_valid.map(pre_process_image).repeat().batch(1000)

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./log/standard_model', update_freq='batch')]

    standard_model.fit(cat_train, steps_per_epoch = STEPS_FACTOR//TRAIN_BATCH_SIZE, epochs=EPOCHS, 
                    validation_data=cat_valid, validation_steps=2, callbacks=callbacks)

    standard_model.save(filename)

    tfjs.converters.save_keras_model(standard_model, './converted-keras/' + name + '/')

def load_saved_model():

    print('here')

    saved_model = tf.keras.models.load_model('./python-scripts/standard_model_3.h5')

    saved_model.summary()

EPOCHS = 1
# STEPS_FACTOR = 23262
STEPS_FACTOR = 500
IMAGE_SIZE = 100
TRAIN_BATCH_SIZE = 64

#load_saved_model()

#train_and_save(create_model1(), './python-scripts/standard_model_1.h5', 'standard_model_1')
#train_and_save(create_model2(), './python-scripts/standard_model_2.h5', 'standard_model_2')
train_and_save(create_model3(), './python-scripts/standard_model_3.h5', 'standard_model_3')

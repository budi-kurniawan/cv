import time
import sys
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

BATCH_SIZE = 128
def rescaling (image):
    return tf.cast(image, tf.float32) / 255

def convert (y):
    return to_categorical(y)

def input_pipelines(xTrain, yTrain, xVal, yVal, xTest, yTest, batchSize = BATCH_SIZE):
    BUFFER_SIZE = len(xTrain)
    # Train dataset
    train_ds = tf.data.Dataset.from_tensor_slices((xTrain, yTrain))\
          .cache()\
          .shuffle(buffer_size=BUFFER_SIZE)\
          .batch(batch_size=batchSize)\
          .prefetch(tf.data.experimental.AUTOTUNE)

    # Validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices((xVal, yVal))\
          .batch(batch_size=batchSize)\
          .cache()\
          .prefetch(tf.data.experimental.AUTOTUNE)

    # Test dataset
    test_ds = tf.data.Dataset.from_tensor_slices((xTest, yTest))\
          .batch(batch_size=batchSize)\
          .cache()\
          .prefetch(tf.data.experimental.AUTOTUNE)
    return train_ds, val_ds, test_ds


def compile_and_fit(model, name, optimizer, max_epochs, batchSize, trainDs):
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #gradient_cb = get_callbacks(name)
    history = model.fit(
        trainDs,
        batch_size=batchSize,
        epochs=max_epochs,
        #validation_data=val_ds,
        #callbacks=gradient_cb,
        verbose=2)
    return history


# Function to return Adam optimizer with schedule learning rate
def get_optimizer(lr_schedule):
    return tf.keras.optimizers.Adam(lr_schedule)

def create_model(num_neurons):
    model_basic = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(num_neurons, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model_basic

def prepare_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    X_train = rescaling(X_train)
    X_test = rescaling(X_test)
    X_val = rescaling(X_val)
    y_train = convert(y_train)
    y_test = convert(y_test)
    y_val = convert(y_val)
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)

def run(X_train, y_train, X_test, y_test, X_val, y_val, max_epochs):
    STEPS_PER_EPOCH = len(X_train) / BATCH_SIZE
    train_ds, val_ds, test_ds = input_pipelines(X_train, y_train, X_val, y_val, X_test, y_test, BATCH_SIZE)
    # set learning schedule
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps=STEPS_PER_EPOCH*1000,
        decay_rate=1,
        staircase=False)
    model_basic = create_model(num_neurons=128)
    histories = {}
    histories['model_basic'] = compile_and_fit(model_basic, 'model_basic', get_optimizer(lr_schedule),                                               max_epochs=max_epochs, batchSize=BATCH_SIZE, trainDs=train_ds)
    eval_result = model_basic.evaluate(X_test, y_test)
    print("[test loss, test accuracy]:", eval_result)

if __name__ == '__main__':
    print(sys.version)
    max_epochs = 5
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = prepare_data()
    start = time.time()
    run(X_train, y_train, X_test, y_test, X_val, y_val, max_epochs)
    end = time.time()
    print('GPU available ?', tf.test.is_gpu_available())  # True/False
    print('GPU with CUDA support available?' , tf.test.is_gpu_available(cuda_only=True))
    print("Execution time: %5.2f seconds" % (end - start))


    print(tf.config.list_physical_devices('GPU'))
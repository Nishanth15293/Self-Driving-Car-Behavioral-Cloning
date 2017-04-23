import csv
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.models import load_model
import matplotlib.pyplot as plt
from augment import augment_batch

PICS_PATH = './Training/IMG/'
DRIVING_LOG = './Training/driving_log.csv'
LOAD_MODEL_PATH = 'model.h5'
SAVE_MODEL_PATH = 'model.h5'
DISPLAY_SAMPLES = False
RESUME = False

samples = []
with open(DRIVING_LOG) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images, angles = augment_batch(batch_samples, PICS_PATH=PICS_PATH, DISPLAY_SAMPLES=False)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)
ch, row, col = 3, 160, 320

if RESUME:
    model = load_model(LOAD_MODEL_PATH)
    model.compile(loss='mse', optimizer='adam', verbose=1)
    history = model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
                                  validation_data=validation_generator,
                                  validation_steps=len(validation_samples), epochs=1, verbose=1)
    model.save(SAVE_MODEL_PATH)
else:
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(filters=3, kernel_size=5, strides=(2, 2)))
    model.add(Activation('elu'))
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2)))
    model.add(Activation('elu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2)))
    model.add(Activation('elu'))
    model.add(Conv2D(filters=48, kernel_size=3, strides=(1, 1)))
    model.add(Activation('elu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(Activation('elu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(rate=0.5))
    model.add(Dense(100))
    model.add(Dropout(rate=0.5))
    model.add(Dense(50))
    model.add(Dropout(rate=0.5))
    model.add(Dense(10))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', verbose=1)
    history = model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator,
                                  validation_steps=len(validation_samples), epochs=3, verbose=1)
    model.save(SAVE_MODEL_PATH)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


import os
import csv
from sklearn.cross_validation import train_test_split
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

samples = []
with open('./Track2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=1):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                center_img_path = './Track2/IMG/' + batch_sample[0].split('\\')[-1]
                left_img_path = './Track2/IMG/' + batch_sample[1].split('\\')[-1]
                right_img_path = './Track2/IMG/' + batch_sample[2].split('\\')[-1]

                correction = 0.2
                center_image = cv2.imread(center_img_path)
                center_angle = float(batch_sample[3])

                left_image = cv2.imread(left_img_path)
                left_angle = float(batch_sample[3]) + correction

                right_image = cv2.imread(right_img_path)
                right_angle = float(batch_sample[3]) - correction

                center_image_flipped = np.fliplr(center_image)
                center_angle_flipped = -center_angle

                left_image_flipped = np.fliplr(left_image)
                left_angle_flipped = -left_angle

                right_image_flipped = np.fliplr(right_image)
                right_angle_flipped = -right_angle

                images.extend([center_image, left_image, right_image, center_image_flipped, left_image_flipped, right_image_flipped])
                angles.extend([center_angle, left_angle, right_angle, center_angle_flipped, left_angle_flipped, right_angle_flipped])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)
print(next(train_generator))
ch, row, col = 3, 160, 320

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(3,5,2,activation='relu'))
model.add(Convolution2D(24,5,2,activation='relu'))
model.add(Convolution2D(36,5,2,activation='relu'))
model.add(Convolution2D(48,3,1,activation='relu'))
model.add(Convolution2D(64,3,1,activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                              nb_val_samples=len(validation_samples), nb_epoch=3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
model.save('model.h5')



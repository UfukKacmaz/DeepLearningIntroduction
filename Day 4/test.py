import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cv2

################
## PREPARATION #
################
import tensorflow as tf
from load_gtsrb import *  
from layers import *
from helper import *

# Load dataset
dataset_path = "C:/Users/schaf/Documents/GTSRB/Final_Training/Images/"
#dataset_path = "C:/Users/jan/Documents/GTSRB/Final_Training/Images/"

num_classes = 43
imgs_per_class = 10000
classes = range(num_classes)
save_dataset(dataset_path, classes, imgs_per_class=imgs_per_class)
data = GTSRB(dataset_path, num_classes)
data.data_augmentation(augment_size=10000)
# Model variables
train_size, valid_size, test_size = data.train_size, data.valid_size, data.test_size
img_width, img_height, img_depth = data.img_width, data.img_height, data.img_depth
x_train, y_train = data.x_train, data.y_train
x_valid, y_valid = data.x_valid, data.y_valid
x_test, y_test = data.x_test, data.y_test

################
##     CNN     #
################

from keras.models import Sequential
from keras.layers import *
from keras.layers.normalization import *
from keras.optimizers import *

# Hyperparameters
epochs = 50
batch_size = 32
learning_rate = 1e-4

# Define the CNN
model = Sequential()
model.add(Conv2D(32, 3, padding="same", input_shape=(img_width, img_height, img_depth)))
model.add(Dropout(0.3))
model.add(Activation("relu"))
model.add(Conv2D(32, 3, padding="same"))
model.add(Dropout(0.3))
model.add(Activation("relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same"))
model.add(Dropout(0.3))
model.add(Activation("relu"))
model.add(Conv2D(64, 3, padding="same"))
model.add(Dropout(0.3))
model.add(Activation("relu"))
model.add(MaxPool2D())

model.add(Conv2D(128, 3, padding="same"))
model.add(Dropout(0.3))
model.add(Activation("relu"))
model.add(Conv2D(128, 3, padding="same"))
model.add(Dropout(0.3))
model.add(Activation("relu"))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Activation("relu"))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

# Print the CNN layers
model.summary()

# Train the CNN
optimizer = RMSprop(lr=learning_rate)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(x_train, y_train, verbose=1,
        batch_size=batch_size, nb_epoch=epochs,
        validation_data=(x_valid, y_valid))

# Test the CNN
score = model.evaluate(x_test, y_test)
print("Test accuracy: ", score[1])
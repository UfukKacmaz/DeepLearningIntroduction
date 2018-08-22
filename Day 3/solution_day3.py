import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cv2

from load_gtsrb import *  

################
## PREPARATION #
################

# Load dataset
dataset_path = "C:/Users/schaf/Documents/GTSRB/Final_Training/Images/"
classes = range(5)

################
## EXERCISE 2 ##
################
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load RGB images
[imgs, labels, class_descs, sign_ids] = load_gtsrb_images(dataset_path, classes, 10000)  
imgs = imgs.astype(np.uint8)

# Transform y labels to one-hot array
x, y = imgs, labels
# Shuffle data first
idx = np.random.randint(0, x.shape[0], x.shape[0])
x, y = x[idx], y[idx]
y = tf.keras.utils.to_categorical(y, num_classes=len(classes))
# Transform x images to vector
x = np.array([x_k[:].flatten() for x_k in x])

# Helper function to split the dataset
def get_train_valid_test(x, y):
    test_size = 0.3
    valid_size = 0.2 / (1 - test_size)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=valid_size, random_state=42)
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

# Split the dataset
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_train_valid_test(x, y)

# Whiten the data
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_train = np.array([(x_k - x_train_mean) / x_train_std for x_k in x_train])

################
## EXERCISE 3 ##
################
from layers import *

# Helper function to get minibatches
def next_batch(x, y, batch_size):
    idx = np.random.randint(0, x.shape[0], batch_size)
    x_batch = x[idx]
    y_batch = y[idx]
    return x_batch, y_batch

# Model variables
num_classes = len(classes)
num_features = x.shape[1]
train_size, valid_size, test_size = x_train.shape[0], x_valid.shape[0], x_test.shape[0]
epochs = 15
batch_size = 16
learning_rate = 5e-5
train_mini_batches = train_size // batch_size
valid_mini_batches = valid_size // batch_size
test_mini_batches = test_size // batch_size

# Input and Output of the NN
x = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

# Model definition
def model(x):
    x = fc_layer(x, num_features, 2048, name="fc1", initializer="normal")
    x = tf.nn.relu(x)
    x = fc_layer(x, 2048, 1024, name="fc2", initializer="normal")
    x = tf.nn.relu(x)
    x = fc_layer(x, 1024, 1024, name="fc3", initializer="normal")
    x = tf.nn.relu(x)
    x = fc_layer(x, 1024, 1024, name="fc4", initializer="normal")
    x = tf.nn.relu(x)
    x = fc_layer(x, 1024, num_classes, name="fc5", initializer="normal")
    x = tf.nn.softmax(x)
    return x

# TensorFlow Ops to train/test
pred_op = model(x)
loss_op = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=pred_op)
correct_result_op = tf.equal(tf.argmax(pred_op, axis=1), tf.argmax(y, axis=1))
accuracy_op = tf.reduce_mean(tf.cast(correct_result_op , tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Start Training and Testing
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Training
    print("\n\n Start training!")
    for epoch in range(epochs):
        train_acc, valid_acc = 0.0, 0.0
        train_loss, valid_loss = 0.0, 0.0
        # Train the weights
        for i in range(train_mini_batches):
            epoch_x, epoch_y = next_batch(x_train, y_train, batch_size)
            _, c = sess.run([train_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
        # Check the performance of the train set
        for i in range(train_mini_batches):
            epoch_x, epoch_y = next_batch(x_train, y_train, batch_size)
            a, c = sess.run([accuracy_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
            train_acc += a
            train_loss += c
        train_acc = train_acc / train_mini_batches
        # Check the performance of the valid set
        for i in range(valid_mini_batches):
            epoch_x, epoch_y = epoch_x, epoch_y = next_batch(x_valid, y_valid, batch_size)
            a, c = sess.run([accuracy_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
            valid_acc += a
            valid_loss += c
        valid_acc = valid_acc / valid_mini_batches
        print("Epoch: ", epoch+1, " of ", epochs, "- Train loss: ", round(train_loss, 3), 
            " Valid loss: ", round(valid_loss, 3), " Train Acc: ", round(train_acc, 3), 
            " Valid Acc: ", round(valid_acc, 3))
    # Testing
    test_acc = 0.0
    test_loss = 0.0
    print("\n\nFinal testing!")
    for i in range(test_mini_batches):
            epoch_x, epoch_y = epoch_x, epoch_y = next_batch(x_test, y_test, batch_size)
            a, c = sess.run([accuracy_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
            test_acc += a
            test_loss += c
    test_acc = test_acc / test_mini_batches
    print("Test Accuracy:\t", test_acc)
    print("Test Loss:\t", test_loss)
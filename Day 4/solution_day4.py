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

# Load dataset
dataset_path = "C:/Users/schaf/Documents/GTSRB/Final_Training/Images/"
#dataset_path = "C:/Users/jan/Documents/GTSRB/Final_Training/Images/"

num_classes = 5
imgs_per_class = 10000
classes = range(num_classes)
save_dataset(dataset_path, classes, imgs_per_class=imgs_per_class)
data = GTSRB(dataset_path, num_classes)
data.data_augmentation(augment_size=1000)
# Model variables
train_size, valid_size, test_size = data.train_size, data.valid_size, data.test_size
img_width, img_height, img_depth = data.img_width, data.img_height, data.img_depth
print("Train Size and Shape:", train_size, data.x_train[0].shape)
print("Valid Size and Shape:", valid_size, data.x_valid[0].shape)
print("Test Size and Shape:", test_size, data.x_test[0].shape)

################
## EXERCISE 1 ##
################

# Hyperparameters
epochs = 50
batch_size = 64
learning_rate = 2e-4
optimizer = tf.train.RMSPropOptimizer
# Helper variables
train_mini_batches = train_size // batch_size
valid_mini_batches = valid_size // batch_size
test_mini_batches = test_size // batch_size

cnn_graph = tf.Graph()
with cnn_graph.as_default():
    # Input and Output of the NN
    x = tf.placeholder(dtype=tf.float32, shape=[None, img_width, img_height, img_depth])
    y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

    # Model definition
    def model(x):
        # Aufgabe 1: Das Netzwerk hier
        # Aufgabe 2: Xavier Init ist standard für die Conv und Fc Layer
        # Aufgabe 3: Data Augmentation weiter oben angewendet
        # Aufgabe 4: Siehe oben Parameterauswahl
        # Aufgabe 5: Dropout und BatchNorm siehe unten
        # Aufgabe 6: Early stopping weiter unten definiert
        x = conv_layer(x, 3, 32, k_size=3, name="conv1") # 32x32
        x = tf.nn.relu(x)
        x = conv_layer(x, 32, 32, k_size=3, name="conv11")
        x = tf.nn.relu(x)
        x = max_pool(x) #16x16
        x = conv_layer(x, 32, 64, k_size=3, name="conv2")
        x = tf.nn.relu(x)
        x = conv_layer(x, 64, 64, k_size=3, name="conv22")
        x = tf.nn.relu(x)
        x = max_pool(x) # 8x8
        x = conv_layer(x, 64, 128, k_size=3, name="conv3")
        x = tf.nn.relu(x)
        x = conv_layer(x, 128, 128, k_size=3, name="conv33")
        x = tf.nn.relu(x)
        x = max_pool(x) # 4x4
        x = flatten(x)
        x = fc_layer(x, 128*4*4, 512, name="fc1")
        x = fc_layer(x, 512, num_classes, name="fc2")
        return x

    # TensorFlow Ops to train/test
    pred_op = model(x)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_op, labels=y))
    correct_result_op = tf.equal(tf.argmax(pred_op, axis=1), tf.argmax(y, axis=1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_result_op , tf.float32))
    optimizer = optimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Start Training and Testing
    with tf.Session(graph=cnn_graph) as sess:
        sess.run(tf.global_variables_initializer())
        # Training
        print("\n\n Start training!")
        for epoch in range(epochs):
            train_acc, valid_acc = 0.0, 0.0
            train_loss, valid_loss = 0.0, 0.0
            # Train the weights
            for i in range(train_mini_batches):
                epoch_x, epoch_y = data.next_train_batch(batch_size)
                _, c = sess.run([train_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
            # Check the performance of the train set
            for i in range(train_mini_batches):
                epoch_x, epoch_y = data.next_train_batch(batch_size)
                a, c = sess.run([accuracy_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
                train_acc += a
                train_loss += c
            train_acc = train_acc / train_mini_batches
            # Check the performance of the valid set
            for i in range(valid_mini_batches):
                epoch_x, epoch_y = data.next_valid_batch(batch_size)
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
                epoch_x, epoch_y = data.next_test_batch(batch_size)
                a, c = sess.run([accuracy_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
                test_acc += a
                test_loss += c
        test_acc = test_acc / test_mini_batches
        print("Test Accuracy:\t", test_acc)
        print("Test Loss:\t", test_loss)
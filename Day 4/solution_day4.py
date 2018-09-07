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
#dataset_path = "C:/Users/schaf/Documents/GTSRB/Final_Training/Images/"
dataset_path = "C:/Users/jan/Documents/GTSRB/Final_Training/Images/"

num_classes = 43
imgs_per_class = 10000
classes = range(num_classes)
save_dataset(dataset_path, classes, imgs_per_class=imgs_per_class)
data = GTSRB(dataset_path, num_classes)
data.data_augmentation(augment_size=10000)
# Model variables
train_size, valid_size, test_size = data.train_size, data.valid_size, data.test_size
img_width, img_height, img_depth = data.img_width, data.img_height, data.img_depth

################
## EXERCISE 1 ##
################
 
# Hyperparameters
epochs = 4
batch_size = 32
learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer
# Helper variables
train_mini_batches = train_size // batch_size
valid_mini_batches = valid_size // batch_size
test_mini_batches = test_size // batch_size
# Variables for early stopping in training
early_stopping_crit = "accuracy"
early_stopping_patience = 10

cnn_graph = tf.Graph()
with cnn_graph.as_default():
    # Input and Output of the NN
    x = tf.placeholder(dtype=tf.float32, shape=[None, img_width, img_height, img_depth])
    y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])
    training = tf.placeholder_with_default(False, shape=())
    keep_prob = tf.placeholder_with_default(1.0, shape=())

# Model definition
def cnn_model(x, training, keep_prob, cnn_graph):
    # Aufgabe 1: Das Netzwerk ist hier
    # Aufgabe 2: Xavier Init ist standard für die Conv und Fc Layer, aus layers.py
    # Aufgabe 3: Data Augmentation weiter oben angewendet und kommt aus load_gtsrb.py
    # Aufgabe 4: Siehe oben Parameterauswahl
    # Aufgabe 5: Dropout und BatchNorm siehe unten, aus layers.py
    # Aufgabe 6: Early stopping unten verwendet
    with cnn_graph.as_default():
        x = conv_layer(x, filters=16, k_size=3, name="conv1") # 32x32
        x = tf.nn.relu(x)
        heatmap1 = heatmap(x)

        x = conv_layer(x, filters=16, k_size=3, name="conv2")
        x = tf.nn.relu(x)
        x = max_pool(x) #16x16
        x = dropout(x, 0.25, training=training)
        heatmap2 = heatmap(x)

        x = conv_layer(x, filters=32, k_size=3, name="conv3")
        x = tf.nn.relu(x)
        heatmap3 = heatmap(x)

        x = conv_layer(x, filters=32, k_size=3, name="conv4")
        x = tf.nn.relu(x)
        x = max_pool(x) # 8x8
        x = dropout(x, 0.25, training=training)
        heatmap4 = heatmap(x)

        x = conv_layer(x, filters=16, k_size=3, name="conv5")
        x = tf.nn.relu(x)
        heatmap5 = heatmap(x)

        x = conv_layer(x, filters=16, k_size=3, name="conv6")
        x = tf.nn.relu(x)
        x = max_pool(x) # 4x4
        heatmap6 = heatmap(x)

        x = flatten(x)
        x = dropout(x, 0.25, training=training)
        x = dense_layer(x, units=128, name="fc1")
        x = tf.nn.relu(x)
        x = dropout(x, 0.5, training=training)
        x = dense_layer(x, units=num_classes, name="fc2")

        heatmaps = {"1": heatmap1, "2": heatmap2, "3": heatmap3, 
                    "4": heatmap4, "5": heatmap5, "6": heatmap6}

        return (x, heatmaps)

with cnn_graph.as_default():
    # TensorFlow Ops
    (pred_op, heatmaps) = cnn_model(x, training, keep_prob, cnn_graph)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_op, labels=y))
    correct_result_op = tf.equal(tf.argmax(pred_op, axis=1), tf.argmax(y, axis=1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_result_op , tf.float32))
    optimizer = optimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_op)

# Start Training and Testing
with tf.Session(graph=cnn_graph) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    # Training
    print("\n\n Start training!")
    train_accs, valid_accs = [], []
    train_losses, valid_losses = [], []
    last_valid_acc = 0.0
    best_valid_acc = 0.0
    best_model_params = None
    for epoch in range(epochs):
        # Variables to track performance
        train_acc, valid_acc = 0.0, 0.0
        train_loss, valid_loss = 0.0, 0.0
        # Train the weights
        for i in range(train_mini_batches):
            epoch_x, epoch_y = data.next_train_batch(batch_size)
            _, c = sess.run([train_op, loss_op], 
                feed_dict={x: epoch_x, y: epoch_y, training: True, keep_prob: 0.8})
        # Check the performance of the train set
        for i in range(train_mini_batches):
            epoch_x, epoch_y = data.next_train_batch(batch_size)
            a, c = sess.run([accuracy_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
            train_acc += a
            train_loss += c
        train_acc = train_acc / train_mini_batches
        # Check the performance of the valid set
        for i in range(valid_mini_batches):
            epoch_x, epoch_y = data.next_valid_batch(i, batch_size)
            a, c = sess.run([accuracy_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
            valid_acc += a
            valid_loss += c
        valid_acc = valid_acc / valid_mini_batches
        print("Epoch: ", epoch+1, " of ", epochs, "- Train loss: ", round(train_loss, 3), 
            " Valid loss: ", round(valid_loss, 3), " Train Acc: ", round(train_acc, 3), 
            " Valid Acc: ", round(valid_acc, 3))
        # Early stopping check
        if valid_acc < last_valid_acc:
            early_stopping_patience -= 1
        if early_stopping_patience == 0:
            print("Early Stopping!")
            break
        if valid_acc > best_valid_acc:
           best_model_params = get_model_params()
        last_valid_acc = valid_acc
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
    display_convergence_error(train_losses, valid_losses)
    display_convergence_acc(train_accs, valid_accs)
    # Testing
    if best_model_params:
        restore_model_params(best_model_params)
    test_acc = 0.0
    test_loss = 0.0
    print("\n\nFinal testing!")
    for i in range(test_mini_batches):
            epoch_x, epoch_y = data.next_test_batch(i, batch_size)
            a, c = sess.run([accuracy_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
            test_acc += a
            test_loss += c
    test_acc = test_acc / test_mini_batches
    print("Test Accuracy:\t", test_acc)
    print("Test Loss:\t", test_loss)
    # Get sample from testset
    img, label = data.x_test[11], data.y_test[11]
    # Occlusion Map
    box_size = 5
    get_occlusion(img, label, box_size, sess, pred_op, x)
    # Heatmap
    get_heatmap(img, sess, heatmaps, x)
    # Save model
    acc_str = str(test_acc)
    save_path = saver.save(sess, "./saves_cnn/cnn_model_save"+acc_str[2:]+".ckpt")
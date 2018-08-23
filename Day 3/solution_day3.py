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
#dataset_path = "C:/Users/jan/Documents/GTSRB/Final_Training/Images/"
classes = range(43)

################
## EXERCISE 2 ##
################
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Helper function to split the dataset
def get_train_valid_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def save_dataset():
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
    # Split the dataset
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_train_valid_test(x, y)
    # Whiten the data
    x_train_mean = np.mean(x_train)
    x_train_std = np.std(x_train)
    x_train = np.array([(x_k - x_train_mean) / x_train_std for x_k in x_train])
    # Save the data to npy files
    np.save(dataset_path+"x_train.npy", x_train)
    np.save(dataset_path+"y_train.npy", y_train)
    np.save(dataset_path+"x_valid.npy", x_valid)
    np.save(dataset_path+"y_valid.npy", y_valid)
    np.save(dataset_path+"x_test.npy", x_test)
    np.save(dataset_path+"y_test.npy", y_test)

def load_dataset():
    x_train = np.load(dataset_path+"x_train.npy")
    y_train = np.load(dataset_path+"y_train.npy")
    x_valid = np.load(dataset_path+"x_valid.npy")
    y_valid = np.load(dataset_path+"y_valid.npy")
    x_test = np.load(dataset_path+"x_test.npy")
    y_test = np.load(dataset_path+"y_test.npy")
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

save_dataset()
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_dataset()


################
##    DNN     ##
################

################
## EXERCISE 3 ##
################
from layers import *

# Helper function to get minibatches by random sampling
def next_batch(x, y, batch_size):
    idx = np.random.randint(0, x.shape[0], batch_size)
    x_batch = x[idx]
    y_batch = y[idx]
    return x_batch, y_batch

# Model variables
num_classes = y_train.shape[1]
num_features = x_train.shape[1]
train_size, valid_size, test_size = x_train.shape[0], x_valid.shape[0], x_test.shape[0]
epochs = 40
batch_size = 64
learning_rate = 1e-2
train_mini_batches = train_size // batch_size
valid_mini_batches = valid_size // batch_size
test_mini_batches = test_size // batch_size

dnn_graph = tf.Graph()
with dnn_graph.as_default():
    # Input and Output of the NN
    x = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
    y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

    # Model definition
    def model(x):
        x = fc_layer(x, num_features, 3072, name="fc1")
        x = tf.nn.relu(x)
        x = fc_layer(x, 3072, 6144, name="fc2")
        x = tf.nn.relu(x)
        x = fc_layer(x, 6144, 2000, name="fc3")
        x = tf.nn.relu(x)
        x = fc_layer(x, 2000, num_classes, name="fc4")
        return x

    # TensorFlow Ops to train/test
    pred_op = model(x)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_op, labels=y))
    correct_result_op = tf.equal(tf.argmax(pred_op, axis=1), tf.argmax(y, axis=1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_result_op , tf.float32))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # # Start Training and Testing
    # with tf.Session(graph=dnn_graph) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     # Training
    #     print("\n\n Start training!")
    #     for epoch in range(epochs):
    #         train_acc, valid_acc = 0.0, 0.0
    #         train_loss, valid_loss = 0.0, 0.0
    #         # Train the weights
    #         for i in range(train_mini_batches):
    #             epoch_x, epoch_y = next_batch(x_train, y_train, batch_size)
    #             _, c = sess.run([train_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
    #         # Check the performance of the train set
    #         for i in range(train_mini_batches):
    #             epoch_x, epoch_y = next_batch(x_train, y_train, batch_size)
    #             a, c = sess.run([accuracy_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
    #             train_acc += a
    #             train_loss += c
    #         train_acc = train_acc / train_mini_batches
    #         # Check the performance of the valid set
    #         for i in range(valid_mini_batches):
    #             epoch_x, epoch_y = epoch_x, epoch_y = next_batch(x_valid, y_valid, batch_size)
    #             a, c = sess.run([accuracy_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
    #             valid_acc += a
    #             valid_loss += c
    #         valid_acc = valid_acc / valid_mini_batches
    #         print("Epoch: ", epoch+1, " of ", epochs, "- Train loss: ", round(train_loss, 3), 
    #             " Valid loss: ", round(valid_loss, 3), " Train Acc: ", round(train_acc, 3), 
    #             " Valid Acc: ", round(valid_acc, 3))
    #     # Testing
    #     test_acc = 0.0
    #     test_loss = 0.0
    #     print("\n\nFinal testing!")
    #     for i in range(test_mini_batches):
    #             epoch_x, epoch_y = epoch_x, epoch_y = next_batch(x_test, y_test, batch_size)
    #             a, c = sess.run([accuracy_op, loss_op], feed_dict={x: epoch_x, y: epoch_y})
    #             test_acc += a
    #             test_loss += c
    #     test_acc = test_acc / test_mini_batches
    #     print("Test Accuracy:\t", test_acc)
    #     print("Test Loss:\t", test_loss)



################
##     CNN    ##
################
################
## EXERCISE 4 ##
################

# Model variables
num_classes = y_train.shape[1]
num_features = x_train.shape[1]
train_size, valid_size, test_size = x_train.shape[0], x_valid.shape[0], x_test.shape[0]
epochs = 50
batch_size = 64
train_mini_batches = train_size // batch_size
valid_mini_batches = valid_size // batch_size
test_mini_batches = test_size // batch_size

learning_rate = 2e-4
optimizer = tf.train.RMSPropOptimizer

cnn_graph = tf.Graph()
with cnn_graph.as_default():
    # Input and Output of the NN
    x = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
    y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

    # Model definition
    def model(x):
        x = tf.reshape(x, shape=[-1, 32, 32, 3]) #32x32
        x = conv_layer(x, 3, 32, k_size=3, name="conv1")
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
# Imports
import tensorflow as tf
import numpy as np
import os
 
MEAN                = 0.000
STD_DEV             = 0.100
BN_EPSILON          = 0.001

activation_functions = {"relu": tf.nn.relu, "sigmoid": tf.nn.sigmoid, "lrelu": tf.nn.leaky_relu,
                        "tanh": tf.nn.tanh, "relu6": tf.nn.relu6}
weight_initializers = {"truncated_normal": tf.truncated_normal_initializer,
                        "normal": tf.random_normal_initializer,
                       "xavier": tf.contrib.layers.xavier_initializer}
 
# Conv Layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W,
                        strides=[1,1,1,1],
                        padding="SAME")

# Max Pool
def max_pool(x):
    return tf.nn.max_pool(x,
                        ksize=[1,2,2,1],
                        strides=[1,2,2,1],
                        padding="SAME")

# Avg Pool
def avg_pool(x):
    return tf.nn.average_pool(x,
                        ksize=[1,2,2,1],
                        strides=[1,2,2,1],
                        padding="SAME")

# Dropout
def dropout(x, keep_prob=1.0):
    return tf.nn.dropout(x, keep_prob=keep_prob)

# Batch Normalization Layer
def batch_norm(x, is_training):
    x = tf.contrib.layers.batch_norm(x, is_training=is_training)
    return x

# Init weights
def weight_initializer(name, shape, initializer="xavier"):
    with tf.name_scope(name):
        if initializer != "xavier":
            initializer = weight_initializers[initializer](mean=MEAN, stddev=STD_DEV)
        else:
            initializer = weight_initializers[initializer]()
        W = tf.get_variable(name+"W", shape=shape, initializer=initializer)
        tf.summary.histogram(name+"W", W)
        return W

# Init weights
def bias_initializer(name, shape, bias_init=0.0):
    with tf.name_scope(name):
        b = tf.get_variable(name+"b", shape=shape, initializer=tf.constant_initializer(bias_init))
        tf.summary.histogram(name+"b", b)
        return b

# Define a conv layer
def conv_layer(x, size_in, size_out, bias_init=0.0, k_size=3, name="conv", act="relu", initializer="xavier", activation=False):
    W = weight_initializer(name=name, shape=[k_size, k_size, size_in, size_out], initializer=initializer)
    b = bias_initializer(name=name, shape=[size_out], bias_init=bias_init)
    val = conv2d(x, W)
    val = tf.add(val, b)
    if activation:
        activation_func = activation_functions[act]
        val = activation_func(val)
    return val

# Define a fc layer
def fc_layer(x, size_in, size_out, bias_init=0.0, name="fc", act="relu", initializer="xavier", activation=False):
    with tf.name_scope(name):
        W = weight_initializer(name=name, shape=[size_in, size_out], initializer=initializer)
        b = bias_initializer(name=name, shape=[size_out], bias_init=bias_init)
        val = tf.matmul(x, W)
        val = tf.add(val, b)
        if activation:
            activation_func = activation_functions[act]
            val = activation_func(val)
        return val

# Transform to flattened layer
def flatten(x):
    shape = tf.shape(x)
    new_shape = shape[1] * shape[2] * shape[3]
    return tf.reshape(x, shape=[-1, new_shape])
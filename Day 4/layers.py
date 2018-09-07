# Imports
import tensorflow as tf
import numpy as np
import os
 
MEAN                = 0.000
STD_DEV             = 0.100

activation_functions = {"relu": tf.nn.relu, "sigmoid": tf.nn.sigmoid, "lrelu": tf.nn.leaky_relu,
                        "tanh": tf.nn.tanh, "relu6": tf.nn.relu6, "softmax": tf.nn.softmax}
weight_initializers = {"truncated_normal": tf.truncated_normal_initializer,
                        "normal": tf.random_normal_initializer,
                       "xavier": tf.glorot_normal_initializer}
 
# Max Pool
def max_pool(x, pool_size=(2,2), strides=(2,2), padding="same"):
    return tf.layers.max_pooling2d(x, pool_size=pool_size,
                                    strides=strides, padding=padding)

# Avg Pool
def avg_pool(x, pool_size=(2,2), strides=(2,2), padding="same"):
    return tf.layers.average_pooling2d(x, pool_size=pool_size,
                                    strides=strides, padding=padding)

# Dropout
def dropout(x, keep_prob=1.0, training=False):
    return tf.layers.dropout(x, keep_prob, training=training)

# Batch Normalization Layer
def batch_norm(x, training=False):
    return tf.layers.batch_normalization(x, training=training)

# Define a conv layer
def conv_layer(x, filters, k_size, name="conv", padding="same", strides=(1,1), w_initializer="xavier"):
    kernel_initializer = weight_initializers[w_initializer]
    return tf.layers.conv2d(x, filters=filters, kernel_size=k_size,
                            strides=strides, padding=padding, name=name,
                            kernel_initializer=kernel_initializer(),
                            bias_initializer=tf.zeros_initializer())

# Define a fc layer
def dense_layer(x, units, name="fc", w_initializer="xavier"):
    kernel_initializer = weight_initializers[w_initializer]
    return tf.layers.dense(x, units=units, kernel_initializer=kernel_initializer(),
                            bias_initializer=tf.zeros_initializer(), name=name)

# Transform to flattened layer
def flatten(x):
    return tf.layers.flatten(x)

# Activation/Heatmaps
def heatmap(x):
    heatmap = tf.identity(x)
    heatmap = tf.nn.relu(heatmap)
    return heatmap
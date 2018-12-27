"""
Artificial Neural Network Code v1.0  (Feed-forward neural network)

By:  Rui Nian

Date of last edit: December 28th, 2018

Patch notes: -

Known issues: -

Features:  normalization, shuffle, prec, recall, testing

To add:  He initialization, drop out, regularization
To test: Load labeled_data.csv, remove shuffle, feed t and t - 1, pickle in min_max_normalization, restore graph,
         uncomment test
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import pickle

import gc

from copy import deepcopy

import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Min max normalization
class MinMaxNormalization:
    """
    Inputs
       -----
            data:  Input feature vectors from the training data

    Attributes
       -----
         col_min:  The minimum value per feature
         col_max:  The maximum value per feature
     denominator:  col_max - col_min

     Methods
        -----
     init:  Builds the col_min, col_max, and denominator
     call:  Normalizes data based on init attributes
    """

    def __init__(self, data):
        self.col_min = np.min(data, axis=0).reshape(1, data.shape[1])
        self.col_max = np.max(data, axis=0).reshape(1, data.shape[1])
        self.denominator = abs(self.col_max - self.col_min)

        # Fix divide by zero, replace value with 1 because these usually happen for boolean columns
        for index, value in enumerate(self.denominator[0]):
            if value == 0:
                self.denominator[0][index] = 1

    def __call__(self, data):
        return np.divide((data - self.col_min), self.denominator)


# Load data
path = '/Users/ruinian/Documents/Logistic_Reg_TF/'
# path = '/home/rui/Documents/logistic_regression_tf'

raw_data = pd.read_csv(path + 'data/syn_10_data.csv', header=None)
# raw_data = pd.read_csv(path + 'data/labeled_data.csv', header=None)

# Get all feature names
feature_names = list(raw_data)

# Delete the label column name
del feature_names[0]

# Turn pandas dataframe into NumPy array
raw_data = raw_data.values
np.random.shuffle(raw_data)
print("Raw data has {} features and {} examples.".format(raw_data.shape[1], raw_data.shape[0]))

# Features and labels split
features = raw_data[:, 1:]
labels = raw_data[:, 0].reshape(features.shape[0], 1)

train_size = 0.9
train_index = int(train_size * raw_data.shape[0])

train_X = features[0:train_index, :]
train_y = labels[0:train_index]

test_X = features[train_index:, :]
test_y = labels[train_index:]

"""
To feed t and t - 1
"""
# train_X = np.concatenate([train_X[:-1, :], train_X[1:, :]], axis=1)
# train_y = train_y[:-1, :]
#
# test_X = np.concatenate([test_X[:-1, :], test_X[1:, :]], axis=1)
# test_y = test_y[:-1, :]

min_max_normalization = MinMaxNormalization(train_X)
train_X = min_max_normalization(train_X)
test_X = min_max_normalization(test_X)

assert(not np.isnan(train_X).any())
assert(not np.isnan(test_X).any())

input_size = features.shape[1]
nodes_h1 = 20
nodes_h2 = 20
nodes_h3 = 20
output_size = 1

batch_size = 256
total_batch_number = int(train_X.shape[0] / batch_size)

X = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

hidden_layer_1 = {'weights': tf.get_variable('h1_weights', shape=[input_size, nodes_h1],
                                             initializer=tf.contrib.layers.xavier_initializer()),
                  'biases': tf.get_variable('h1_biases', shape=[nodes_h1],
                                            initializer=tf.contrib.layers.xavier_initializer())}

hidden_layer_2 = {'weights': tf.get_variable('h2_weights', shape=[nodes_h1, nodes_h2],
                                             initializer=tf.contrib.layers.xavier_initializer()),
                  'biases': tf.get_variable('h2_biases', shape=[nodes_h2],
                                            initializer=tf.contrib.layers.xavier_initializer())}

hidden_layer_3 = {'weights': tf.get_variable('h3_weights', shape=[nodes_h2, nodes_h3],
                                             initializer=tf.contrib.layers.xavier_initializer()),
                  'biases': tf.get_variable('h3_biases', shape=[nodes_h3],
                                            initializer=tf.contrib.layers.xavier_initializer())}

output_layer = {'weights': tf.get_variable('output_weights', shape=[nodes_h3, output_size],
                                           initializer=tf.contrib.layers.xavier_initializer()),
                'biases': tf.get_variable('output_biases', shape=[output_size],
                                          initializer=tf.contrib.layers.xavier_initializer())}

l1 = tf.add(tf.matmul(X, hidden_layer_1['weights']), hidden_layer_1['biases'])
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
l2 = tf.nn.relu(l2)

l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
l3 = tf.nn.relu(l3)

output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# For output of 0 or 1
pred = tf.round(tf.sigmoid(output))

correct = tf.cast(tf.equal(pred, y), dtype=tf.float32)
accuracy = tf.reduce_mean(correct)
precision, prec_ops = tf.metrics.precision(labels=y, predictions=pred)
recall, recall_ops = tf.metrics.recall(labels=y, predictions=pred)

init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

epochs = 5
loss_history = []

saver = tf.train.Saver()

with tf.Session() as sess:

    # saver.restore(sess, 'checkpoints/test.ckpt')

    sess.run(init)
    sess.run(init_l)

    for epoch in range(epochs):

        for i in range(total_batch_number):
            batch_index = i * batch_size
            batch_X = train_X[batch_index:batch_index + batch_size, :]
            batch_y = train_y[batch_index:batch_index + batch_size, :]

            sess.run(optimizer, feed_dict={X: batch_X, y: batch_y})
            current_loss = sess.run(loss, feed_dict={X: batch_X, y: batch_y})
            loss_history.append(current_loss)

            if i % 18 == 0 and i != 0:
                Acc = sess.run(accuracy, feed_dict={X: batch_X, y: batch_y})
                Prec, Recall = sess.run([prec_ops, recall_ops], feed_dict={X: batch_X, y: batch_y})
                print('Acc: {:5f} | Prec: {:5f} | Recall: {:5f}'.format(Acc, Prec, Recall))

            # predictions = sess.run(pred, feed_dict={X: batch_X, y: batch_y})
            # Acc = sess.run(accuracy, feed_dict={X: features, y: labels})
            # Prec, Recall = sess.run([prec_ops, recall_ops], feed_dict={X: features, y: labels})
            # print('Acc: {:5f} | Prec: {:5f} | Recall: {:5f}'.format(Acc, Prec, Recall))
            # break

    saver.save(sess, 'checkpoints/test.ckpt')

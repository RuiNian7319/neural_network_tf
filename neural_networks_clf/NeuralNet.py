"""
Artificial Neural Network Code v1.0  (Feed-forward neural network)

By:  Rui Nian

Date of last edit: January 3rd, 2018

Patch notes: -

Known issues: -

Features:  Normalization, Shuffle, Precision, Recall, He init, L2 Regularization, Drop out, batch normalization
notes: for batch norm, batch size has to be identical, also, add is_train=True to feed dict

To add:  proper testing procedure, batch normalization

To test: Load labeled_data.csv, remove shuffle, feed t and t - 1, pickle in min_max_normalization, restore graph,
         uncomment test
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import pickle

from EVAL_NeuralNetClf import *

import gc

from copy import deepcopy

import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)


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
# path = '/Users/ruinian/Documents/Willowglen/'
path = '/home/rui/Documents/Willowglen/'

# raw_data = pd.read_csv(path + 'data/10_data_12.csv', header=None)
raw_data = pd.read_csv(path + 'data/labeled_data.csv')

# Get all feature names
feature_names = list(raw_data)

# Delete the label column name
del feature_names[0]

# Turn pandas dataframe into NumPy array
raw_data = raw_data.values
# np.random.shuffle(raw_data)
print("Raw data has {} features and {} examples.".format(raw_data.shape[1], raw_data.shape[0]))

# Features and labels split
features = raw_data[:, 1:]
labels = raw_data[:, 0].reshape(features.shape[0], 1)

train_size = 1
assert(train_size <= 1)
train_index = int(train_size * raw_data.shape[0])

train_X = features[0:train_index, :]
train_y = labels[0:train_index]

test_X = features[train_index:, :]
test_y = labels[train_index:]

"""
To feed t and t - 1
"""
train_X = np.concatenate([train_X[:-1, :], train_X[1:, :]], axis=1)
train_y = train_y[:-1, :]

test_X = np.concatenate([test_X[:-1, :], test_X[1:, :]], axis=1)
test_y = test_y[:-1, :]

# min_max_normalization = MinMaxNormalization(train_X)
pickle_in = open(path + 'neural_net_tf/pickles/norm.pickle', 'rb')
min_max_normalization = pickle.load(pickle_in)
train_X = min_max_normalization(train_X)
test_X = min_max_normalization(test_X)

assert(not np.isnan(train_X).any())
assert(not np.isnan(test_X).any())

input_size = train_X.shape[1]
nodes_h1 = 35
nodes_h2 = 35
nodes_h3 = 35
nodes_h4 = 35
nodes_h5 = 35

output_size = 1

batch_size = 16
total_batch_number = int(train_X.shape[0] / batch_size)

X = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

# Batch Normalization
training = True
is_train = tf.placeholder(tf.bool, name='is_train')

hidden_layer_1 = {'weights': tf.get_variable('h1_weights', shape=[input_size, nodes_h1],
                                             initializer=tf.contrib.layers.variance_scaling_initializer()),
                  'biases': tf.get_variable('h1_biases', shape=[nodes_h1],
                                            initializer=tf.contrib.layers.variance_scaling_initializer())}

hidden_layer_2 = {'weights': tf.get_variable('h2_weights', shape=[nodes_h1, nodes_h2],
                                             initializer=tf.contrib.layers.variance_scaling_initializer()),
                  'biases': tf.get_variable('h2_biases', shape=[nodes_h2],
                                            initializer=tf.contrib.layers.variance_scaling_initializer())}

hidden_layer_3 = {'weights': tf.get_variable('h3_weights', shape=[nodes_h2, nodes_h3],
                                             initializer=tf.contrib.layers.variance_scaling_initializer()),
                  'biases': tf.get_variable('h3_biases', shape=[nodes_h3],
                                            initializer=tf.contrib.layers.variance_scaling_initializer())}

hidden_layer_4 = {'weights': tf.get_variable('h4_weights', shape=[nodes_h3, nodes_h4],
                                             initializer=tf.contrib.layers.variance_scaling_initializer()),
                  'biases': tf.get_variable('h4_biases', shape=[nodes_h4],
                                            initializer=tf.contrib.layers.variance_scaling_initializer())}

hidden_layer_5 = {'weights': tf.get_variable('h5_weights', shape=[nodes_h4, nodes_h5],
                                             initializer=tf.contrib.layers.variance_scaling_initializer()),
                  'biases': tf.get_variable('h5_biases', shape=[nodes_h5],
                                            initializer=tf.contrib.layers.variance_scaling_initializer())}

output_layer = {'weights': tf.get_variable('output_weights', shape=[nodes_h5, output_size],
                                           initializer=tf.contrib.layers.variance_scaling_initializer()),
                'biases': tf.get_variable('output_biases', shape=[output_size],
                                          initializer=tf.contrib.layers.variance_scaling_initializer())}

# drop_out_prob = 0.2

l1 = tf.add(tf.matmul(X, hidden_layer_1['weights']), hidden_layer_1['biases'])
l1 = tf.nn.relu(l1)
# l1 = tf.nn.dropout(l1, keep_prob=drop_out_prob)
l1 = tf.layers.batch_normalization(l1, training=is_train)

l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
l2 = tf.nn.relu(l2)
# l2 = tf.nn.dropout(l2, keep_prob=drop_out_prob)
l2 = tf.layers.batch_normalization(l2, training=is_train)

l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
l3 = tf.nn.relu(l3)
# l3 = tf.nn.dropout(l3, keep_prob=drop_out_prob)
l3 = tf.layers.batch_normalization(l3, training=is_train)

output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

# L2 Regularization
lambd = 0.001
trainable_vars = tf.trainable_variables()
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]) * lambd

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y) + lossL2)

# Batch Normalization
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# For output of 0 or 1
pred = tf.round(tf.sigmoid(output))

correct = tf.cast(tf.equal(pred, y), dtype=tf.float32)
accuracy = tf.reduce_mean(correct)
precision, prec_ops = tf.metrics.precision(labels=y, predictions=pred)
recall, recall_ops = tf.metrics.recall(labels=y, predictions=pred)

init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

epochs = 1
loss_history = []

saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, path + 'neural_net_tf/checkpoints/test.ckpt')

    # sess.run(init)
    sess.run(init_l)

    for epoch in range(epochs):

        for i in range(total_batch_number):
            batch_index = i * batch_size
            batch_X = train_X[batch_index:batch_index + batch_size, :]
            batch_y = train_y[batch_index:batch_index + batch_size, :]

            sess.run(optimizer, feed_dict={X: batch_X, y: batch_y, is_train: training})

            if i % int(total_batch_number / 2) == 0 and i != 0:
                train_acc = sess.run(accuracy, feed_dict={X: train_X, y: train_y, is_train: training})
                test_acc = sess.run(accuracy, feed_dict={X: test_X, y: test_y, is_train: training})
                Prec, Recall = sess.run([prec_ops, recall_ops], feed_dict={X: np.concatenate([train_X, test_X], axis=0),
                                                                           y: np.concatenate([train_y, test_y], axis=0),
                                                                           is_train: training})

                current_loss = sess.run(loss, feed_dict={X: batch_X, y: batch_y, is_train: training})
                loss_history.append(current_loss)

                print('Epoch: {} | Loss: {:5f} | Train_Acc: {:5f} | Test_Acc {:5f} | Prec: {:5f} | Recall: {:5f}'
                      .format(epoch, current_loss, train_acc, test_acc, Prec, Recall))

            pred_perc = tf.sigmoid(output)
            predictions = sess.run(pred_perc, feed_dict={X: train_X, y: train_y, is_train: training})
            Acc = sess.run(accuracy, feed_dict={X: train_X, y: train_y, is_train: training})
            Prec, Recall = sess.run([prec_ops, recall_ops], feed_dict={X: np.concatenate([train_X, test_X], axis=0),
                                                                       y: np.concatenate([train_y, test_y], axis=0),
                                                                       is_train: training})
            print('Acc: {:5f} | Prec: {:5f} | Recall: {:5f}'.format(Acc, Prec, Recall))
            break

    # saver.save(sess, path + 'neural_net_tf/checkpoints/test.ckpt')

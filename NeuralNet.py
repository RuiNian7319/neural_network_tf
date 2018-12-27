"""
Artificial Neural Network Code v1.0  (Feed-forward neural network)

By:  Rui Nian

Date of last edit: December 26th, 2018

Patch notes: -

Known issues: -

shuffle, normalize
to beat: precision = 96.1%, recall = 94.6%
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

df = pd.read_csv("test_datasets/breast-cancer-wisconsin.data.txt")
# Remove missing values
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Remove ID column
df.drop(['id'], axis=1, inplace=True)
df.iloc[:, 5] = pd.to_numeric(df.iloc[:, 5])

# Replace labels booleans column with 0 and 1
df.loc[:, 'class'].replace(2, 0, inplace=True)
df.loc[:, 'class'].replace(4, 1, inplace=True)

# Sort labels, with minority class on top
df.sort_values(['class'], ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)
df = df.values

features = df[:, 0:-1]
labels = df[:, -1].reshape(features.shape[0], 1)

input_size = features.shape[1]
nodes_h1 = 10
nodes_h2 = 10
nodes_h3 = 10
output_size = 1

batch_size = 32
total_batch_number = int(features.shape[0] / batch_size)

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

epochs = 200
loss_history = []

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    sess.run(init_l)

    # saver.restore(sess, 'checkpoints/test.ckpt')

    for epoch in range(epochs):

        for i in range(total_batch_number):
            batch_index = i * batch_size
            batch_X = features[batch_index:batch_index + batch_size, :]
            batch_y = labels[batch_index:batch_index + batch_size, :]

            sess.run(optimizer, feed_dict={X: batch_X, y: batch_y})
            current_loss = sess.run(loss, feed_dict={X: batch_X, y: batch_y})
            loss_history.append(current_loss)

            if i % 10 == 0:
                Acc = sess.run(accuracy, feed_dict={X: batch_X, y: batch_y})
                Prec, Recall = sess.run([prec_ops, recall_ops], feed_dict={X: batch_X, y: batch_y})
                print('Acc: {:5f} | Prec: {:5f} | Recall: {:5f}'.format(Acc, Prec, Recall))

            # predictions = sess.run(pred, feed_dict={X: batch_X, y: batch_y})
            # print(predictions)
            # print(batch_y)
            # break

    saver.save(sess, 'checkpoints/test.ckpt')

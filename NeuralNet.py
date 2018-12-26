"""
Artificial Neural Network Code v1.0  (Feed-forward neural network)

By:  Rui Nian

Date of last edit: December 26th, 2018

Patch notes: -

Known issues: -
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import pickle

import gc

from copy import deepcopy

import warnings

df = pd.read_csv("test_datasets/breast-cancer-wisconsin.data.txt")
# Remove missing values
df.replace('?', -99999, inplace=True)
# Remove ID column
df.drop(['id'], axis=1, inplace=True)
df.iloc[:, 5] = pd.to_numeric(df.iloc[:, 5])
# Replace labels booleans column with 0 and 1
df.loc[:, 'class'].replace(2, 0, inplace=True)
df.loc[:, 'class'].replace(4, 1, inplace=True)
# Sort labels, with minority class on top
df.sort_values(['class'], ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

input_size = 9
nodes_h1 = 100
nodes_h2 = 100
nodes_h3 = 100
output_size = 1

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

l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
l2 = tf.nn.relu(l2)

l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
l3 = tf.nn.relu(l3)

output = tf.matmul(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

epochs = 10

with
    








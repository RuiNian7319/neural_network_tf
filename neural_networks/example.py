import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

"""
Input > Weight, Bias > Hidden Layer 1 (Activation Function) > Weight, Bias > Hidden Layer 2 (Activation Function) > ...
z = wx + b
y = sigma(z)

Compare output to intended output via cost or loss function, like cross entropy or mean squared error
Optimization function to minimize cost via computing gradients, then applying gradient descent. (Adam, SGD, AdaGrad)
-> Backpropagation

After one pass through ALL the data -> Epoch
"""

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 10 classes, 0-9, so the actual value is 1 and the rest are zeros.

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', shape=[None, 28 * 28])
y = tf.placeholder('float', shape=[None, n_classes])


def neural_network_model(data):
    # hidden_1_layer = {'weights': tf.get_variable('l1_weights', shape=[784, n_nodes_hl1],
    #                                              initializer=tf.contrib.layers.xavier_initializer())}

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    print(prediction)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                epoch_loss += c

            print("Epoch {} completed out of {}. Loss: {}".format(epoch, epochs, epoch_loss))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy: {}'.format(accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))

    return correct, prediction, accuracy


correct, prediction, accuracy = train_neural_network(x)

# -*- coding: utf-8 -*-

# Sample code to use string producer.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    # if type(x) == list:
    #    x = np.array(x)
    # x = x.flatten()
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 3
batch_size = 5


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------


def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), num_classes)
        # image = tf.image.resize_image_with_crop_or_pad(image, 128, 128)
        image = tf.reshape(image, [128, 128, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 30 * 30 * 64]), units=20, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


train_batch, label_batch_train = dataSource(["data/0/train/*.jpg",
                                             "data/1/train/*.jpg",
                                             "data/2/train/*.jpg"],
                                            batch_size=batch_size)

valid_batch, label_batch_valid = dataSource(["data/0/validation/*.jpg",
                                             "data/1/validation/*.jpg",
                                             "data/2/validation/*.jpg"],
                                            batch_size=batch_size)

test_batch, label_batch_test = dataSource(["data/0/test/*.jpg",
                                           "data/1/test/*.jpg",
                                           "data/2/test/*.jpg"],
                                          batch_size=batch_size)

batch_train_predicted = myModel(train_batch, reuse=False)
batch_valid_predicted = myModel(valid_batch, reuse=True)
batch_test_predicted = myModel(test_batch, reuse=True)

cost = tf.reduce_sum(tf.square(batch_train_predicted - tf.cast(label_batch_train, dtype=tf.float32)))
cost_valid = tf.reduce_sum(tf.square(batch_valid_predicted - tf.cast(label_batch_valid, dtype=tf.float32)))
cost_test = tf.reduce_sum(tf.square(batch_test_predicted - tf.cast(label_batch_test, dtype=tf.float32)))

# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    error = 0
    validError = 0
    previousLoss = 10
    lossesVal = []
    losses = []

    for _ in range(10000):
        sess.run(optimizer)
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(label_batch_valid))
            print(sess.run(batch_valid_predicted))
            validError = sess.run(cost_valid)
            lossesVal.append(validError)
            print("Error validación: ", validError)
            error = sess.run(cost)
            losses.append(error)
            print("Error", error)
            if abs(validError - previousLoss) < 0.0001:
                break
            previousLoss = validError

    print('/************* TESTING *************/')

    expectedResult = sess.run(label_batch_test)
    result = sess.run(batch_test_predicted)

    rightGuess = 0
    failGuess = 0

    for b, r in zip(expectedResult, result):
        if np.argmax(b) == np.argmax(r):
            rightGuess += 1
        else:
            failGuess += 1
        print(b, "-->", r)
        print("Good: ", rightGuess)
        print("Bad: ", failGuess)
        Total = rightGuess + failGuess
        print("Percentage of right guesses: ", (float(rightGuess) / float(Total)) * 100, "%")
        print("----------------------------------------------------------------------------------")

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    plt.plot(losses)
    plt.plot(lossesVal)
    plt.legend(['E. Validación', 'E. Entrenamiento'], loc='upper right')
    plt.show()

    coord.request_stop()
    coord.join(threads)

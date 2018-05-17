# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

filenames0 = tf.train.match_filenames_once("data/0/*")
filenames1 = tf.train.match_filenames_once("data/1/*")
filenames2 = tf.train.match_filenames_once("data/2/*")

filename_queue0 = tf.train.string_input_producer(filenames0, shuffle=False)
filename_queue1 = tf.train.string_input_producer(filenames1, shuffle=False)
filename_queue2 = tf.train.string_input_producer(filenames2, shuffle=False)

reader0 = tf.WholeFileReader()
reader1 = tf.WholeFileReader()
reader2 = tf.WholeFileReader()

# Here we have the files in binary format
key0, file_image0 = reader0.read(filename_queue0)
key1, file_image1 = reader1.read(filename_queue1)
key2, file_image2 = reader2.read(filename_queue2)

image0, label0 = tf.image.decode_jpeg(file_image0), [1., 0., 0.]  # key0
# image0 = tf.image.resize_image_with_crop_or_pad(image0, 80,80)...
image0 = tf.reshape(image0, [128, 128, 1])

image1, label1 = tf.image.decode_jpeg(file_image1), [0., 1., 0.]  # key1
image1 = tf.reshape(image1, [128, 128, 1])

image2, label2 = tf.image.decode_jpeg(file_image2), [0., 0., 1.]  # Key 3
image2 = tf.reshape(image2, [128, 128, 1])

image0 = tf.to_float(image0) / 256. - 0.5
image1 = tf.to_float(image1) / 256. - 0.5
image2 = tf.to_float(image2) / 256. - 0.5

batch_size = 5
min_after_dequeue = 10  # 10000
capacity = min_after_dequeue + 3 * batch_size


example_batch0, label_batch0 = tf.train.shuffle_batch([image0, label0], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

example_batch1, label_batch1 = tf.train.shuffle_batch([image1, label1], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

example_batch2, label_batch2 = tf.train.shuffle_batch([image2, label2], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

example_batch = tf.concat(values=[example_batch0, example_batch1, example_batch2], axis=0)
label_batch = tf.concat(values=[label_batch0, label_batch1, label_batch2], axis=0)

# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

# Here we get the tensor with dimensions 78x138x32(at least its what was said) convolutional layer
o1 = tf.layers.conv2d(inputs=example_batch, filters=32, kernel_size=3, activation=tf.nn.relu)
# Here we get the reduced tensor with the dimensions 39x69x32 max pool
o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
# Here we get the tensor with dimensions 37x67x64 convolutional
o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
# Here we pass the volume through another pool layer, getting the volume with dimensions 18x33x64
o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

# Here we use the "ReLU" function for something i don't know what
h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 30 * 30 * 64]), units=5, activation=tf.nn.relu)
# The activation function is the sigmoid
# y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.sigmoid)
# This is the model 'y'
y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)

cost = tf.reduce_sum(tf.square(y - label_batch))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

# Object implemented in tensorflow which saves the results in the disk
saver = tf.train.Saver()
currentLoss = 0
losses = [None]
previousLoss = 10

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for _ in range(3000):
        sess.run(optimizer)
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(y))
            currentLoss = sess.run(label_batch)
            losses.append(currentLoss)
            print("Error:", sess.run(cost))

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)


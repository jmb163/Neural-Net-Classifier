#!/usr/local/bin/python3

from __future__ import print_function
import numpy as np
import tensorflow as tf
import math as math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
args = parser.parse_args()

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def read_from_csv(filename_queue):
  reader = tf.TextLineReader(skip_header_lines=1)
  _, csv_row = reader.read(filename_queue)
  record_defaults = [[5.0] for col in range(11)]
  cols = []
  cols = tf.decode_csv(csv_row, record_defaults=record_defaults, field_delim=',')
  features = tf.stack(cols[1:10])
  #tf.reshape(features, shape=[1,8])
  label = tf.stack(cols[10])
  return features, label

def input_pipeline(batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer([args.dataset], num_epochs=num_epochs, shuffle=False)
  print(args.dataset)
  example, label = read_from_csv(filename_queue)
  print("***************************************************************")
  print(example)
  print(label)
  print("***************************************************************")
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch

def def_graph():
  with tf.Graph().as_default():
    input_placeholder = tf.placeholder(shape=[1, 9], dtype=tf.float32, name="input_placeholder")
    target_placeholder = tf.placeholder(shape=[1], dtype=tf.float32, name="target_placeholder")
    sess = tf.Session()
    w1 = tf.Variable(tf.random_uniform(shape=[9, 30], dtype=tf.float32), name="weights1")
    b1 = tf.Variable(tf.random_uniform(shape=[1, 30], dtype=tf.float32), name="bias1")

    h1 = tf.sigmoid(tf.matmul(input_placeholder, w1) + b1, name="hidden1")
    w2 = tf.Variable(tf.random_uniform(shape=[30, 1], dtype=tf.float32), name="weigts2")
    output = tf.matmul(h1, w2, name="output")
    loss = tf.reduce_mean(tf.squared_difference(target_placeholder, output), name="loss")

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, name="train_op")

    init = tf.global_variables_initializer()
    sess.run(init)
  return sess, train_op

batch_size=20

sess, train_op = def_graph()

with sess as sess:
  init=tf.global_variables_initializer()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  goal=tf.constant([0.2])

  # start populating filename queue

  try:
    while not coord.should_stop():
      print("now training batch")
      input_batch, target_batch = input_pipeline(batch_size)
      tf.Print(input_batch, [input_batch])
      print(input_batch.eval())
      for step in range(batch_size):
        instance = tf.get_session_handle(tf.slice(input_batch, [step, 0], [step, 10]))
        instance_target = tf.get_session_handle(tf.slice(target_batch, [step], [1]))
        instance=sess.run(instance)
        instance_target=sess.run(instance_target)
        feed_dict = {tf.get_variable(name="input_placeholder", shape=[1,9]): instance,
              tf.get_variable(name="target_placeholder", shape=[1]): instance_target}
        sess.run(train_op, feed_dict=feed_dict)
        if (goal > tf.get_variable(name="loss", shape=[1])):
          print("*************************************")
          print("\n\n")
          print("convergence reached!")
          print("\n\n")
          print("*************************************")
          break

  except tf.errors.OutOfRangeError:
    print('Done training, epoch reached')
  finally:
    coord.request_stop()

  coord.join(threads)

# with tf.Session() as sess:
#   init=tf.global_variables_initializer()
#   coord = tf.train.Coordinator()
#   threads = tf.train.start_queue_runners(coord=coord)
#   goal=tf.constant([0.2])
#
#   for step in range(batch_size):
#
#   # start populating filename queue
#   input, target = input_pipeline(batch_size)
#   print(input.__class__.__name__)
#   print(target.__class__.__name__)
#
#   tf.Print([input], [target], "this is a")
#   input=tf.identity(input, name="input")
#   target=tf.identity(target, name="target")
#
#
#
#   weights=tf.Variable(tf.random_uniform(shape=[9,30]))
#   bias=tf.Variable(tf.random_uniform(shape=[1,30]))
#
#   h1=tf.nn.sigmoid(tf.matmul(input,weights)+bias)
#   w2=tf.Variable(tf.random_uniform(shape=[30,1]))
#
#   output=tf.matmul(h1, w2)
#
#   loss = tf.reduce_mean(tf.squared_difference(target, output))
#   tf.Print(loss, [loss])
#
#   try:
#     while not coord.should_stop():
#
#       example_batch, label_batch = sess.run(init, loss, feed_dict={"input":input,"target": target})
#       if (goal > loss):
#         print("*************************************")
#         print("\n\n")
#         print("convergence reached!")
#         print("\n\n")
#         print("*************************************")
#         break
#
#       print(example_batch)
#   except tf.errors.OutOfRangeError:
#     print('Done training, epoch reached')
#   finally:
#     coord.request_stop()
#
#   coord.join(threads)

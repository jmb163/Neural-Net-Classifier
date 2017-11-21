#!/usr/local/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd

'''
to implement:
  main
  run_training
  do_eval
  fill_feed_dict *
  placeholder_inputs *
'''

batch_size = 20
hidden_layer1_nodes = 30
hidden_layer2_nodes = 1
target_size = 1
attr_size = 9

#config = tf.ConfigProto(device_count = {'GPU': 0})

v_deteriorations = 0
'''
stopping rule:
if the validation does worse a couple times in a row while the network error stagnates,
then the network should stop training, and save the weights that it has trained.
'''
run_options = tf.RunOptions(timeout_in_ms=10000)

#model params (2 layer network)
input_placeholder = tf.placeholder(shape=[None, attr_size], dtype=tf.float32, name="input")
target_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="target")

w1 = tf.Variable(tf.random_normal(shape=[attr_size, hidden_layer1_nodes]), name="w1")
b1 = tf.Variable(tf.random_normal(shape=[1, hidden_layer1_nodes]), name="b1")

w2 = tf.Variable(tf.random_normal(shape=[hidden_layer1_nodes, target_size]), name="w2")
b2 = tf.Variable(tf.random_normal(shape=[1, target_size]), name="b2")

tf.random_shuffle(input_placeholder)

h1 = tf.sigmoid(tf.matmul(input_placeholder, w1) + b1)
output = tf.matmul(h1, w2) + b2

loss = tf.reduce_mean(tf.squared_difference(target_placeholder, output))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
truncated_output = tf.cast(output, dtype=tf.int32)
truncated_target = tf.cast(target_placeholder, dtype=tf.int32)

correct_prediction = tf.equal(tf.cast(output, dtype=tf.int32), tf.cast(target_placeholder, dtype=tf.int32))
accuracy, acc_op = tf.metrics.accuracy(labels=tf.cast(output, dtype=tf.int32), predictions=tf.cast(target_placeholder, dtype=tf.int32))


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
  features = tf.reshape(features, shape=[1,9])
  label = tf.stack(cols[10])
  label = tf.reshape(label, shape=[1, 1])
  # print("****************************")
  # print(tf.shape(features), tf.shape(label))
  # print("****************************")

  return features, label

def get_data_set(fqueue):
    return read_from_csv(fqueue)

def single_read(filename):
    df = pd.read_csv(filename, sep=',', header=None)
    arr = np.array(df)
    return arr[:, 1:10], arr[:, 10:] #labels, targets

def placeholder_inputs(batch_size):
    input_placeholder = tf.placeholder(tf.float32, shape=[batch_size, 9])
    target_placeholder = tf.placeholder(tf.float32, shape=[batch_size])
    return input_placeholder, target_placeholder

def read_data_set(filenames):
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    return filename_queue

def input_pipeline(batch_size, filenames):
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    example, label = read_from_csv(filename_queue)
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, min_after_dequeue=250, capacity=300)
    return example_batch, label_batch

init = tf.global_variables_initializer()

#we need the threads to be started so that the files can be read

# inputs, targets = input_pipeline(batch_size, ["./sets/processedtrain.csv"])
# validation_inputs, validation_targets = input_pipeline(batch_size, ["processedvalid.csv"])

#when the model starts to perform worse on the validation set, during training, we stop
goal = tf.constant([0.9])

with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    while not coord.should_stop():
        i = i + 1
        input_batch, target_batch = single_read("./sets/processedtrain.csv")
        fdict1 = {input_placeholder:input_batch, target_placeholder:target_batch}
        sess.run([train_step, acc_op, accuracy], feed_dict=fdict1, options=run_options)
        if (i % 100 == 0):
            print("Checking model values")
            #now we run the validation
            sess.run(acc_op, feed_dict=fdict1, options=run_options)
            a, l = sess.run([accuracy, loss], feed_dict=fdict1, options=run_options)
            print("Accuracy is {}, loss is {}".format(a, l))
            g = sess.run(goal, options=run_options)
            if(a > g):
                print("We have reached convergence", flush=True)
                saver.save(sess, "./checkpoints/{}nodes".format(hidden_layer1_nodes))
                print("model saved", flush=True)
                coord.request_stop()



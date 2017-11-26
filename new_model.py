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

tbatch_size = 6414
vbatch_size = 1609
batch_size = 128
hidden_layer1_nodes = 10
hidden_layer2_nodes = 10
hidden_layer3_nodes = 10
hidden_layer4_nodes = 15
hidden_layer5_nodes = 10
target_size = 1
attr_size = 14
v_deteriorations = 0
v_max = 0
patience = 100

training_data = "./sets/wages/alt_adult.datatrain.csv"
testing_data = "./sets/wages/alt_adult.datatest.csv"
validation_data = "./sets/wages/alt_adult.datavalid.csv"
logfile = "./sets/wages/deepnet.log"

#config = tf.ConfigProto(device_count = {'GPU': 1})

v_deteriorations = 0
'''
stopping rule:
if the validation does worse a couple times in a row while the network error stagnates,
then the network should stop training, and save the weights that it has trained.
'''
run_options = tf.RunOptions(timeout_in_ms=10000)

#model params (2 layer network)
probability_placeholder = tf.placeholder(dtype=tf.float32)

input_placeholder = tf.placeholder(shape=[None, attr_size], dtype=tf.float32, name="input")
target_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="target")

# w1 = tf.Variable(tf.random_normal(shape=[attr_size, hidden_layer1_nodes]),name="w1")
# b1 = tf.Variable(tf.random_normal(shape=[1, hidden_layer1_nodes]), name="b1")
#
# w2 = tf.Variable(tf.random_normal(shape=[hidden_layer1_nodes, hidden_layer2_nodes]), name="w2")
# b2 = tf.Variable(tf.random_normal(shape=[1, hidden_layer2_nodes]), name="b2")
#
# w3 = tf.Variable(tf.random_normal(shape=[hidden_layer2_nodes, hidden_layer3_nodes]), name="w3")
# b3 = tf.Variable(tf.random_normal(shape=[1, hidden_layer3_nodes]), name="b3")
#
# w4 = tf.Variable(tf.random_normal(shape=[hidden_layer3_nodes, hidden_layer4_nodes]), name="w4")
# b4 = tf.Variable(tf.random_normal(shape=[1, hidden_layer4_nodes]), name="b4")
#
# w5 = tf.Variable(tf.random_normal(shape=[hidden_layer4_nodes, hidden_layer5_nodes]), name="w5")
# b5 = tf.Variable(tf.random_normal(shape=[1, hidden_layer5_nodes]), name="b5")
#
# w6 = tf.Variable(tf.random_normal(shape=[hidden_layer5_nodes, target_size]), name="w6")
# b6 = tf.Variable(tf.random_normal(shape=[1, target_size]), name="b6")

w1 = tf.Variable(tf.zeros(shape=[attr_size, hidden_layer1_nodes]),name="w1")
b1 = tf.Variable(tf.zeros(shape=[1, hidden_layer1_nodes]), name="b1")

w2 = tf.Variable(tf.zeros(shape=[hidden_layer1_nodes, hidden_layer2_nodes]), name="w2")
b2 = tf.Variable(tf.zeros(shape=[1, hidden_layer2_nodes]), name="b2")

w3 = tf.Variable(tf.zeros(shape=[hidden_layer2_nodes, hidden_layer3_nodes]), name="w3")
b3 = tf.Variable(tf.zeros(shape=[1, hidden_layer3_nodes]), name="b3")

w4 = tf.Variable(tf.zeros(shape=[hidden_layer3_nodes, hidden_layer4_nodes]), name="w4")
b4 = tf.Variable(tf.zeros(shape=[1, hidden_layer4_nodes]), name="b4")

w5 = tf.Variable(tf.zeros(shape=[hidden_layer4_nodes, hidden_layer5_nodes]), name="w5")
b5 = tf.Variable(tf.zeros(shape=[1, hidden_layer5_nodes]), name="b5")

w6 = tf.Variable(tf.zeros(shape=[hidden_layer5_nodes, target_size]), name="w6")
b6 = tf.Variable(tf.zeros(shape=[1, target_size]), name="b6")

# w1 = tf.Variable(tf.zeros(shape=[attr_size, hidden_layer1_nodes]), name="w1")
# b1 = tf.Variable(tf.zeros(shape=[1, hidden_layer1_nodes]), name="b1")
#
# w2 = tf.Variable(tf.zeros(shape=[hidden_layer1_nodes, target_size]), name="w2")
# b2 = tf.Variable(tf.zeros(shape=[1, target_size]), name="b2")

# tf.random_shuffle(input_placeholder)

h1 = tf.nn.relu(tf.matmul(input_placeholder, tf.nn.dropout(w1, keep_prob=probability_placeholder)) + b1)
h2 = tf.nn.relu(tf.matmul(h1, tf.nn.dropout(w2, keep_prob=probability_placeholder)) + b2)
h3 = tf.nn.relu(tf.matmul(h2, tf.nn.dropout(w3, keep_prob=probability_placeholder)) + b3)
h4 = tf.nn.relu(tf.matmul(h3, tf.nn.dropout(w4, keep_prob=probability_placeholder)) + b4)
h5 = tf.nn.relu(tf.matmul(h4, tf.nn.dropout(w5, keep_prob=probability_placeholder)) + b5)
output = tf.sigmoid(tf.matmul(h5, tf.nn.dropout(w6, keep_prob=probability_placeholder)) + b6)

# h1 = tf.sigmoid(tf.matmul(input_placeholder, w1) + b1)
# h2 = tf.sigmoid(tf.matmul(h1, w2) + b2)
# h3 = tf.sigmoid(tf.matmul(h2, w3) + b3)
# h4 = tf.sigmoid(tf.matmul(h3, w4) + b4)
# h5 = tf.sigmoid(tf.matmul(h4, w5) + b5)
# output = tf.matmul(h5, w6) + b6

# h1 = tf.sigmoid(tf.matmul(input_placeholder, w1) + b1)
# output = (tf.matmul(h1, w2) + b2)

# vh1 = tf.sigmoid(tf.matmul(input_placeholder, w1) + b1)
# voutput = (tf.matmul(h1, w2) + b2)

vh1 = tf.nn.relu(tf.add(tf.matmul(input_placeholder, w1), b1))
vh2 = tf.nn.relu(tf.add(tf.matmul(vh1, w2), b2))
vh3 = tf.nn.relu(tf.add(tf.matmul(vh2, w3), b3))
vh4 = tf.nn.relu(tf.add(tf.matmul(vh3, w4), b4))
vh5 = tf.nn.relu(tf.add(tf.matmul(vh4, w5), b5))
voutput = tf.sigmoid(tf.add(tf.matmul(vh5, w6), b6))


loss = tf.reduce_mean(tf.squared_difference(target_placeholder, output))
train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
truncated_output = tf.round(output)
truncated_target = tf.round(target_placeholder)
truncated_voutput = tf.round(voutput)

correct_prediction = tf.equal(truncated_output, truncated_target)
vcorrect_prediction = tf.equal(truncated_voutput, truncated_target)

accuracy, acc_op = tf.metrics.accuracy(labels=truncated_output, predictions=truncated_target)

df = pd.read_csv(training_data, sep=',', header=None)
dfarr = np.array(df)

def read_from_arr(size):
    np.random.shuffle(dfarr)
    return dfarr[0:size, 0:attr_size], np.reshape(dfarr[0:size, attr_size], [size, 1]) #labels, targets


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

def num_correct(bool_arr):
    barr = []
    for item in bool_arr:
        for i in item:
            barr.append(i)
    total = 0
    num_items = len(barr)
    for item in barr:
        if(item):
            total+=1
    return total/num_items

def get_data_set(fqueue):
    return read_from_csv(fqueue)

def single_read(filename, size):
    df = pd.read_csv(filename, sep=',', header=None)
    arr = np.array(df)
    np.random.shuffle(arr)
    return arr[0:size, 0:attr_size], np.reshape(arr[0:size, attr_size-1], [size, 1]) #labels, targets

def single_read_valid(filename):
    df = pd.read_csv(filename, sep=',', header=None)
    arr = np.array(df)
    np.random.shuffle(arr)
    return arr[0:vbatch_size, 0:attr_size], np.reshape(arr[0:vbatch_size, attr_size:], np.shape(arr)[0]) #labels, targets

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


'''
if the model accuracy on v deteriorates for 10 batches in a row, then the model shall
cease training, terminating itself, saving the parameters, and reporting it's final
error rate
'''


with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    while not coord.should_stop():
        # if(i == 0):
        #     we1, we2, we3, we4, we5 = sess.run([w1, w2, w3, w4, w5], options=run_options)
        #     print(we1, we2, we3, we4, we5)

        i = i + 1
        # input_batch, target_batch = single_read(training_data, batch_size)
        input_batch, target_batch = read_from_arr(batch_size)
        fdict1 = {input_placeholder:input_batch,
                  target_placeholder:target_batch,
                  probability_placeholder:[0.3]
                  }
        sess.run([train_step, acc_op], feed_dict=fdict1, options=run_options)
        if (i % 10 == 0):
            print("Checking model values")
            vinput_batch, vtarget_batch = single_read(validation_data, vbatch_size)
            vdict = {input_placeholder:vinput_batch,
                     target_placeholder:vtarget_batch,
                     probability_placeholder:[0.5]
                     }
            #now we run the validation
            #sess.run([acc_op], feed_dict=fdict1, options=run_options)
            if(i % 30 == 0):
                a, l, c, out = sess.run([accuracy, loss, correct_prediction, output], feed_dict=fdict1, options=run_options)
                # print(out)
                # print(input_batch)
                # print(target_batch)
                # print("target is: ", target_batch, "input is: ", input_batch)

            else:
                a, l, c = sess.run([accuracy, loss, correct_prediction], feed_dict=fdict1, options=run_options)
            print("Accuracy is {}%, batch correct: {}%".format(a*100, num_correct(c)*100))
            vc = sess.run([vcorrect_prediction], feed_dict=vdict, options=run_options)
            nvc = num_correct(vc)
            if(nvc > v_max):
                v_max = nvc
                v_deteriorations = 0
                print("model improving!")
            else:
                v_deteriorations+=1
            if(v_deteriorations == patience):
                print("We have reached convergence", flush=True)
                #run the training data through the network, check the error rate
                #append to a log file
                print("Checking the test error...")
                tinput_batch, ttarget_batch = single_read(testing_data, tbatch_size )
                tdict = {input_placeholder:tinput_batch,
                         target_placeholder:ttarget_batch,
                         probability_placeholder:[0.5]}
                tc, out = sess.run([vcorrect_prediction, output], feed_dict=tdict, options=run_options)
                print(out)
                tpercentage = num_correct(tc)
                #save the data to a log file and report back
                print("With the {} layers, the accuracy on the test set is: {}".format(hidden_layer1_nodes, tpercentage))
                with open(logfile, "a") as log:
                    log.write("{},{}\n".format(hidden_layer1_nodes, tpercentage))
                saver.save(sess, "./checkpoints/{}nodes".format(hidden_layer1_nodes))
                print("model saved", flush=True)
                coord.request_stop()



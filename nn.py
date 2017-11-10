#!/usr/local/bin/python3


import tensorflow as tf
import numpy as np
import sys
import csv



def load_data(filename):
    attrs=[]
    print("now loading: {}".format(filename))

    with open(filename) as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        i=0
        for row in data:
            attrs.append(row)
            i=i+1
    return attrs

def tf_to_list(tflist):
    return tf.convert_to_tensor(tflist, tf.float32)




class Model:
    '''A class to hold and manage Layers'''
    def __init__(self):
        self._error = None
        self._dataset = None
        self._optimize = None


#it's important that the graph interface class have so over what the dimensions
#of layers can be with regard to each other
#Layer 1 with I = 2, and N = 4 necessitates that
#Layer 2 will have I = 4, and N2 (could be anything)
#Layer 3 will have I = N2 and N3 (see the pattern?)



class Layer:
    '''abstract class to be derived from for more specialized Layers'''
    def __init__(self, inp_size, num_nodes, t_func, pos):
        self.numberNodes = num_nodes
        self.input_size= inp_size
        self.tfunc = t_func
        self.position = pos
        self._input = tf.placeholder(dtype=tf.float32, shape=[1, self.input_size])
        self._bias = tf.Variable(tf.random_normal(shape=[1, self.numberNodes]))
        self._weights = tf.Variable(tf.random_normal(shape=[self.input_size, self.numberNodes]))
        self._output = tf.placeholder(self.tfunc(tf.matmul(self._input, self._weights) + self._bias))





class GraphInterface:
    '''A class more specifically to manage the connections between layers'''
    def __init__(self, num_l, t_size, i_size, sizes):
        self.num_layers = num_l
        self.target_size = t_size
        self.input_size = i_size
        self.sizes = sizes #[[I, N1],[N1, N2],[N2, N3],.... [N(n-1), Target_size]]
        self._layers=[]
        self.attr_input

        for i in range(self.num_layers):
            self._layers.append(Layer(sizes[i][0], sizes[i][1], tf.sigmoid, i))




























def main(argc, argv):
    layer_size=argv[1] #how nodes in the hidden layer
    learning_rate=0.05 #this should stay the same across experiments
    target_size=0
    #sess=tf.Session()  #create a tensorflow session

    #now I need to read the input data and have it loaded into tensors
    attr=load_data(argv[2])
    #define my graph
    #1000025,5,1,1,1,2,1,3,1,1,2
    x=tf.placeholder(tf.float32, shape=[None, 10], name="input")
    t=tf.placeholder(tf.float32, shape=[0], name="target")
    W = tf.Variable(tf.random_normal([10, layer_size], stddev=0.35),
                      name="weights")
    b= tf.Variable(tf.float32, shape=[layer_size], name="bias")
    h = tf.nn.sigmoid(tf.matmul(x, W) + b) #of size
    W2 = tf.Variable(tf.random_normal([]))
    error=tf.sub()

main(len(sys.argv), sys.argv)

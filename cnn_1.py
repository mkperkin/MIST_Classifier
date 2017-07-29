"""
CONVOLUTIONAL NEURAL NETWORK ATTEMPT #1

COGS 118B 

GROUP RAG TAG

Code primarily from
https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/index.html#deep-mnist-for-experts
with minor adjustments to fit our dataset

this file depends on getdata.py
"""

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse



"""Functions for downloading and reading MNIST data."""

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import getdata as p

# Import data

import tensorflow as tf

FLAGS = None


def main(_):
  train, test, validation = p.getdata()
  #mnist = loadlibs.readdata(FLAGS.data_dir, one_hot=True)

  # Create the model
 # x = tf.placeholder(tf.float32, [None, 784])
  #W = tf.Variable(tf.zeros([784, 10]))
  #b = tf.Variable(tf.zeros([10]))

  sess = tf.InteractiveSession()
  x = tf.placeholder(tf.float32, [None, 16384])
  W = tf.Variable(tf.zeros([16384, 2]))
  b = tf.Variable(tf.zeros([2]))
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])
	
  y = tf.matmul(x, W) + b
   
  #sess.run(tf.global_variables_initializer())
  #y = tf.nn.softmax(tf.matmul(x,W) + b)

  #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # Train
  #for _ in range(4000):
   # batch_xs, batch_ys = train.next_batch(100)
    #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  #print(sess.run(accuracy, feed_dict={x: test.images,
                                   #   y_: test.labels}))

  def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

  def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

  def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME')
	
  def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


  #set up convolution 

  W_conv1 = weight_variable([7,7,1,32])
  #W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])

  x_im = tf.reshape(x,[-1,128,128,1])

  h_conv1 = tf.nn.relu(conv2d(x_im, W_conv1) + b_conv1)
  print("after conv of layer1", h_conv1.get_shape())
  h_pool1 = max_pool_2x2(h_conv1)
  print("after pool oflayer 1", h_pool1.get_shape())


  W_conv2 = weight_variable([7,7,32,64])
  #W_conv2 = weight_variable([5, 5, 32, 64])
  
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
  
  print("after conv of layer2", h_conv2.get_shape())
  
  h_pool2 = max_pool_2x2(h_conv2)

  print("after pool oflayer 2", h_pool2.get_shape())
  W_conv3 = weight_variable([7,7,64,128])
  b_conv3 = bias_variable([128])

  h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)
  h_pool3 = max_pool_2x2(h_conv3)

  print("after conv of layer3", h_conv3.get_shape())

  print("after pool oflayer 3", h_pool3.get_shape())
  W_conv4 = weight_variable([7,7,128,256])
  b_conv4 = bias_variable([256])

  h_conv4 = tf.nn.relu(conv2d(h_pool3,W_conv4) + b_conv4)
  h_pool4 = max_pool_2x2(h_conv4)

  print("after conv of layer4", h_conv4.get_shape())

  print("after pool oflayer 4", h_pool4.get_shape())

  #W_fc1 = weight_variable([7 * 7 * 64, 1024])
  #b_fc1 = bias_variable([1024])

  #h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  #h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  #fully connected layer

  W_fc1 = weight_variable([8*8*256,1024])
  b_fc1 = bias_variable([1024])

  h_pool4_flat = tf.reshape(h_pool4, [-1, 8*8*256])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

  print("fully connected, relu of last layer", h_fc1.get_shape())
  #dropout 

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  #readout layer

  W_fc2 = weight_variable([1024,2])
  b_fc2 = bias_variable([2])

  v = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
   
  print("fully connected2, relu of last layer", v.get_shape())
  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  print(y_conv)
  print(y_)
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

  #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess = tf.InteractiveSession()
  tf.initialize_all_variables().run()
  for i in range(6):
    batch = train.next_batch(1)
    print("hi")
    if i%1 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x:batch[0], y_: batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


  print(test.images.shape)
  print(test.labels)
  print("test accuracy %g"%accuracy.eval(feed_dict={
      x: test.images, y_: test.labels, keep_prob: 1.0}))





if __name__ == '__main__':
  tf.app.run()

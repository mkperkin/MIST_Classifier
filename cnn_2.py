"""
CONVOLUTIONAL NEURAL NETWORK ATTEMPT #2 
WORKING MODEL

COGS 118B 

GROUP RAG TAG

Code primarily from (https://ireneli.eu/2016/03/13/tensorflow-04-implement-a-lenet-5-like-nn-to-classify-notmnist-images/)
Only minor changes to adjust for custom dataset

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

import sys
from PIL import Image
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from tensorflow.python.framework import dtypes


IMAGE_SIZE = 128
PIXEL_DEPTH = 255.0



def accuracy(predictions, labels):
  return (100.0 * numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) / predictions.shape[0])




def load_letter(folder, min_num_images):

	image_files = os.listdir(folder)
	dataset = numpy.ndarray(shape=(len(image_files), IMAGE_SIZE, IMAGE_SIZE),
							dtype = numpy.float32)
	print(folder)
	num_images = 0
	for image in image_files:
		img = Image.open(os.path.join(folder, image)).convert('L') 
		#img.show()
		image_file = os.path.join(folder, image)
		try:
			arr = numpy.array(img)
			image_data = numpy.invert(arr)

			numpy.set_printoptions(threshold=numpy.nan)
			print(image_data.shape)
			
			if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE):
				raise Exception('Unexpected image shape: %s' % str(image_data.shape))
			dataset[num_images, :, :] = image_data
			num_images = num_images + 1
		except IOError as e:
			print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

	dataset = dataset[0:num_images, :, :] 
	if num_images < min_num_images:
		raise Exception('Many few images than excpected')
	
	print('Full dataset tensor:', dataset.shape)
	
	return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  print(data_folders)
  for folder in data_folders:
    print(folder)
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(os.path.join(data_folders, folder), min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names



def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = numpy.ndarray((nb_rows, img_size, img_size), dtype=numpy.float32)
    labels = numpy.ndarray(nb_rows, dtype=numpy.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)

  print(num_classes)

  valid_dataset, valid_labels = make_arrays(valid_size, IMAGE_SIZE)
  train_dataset, train_labels = make_arrays(train_size, IMAGE_SIZE)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes

  print(vsize_per_class)
  print(tsize_per_class)

  index = 0;

  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        numpy.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
        

        train_letter = letter_set[:tsize_per_class, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
	index = index + 1
  	print(end_t) 
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels



def randomize(dataset, labels):
  permutation = numpy.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels



def reformat(dataset, labels):
  #WHEN YOUR LABELS CHANGE.... CHANGE ME!!!
  image_size = 128
  num_labels = 2
  num_channels = 1
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(numpy.float32)
  labels = (numpy.arange(num_labels) == labels[:,None]).astype(numpy.float32)
  return dataset, labels



def main(_):
	#CHANGE ME!!!!!
	batch_size = 1
	patch_size = 7
	depth = 16
	num_hidden = 64
	num_hidden2 = 32
	image_size = 128
	num_channels = 1
	num_labels = 2

	graph = tf.Graph()
	
	#THIS IS WHERE YOU PUT THE DIRECTORIES TO WHERE YOU IMAGES ARE
	dataset_folder_train = ['/Users/smokey/train/train_a', '/Users/smokey/train/train_b']
	dataset_folder_test = ['/Users/smokey/test/test_a', '/Users/smokey/test/test_b']
	
	
	#CHANGE ME
	train_datasets = maybe_pickle(dataset_folder_train, 3)
	test_datasets = maybe_pickle(dataset_folder_test, 3)

	#CHANGE ME
	size = 4
	train_size = 6
	valid_size = size
	test_size = size
	
	
	valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
	  train_datasets, train_size, valid_size)
	_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size, 0)

	
	train_dataset, train_labels = reformat(train_dataset, train_labels)
	valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
	test_dataset, test_labels = reformat(test_dataset, test_labels)

	train_dataset, train_labels = randomize(train_dataset, train_labels)
	test_dataset, test_labels = randomize(test_dataset, test_labels)
	valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
	
	print('Test set', test_dataset.shape, test_labels.shape)
	print('Training:', train_dataset.shape, train_labels.shape)
	print('Validation:', valid_dataset.shape, valid_labels.shape)
	print('Testing:', test_dataset.shape, test_labels.shape)
	
   	with graph.as_default():
		# Input data.
		tf_train_dataset = tf.placeholder(
			tf.float32, shape=[batch_size, image_size, image_size, num_channels])
		tf_train_labels = tf.placeholder(tf.float32, shape=[batch_size, num_labels])

		tf_valid_dataset = tf.constant(valid_dataset)
		print(tf_valid_dataset.get_shape())
		tf_test_dataset = tf.constant(test_dataset)
	  
		

		# Variables.
		
		print("layer1_weights= [", patch_size, patch_size, num_channels, depth, sep=" ")
		layer1_weights = tf.Variable(tf.truncated_normal(shape=
			[patch_size, patch_size, num_channels, depth], stddev=0.1))
		
		print("layer1_biases ", depth)
		layer1_biases = tf.Variable(tf.zeros([depth]))

		# dropout
		keep_prob = tf.placeholder("float")

		print("layer2_weights= [", patch_size, patch_size, depth, depth, sep=" ")
		layer2_weights = tf.Variable(tf.truncated_normal(shape=
				[patch_size, patch_size, depth, depth], stddev=0.1))
		
		print("layer2_biases ", depth)
		layer1_biases = tf.Variable(tf.zeros([depth]))
		layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

	
		poop = image_size // 4 * image_size // 4 * depth
		print("layer3_weights= [ 32 * 32 * depth, num_hidden] ",poop , num_hidden, sep=" ")
		layer3_weights = tf.Variable(tf.truncated_normal(
			[image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
		
		print("layer3_biases ", depth)
		layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
	  
		print("layer4_weights= [", num_hidden , num_hidden2, sep=" ")
		layer4_weights = tf.Variable(tf.truncated_normal(
			 [num_hidden, num_hidden2], stddev=0.1))
		
		print("layer4_biases ", num_hidden2)
		layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))

		print("layer5_weights= [", num_hidden2, num_labels, sep=" ")
		layer5_weights = tf.Variable(tf.truncated_normal(
			[num_hidden2, num_labels], stddev=0.1))
	      	
		#for the "readout layer"
		print("layer5_biases ", num_labels)
		layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))



	 # Model.
		def model(data):  
		    #     The convolutional model above uses convolutions with stride 2 to reduce
		    # the dimensionality. Replace the strides by a max pooling operation (nn.max_pool())
		    # of stride 2 and kernel size 2.
		    # I will use 1 x 1 convention then max pooling.


		    #   layer1
		    # layer1: Conv1 layer
		    # patch 5 * 5 , input_channel 1, depth 16 
		    # A cube of [28,28,16]


		    conv = tf.nn.conv2d(data, layer1_weights, strides=[1, 1, 1, 1], padding='SAME')
		    hidden = tf.nn.relu(conv + layer1_biases)
		    print("layer1 hidden dimensions after convolution", hidden.get_shape())
				    
		    #   pooling
		    pool1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
					 padding='SAME', name='pool1')
		    
		    
		    print("layer1 hidden dimensions after pool", pool1.get_shape())
		    
		    norm1 = tf.nn.lrn(pool1, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
				    name='norm1')
		    
		    print("layer1 hidden dimensions after norm", norm1.get_shape())
		   
		    

		    #   layer2
		    conv = tf.nn.conv2d(norm1, layer2_weights, [1, 1, 1, 1], padding='SAME')
		    hidden = tf.nn.relu(conv + layer2_biases)
		    
		    print("layer2 hidden dimensions after convolution... convolution of previous normalization", hidden.get_shape())
		   
		    
		    #   pooling2
		    pool2 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
					 padding='SAME', name='pool1')
		    
		    
		    print("layer2 hidden dimensions after pool", pool2.get_shape())
		    norm2 = tf.nn.lrn(pool2, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
				    name='norm1')
		    
		    #   norm2, alpha=alpha=0.001 / 100.0;  Test accuracy: 90.3% 
		    #   norm2, alpha=alpha=0.1;  Test accuracy: 90.4% 
		    
		    #   layer3
		    conv = tf.nn.conv2d(norm2, layer2_weights, [1, 1, 1, 1], padding='SAME')
		    hidden = tf.nn.relu(conv + layer2_biases)

		    
		    print("layer2 second conv hidden dimensions after convolution....convollution of norm2", hidden.get_shape())
		    
		    
		    shape = hidden.get_shape().as_list()
		    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

		    #   RELU
		    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
		    
		    
		    print("layer3 weights and biasees relu of HIDDEN fully connected layer ", hidden.get_shape())
		    
		    hidden = tf.matmul(hidden, layer4_weights) + layer4_biases
		   
		    print("layer4 weights and biasees relu of HIDDEN fully connected layer ", hidden.get_shape())
		    # #   add a dropout
		    #     hidden = tf.nn.dropout(hidden, keep_prob)
		    
		    result = tf.matmul(hidden, layer5_weights) + layer5_biases

		    print("layer5 weights and biasees relu of fully connected layer FINAL OUTPUT LAYER", result.get_shape())
		    return result

		 
		# Training computation.
		logits = model(tf_train_dataset)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	    
		# Learning rate decay
		global_step = tf.Variable(0, trainable=False)
		starter_learning_rate = 0.001
		learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
						   100000, 0.96, staircase=True)

	  
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	  
		# Predictions for the training, validation, and test data.
		train_prediction = tf.nn.softmax(logits)
		valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
		test_prediction = tf.nn.softmax(model(tf_test_dataset))


		num_steps = 6

		with tf.Session(graph=graph) as session:
			tf.initialize_all_variables().run()
			print('Initialized')
			for step in range(num_steps):
				offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
				batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
				batch_labels = train_labels[offset:(offset + batch_size), :]
				feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,keep_prob:1.0}
	    
				_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
				if (step % 50 == 0):
					print('Minibatch loss at step %d: %f' % (step, l))
					print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
					print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(session=session,feed_dict={keep_prob:0.5}), valid_labels))
		
		
			print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(session=session,feed_dict={keep_prob:1.0}), test_labels))



if __name__ == '__main__':
  tf.app.run()

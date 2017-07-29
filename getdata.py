"""
COGS 118b

This is a helper file to be used with the files nn.py and cnn_1.py
Its purpose is to load in the MIST dataset and prepare the data to be used
with Tensorflow to create a convolutional neural network 

Code is primarily from
https://ireneli.eu/2016/03/13/tensorflow-04-implement-a-lenet-5-like-nn-to-classify-notmnist-images/
with minor adjustments to fit our data
"""
from __future__ import print_function
import numpy
import os
import sys
import tarfile
from PIL import Image
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base


IMAGE_SIZE = 128
PIXEL_DEPTH = 255.0

class DataSet(object):

  def __init__(self,
               images,
               labels,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
      assert images.shape[3] == 1
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    
    
    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      
    if dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
    
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0


  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed



  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]





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
			#flat_arr = arr.ravel()
			#tmp_vector = numpy.matrix(flat_arr)

			numpy.set_printoptions(threshold=numpy.nan)
			print(image_data.shape)
			#image_data = ndimage.imread(image_file).astype(float)
			#print(image_data)
			"""image_data = (ndimage.imread(image_file).astype(float) -
					PIXEL_DEPTH / 2) / PIXEL_DEPTH"""
			
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
	#print("in merge_datasets printing the letter set shape")
	#print(letter_set.shape)
	#numpy.set_printoptions(threshold=numpy.nan)
        #print(letter_set)
	# let's shuffle the letters to have random validation and training set
        numpy.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
        

        train_letter = letter_set[:tsize_per_class, :, :]
        #train_letter = letter_set[vsize_per_class:end_l, :, :]
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
  image_size = 128
  num_labels = 2
  num_channels = 1
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(numpy.float32)
  labels = (numpy.arange(num_labels) == labels[:,None]).astype(numpy.float32)
  return dataset, labels

#def main():
def getdata():
	#folder for the lower case 'a'

	#dataset_folder_train = ['/Users/smokey/train_ab/train_a', '/Users/smokey/train_ab/train_b',  '/Users/smokey/train_ab/train_c',  '/Users/smokey/train_ab/train_d']
	#dataset_folder_test = ['/Users/smokey/test_ab/test_a', '/Users/smokey/test_ab/test_b', '/Users/smokey/test_ab/test_c',  '/Users/smokey/test_ab/test_d']
	
	dataset_folder_train = ['/Users/smokey/train/train_a', '/Users/smokey/train/train_b']
	dataset_folder_test = ['/Users/smokey/test/test_a', '/Users/smokey/test/test_b']
	train_datasets = maybe_pickle(dataset_folder_train, 3)
	test_datasets = maybe_pickle(dataset_folder_test, 3)

	size = 4
	#even af
	train_size = 6
	valid_size = size
	test_size = size
	
	
	valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
	  train_datasets, train_size, valid_size)
	_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size, 0)


	#numpy.set_printoptions(threshold=numpy.nan)
	#print(train_dataset)

	"""new_train_lab = numpy.zeros((6,2))

	row = 0
	for x in numpy.nditer(train_labels):
		new_train_lab[row, x] = 1
		row = row + 1
	train_labels = new_train_lab;	
	

	new_test_lab = numpy.zeros((size,2))

	row = 0
	for y in numpy.nditer(test_labels):
		new_test_lab[row, y] = 1
		row = row + 1
	
	test_labels = new_test_lab;	
	

	new_valid_lab = numpy.zeros((size,2))
	row = 0
	for z in numpy.nditer(valid_labels):
		new_valid_lab[row, z] = 1
		row = row + 1

	valid_labels = new_valid_lab"""
	
	
	train_dataset, train_labels = reformat(train_dataset, train_labels)
	valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
	test_dataset, test_labels = reformat(test_dataset, test_labels)

	train_dataset, train_labels = randomize(train_dataset, train_labels)
	test_dataset, test_labels = randomize(test_dataset, test_labels)
	valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
	
	print(valid_labels)
	print('Test set', test_dataset.shape, test_labels.shape)

	print('Training:', train_dataset.shape, train_labels.shape)
	print('Validation:', valid_dataset.shape, valid_labels.shape)
	print('Testing:', test_dataset.shape, test_labels.shape)

	train = DataSet(train_dataset, 
			train_labels, 
			dtype=dtypes.float32, 
			reshape=True);

	test = DataSet(test_dataset, 
			test_labels, 
			dtype=dtypes.float32, 
			reshape=True);
	
	validation = DataSet(valid_dataset, 
			    valid_labels, 
			    dtype=dtypes.float32, 
			    reshape=True)

	#numpy.set_printoptions(threshold=numpy.nan)
        #print(train.images)
	return train, test, validation 


#if __name__ == '__main__':
#	main()

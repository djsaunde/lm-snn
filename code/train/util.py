'''
Supporting functions for use in training scripts.
'''

import cPickle as p
import numpy as np
import os
from struct import unpack

top_level_path = os.path.join('..', '..')
MNIST_data_path = os.path.join(top_level_path, 'data')
CIFAR10_data_path = os.path.join(top_level_path, 'data', 'cifar-10-batches-py')


def get_labeled_data(pickle_name, train=True, reduced_dataset=False, classes=range(10), examples_per_class=100):
	'''
	Read input-vector (image) and target class (label, 0-9) and return it as 
	a list of tuples.
	'''
	if reduced_dataset:
		pickle_name = '_'.join([pickle_name, 'reduced', '_'.join([ str(class_) for class_ in classes ]), str(examples_per_class)])

	if os.path.isfile('%s.pickle' % pickle_name):
		data = p.load(open('%s.pickle' % pickle_name))
	else:
		# Open the images with gzip in read binary mode
		if train:
			images = open(os.path.join(MNIST_data_path, 'train-images-idx3-ubyte'), 'rb')
			labels = open(os.path.join(MNIST_data_path, 'train-labels-idx1-ubyte'), 'rb')
		else:
			images = open(os.path.join(MNIST_data_path, 't10k-images-idx3-ubyte'), 'rb')
			labels = open(os.path.join(MNIST_data_path, 't10k-labels-idx1-ubyte'), 'rb')

		# Get metadata for images
		images.read(4)  # skip the magic_number
		number_of_images = unpack('>I', images.read(4))[0]
		rows = unpack('>I', images.read(4))[0]
		cols = unpack('>I', images.read(4))[0]

		# Get metadata for labels
		labels.read(4)  # skip the magic_number
		N = unpack('>I', labels.read(4))[0]

		if number_of_images != N:
			raise Exception('number of labels did not match the number of images')

		# Get the data
		print '...Loading MNIST data from disk.'
		print '\n'

		x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
		y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array

		for i in xrange(N):
			if i % 1000 == 0:
				print 'Progress :', i, '/', N
			x[i] = [[unpack('>B', images.read(1))[0] for unused_col in xrange(cols)] for unused_row in xrange(rows) ]
			y[i] = unpack('>B', labels.read(1))[0]

		print 'Progress :', N, '/', N, '\n'

		if reduced_dataset:
			reduced_x = np.zeros((examples_per_class * len(classes), rows, cols), dtype=np.uint8)
			for idx, class_index in enumerate(classes):
				current = examples_per_class * idx
				for example_index, example in enumerate(x):
					if y[example_index] == class_index:
						reduced_x[current, :, :] = x[example_index, :, :]
						current += 1
						if current == examples_per_class * (idx + 1):
							break

			reduced_y = np.array([ label // examples_per_class for label in xrange(examples_per_class * len(classes)) ],
															dtype=np.uint8).reshape((examples_per_class * len(classes), 1))
	
			# Randomize order of data examples
			rng_state = np.random.get_state()
			np.random.shuffle(reduced_x)
			np.random.set_state(rng_state)
			np.random.shuffle(reduced_y)

			# Set data to reduced data
			x, y = reduced_x, reduced_y

		data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}

		p.dump(data, open("%s.pickle" % pickle_name, "wb"))

	return data


def get_labeled_CIFAR10_data(train=True, single_channel=True):
	data = {}
	if train:
		files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
	else:
		files = ['test_batch']
	
	for idx, file in enumerate(files):
		with open(os.path.join(CIFAR10_data_path, file), 'rb') as open_file:
			if idx == 0:
				data = p.load(open_file)

				del data['batch_label']
				del data['filenames']
			else:
				temp_data = p.load(open_file)

				data['data'] = np.vstack([data['data'], temp_data['data']])
				data['labels'] = np.hstack([data['labels'], temp_data['labels']])

	if single_channel:
		data['data'] = np.reshape(data['data'], (data['data'].shape[0], 3, 1024))
		data['data'] = np.mean(data['data'], axis=1)
		data['data'] = data['data'].reshape((data['data'].shape[0], 32, 32))
	else:
		data['data'] = data['data'].reshape((data['data'].shape[0], 3, 32, 32))

	return data


def is_lattice_connection(sqrt, i, j, lattice_structure):
	'''
	Boolean method which checks if two indices in a network correspond to neighboring nodes in a 4-, 8-, or all-lattice.

	n_e: Square root of the number of nodes in population
	i: First neuron's index
	k: Second neuron's index
	lattice_structure: Connectivity pattern between connected patches
	'''
	if lattice_structure == 'none':
		return False
	if lattice_structure == '4':
		return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j
	if lattice_structure == '8':
		return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j \
											or i + sqrt == j + 1 and j % sqrt != 0 or i + sqrt == j - 1 and i % sqrt != 0 \
											or i - sqrt == j + 1 and i % sqrt != 0 or i - sqrt == j - 1 and j % sqrt != 0
	if lattice_structure == 'all':
		return True


def save_connections(weights_dir, connections, input_connections, ending, suffix):
	'''
	Save all synaptic connection parameters out to disk.
	'''

	# merge two dictionaries of connections into one
	connections.update(input_connections)

	# save out each connection's parameters to disk

	for connection_name in connections.keys():		
		# get parameters of this connection
		connection_matrix = connections[connection_name][:].todense()
		# save it out to disk
		np.save(os.path.join(weights_dir, connection_name + '_' + ending + '_' + str(suffix)), connection_matrix)




def save_theta(weights_dir, populations, neuron_groups, ending, suffix):

	'''
	Save the adaptive threshold parameters out to disk.
	'''

	# iterate over population for which to save theta parameters
	for population in populations:
		# save out the theta parameters to file
		np.save(os.path.join(weights_dir, 'theta_' + population + '_' + ending + '_' + str(suffix)), neuron_groups[population + 'e'].theta)


def save_assignments(weights_dir, assignments, ending, suffix):
	'''
	Save neuron class labels out to disk.
	'''

	# save the labels assigned to excitatory neurons out to disk
	np.save(os.path.join(weights_dir, '_'.join(['assignments', ending, str(suffix)])), assignments)


def save_accumulated_rates(weights_dir, accumulated_rates, ending, suffix):
	'''
	Save neuron class labels out to disk.
	'''

	# save the labels assigned to excitatory neurons out to disk
	np.save(os.path.join(weights_dir, '_'.join(['accumulated_rates', ending, str(suffix)])), assignments)
import numpy as np
import matplotlib.cm as cmap
import time, os.path, scipy, math, sys, timeit
import cPickle as p
import brian_no_units
import brian as b

from scipy.sparse import coo_matrix
from struct import unpack
from brian import *

fig_num = 0
wmax_ee = 1.0


def get_matrix_from_file(file_name, n_src, n_tgt):
	'''
	Given the name of a file pointing to a .npy ndarray object, load it into
	'weight_matrix' and return it
	'''

	# load the stored ndarray into 'readout', instantiate 'weight_matrix' as
	# correctly-shaped zeros matrix
	readout = np.load(file_name)
	weight_matrix = np.zeros((n_src, n_tgt))

	# read the 'readout' ndarray values into weight_matrix by (row, column) indices
	weight_matrix[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]

	# return the weight matrix read from file
	return weight_matrix


def get_2d_input_weights():
	'''
	Get the weights from the input to excitatory layer and reshape it to be two
	dimensional and square.
	'''
	# rearranged_weights = np.zeros((conv_features_sqrt * conv_size * n_e_sqrt, conv_features_sqrt * conv_size * n_e_sqrt))
	# connection = input_connections['XeAe'][:]

	# # for each convolution feature
	# for feature in xrange(conv_features):
	# 	# for each excitatory neuron in this convolution feature
	# 	for n in xrange(n_e):
	# 		temp = connection[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)].todense()

	# 		# print ((feature // conv_features_sqrt) * conv_size * n_e_sqrt) + ((n // n_e_sqrt) * conv_size), ((feature // conv_features_sqrt) * conv_size * n_e_sqrt) + ((n // n_e_sqrt) * conv_size) + conv_size, ((feature % conv_features_sqrt) * conv_size * n_e_sqrt) + ((n % n_e_sqrt) * (conv_size)), ((feature % conv_features_sqrt) * conv_size * n_e_sqrt) + ((n % n_e_sqrt) * (conv_size)) + conv_size

	# 		rearranged_weights[ ((feature % conv_features_sqrt) * conv_size * n_e_sqrt) + ((n % n_e_sqrt) * conv_size) : ((feature % conv_features_sqrt) * conv_size * n_e_sqrt) + ((n % n_e_sqrt) * conv_size) + conv_size, ((feature // conv_features_sqrt) * conv_size * n_e_sqrt) + ((n // n_e_sqrt) * (conv_size)) : ((feature // conv_features_sqrt) * conv_size * n_e_sqrt) + ((n // n_e_sqrt) * (conv_size)) + conv_size ] = temp[convolution_locations[n]].reshape((conv_size, conv_size))

	# # return the rearranged weights to display to the user
	# return rearranged_weights.T
	
	rearranged_weights = np.zeros((conv_features * conv_size, conv_size * n_e))

	# counts number of input -> excitatory weights displayed so far
	connection = weight_matrix

	# for each excitatory neuron in this convolution feature
	for n in xrange(n_e):
		# for each convolution feature
		for feature in xrange(conv_features):
			temp = connection[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)]
			rearranged_weights[ feature * conv_size : (feature + 1) * conv_size, n * conv_size : (n + 1) * conv_size ] = \
																		temp[convolution_locations[n]].reshape((conv_size, conv_size))

	# return the rearranged weights to display to the user
	return rearranged_weights.T


def plot_2d_input_weights():
	'''
	Plot the weights from input to excitatory layer to view during training.
	'''
	weights = get_2d_input_weights()
	fig = b.figure(fig_num, figsize=(18, 18))
	im = b.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
	b.colorbar(im)
	b.title('Reshaped input -> convolution weights')
	b.xticks(xrange(conv_size, conv_size * (conv_features + 1), conv_size), xrange(1, conv_features + 1))
	b.yticks(xrange(conv_size, conv_size * (n_e + 1), conv_size), xrange(1, n_e + 1))
	fig.canvas.draw()
	return im, fig


weight_dir = '../weights/conv_patch_connectivity_weights/'

print '\n'
print '\n'.join([ str(idx) + ' | ' + file_name for idx, file_name in enumerate([ file_name for file_name in sorted(os.listdir(weight_dir)) if 'XeAe' in file_name ]) ])
print '\n'

to_plot = raw_input('Enter the index of the file from above which you\'d like to plot: ')
if to_plot == '':
	file_name = [ file_name for file_name in sorted(os.listdir(weight_dir)) if 'XeAe' in file_name ][0]
else:
	file_name = [ file_name for file_name in sorted(os.listdir(weight_dir)) if 'XeAe' in file_name ][int(to_plot)]

# number of inputs to the network
n_input = 784
n_input_sqrt = int(math.sqrt(n_input))

conv_size = int(file_name.split('_')[2])
conv_stride = int(file_name.split('_')[3])
conv_features = int(file_name.split('_')[4])

# number of excitatory neurons (number output from convolutional layer)
n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
n_e_total = n_e * conv_features
n_e_sqrt = int(math.sqrt(n_e))

# number of inhibitory neurons (number of convolutational features (for now))
n_i = n_e

conv_features_sqrt = int(math.ceil(math.sqrt(conv_features)))

print '\n'

# creating convolution locations inside the input image
convolution_locations = {}
for n in xrange(n_e):
	convolution_locations[n] = [ ((n % n_e_sqrt) * conv_stride + (n // n_e_sqrt) * n_input_sqrt * conv_stride) + (x * n_input_sqrt) + y for y in xrange(conv_size) for x in xrange(conv_size) ]

weight_matrix = get_matrix_from_file(weight_dir + file_name, n_input, conv_features * n_e)

plot_2d_input_weights()

b.show()

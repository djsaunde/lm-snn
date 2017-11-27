import numpy as np
import matplotlib.cm as cmap
import time, os, scipy, math, sys, timeit
import cPickle as p
import brian_no_units
import brian as b
import argparse
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import coo_matrix
from struct import unpack
from brian import *


parser = argparse.ArgumentParser()
parser.add_argument('--directory', default='best')
args = parser.parse_args()

directory = args.directory

fig_num = 0

top_level_path = os.path.join('..', '..')
model_name = 'csnn_growing_inhibition'
weight_dir = os.path.join(top_level_path, 'weights', model_name, directory)
plots_dir = os.path.join(top_level_path, 'plots', model_name)

if not os.path.isdir(plots_dir):
	os.makedirs(plots_dir)


def normalize_weights():
	'''
	Squash the input -> excitatory weights to sum to a prespecified number.
	'''
	for feature in xrange(conv_features):
		feature_connection = weight_matrix[:, feature * n_e : (feature + 1) * n_e]
		column_sums = np.sum(feature_connection, axis=0)
		column_factors = weight['ee_input'] / column_sums

		for n in xrange(n_e):
			dense_weights = weight_matrix[:, feature * n_e + n]
			dense_weights[convolution_locations[n]] *= column_factors[n]
			weight_matrix[:, feature * n_e + n] = dense_weights


def get_2d_input_weights():
	'''
	Get the weights from the input to excitatory layer and reshape it to be two
	dimensional and square.
	'''
	# specify the desired shape of the reshaped input -> excitatory weights
	rearranged_weights = np.zeros((conv_features * conv_size, conv_size * n_e))

	# get the input -> excitatory synaptic weights
	connection = weight_matrix

	if sort_euclidean:
		# for each excitatory neuron in this convolution feature
		euclid_dists = np.zeros((n_e, conv_features))
		temps = np.zeros((n_e, conv_features, n_input))
		for n in xrange(n_e):
			# for each convolution feature
			for feature in xrange(conv_features):
				temp = connection[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)]
				if feature == 0:
					if n == 0:
						euclid_dists[n, feature] = 0.0
					else:
						euclid_dists[n, feature] = np.linalg.norm(temps[0, 0, convolution_locations[n]] - temp[convolution_locations[n]])
				else:
					euclid_dists[n, feature] = np.linalg.norm(temps[n, 0, convolution_locations[n]] - temp[convolution_locations[n]])

				temps[n, feature, :] = temp.ravel()

			for idx, feature in enumerate(np.argsort(euclid_dists[n])):
				temp = temps[n, feature]
				rearranged_weights[ idx * conv_size : (idx + 1) * conv_size, n * conv_size : (n + 1) * conv_size ] = \
																temp[convolution_locations[n]].reshape((conv_size, conv_size))

	else:
		for n in xrange(n_e):
			for feature in xrange(conv_features):
				temp = connection[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)]
				rearranged_weights[ feature * conv_size : (feature + 1) * conv_size, n * conv_size : (n + 1) * conv_size ] = \
																temp[convolution_locations[n]].reshape((conv_size, conv_size))

	# return the rearranged weights to display to the user
	if n_e == 1:
		ceil_sqrt = int(math.ceil(math.sqrt(conv_features)))
		square_weights = np.zeros((28 * ceil_sqrt, 28 * ceil_sqrt))
		for n in xrange(conv_features):
			square_weights[(n // ceil_sqrt) * 28 : ((n // ceil_sqrt) + 1) * 28, (n % ceil_sqrt) * 28 : ((n % ceil_sqrt) + 1) * 28] = rearranged_weights[n * 28 : (n + 1) * 28, :]

		return square_weights.T
	else:
		return rearranged_weights.T


def plot_2d_input_weights(title):
	'''
	Plot the weights from input to excitatory layer to view during training.
	'''
	weights = get_2d_input_weights()

	if n_e != 1:
		fig = plt.figure(fig_num, figsize=(18, 9))
	else:
		fig = plt.figure(fig_num, figsize=(9, 9))

	plt.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom='off',      # ticks along the bottom edge are off
	    top='off',         # ticks along the top edge are off
	    labelbottom='off') # labels along the bottom edge are off

	plt.tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    left='off',      # ticks along the bottom edge are off
	    right='off',         # ticks along the top edge are off
	    labelleft='off') # labels along the bottom edge are off

	ax = plt.gca()

	im = ax.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
	
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.1)
	
	plt.colorbar(im, cax=cax)

	if n_e != 1:
		plt.xticks(xrange(conv_size, conv_size * (conv_features + 1), conv_size), xrange(1, conv_features + 1))
		plt.yticks(xrange(conv_size, conv_size * (n_e + 1), conv_size), xrange(1, n_e + 1))
		plt.xlabel('Convolution patch')
		plt.ylabel('Location in input (from top left to bottom right')
	
	fig.canvas.draw()
	return im, fig


print '\n'
print '\n'.join([ str(idx) + ' | ' + file_name for idx, file_name in enumerate([ file_name for file_name in sorted(os.listdir(weight_dir)) if 'XeAe' in file_name ]) ])
print '\n'

to_plot = raw_input('Enter the index of the file from above which you\'d like to plot: ')
if to_plot == '':
	file_name = [ file_name for file_name in sorted(os.listdir(weight_dir)) if 'XeAe' in file_name ][0]
else:
	file_name = [ file_name for file_name in sorted(os.listdir(weight_dir)) if 'XeAe' in file_name ][int(to_plot)]

sort_euclidean = raw_input('Sort plot by Euclidean distance? (y / n, default no): ')
if sort_euclidean in ['', 'n']:
	sort_euclidean = False
elif sort_euclidean == 'y':
	sort_euclidean = True
else:
	raise Exception('Expecting one of "", "y", or "n".')

# number of inputs to the network
n_input = 784
n_input_sqrt = int(math.sqrt(n_input))

conv_size = int(file_name.split('_')[1])
conv_stride = int(file_name.split('_')[2])
conv_features = int(file_name.split('_')[3])

# number of excitatory neurons (number output from convolutional layer)
if conv_size == 28 and conv_stride == 0:
	n_e = n_e_sqrt = 1
	n_e_total = conv_features
else:
	n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
	n_e_total = n_e * conv_features
	n_e_sqrt = int(math.sqrt(n_e))

# number of inhibitory neurons (number of convolutational features (for now))
n_i = n_e

weight = {}

if conv_size == 28 and conv_stride == 0:
	weight['ee_input'] = n_e_total * 0.15
else:
	weight['ee_input'] = (conv_size ** 2) * 0.1625

conv_features_sqrt = int(math.ceil(math.sqrt(conv_features)))

print '\n'

# creating convolution locations inside the input image
convolution_locations = {}
for n in xrange(n_e):
	convolution_locations[n] = [ ((n % n_e_sqrt) * conv_stride + (n // n_e_sqrt) * n_input_sqrt * conv_stride) + (x * n_input_sqrt) + y for y in xrange(conv_size) for x in xrange(conv_size) ]

weight_matrix = np.load(os.path.join(weight_dir, file_name))

wmax_ee = np.max(weight_matrix)

input_weight_monitor, fig_weights = plot_2d_input_weights(' '.join(file_name[:-4].split('_')))

plt.savefig(os.path.join(plots_dir, file_name[:-4] + '.png'))
plt.show()

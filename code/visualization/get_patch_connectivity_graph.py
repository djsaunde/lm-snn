import numpy as np
import matplotlib.cm as cmap
import time, os.path, scipy, math, sys, timeit
import cPickle as p
import brian_no_units
import brian as b
import networkx as nx

from scipy.sparse import coo_matrix
from struct import unpack
from brian import *

fig_num = 0
wmax_ee = 1.0

np.set_printoptions(threshold=np.nan)

top_level_path = '../../'
weight_dir = top_level_path + 'weights/csnn_pc/'


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


def get_patch_weights():
	'''
	Get the weights from the input to excitatory layer and reshape them.
	'''
	rearranged_weights = np.zeros((conv_features * n_e, conv_features * n_e))
	connection = weight_matrix

	for feature in xrange(conv_features):
		for other_feature in xrange(conv_features):
			if feature != other_feature:
				for this_n in xrange(n_e):
					for other_n in xrange(n_e):
						if is_lattice_connection(n_e_sqrt, this_n, other_n):
							rearranged_weights[feature * n_e + this_n, other_feature * n_e + other_n] = connection[feature * n_e + this_n, other_feature * n_e + other_n]

	return rearranged_weights


def plot_patch_weights():
	'''
	Plot the weights between convolution patches to view during training.
	'''
	weights = get_patch_weights()
	fig = b.figure(fig_num, figsize=(8, 8))
	im = b.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
	# for idx in xrange(n_e, n_e * conv_features, n_e):
	# 	b.axvline(idx, ls='--', lw=1)
	# 	b.axhline(idx, ls='--', lw=1)
	b.colorbar(im)
	b.title('Between-patch connectivity')
	fig.canvas.draw()
	return im, fig


def is_lattice_connection(sqrt, i, j):
	'''
	Boolean method which checks if two indices in a network correspond to neighboring nodes in a 4-, 8-, or all-lattice.

	sqrt: square root of the number of nodes in population
	i: First neuron's index
	k: Second neuron's index
	'''
	if lattice_structure == 'none':
		return False
	if lattice_structure == '4':
		return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j
	if lattice_structure == '8':
		return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j or i + sqrt == j + 1 and j % sqrt != 0 or i + sqrt == j - 1 and \
																						i % sqrt != 0 or i - sqrt == j + 1 and i % sqrt != 0 or i - sqrt == j - 1 and j % sqrt != 0
	if lattice_structure == 'all':
		return True


print '\n'
print '\n'.join([ str(idx) + ' | ' + file_name for idx, file_name in enumerate([ file_name for file_name in sorted(os.listdir(weight_dir)) if 'AeAe' in file_name and 'all' in file_name and '.npy' in file_name ]) ])
print '\n'

to_plot = raw_input('Enter the index of the file from above which you\'d like to use: ')
if to_plot == '':
	file_name = [ file_name for file_name in sorted(os.listdir(weight_dir)) if 'AeAe' in file_name and 'all' in file_name and '.npy' in file_name ][0]
else:
	file_name = [ file_name for file_name in sorted(os.listdir(weight_dir)) if 'AeAe' in file_name and 'all' in file_name and '.npy' in file_name ][int(to_plot)]

# number of inputs to the network
n_input = 784
n_input_sqrt = int(math.sqrt(n_input))

conv_size = int(file_name.split('_')[2])
conv_stride = int(file_name.split('_')[3])
conv_features = int(file_name.split('_')[4])
lattice_structure = file_name.split('sharing')[1].split('_')[1]

# number of excitatory neurons (number output from convolutional layer)
n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
n_e_total = n_e * conv_features
n_e_sqrt = int(math.sqrt(n_e))

# number of inhibitory neurons (number of convolutational features (for now))
n_i = n_e

print '\n'
print 'Number of nodes:', conv_features * n_e

conv_features_sqrt = int(math.sqrt(conv_features))

print '\n'

weight_matrix = get_matrix_from_file(weight_dir + file_name, conv_features * n_e, conv_features * n_e)

plot_patch_weights()
b.show()

nonzero = np.count_nonzero(weight_matrix)
weight_matrix[weight_matrix < 0.99] = 0.0
new_nonzero = np.count_nonzero(weight_matrix)

print '\n'
print nonzero, new_nonzero, new_nonzero / float(nonzero)
print '\n'

weight_matrix[weight_matrix > 0.0] = 1

G = nx.Graph(weight_matrix)

plt.figure(figsize=(18.5, 10))
nx.draw_circular(G, node_color='g', edge_color='#909090', edge_size=1, node_size=10)
plt.axis('equal')

plt.show()

plt.figure(figsize=(18.5, 10))
nx.draw_spectral(G, node_color='g', edge_color='#909090', edge_size=1, node_size=10)
plt.axis('equal')

plt.show()

plt.figure(figsize=(18.5, 10))
nx.draw_spring(G, node_color='g', edge_color='#909090', edge_size=1, node_size=10)
plt.axis('equal')

plt.show()

import numpy as np
import matplotlib.cm as cmap
import cPickle as p
import brian_no_units
import brian as b
import networkx as nx
import time, os.path, scipy, math, sys, timeit, random, argparse

from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix
from struct import unpack
from brian import *


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

weight_path = '../../weights/conv_patch_connectivity_weights/'

print '\n'
print '\n'.join([ str(idx + 1) + ' | ' + file_name for idx, file_name in enumerate([ file_name for file_name in sorted(os.listdir(weight_path)) if 'AeAe' in file_name and 'all' in file_name and '.npy' in file_name ]) ])
print '\n'

to_get = raw_input('Enter the index of the file from above which you\'d like to plot: ')

file_name = [ file_name for file_name in sorted(os.listdir(weight_path)) if 'AeAe' in file_name and 'all' in file_name and '.npy' in file_name ][int(to_get) - 1]

n_input = 784
n_input_sqrt = int(math.sqrt(n_input))

conv_size = int(file_name.split('_')[2])
conv_stride = int(file_name.split('_')[3])
conv_features = int(file_name.split('_')[4])

n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
n_e_total = n_e * conv_features
n_e_sqrt = int(math.sqrt(n_e))
n_i = n_e

weight_matrix = get_matrix_from_file(weight_path + file_name, conv_features * n_e, conv_features * n_e)

weight_matrix[weight_matrix < np.percentile(weight_matrix[np.where(weight_matrix != 0)], 99)] = 0.0
weight_matrix[weight_matrix > 0.0] = 1

np.savetxt('../../data/patch_connectivity_matrices/' + file_name.split('.npy')[0] + '.txt', weight_matrix)

print 'Total number of possible connections:', (conv_features * n_e) ** 2
print 'Shape of connectivity matrix:', weight_matrix.shape
print 'Number of non-zero connections:', np.size(np.nonzero(weight_matrix))
print 'Approximate percentage of non-zero connections:', np.size(np.nonzero(weight_matrix)) / float((conv_size ** 2) * 4 * conv_features)
print '\n'
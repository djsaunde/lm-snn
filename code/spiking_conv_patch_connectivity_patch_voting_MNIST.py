'''
Much the same as 'spiking_MNIST.py', but we instead use a number of convolutional
windows to map the input to a reduced space.
'''

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

np.set_printoptions(threshold=np.nan)

# only show log messages of level ERROR or higher
b.log_level_error()

MNIST_data_path = '../data/'
top_level_path = '../'


def get_labeled_data(picklename, b_train=True):
	'''
	Read input-vector (image) and target class (label, 0-9) and return it as 
	a list of tuples.
	'''
	if os.path.isfile('%s.pickle' % picklename):
		data = p.load(open('%s.pickle' % picklename))
	else:
		# Open the images with gzip in read binary mode
		if b_train:
			images = open(MNIST_data_path + 'train-images-idx3-ubyte', 'rb')
			labels = open(MNIST_data_path + 'train-labels-idx1-ubyte', 'rb')
		else:
			images = open(MNIST_data_path + 't10k-images-idx3-ubyte', 'rb')
			labels = open(MNIST_data_path + 't10k-labels-idx1-ubyte', 'rb')

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
		x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
		y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
		for i in xrange(N):
			if i % 1000 == 0:
				print("i: %i" % i)
			x[i] = [[unpack('>B', images.read(1))[0] for unused_col in xrange(cols)]  for unused_row in xrange(rows) ]
			y[i] = unpack('>B', labels.read(1))[0]

		data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
		p.dump(data, open("%s.pickle" % picklename, "wb"))
	return data


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
		return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j or i + sqrt == j + 1 and j % sqrt != 0 or i + sqrt == j - 1 and i % sqrt != 0 or i - sqrt == j + 1 and i % sqrt != 0 or i - sqrt == j - 1 and j % sqrt != 0
	if lattice_structure == 'all':
		return True

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


def save_connections():
	'''
	Save all connections in 'save_conns'; ending may be set to the index of the last
	example run through the network
	'''

	# print out saved connections
	print '...saving connections: weights/conv_patch_connectivity_patch_voting_weights/' + save_conns[0] + '_' + ending + ' and ' + 'weights/conv_patch_connectivity_weights/' + save_conns[1] + '_' + stdp_input

	# iterate over all connections to save
	for conn_name in save_conns:
		if conn_name == 'AeAe':
			conn_matrix = connections[conn_name][:]
		else:
			conn_matrix = input_connections[conn_name][:]
		# sparsify it into (row, column, entry) tuples
		conn_list_sparse = ([(i, j, conn_matrix[i, j]) for i in xrange(conn_matrix.shape[0]) for j in xrange(conn_matrix.shape[1]) ])
		# save it out to disk
		np.save(top_level_path + 'weights/conv_patch_connectivity_patch_voting_weights/' + conn_name + '_' + ending, conn_list_sparse)


def save_theta():
	'''
	Save the adaptive threshold parameters to a file.
	'''

	# iterate over population for which to save theta parameters
	for pop_name in population_names:
		# print out saved theta populations
		print '...saving theta: weights/conv_patch_connectivity_patch_voting_weights/theta_' + pop_name + '_' + ending

		# save out the theta parameters to file
		np.save(top_level_path + 'weights/conv_patch_connectivity_patch_voting_weights/theta_' + pop_name + '_' + ending, neuron_groups[pop_name + 'e'].theta)


def set_weights_most_fired(current_spike_count):
	'''
	For each convolutional patch, set the weights to those of the neuron which
	fired the most in the last iteration.
	'''

	for conn_name in input_connections:
		for feature in xrange(conv_features):
			# count up the spikes for the neurons in this convolution patch
			column_sums = np.sum(current_spike_count[feature : feature + 1, :], axis=0)

			# find the excitatory neuron which spiked the most
			most_spiked = np.argmax(column_sums)

			# create a "dense" version of the most spiked excitatory neuron's weight
			most_spiked_dense = input_connections[conn_name][:, feature * n_e + most_spiked].todense()

			# set all other neurons' (in the same convolution patch) weights the same as the most-spiked neuron in the patch
			for n in xrange(n_e):
				if n != most_spiked:
					other_dense = input_connections[conn_name][:, feature * n_e + n].todense()
					other_dense[convolution_locations[n]] = most_spiked_dense[convolution_locations[most_spiked]]
					input_connections[conn_name][:, feature * n_e + n] = other_dense


def normalize_weights():
	'''
	Squash the input -> excitatory weights to sum to a prespecified number.
	'''
	for conn_name in input_connections:
		connection = input_connections[conn_name][:].todense()
		for feature in xrange(conv_features):
			feature_connection = connection[:, feature * n_e : (feature + 1) * n_e]
			column_sums = np.sum(feature_connection, axis=0)
			column_factors = weight['ee_input'] / column_sums

			for n in xrange(n_e):
				dense_weights = input_connections[conn_name][:, feature * n_e + n].todense()
				dense_weights[convolution_locations[n]] *= column_factors[n]
				input_connections[conn_name][:, feature * n_e + n] = dense_weights

	for conn_name in connections:
		if 'AeAe' in conn_name and lattice_structure != 'none':
			connection = connections[conn_name][:].todense()
			for feature in xrange(conv_features):
				feature_connection = connection[feature * n_e : (feature + 1) * n_e, :]
				column_sums = np.sum(feature_connection)
				column_factors = weight['ee_recurr'] / column_sums

				for idx in xrange(feature * n_e, (feature + 1) * n_e):
					connections[conn_name][idx, :] *= column_factors


def plot_input(rates):
	'''
	Plot the current input example during the training procedure.
	'''
	fig = b.figure(fig_num, figsize = (5, 5))
	im = b.imshow(rates.reshape((28, 28)), interpolation = 'nearest', vmin=0, vmax=64, cmap=cmap.get_cmap('gray'))
	b.colorbar(im)
	b.title('Current input example')
	fig.canvas.draw()
	return im, fig


def update_input(rates, im, fig):
	'''
	Update the input image to use for input plotting.
	'''
	im.set_array(rates.reshape((28, 28)))
	fig.canvas.draw()
	return im


def plot_cluster_centers(cluster_centers):
	'''
	Plot the cluster centers for the input to excitatory layer weights during training.
	'''
	fig = b.figure(fig_num, figsize=(8, 8))
	centers_sqrt = int(math.sqrt(len(cluster_centers)))
	to_show = np.zeros((conv_size * centers_sqrt, conv_size * centers_sqrt))
	for i in xrange(centers_sqrt):
		for j in xrange(centers_sqrt):
			to_show[i * conv_size : (i + 1) * conv_size, j * conv_size : (j + 1) * conv_size] = cluster_centers[i * centers_sqrt + j].reshape((conv_size, conv_size))
	im = b.imshow(to_show, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
	b.colorbar(im)
	b.title('Cluster centers')
	fig.canvas.draw()
	return im, fig


def update_cluster_centers(cluster_centers, im, fig):
	'''
	Update the plot of the cluster centers (input to excitatory weights).
	'''
	centers_sqrt = int(math.sqrt(len(cluster_centers)))
	to_show = np.zeros((conv_size * centers_sqrt, conv_size * centers_sqrt))
	for i in xrange(centers_sqrt):
		for j in xrange(centers_sqrt):
			to_show[i * conv_size : (i + 1) * conv_size, j * conv_size : (j + 1) * conv_size] = cluster_centers[i * centers_sqrt + j].reshape((conv_size, conv_size)).T
	im.set_array(to_show)
	fig.canvas.draw()
	return im


def get_2d_input_weights():
	'''
	Get the weights from the input to excitatory layer and reshape it to be two
	dimensional and square.
	'''
	rearranged_weights = np.zeros((conv_features_sqrt * conv_size * n_e_sqrt, conv_features_sqrt * conv_size * n_e_sqrt))
	connection = input_connections['XeAe'][:]

	# for each convolution feature
	for feature in xrange(conv_features):
		# for each excitatory neuron in this convolution feature
		for n in xrange(n_e):
			temp = connection[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)].todense()

			# print ((feature // conv_features_sqrt) * conv_size * n_e_sqrt) + ((n // n_e_sqrt) * conv_size), ((feature // conv_features_sqrt) * conv_size * n_e_sqrt) + ((n // n_e_sqrt) * conv_size) + conv_size, ((feature % conv_features_sqrt) * conv_size * n_e_sqrt) + ((n % n_e_sqrt) * (conv_size)), ((feature % conv_features_sqrt) * conv_size * n_e_sqrt) + ((n % n_e_sqrt) * (conv_size)) + conv_size
			rearranged_weights[ ((feature % conv_features_sqrt) * conv_size * n_e_sqrt) + ((n % n_e_sqrt) * conv_size) : ((feature % conv_features_sqrt) * conv_size * n_e_sqrt) + ((n % n_e_sqrt) * conv_size) + conv_size, ((feature // conv_features_sqrt) * conv_size * n_e_sqrt) + ((n // n_e_sqrt) * (conv_size)) : ((feature // conv_features_sqrt) * conv_size * n_e_sqrt) + ((n // n_e_sqrt) * (conv_size)) + conv_size ] = temp[convolution_locations[n]].reshape((conv_size, conv_size))

	# return the rearranged weights to display to the user
	return rearranged_weights.T

def get_input_weights(weight_matrix):
	'''
	Get the weights from the input to excitatory layer and reshape it to be two
	dimensional and square.
	'''
	weights = []

	# for each convolution feature
	for feature in xrange(conv_features):
		# for each excitatory neuron in this convolution feature
		for n in xrange(n_e):
			temp = weight_matrix[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)]
			weights.append(np.ravel(temp[convolution_locations[n]]))

	# return the rearranged weights to display to the user
	return weights


def plot_2d_input_weights():
	'''
	Plot the weights from input to excitatory layer to view during training.
	'''
	weights = get_2d_input_weights()
	fig = b.figure(fig_num, figsize=(18, 18))
	im = b.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
	for idx in xrange(conv_size * n_e_sqrt, conv_size * conv_features_sqrt * n_e_sqrt, conv_size * n_e_sqrt):
		b.axvline(idx, ls='--', lw=1)
		b.axhline(idx, ls='--', lw=1)
	b.colorbar(im)
	b.title('Reshaped input -> convolution weights')
	b.xticks(xrange(0, conv_size * conv_features_sqrt * n_e_sqrt, conv_size * n_e_sqrt))
	b.yticks(xrange(0, conv_size * conv_features_sqrt * n_e_sqrt, conv_size * n_e_sqrt))
	fig.canvas.draw()
	return im, fig


def update_2d_input_weights(im, fig):
	'''
	Update the plot of the weights from input to excitatory layer to view during training.
	'''
	weights = get_2d_input_weights()
	im.set_array(weights)
	fig.canvas.draw()
	return im


def get_patch_weights():
	'''
	Get the weights from the input to excitatory layer and reshape them.
	'''
	rearranged_weights = np.zeros((conv_features * n_e, conv_features * n_e))
	connection = connections['AeAe'][:]

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
	for idx in xrange(n_e, n_e * conv_features, n_e):
		b.axvline(idx, ls='--', lw=1)
		b.axhline(idx, ls='--', lw=1)
	b.colorbar(im)
	b.title('Between-patch connectivity')
	fig.canvas.draw()
	return im, fig

def update_patch_weights(im, fig):
	'''
	Update the plot of the weights between convolution patches to view during training.
	'''
	weights = get_patch_weights()
	im.set_array(weights)
	fig.canvas.draw()
	return im

def plot_neuron_votes(assignments, spike_rates):
	'''
	Plot the votes of the neurons per label.
	'''
	all_summed_rates = [0] * 10
	num_assignments = [0] * 10

	for i in xrange(10):
		num_assignments[i] = len(np.where(assignments == i)[0])
		if num_assignments[i] > 0:
			all_summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]

	fig = b.figure(fig_num, figsize=(6, 4))
	rects = b.bar(xrange(10), [ 0.1 ] * 10)
	b.ylim([0, 1])
	b.title('Percentage votes per label')
	fig.canvas.draw()
	return rects, fig


def update_neuron_votes(rects, fig, spike_rates):
	'''
	Update the plot of the votes of the neurons by label.
	'''
	all_summed_rates = [0] * 10
	num_assignments = [0] * 10

	for i in xrange(10):
		num_assignments[i] = len(np.where(assignments == i)[0])
		if num_assignments[i] > 0:
			all_summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]

	total_votes = np.sum(all_summed_rates)

	if total_votes != 0:
		for rect, h in zip(rects, all_summed_rates):
			rect.set_height(h / float(total_votes))

	fig.canvas.draw()
	return rects


def get_current_performance(performances, current_example_num):
	'''
	Evaluate the performance of the network on the past 'update_interval' training
	examples.
	'''
	global all_output_numbers, most_spiked_output_numbers, top_percent_output_numbers, input_numbers

	current_evaluation = int(current_example_num / update_interval)
	start_num = current_example_num - update_interval
	end_num = current_example_num

	for performance in performances.keys():
		difference = output_numbers[performance][start_num : end_num, 0] - input_numbers[start_num : end_num]
		correct = len(np.where(difference == 0)[0])
		performances[performance][current_evaluation] = correct / float(update_interval) * 100

	return performances


def plot_performance(fig_num, performances, num_evaluations):
	'''
	Set up the performance plot for the beginning of the simulation.
	'''
	time_steps = range(0, num_evaluations)

	fig = b.figure(fig_num, figsize = (15, 5))
	fig_num += 1

	for performance in performances.keys():
		im, = plt.plot(time_steps, performances[performance])

	b.ylim(ymax = 100)
	b.title('Classification performance')
	fig.canvas.draw()

	return im, fig_num, fig


def update_performance_plot(im, performances, current_example_num, fig):
	'''
	Update the plot of the performance based on results thus far.
	'''
	performances = get_current_performance(performances, current_example_num)
	im.set_ydata(performances.values())
	fig.canvas.draw()
	return im, performances


def get_recognized_number_ranking(assignments, kmeans_assignments, kmeans, simple_clusters, spike_rates, average_firing_rate):
	'''
	Given the label assignments of the excitatory layer and their spike rates over
	the past 'update_interval', get the ranking of each of the categories of input.
	'''
	most_spiked_summed_rates = [0] * 10
	num_assignments = [0] * 10

	most_spiked_array = np.array(np.zeros((conv_features, n_e)), dtype=bool)

	for feature in xrange(conv_features):
		# count up the spikes for the neurons in this convolution patch
		column_sums = np.sum(spike_rates[feature : feature + 1, :], axis=0)

		# find the excitatory neuron which spiked the most
		most_spiked_array[feature, np.argmax(column_sums)] = True

	# for each label
	for i in xrange(10):
		# get the number of label assignments of this type
		num_assignments[i] = len(np.where(assignments[most_spiked_array] == i)[0])

		if len(spike_rates[np.where(assignments[most_spiked_array] == i)]) > 0:
			# sum the spike rates of all excitatory neurons with this label, which fired the most in its patch
			most_spiked_summed_rates[i] = np.sum(spike_rates[np.where(np.logical_and(assignments == i, most_spiked_array))]) / float(np.sum(spike_rates[most_spiked_array]))

	all_summed_rates = [0] * 10
	num_assignments = [0] * 10

	for i in xrange(10):
		num_assignments[i] = len(np.where(assignments == i)[0])
		if num_assignments[i] > 0:
			all_summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]

	top_percent_summed_rates = [0] * 10
	num_assignments = [0] * 10
	
	top_percent_array = np.array(np.zeros((conv_features, n_e)), dtype=bool)
	top_percent_array[np.where(spike_rates > np.percentile(spike_rates, 100 - top_percent))] = True

	# for each label
	for i in xrange(10):
		# get the number of label assignments of this type
		num_assignments[i] = len(np.where(assignments[top_percent_array] == i)[0])

		if len(np.where(assignments[top_percent_array] == i)) > 0:
			# sum the spike rates of all excitatory neurons with this label, which fired the most in its patch
			top_percent_summed_rates[i] = len(spike_rates[np.where(np.logical_and(assignments == i, top_percent_array))])

	# cluster_summed_rates = [0] * 10
	# num_assignments = [0] * 10

	# spike_rates_flat = np.copy(np.ravel(spike_rates))

	# for i in xrange(10):
	# 	num_assignments[i] = 0
	# 	for assignment in cluster_assignments.keys():
	# 		if cluster_assignments[assignment] == i and len(clusters[assignment]) > 1:
	# 			num_assignments[i] += 1
	# 	if num_assignments[i] > 0:
	# 		for assignment in cluster_assignments.keys():
	# 			if cluster_assignments[assignment] == i and len(clusters[assignment]) > 1:
	# 				cluster_summed_rates[i] += np.sum(spike_rates_flat[clusters[assignment]]) / float(len(clusters[assignment]))

	spike_rates_flat = np.copy(np.ravel(spike_rates))

	kmeans_summed_rates = [0] * 10
	num_assignments = [0] * 10

	for i in xrange(10):
		num_assignments[i] = 0
		for assignment in kmeans_assignments.keys():
			if kmeans_assignments[assignment] == i:
				num_assignments[i] += 1
		if num_assignments[i] > 0:
			for cluster, assignment in enumerate(kmeans_assignments.keys()):
				if kmeans_assignments[assignment] == i:
					kmeans_summed_rates[i] += sum([ spike_rates_flat[idx] for idx, label in enumerate(kmeans.labels_) if label == cluster ]) / float(len([ label for label in kmeans.labels_ if label == i ]))

	simple_cluster_summed_rates = [0] * 10
	num_assignments = [0] * 10

	for i in xrange(10):
		if i in simple_clusters.keys() and len(simple_clusters[i]) > 1:
			# simple_cluster_summed_rates[i] = np.sum(spike_rates_flat[simple_clusters[i]]) / float(len(simple_clusters[i]))
			this_spike_rates = spike_rates_flat[simple_clusters[i]]
			simple_cluster_summed_rates[i] = np.sum(this_spike_rates[np.argpartition(this_spike_rates, -5)][-10:])

	# print simple_cluster_summed_rates

	# simple_cluster_summed_rates = simple_cluster_summed_rates / average_firing_rate

	return ( np.argsort(summed_rates)[::-1] for summed_rates in (all_summed_rates, most_spiked_summed_rates, top_percent_summed_rates, kmeans_summed_rates, simple_cluster_summed_rates) )


def get_new_assignments(result_monitor, input_numbers):
	'''
	Based on the results from the previous 'update_interval', assign labels to the
	excitatory neurons.
	'''
	assignments = np.ones((conv_features, n_e))
	input_nums = np.asarray(input_numbers)
	maximum_rate = np.zeros(conv_features * n_e)
	
	for j in xrange(10):
		num_assignments = len(np.where(input_nums == j)[0])
		if num_assignments > 0:
			rate = np.sum(result_monitor[input_nums == j], axis=0) / num_assignments
			for i in xrange(conv_features * n_e):
				if rate[i // n_e, i % n_e] > maximum_rate[i]:
					maximum_rate[i] = rate[i // n_e, i % n_e]
					assignments[i // n_e, i % n_e] = j

	weight_matrix = np.copy(np.array(connections['AeAe'][:].todense()))
	
	# print '\n'
	# print 'Maximum between-patch edge weight:', np.max(weight_matrix)
	# print '\n'
	
	# print '99-th percentile:', np.percentile(weight_matrix[np.where(weight_matrix != 0)], 99)
	# print '99.5-th percentile:', np.percentile(weight_matrix[np.where(weight_matrix != 0)], 99.5)
	# print '99.9-th percentile:', np.percentile(weight_matrix[np.where(weight_matrix != 0)], 99.9)

	# weight_matrix[weight_matrix < np.percentile(weight_matrix[np.where(weight_matrix != 0)], 99)] = 0.0
	# weight_matrix[weight_matrix > 0.0] = 1

	# recurrent_graph = nx.Graph(weight_matrix)

	# plt.figure(figsize=(18.5, 10))
	# nx.draw_circular(recurrent_graph, node_color='g', edge_color='#909090', edge_size=1, node_size=10)
	# plt.axis('equal')

	# plt.show()

	# _, temp = networkx_mcl(recurrent_graph, expand_factor=2, inflate_factor=2, mult_factor=2)

	# clusters = {}
	# for key, value in temp.items():
	# 	if value not in clusters.values():
	# 		clusters[key] = value

	# # print '\n'
	# # print 'Number of qualifying clusters:', len([ cluster for cluster in clusters.values() if len(cluster) > 1 ])
	# # print 'Average size of qualifying clusters:', sum([ len(cluster) for cluster in clusters.values() if len(cluster) > 1 ]) / float(len(clusters))
	# # print 'Nodes per cluster:', sorted([ len(cluster) for cluster in clusters.values() ], reverse=True)

	# cluster_assignments = {}
	# votes_vector = {}

	# for cluster in clusters.keys():
	# 	cluster_assignments[cluster] = -1
	# 	votes_vector[cluster] = np.zeros(10)

	# for j in xrange(10):
	# 	num_assignments = len(np.where(input_nums == j)[0])
	# 	if num_assignments > 0:
	# 		rate = np.sum(result_monitor[input_nums == j], axis=0) / float(num_assignments)
	# 		rate = np.ravel(rate)
	# 		for cluster in clusters.keys():
	# 			if len(clusters[cluster]) > 1:
	# 				votes_vector[cluster][j] += np.sum(rate[clusters[cluster]]) / float(rate[clusters[cluster]].size)
	# 		if j in cluster_assignments.values():
	# 			votes_vector[j] / float(len([ value for value in cluster_assignments.values() if value == j ]))
	
	# for cluster in clusters.keys():
	# 	cluster_assignments[cluster] = np.argmax(votes_vector[cluster])

	# print 'Qualifying cluster assignments (in order of label):', sorted([ value for key, value in cluster_assignments.items() if value != -1 and len(clusters[key]) > 1 ]), '\n'
	# for idx in xrange(10):
	# 	print 'There are', len([ value for key, value in cluster_assignments.items() if value == idx and len(clusters[key]) > 1 ]), str(idx) + '-labeled qualifying clusters'
	# print '\n'

	kmeans_assignments = {}
	votes_vector = {}

	# get the list of flattened input weights per neuron per feature
	weights = get_input_weights(np.copy(input_connections['XeAe'][:].todense()))

	# create and fit a KMeans model
	kmeans = KMeans(n_clusters=25).fit(weights)

	for cluster in xrange(kmeans.n_clusters):
		kmeans_assignments[cluster] = -1
		votes_vector[cluster] = np.zeros(10)

	for j in xrange(10):
		num_assignments = len(np.where(input_nums == j)[0])
		if num_assignments > 0:
			rate = np.sum(result_monitor[input_nums == j], axis=0) / float(num_assignments)
			rate = np.ravel(rate)
			for cluster in xrange(kmeans.n_clusters):
				votes_vector[cluster][j] += sum([ rate[idx] for idx, label in enumerate(kmeans.labels_) if label == cluster ]) / float(len([ label for label in kmeans.labels_ if label == j ]))

	for cluster in xrange(kmeans.n_clusters):
		kmeans_assignments[cluster] = np.argmax(votes_vector[cluster])

	# print 'kmeans cluster assignments (in order of label):', sorted([ value for key, value in kmeans_assignments.items() if value != -1 ]), '\n'
	# for idx in xrange(10):
	# 	print 'There are', len([ value for key, value in kmeans_assignments.items() if value == idx ]), str(idx) + '-labeled KMeans clusters'
	# print '\n'

	simple_clusters = {}
	votes_vector = {}

	for cluster in simple_clusters.keys():
		votes_vector[cluster] = np.zeros(10)

	average_firing_rate = np.zeros(10)

	for j in xrange(10):
		this_result_monitor = result_monitor[input_nums == j]
		average_firing_rate[j] = np.sum(this_result_monitor[np.nonzero(this_result_monitor)]) \
							/ float(np.size(this_result_monitor[np.nonzero(this_result_monitor)]))

	print '\n', average_firing_rate, '\n'

	for j in xrange(10):
		num_assignments = len(np.where(input_nums == j)[0])
		if num_assignments > 0:
			rate = np.sum(result_monitor[input_nums == j], axis=0) / float(num_assignments)
			this_result_monitor = result_monitor[input_nums == j]
			# simple_clusters[j] = np.argwhere(np.sum(result_monitor[input_nums == j], axis=0) > np.percentile(this_result_monitor[np.nonzero(this_result_monitor)], 99))
			# simple_clusters[j] = np.array([ node[0] * n_e + node[1] for node in simple_clusters[j] ])
			# print '99-th percentile for cluster', j, ':', np.percentile(this_result_monitor[np.nonzero(this_result_monitor)], 99)
			simple_clusters[j] = np.argsort(np.ravel(np.sum(this_result_monitor, axis=0)))[::-1][:40]
			# simple_clusters[j] = np.array([ node[0] * n_e + node[1] for node in simple_clusters[j] ])
			print simple_clusters[j]

	# np.savetxt('activity.txt', result_monitor[j])

	print '\n'
	for j in xrange(10):
		if j in simple_clusters.keys():
			print 'There are', len(simple_clusters[j]), 'neurons in the cluster for digit', j, '\n'
	print '\n'

	return assignments, kmeans, kmeans_assignments, simple_clusters, weights, average_firing_rate


def build_network():
	global fig_num

	neuron_groups['e'] = b.NeuronGroup(n_e_total, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=scr_e, compile=True, freeze=True)
	neuron_groups['i'] = b.NeuronGroup(n_e_total, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i, compile=True, freeze=True)

	########################################################
	# CREATE NETWORK POPULATIONS AND RECURRENT CONNECTIONS #
	########################################################

	for name in population_names:
		print '...creating neuron group:', name

		# get a subgroup of size 'n_e' from all exc
		neuron_groups[name + 'e'] = neuron_groups['e'].subgroup(conv_features * n_e)
		# get a subgroup of size 'n_i' from the inhibitory layer
		neuron_groups[name + 'i'] = neuron_groups['i'].subgroup(conv_features)

		# start the membrane potentials of these groups 40mV below their resting potentials
		neuron_groups[name + 'e'].v = v_rest_e - 40. * b.mV
		neuron_groups[name + 'i'].v = v_rest_i - 40. * b.mV

	print '...creating recurrent connections'

	for name in population_names:
		# if we're in test mode / using some stored weights
		if test_mode or weight_path[-8:] == 'weights/conv_patch_connectivity_patch_voting_weights/':
			# load up adaptive threshold parameters
			neuron_groups['e'].theta = np.load(weight_path + 'theta_A' + '_' + ending +'.npy')
		else:
			# otherwise, set the adaptive additive threshold parameter at 20mV
			neuron_groups['e'].theta = np.ones((n_e_total)) * 20.0 * b.mV

		for conn_type in recurrent_conn_names:
			if conn_type == 'ei':
				# create connection name (composed of population and connection types)
				conn_name = name + conn_type[0] + name + conn_type[1]
				# create a connection from the first group in conn_name with the second group
				connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
				# instantiate the created connection
				for feature in xrange(conv_features):
					for n in xrange(n_e):
						connections[conn_name][feature * n_e + n, feature] = 10.4 / ( n_e / 2.0 )

			elif conn_type == 'ie':
				# create connection name (composed of population and connection types)
				conn_name = name + conn_type[0] + name + conn_type[1]
				# create a connection from the first group in conn_name with the second group
				connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
				# instantiate the created connection
				for feature in xrange(conv_features):
					for other_feature in xrange(conv_features):
						if feature != other_feature:
							for n in xrange(n_e):
								connections[conn_name][feature, other_feature * n_e + n] = 17.4

				if random_inhibition_prob != 0.0:
					for feature in xrange(conv_features):
						for other_feature in xrange(conv_features):
							for n_this in xrange(n_e):
								for n_other in xrange(n_e):
									if n_this != n_other:
										if b.random() < random_inhibition_prob:
											connections[conn_name][feature * n_e + n_this, other_feature * n_e + n_other] = 17.4

			elif conn_type == 'ee':
				# create connection name (composed of population and connection types)
				conn_name = name + conn_type[0] + name + conn_type[1]
				# get weights from file if we are in test mode
				if test_mode:
					weight_matrix = get_matrix_from_file(weight_path + conn_name + '_' + ending + '.npy', conv_features * n_e, conv_features * n_e)
				# create a connection from the first group in conn_name with the second group
				connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
				# instantiate the created connection
				if connectivity == 'all':
					for feature in xrange(conv_features):
						for other_feature in xrange(conv_features):
							if feature != other_feature:
								for this_n in xrange(n_e):
									for other_n in xrange(n_e):
										if is_lattice_connection(n_e_sqrt, this_n, other_n):
											if test_mode:
												connections[conn_name][feature * n_e + this_n, other_feature * n_e + other_n] = weight_matrix[feature * n_e + this_n, other_feature * n_e + other_n]
											else:
												connections[conn_name][feature * n_e + this_n, other_feature * n_e + other_n] = (b.random() + 0.01) * 0.3

				elif connectivity == 'pairs':
					for feature in xrange(conv_features):
						if feature % 2 == 0:
							for this_n in xrange(n_e):
								for other_n in xrange(n_e):
									if is_lattice_connection(n_e_sqrt, this_n, other_n):
										if test_mode:
											connections[conn_name][feature * n_e + this_n, (feature + 1) * n_e + other_n] = weight_matrix[feature * n_e + this_n, (feature + 1) * n_e + other_n]
										else:
											connections[conn_name][feature * n_e + this_n, (feature + 1) * n_e + other_n] = (b.random() + 0.01) * 0.3
						elif feature % 2 == 1:
							for this_n in xrange(n_e):
								for other_n in xrange(n_e):
									if is_lattice_connection(n_e_sqrt, this_n, other_n):
										if test_mode:
											connections[conn_name][feature * n_e + this_n, (feature - 1) * n_e + other_n] = weight_matrix[feature * n_e + this_n, (feature - 1) * n_e + other_n]
										else:
											connections[conn_name][feature * n_e + this_n, (feature - 1) * n_e + other_n] = (b.random() + 0.01) * 0.3

				elif connectivity == 'linear':
					for feature in xrange(conv_features):
						if feature != conv_features - 1:
							for this_n in xrange(n_e):
								for other_n in xrange(n_e):
									if is_lattice_connection(n_e_sqrt, this_n, other_n):
										if test_mode:
											connections[conn_name][feature * n_e + this_n, (feature + 1) * n_e + other_n] = weight_matrix[feature * n_e + this_n, (feature + 1) * n_e + other_n]
										else:
											connections[conn_name][feature * n_e + this_n, (feature + 1) * n_e + other_n] = (b.random() + 0.01) * 0.3
						if feature != 0:
							for this_n in xrange(n_e):
								for other_n in xrange(n_e):
									if is_lattice_connection(n_e_sqrt, this_n, other_n):
										if test_mode:
											connections[conn_name][feature * n_e + this_n, (feature - 1) * n_e + other_n] = weight_matrix[feature * n_e + this_n, (feature - 1) * n_e + other_n]
										else:
											connections[conn_name][feature * n_e + this_n, (feature - 1) * n_e + other_n] = (b.random() + 0.01) * 0.3

				elif connectivity == 'none':
					pass

		# if STDP from excitatory -> excitatory is on and this connection is excitatory -> excitatory
		if ee_STDP_on and 'ee' in recurrent_conn_names:
			stdp_methods[name + 'e' + name + 'e'] = b.STDP(connections[name + 'e' + name + 'e'], eqs=eqs_stdp_ee, pre=eqs_stdp_pre_ee, post=eqs_stdp_post_ee, wmin=0., wmax=wmax_ee)

		print '...creating monitors for:', name

		# spike rate monitors for excitatory and inhibitory neuron populations
		rate_monitors[name + 'e'] = b.PopulationRateMonitor(neuron_groups[name + 'e'], bin=(single_example_time + resting_time) / b.second)
		rate_monitors[name + 'i'] = b.PopulationRateMonitor(neuron_groups[name + 'i'], bin=(single_example_time + resting_time) / b.second)
		spike_counters[name + 'e'] = b.SpikeCounter(neuron_groups[name + 'e'])

		# record neuron population spikes if specified
		if record_spikes:
			spike_monitors[name + 'e'] = b.SpikeMonitor(neuron_groups[name + 'e'])
			spike_monitors[name + 'i'] = b.SpikeMonitor(neuron_groups[name + 'i'])

	if record_spikes and do_plot:
		b.figure(fig_num)
		fig_num += 1
		b.ion()
		b.subplot(211)
		b.raster_plot(spike_monitors['Ae'], refresh=1000 * b.ms, showlast=1000 * b.ms)
		b.subplot(212)
		b.raster_plot(spike_monitors['Ai'], refresh=1000 * b.ms, showlast=1000 * b.ms)

	# creating lattice locations for each patch
	if connectivity == 'all':
		lattice_locations = {}
		for this_n in xrange(conv_features * n_e):
			lattice_locations[this_n] = [ other_n for other_n in xrange(conv_features * n_e) if is_lattice_connection(n_e_sqrt, this_n % n_e, other_n % n_e) ]
	elif connectivity == 'pairs':
		lattice_locations = {}
		for this_n in xrange(conv_features * n_e):
			lattice_locations[this_n] = []
			for other_n in xrange(conv_features * n_e):
				if this_n // n_e % 2 == 0:
					if is_lattice_connection(n_e_sqrt, this_n % n_e, other_n % n_e) and other_n // n_e == this_n // n_e + 1:
						lattice_locations[this_n].append(other_n)
				elif this_n // n_e % 2 == 1:
					if is_lattice_connection(n_e_sqrt, this_n % n_e, other_n % n_e) and other_n // n_e == this_n // n_e - 1:
						lattice_locations[this_n].append(other_n)
	elif connectivity == 'linear':
		lattice_locations = {}
		for this_n in xrange(conv_features * n_e):
			lattice_locations[this_n] = []
			for other_n in xrange(conv_features * n_e):
				if this_n // n_e != conv_features - 1:
					if is_lattice_connection(n_e_sqrt, this_n % n_e, other_n % n_e) and other_n // n_e == this_n // n_e + 1:
						lattice_locations[this_n].append(other_n)
				elif this_n // n_e != 0:
					if is_lattice_connection(n_e_sqrt, this_n % n_e, other_n % n_e) and other_n // n_e == this_n // n_e - 1:
						lattice_locations[this_n].append(other_n)
	elif connectivity == 'none':
		lattice_locations = {}

	# setting up parameters for weight normalization between patches
	num_lattice_connections = sum([ len(value) for value in lattice_locations.values() ])
	weight['ee_recurr'] = (num_lattice_connections / conv_features) * 0.15

	# creating Poission spike train from input image (784 vector, 28x28 image)
	for name in input_population_names:
		input_groups[name + 'e'] = b.PoissonGroup(n_input, 0)
		rate_monitors[name + 'e'] = b.PopulationRateMonitor(input_groups[name + 'e'], bin=(single_example_time + resting_time) / b.second)

	# creating connections from input Poisson spike train to convolution patch populations
	for name in input_connection_names:
		print '\n...creating connections between', name[0], 'and', name[1]
		
		# for each of the input connection types (in this case, excitatory -> excitatory)
		for conn_type in input_conn_names:
			# saved connection name
			conn_name = name[0] + conn_type[0] + name[1] + conn_type[1]

			# get weight matrix depending on training or test phase
			if test_mode:
				weight_matrix = get_matrix_from_file(weight_path + conn_name + '_' + ending + '.npy', n_input, conv_features * n_e)

			# create connections from the windows of the input group to the neuron population
			input_connections[conn_name] = b.Connection(input_groups['Xe'], neuron_groups[name[1] + conn_type[1]], structure='sparse', state='g' + conn_type[0], delay=True, max_delay=delay[conn_type][1])
			
			if test_mode:
				for feature in xrange(conv_features):
					for n in xrange(n_e):
						for idx in xrange(conv_size ** 2):
							input_connections[conn_name][convolution_locations[n][idx], feature * n_e + n] = weight_matrix[convolution_locations[n][idx], feature * n_e + n]
			else:
				for feature in xrange(conv_features):
					for n in xrange(n_e):
						for idx in xrange(conv_size ** 2):
							input_connections[conn_name][convolution_locations[n][idx], feature * n_e + n] = (b.random() + 0.01) * 0.3

		# if excitatory -> excitatory STDP is specified, add it here (input to excitatory populations)
		if ee_STDP_on:
			print '...creating STDP for connection', name
			
			# STDP connection name
			conn_name = name[0] + conn_type[0] + name[1] + conn_type[1]
			# create the STDP object
			stdp_methods[conn_name] = b.STDP(input_connections[conn_name], eqs=eqs_stdp_ee, pre=eqs_stdp_pre_ee, post=eqs_stdp_post_ee, wmin=0., wmax=wmax_ee)

	print '\n'


def run_simulation():
	'''
	Logic for running the simulation itself.
	'''
	global fig_num, input_intensity, previous_spike_count, rates, assignments, clusters, cluster_assignments, \
				kmeans, kmeans_assignments, simple_clusters, simple_cluster_assignments

	# plot input weights
	if not test_mode and do_plot:
		input_weight_monitor, fig_weights = plot_2d_input_weights()
		fig_num += 1
		patch_weight_monitor, fig2_weights = plot_patch_weights()
		fig_num += 1
		neuron_rects, fig_neuron_votes = plot_neuron_votes(assignments, result_monitor[:])
		fig_num += 1

	average_firing_rate = np.ones(10)
	if do_plot:
		cluster_monitor, cluster_fig = plot_cluster_centers([ np.zeros((conv_size, conv_size)) ] * 25)
		fig_num += 1

	# plot input intensities
	if do_plot:
		input_image_monitor, input_image = plot_input(rates)
		fig_num += 1

	# plot performance
	num_evaluations = int(num_examples / update_interval)
	performances = {}
	performances['all'], performances['most_spiked'], performances['top_percent'], performances['kmeans'], performances['simple_clusters'] = np.zeros(num_evaluations), np.zeros(num_evaluations), np.zeros(num_evaluations), np.zeros(num_evaluations), np.zeros(num_evaluations)
	if do_plot_performance and do_plot:
		performance_monitor, fig_num, fig_performance = plot_performance(fig_num, performances, num_evaluations)
	else:
		performances = get_current_performance(performances, 0)

	# set firing rates to zero initially
	for name in input_population_names:
		input_groups[name + 'e'].rate = 0

	# initialize network
	j = 0
	num_retries = 0
	b.run(0)

	# start recording time
	start_time = timeit.default_timer()

	while j < num_examples:
		# fetched rates depend on training / test phase, and whether we use the 
		# testing dataset for the test phase
		if test_mode:
			if use_testing_set:
				rates = (testing['x'][j % 10000, :, :] / 8.0) * input_intensity
			else:
				rates = (training['x'][j % 60000, :, :] / 8.0) * input_intensity
		
		else:
			# ensure weights don't grow without bound
			normalize_weights()
			# get the firing rates of the next input example
			rates = (training['x'][j % 60000, :, :] / 8.0) * input_intensity
		
		# plot the input at this step
		if do_plot:
			input_image_monitor = update_input(rates, input_image_monitor, input_image)
		
		# sets the input firing rates
		input_groups['Xe'].rate = rates.reshape(n_input)
		
		# run the network for a single example time
		b.run(single_example_time)
		
		# get new neuron label assignments every 'update_interval'
		if j % update_interval == 0 and j > 0:
			assignments, kmeans, kmeans_assignments, simple_clusters, weights, average_firing_rate = get_new_assignments(result_monitor[:], input_numbers[j - update_interval : j])
			if do_plot:
				update_cluster_centers(kmeans.cluster_centers_, cluster_monitor, cluster_fig)

		# get count of spikes over the past iteration
		current_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e)) - previous_spike_count
		previous_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e))

		# set weights to those of the most-fired neuron
		if not test_mode and weight_sharing == 'weight_sharing':
			set_weights_most_fired(current_spike_count)

		# update weights every 'weight_update_interval'
		if j % weight_update_interval == 0 and not test_mode and do_plot:
			update_2d_input_weights(input_weight_monitor, fig_weights)
			update_patch_weights(patch_weight_monitor, fig2_weights)
			
		if do_plot:
			update_neuron_votes(neuron_rects, fig_neuron_votes, result_monitor[:])

		# if the neurons in the network didn't spike more than four times
		if np.sum(current_spike_count) < 5 and num_retries < 3:
			# increase the intensity of input
			input_intensity += 2
			num_retries += 1
			
			# set all network firing rates to zero
			for name in input_population_names:
				input_groups[name + 'e'].rate = 0

			# let the network relax back to equilibrium
			b.run(resting_time)
		# otherwise, record results and continue simulation
		else:
			num_retries = 0
			# record the current number of spikes
			result_monitor[j % update_interval, :] = current_spike_count
			
			# decide whether to evaluate on test or training set
			if test_mode and use_testing_set:
				input_numbers[j] = testing['y'][j % 10000][0]
			else:
				input_numbers[j] = training['y'][j % 60000][0]
			
			# get the output classifications of the network
			output_numbers['all'][j, :], output_numbers['most_spiked'][j, :], output_numbers['top_percent'][j, :], \
							output_numbers['kmeans'][j, :], output_numbers['simple_clusters'][j, :] = \
							get_recognized_number_ranking(assignments, kmeans_assignments, kmeans, simple_clusters, 
							result_monitor[j % update_interval, :], average_firing_rate)
			
			# print progress
			if j % print_progress_interval == 0 and j > 0:
				print 'runs done:', j, 'of', int(num_examples), '(time taken for past', print_progress_interval, 'runs:', str(timeit.default_timer() - start_time) + ')'
				start_time = timeit.default_timer()
			
			# plot performance if appropriate
			if j % update_interval == 0 and j > 0:
				if do_plot_performance and do_plot:
					# updating the performance plot
					perf_plot, performances = update_performance_plot(performance_monitor, performances, j, fig_performance)
				else:
					performances = get_current_performance(performances, j)

				# printing out classification performance results so far
				target = open('../performance/conv_patch_connectivity_patch_voting_performance/' + ending + '.txt', 'w')
				target.truncate()
				target.write('Iteration ' + str(j) + '\n')

				for performance in performances:
					print '\nClassification performance (' + performance + ')', performances[performance][1:int(j / float(update_interval)) + 1], '\nAverage performance:', sum(performances[performance][1:int(j / float(update_interval)) + 1]) / float(len(performances[performance][1:int(j / float(update_interval)) + 1])), '\n'		
					target.write(performance + ' : ' + ' '.join([ str(item) for item in performances[performance][1:int(j / float(update_interval)) + 1] ]) + '\n')
				
				target.close()
					
			# set input firing rates back to zero
			for name in input_population_names:
				input_groups[name + 'e'].rate = 0
			
			# run the network for 'resting_time' to relax back to rest potentials
			b.run(resting_time)
			# reset the input firing intensity
			input_intensity = start_input_intensity
			# increment the example counter
			j += 1

	# set weights to those of the most-fired neuron
	if not test_mode and weight_sharing == 'weight_sharing':
		set_weights_most_fired(current_spike_count)

	# ensure weights don't grow without bound
	normalize_weights()


def save_and_plot_results():
	'''
	Logic for saving and plotting results of the simulation.
	'''
	global fig_num
	
	print '...saving results'

	if not test_mode:
		save_theta()
	if not test_mode:
		save_connections()
	else:
		np.save(top_level_path + 'activity/conv_patch_connectivity_patch_voting_activity/results_' + str(num_examples) + '_' + ending, result_monitor)
		np.save(top_level_path + 'activity/conv_patch_connectivity_patch_voting_activity/input_numbers_' + str(num_examples) + '_' + ending, input_numbers)

	if do_plot:
		if rate_monitors:
			b.figure(fig_num)
			fig_num += 1
			for i, name in enumerate(rate_monitors):
				b.subplot(len(rate_monitors), 1, i + 1)
				b.plot(rate_monitors[name].times / b.second, rate_monitors[name].rate, '.')
				b.title('Rates of population ' + name)

		if spike_monitors:
			b.figure(fig_num)
			fig_num += 1
			for i, name in enumerate(spike_monitors):
				b.subplot(len(spike_monitors), 1, i + 1)
				b.raster_plot(spike_monitors[name])
				b.title('Spikes of population ' + name)

		if spike_counters:
			b.figure(fig_num)
			fig_num += 1
			for i, name in enumerate(spike_counters):
				b.subplot(len(spike_counters), 1, i + 1)
				b.plot(np.asarray(spike_counters['Ae'].count[:]))
				b.title('Spike count of population ' + name)

		plot_2d_input_weights()
		plot_patch_weights()

		b.ioff()
		b.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', default='train')
	parser.add_argument('--connectivity', default='none')
	parser.add_argument('--weight_dependence', default='no_weight_dependence')
	parser.add_argument('--post_pre', default='postpre')
	parser.add_argument('--conv_size', type=int, default=16)
	parser.add_argument('--conv_stride', type=int, default=4)
	parser.add_argument('--conv_features', type=int, default=50)
	parser.add_argument('--weight_sharing', default='no_weight_sharing')
	parser.add_argument('--lattice_structure', default='4')
	parser.add_argument('--random_lattice_prob', type=float, default=0.0)
	parser.add_argument('--random_inhibition_prob', type=float, default=0.0)
	parser.add_argument('--top_percent', type=int, default=10)
	
	args = parser.parse_args()
	mode, connectivity, weight_dependence, post_pre, conv_size, conv_stride, conv_features, weight_sharing, lattice_structure, random_lattice_prob, random_inhibition_prob, top_percent = \
		args.mode, args.connectivity, args.weight_dependence, args.post_pre, args.conv_size, args.conv_stride, args.conv_features, args.weight_sharing, \
		args.lattice_structure, args.random_lattice_prob, args.random_inhibition_prob, args.top_percent

	print '\n'

	print args.mode, args.connectivity, args.weight_dependence, args.post_pre, args.conv_size, args.conv_stride, args.conv_features, args.weight_sharing, \
		args.lattice_structure, args.random_lattice_prob, args.random_inhibition_prob, args.top_percent

	print '\n'

	# set global preferences
	b.set_global_preferences(defaultclock = b.Clock(dt=0.5*b.ms), useweave = True, gcc_options = ['-ffast-math -march=native'], usecodegen = True,
		usecodegenweave = True, usecodegenstateupdate = True, usecodegenthreshold = False, usenewpropagate = True, usecstdp = True, openmp = False,
		magic_useframes = False, useweave_linear_diffeq = True)

	# for reproducibility's sake
	np.random.seed(0)

	# setting test / train mode
	if mode == 'test':
		test_mode = True
	else:
		test_mode = False

	if not test_mode:
	    start = time.time()
	    training = get_labeled_data(MNIST_data_path + 'training', b_train=True)
	    end = time.time()
	    print 'time needed to load training set:', end - start

	else:
	    start = time.time()
	    testing = get_labeled_data(MNIST_data_path + 'testing', b_train=False)
	    end = time.time()
	    print 'time needed to load test set:', end - start

	# set parameters for simulation based on train / test mode
	if test_mode:
		weight_path = top_level_path + 'weights/conv_patch_connectivity_patch_voting_weights/'
		num_examples = 10000 * 1
		use_testing_set = True
		do_plot_performance = False
		record_spikes = True
		ee_STDP_on = False
	else:
		weight_path = top_level_path + 'random/conv_patch_connectivity_patch_voting_random/'
		num_examples = 60000 * 1
		use_testing_set = False
		do_plot_performance = False
		record_spikes = True
		ee_STDP_on = True

	# plotting or not
	do_plot = False

	# number of inputs to the network
	n_input = 784
	n_input_sqrt = int(math.sqrt(n_input))

	# number of neurons parameters
	n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
	n_e_total = n_e * conv_features
	n_e_sqrt = int(math.sqrt(n_e))
	n_i = n_e
	conv_features_sqrt = int(math.ceil(math.sqrt(conv_features)))

	# time (in seconds) per data example presentation and rest period in between, used to calculate total runtime
	single_example_time = 0.35 * b.second
	resting_time = 0.15 * b.second
	runtime = num_examples * (single_example_time + resting_time)

	# set the update interval
	if test_mode:
		update_interval = num_examples
	else:
		update_interval = 100

	# weight updates and progress printing intervals
	weight_update_interval = 10
	print_progress_interval = 10

	# rest potential parameters, reset potential parameters, threshold potential parameters, and refractory periods
	v_rest_e, v_rest_i = -65. * b.mV, -60. * b.mV
	v_reset_e, v_reset_i = -65. * b.mV, -45. * b.mV
	v_thresh_e, v_thresh_i = -52. * b.mV, -40. * b.mV
	refrac_e, refrac_i = 5. * b.ms, 2. * b.ms

	# dictionaries for weights and delays
	weight, delay = {}, {}

	# populations, connections, saved connections, etc.
	input_population_names = [ 'X' ]
	population_names = [ 'A' ]
	input_connection_names = [ 'XA' ]
	save_conns = [ 'XeAe', 'AeAe' ]

	# weird and bad names for variables, I think
	input_conn_names = [ 'ee_input' ]
	recurrent_conn_names = [ 'ei', 'ie', 'ee' ]
	
	# setting weight, delay, and intensity parameters
	weight['ee_input'] = (conv_size ** 2) * 0.1625
	delay['ee_input'] = (0 * b.ms, 10 * b.ms)
	delay['ei_input'] = (0 * b.ms, 5 * b.ms)
	input_intensity = start_input_intensity = 2.0

	# time constants, learning rates, max weights, weight dependence, etc.
	tc_pre_ee, tc_post_ee = 20 * b.ms, 20 * b.ms
	nu_ee_pre, nu_ee_post = 0.0001, 0.01
	wmax_ee = 1.0
	exp_ee_post = exp_ee_pre = 0.2
	w_mu_pre, w_mu_post = 0.2, 0.2

	# setting up differential equations (depending on train / test mode)
	if test_mode:
		scr_e = 'v = v_reset_e; timer = 0*ms'
	else:
		tc_theta = 1e7 * b.ms
		theta_plus_e = 0.05 * b.mV
		scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

	offset = 20.0 * b.mV
	v_thresh_e = '(v>(theta - offset + ' + str(v_thresh_e) + ')) * (timer>refrac_e)'

	# equations for neurons
	neuron_eqs_e = '''
			dv/dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / (100 * ms)  : volt
			I_synE = ge * nS *         -v                           : amp
			I_synI = gi * nS * (-100.*mV-v)                          : amp
			dge/dt = -ge/(1.0*ms)                                   : 1
			dgi/dt = -gi/(2.0*ms)                                  : 1
			'''
	if test_mode:
		neuron_eqs_e += '\n  theta      :volt'
	else:
		neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'

	neuron_eqs_e += '\n  dtimer/dt = 100.0 : ms'

	neuron_eqs_i = '''
			dv/dt = ((v_rest_i - v) + (I_synE + I_synI) / nS) / (10*ms)  : volt
			I_synE = ge * nS *         -v                           : amp
			I_synI = gi * nS * (-85.*mV-v)                          : amp
			dge/dt = -ge/(1.0*ms)                                   : 1
			dgi/dt = -gi/(2.0*ms)                                  : 1
			'''

	# STDP rule
	stdp_input = weight_dependence + '_' + post_pre
	if weight_dependence == 'weight_dependence':
		use_weight_dependence = True
	else:
		use_weight_dependence = False
	if post_pre == 'postpre':
		use_post_pre = True
	else:
		use_post_pre = False

	# STDP synaptic traces
	eqs_stdp_ee = '''
				dpre/dt = -pre / tc_pre_ee : 1.0
				dpost/dt = -post / tc_post_ee : 1.0
				'''

	# setting STDP update rule
	if use_weight_dependence:
		if post_pre:
			eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post * w ** exp_ee_pre'
			eqs_stdp_post_ee = 'w += nu_ee_post * pre * (wmax_ee - w) ** exp_ee_post; post = 1.'

		else:
			eqs_stdp_pre_ee = 'pre = 1.'
			eqs_stdp_post_ee = 'w += nu_ee_post * pre * (wmax_ee - w) ** exp_ee_post; post = 1.'

	else:
		if use_post_pre:
			eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post'
			eqs_stdp_post_ee = 'w += nu_ee_post * pre; post = 1.'

		else:
			eqs_stdp_pre_ee = 'pre = 1.'
			eqs_stdp_post_ee = 'w += nu_ee_post * pre; post = 1.'

	print '\n'

	# set ending of filename saves
	ending = connectivity + '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e) + '_' + weight_dependence + '_' + post_pre + '_' + weight_sharing + '_' + lattice_structure + '_' + str(random_lattice_prob) # + '_' + str(random_inhibition_prob)

	b.ion()
	fig_num = 1
	
	# creating dictionaries for various objects
	neuron_groups, input_groups, connections, input_connections, stdp_methods, \
		rate_monitors, spike_monitors, spike_counters, output_numbers = {}, {}, {}, {}, {}, {}, {}, {}, {}

	# creating convolution locations inside the input image
	convolution_locations = {}
	for n in xrange(n_e):
		convolution_locations[n] = [ ((n % n_e_sqrt) * conv_stride + (n // n_e_sqrt) * n_input_sqrt * conv_stride) + (x * n_input_sqrt) + y for y in xrange(conv_size) for x in xrange(conv_size) ]
	
	# instantiating neuron "vote" monitor
	result_monitor = np.zeros((update_interval, conv_features, n_e))

	# build the spiking neural network
	build_network()

	# bookkeeping variables
	previous_spike_count = np.zeros((conv_features, n_e))
	assignments = np.zeros((conv_features, n_e))
	kmeans = KMeans()
	kmeans_assignments = {}
	simple_clusters = {}
	input_numbers = [0] * num_examples
	output_numbers['all'] = np.zeros((num_examples, 10))
	output_numbers['most_spiked'] = np.zeros((num_examples, 10))
	output_numbers['top_percent'] = np.zeros((num_examples, 10))
	output_numbers['kmeans'] = np.zeros((num_examples, 10))
	output_numbers['simple_clusters'] = np.zeros((num_examples, 10))
	rates = np.zeros((n_input_sqrt, n_input_sqrt))

	# run the simulation of the network
	run_simulation()

	# save and plot results
	save_and_plot_results()

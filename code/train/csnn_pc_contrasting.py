import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.cm as cmap
import cPickle as p
import brian_no_units
import brian as b
import networkx as nx
import pandas as pd
import time, os.path, scipy, math, sys, timeit, random, argparse

from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix
from struct import unpack
from brian import *

from util import *

np.set_printoptions(threshold=np.nan, linewidth=200)

# only show log messages of level ERROR or higher
b.log_level_error()

# set these appropriate to your directory structure
top_level_path = '../../'
MNIST_data_path = os.path.join(top_level_path, 'data')
results_path = os.path.join(top_level_path, 'results')
model_name = 'csnn_pc_contrasting'

performance_dir = os.path.join(top_level_path, 'performance', model_name)
activity_dir = os.path.join(top_level_path, 'activity', model_name)
weights_dir = os.path.join(top_level_path, 'weights', model_name)
random_dir = os.path.join(top_level_path, 'random', model_name)

for d in [ performance_dir, activity_dir, weights_dir, random_dir, MNIST_data_path, results_path ]:
	if not os.path.isdir(d):
		os.makedirs(d)


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
			column_sums = np.sum(np.asarray(feature_connection), axis=0)
			column_factors = weight['ee_input'] / column_sums

			for n in xrange(n_e):
				dense_weights = input_connections[conn_name][:, feature * n_e + n].todense()
				dense_weights[convolution_locations[n]] *= column_factors[n]
				input_connections[conn_name][:, feature * n_e + n] = dense_weights

	for conn_name in connections:
		if 'AeAe' in conn_name and lattice_structure != 'none' and lattice_structure != 'none':
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
	rearranged_weights = np.zeros((conv_features * conv_size, conv_size * n_e))

	# counts number of input -> excitatory weights displayed so far
	connection = input_connections['XeAe'][:]

	# for each excitatory neuron in this convolution feature
	euclid_dists = np.zeros((n_e, conv_features))
	temps = np.zeros((n_e, conv_features, n_input))
	for n in xrange(n_e):
		# for each convolution feature
		for feature in xrange(conv_features):
			temp = connection[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)].todense()
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
	b.colorbar(im)
	b.title('Reshaped input -> convolution weights')
	b.xticks(xrange(conv_size, conv_size * (conv_features + 1), conv_size), xrange(1, conv_features + 1))
	b.yticks(xrange(conv_size, conv_size * (n_e + 1), conv_size), xrange(1, n_e + 1))
	b.xlabel('Convolution patch')
	b.ylabel('Location in input (from top left to bottom right')
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
						if is_lattice_connection(n_e_sqrt, this_n, other_n, lattice_structure):
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


def predict_label(assignments, kmeans_assignments, kmeans, simple_clusters, index_matrix, input_numbers, spike_rates, average_firing_rate):
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
			this_spike_rates = spike_rates_flat[simple_clusters[i]]
			simple_cluster_summed_rates[i] = np.sum(this_spike_rates[np.argpartition(this_spike_rates, -1)][-1:])

	spatial_cluster_index_vector = np.empty(n_e)
	spatial_cluster_index_vector[:] = np.nan

	for idx in xrange(n_e):
		this_spatial_location = spike_rates_flat[idx::n_e]
		if np.size(np.where(this_spatial_location > 0.9 * np.max(spike_rates_flat))) > 0:
			spatial_cluster_index_vector[idx] = np.argmax(this_spatial_location)

	spatial_cluster_summed_rates = [0] * 10
	if input_numbers != []:
		if np.count_nonzero([[ x == y for (x, y) in zip(spatial_cluster_index_vector, index_matrix[idx]) ] for idx in xrange(update_interval) ]) > 0:
			best_col_idx = np.argmax([ sum([ 1.0 if x == y else 0.0 for (x, y) in zip(spatial_cluster_index_vector, index_matrix[idx]) ]) for idx in xrange(update_interval) ])
			spatial_cluster_summed_rates[input_numbers[best_col_idx]] = 1.0
			# for idx in xrange(update_interval):
			# 	# print spatial_cluster_index_vector == index_matrix[idx]
			# 	spatial_cluster_summed_rates[input_numbers[idx]] += np.count_nonzero([ x == y for (x, y) in zip(spatial_cluster_index_vector, index_matrix[idx]) ])

			# print '->', [ input_numbers.count(i) for i in xrange(10) ]
			# spatial_cluster_summed_rates = [ x / float(y) if y != 0 else x for (x, y) in zip(spatial_cluster_summed_rates, [ input_numbers.count(i) for i in xrange(10) ]) ]

	# if spatial_cluster_summed_rates == [0] * 10:
	# 	print '>', spatial_cluster_index_vector

	# print spatial_cluster_summed_rates

	return ( np.argsort(summed_rates)[::-1] for summed_rates in (all_summed_rates, most_spiked_summed_rates, top_percent_summed_rates, \
																	kmeans_summed_rates, simple_cluster_summed_rates, spatial_cluster_summed_rates) )


def assign_labels(result_monitor, input_numbers):
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
				votes_vector[cluster][j] += sum([ rate[idx] for idx, label in enumerate(kmeans.labels_) if label == cluster ]) / \
														float(len([ label for label in kmeans.labels_ if label == j ]))

	for cluster in xrange(kmeans.n_clusters):
		kmeans_assignments[cluster] = np.argmax(votes_vector[cluster])

	simple_clusters = {}
	votes_vector = {}

	for cluster in simple_clusters.keys():
		votes_vector[cluster] = np.zeros(10)

	average_firing_rate = np.zeros(10)

	for j in xrange(10):
		this_result_monitor = result_monitor[input_nums == j]
		average_firing_rate[j] = np.sum(this_result_monitor[np.nonzero(this_result_monitor)]) \
							/ float(np.size(this_result_monitor[np.nonzero(this_result_monitor)]))

	print '\n', average_firing_rate

	for j in xrange(10):
		num_assignments = len(np.where(input_nums == j)[0])
		if num_assignments > 0:
			rate = np.sum(result_monitor[input_nums == j], axis=0) / float(num_assignments)
			this_result_monitor = result_monitor[input_nums == j]
			simple_clusters[j] = np.argsort(np.ravel(np.sum(this_result_monitor, axis=0)))[::-1][:int(0.025 * (np.size(result_monitor) / float(10000)))]

	# print '\n'
	# for j in xrange(10):
	# 	if j in simple_clusters.keys():
	# 		print 'There are', len(simple_clusters[j]), 'neurons in the cluster for digit', j, '\n'

	index_matrix = np.empty((update_interval, n_e))
	index_matrix[:] = np.nan

	for idx in xrange(update_interval):
		this_result_monitor_flat = np.ravel(result_monitor[idx, :])
		for n in xrange(n_e):
			this_spatial_result_monitor_flat = this_result_monitor_flat[n::n_e]
			if np.size(np.where(this_spatial_result_monitor_flat > 0.9 * np.max(this_result_monitor_flat))) > 0:
				index_matrix[idx, n] = np.argmax(this_spatial_result_monitor_flat)

	# print index_matrix

	return assignments, kmeans, kmeans_assignments, simple_clusters, weights, average_firing_rate, index_matrix


def build_network():
	global fig_num

	neuron_groups['e'] = b.NeuronGroup(n_e_total, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=scr_e, compile=True, freeze=True)
	neuron_groups['i'] = b.NeuronGroup(n_e_total, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i, compile=True, freeze=True)

	for name in population_names:
		print '...creating neuron group:', name

		# get a subgroup of size 'n_e' from all exc
		neuron_groups[name + 'e'] = neuron_groups['e'].subgroup(conv_features * n_e)
		# get a subgroup of size 'n_i' from the inhibitory layer
		neuron_groups[name + 'i'] = neuron_groups['i'].subgroup(conv_features * n_e)

		# start the membrane potentials of these groups 40mV below their resting potentials
		neuron_groups[name + 'e'].v = v_rest_e - 40. * b.mV
		neuron_groups[name + 'i'].v = v_rest_i - 40. * b.mV

	print '...creating recurrent connections'

	for name in population_names:
		# if we're in test mode / using some stored weights
		if test_mode:
			# load up adaptive threshold parameters
			neuron_groups['e'].theta = np.ones((n_e_total)) * 35.0 * b.mV
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
						connections[conn_name][feature * n_e + n, feature * n_e + n] = 10.4

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
								connections[conn_name][feature * n_e + n, other_feature * n_e + n] = 17.4

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
					weight_matrix = np.load(os.path.join(weights_dir, conn_name + '_' + ending + '.npy')) 
				# create a connection from the first group in conn_name with the second group
				connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
				# instantiate the created connection
				if connectivity == 'all':
					for feature in xrange(conv_features):
						for other_feature in xrange(conv_features):
							if feature != other_feature:
								for this_n in xrange(n_e):
									for other_n in xrange(n_e):
										if is_lattice_connection(n_e_sqrt, this_n, other_n, lattice_structure):
											if test_mode:
												connections[conn_name][feature * n_e + this_n, other_feature * n_e + other_n] = weight_matrix[feature * n_e + this_n, other_feature * n_e + other_n]
											else:
												connections[conn_name][feature * n_e + this_n, other_feature * n_e + other_n] = (b.random() + 0.01) * 0.3

				elif connectivity == 'pairs':
					for feature in xrange(conv_features):
						if feature % 2 == 0:
							for this_n in xrange(n_e):
								for other_n in xrange(n_e):
									if is_lattice_connection(n_e_sqrt, this_n, other_n, lattice_structure):
										if test_mode:
											connections[conn_name][feature * n_e + this_n, (feature + 1) * n_e + other_n] = weight_matrix[feature * n_e + this_n, (feature + 1) * n_e + other_n]
										else:
											connections[conn_name][feature * n_e + this_n, (feature + 1) * n_e + other_n] = (b.random() + 0.01) * 0.3
						elif feature % 2 == 1:
							for this_n in xrange(n_e):
								for other_n in xrange(n_e):
									if is_lattice_connection(n_e_sqrt, this_n, other_n, lattice_structure):
										if test_mode:
											connections[conn_name][feature * n_e + this_n, (feature - 1) * n_e + other_n] = weight_matrix[feature * n_e + this_n, (feature - 1) * n_e + other_n]
										else:
											connections[conn_name][feature * n_e + this_n, (feature - 1) * n_e + other_n] = (b.random() + 0.01) * 0.3

				elif connectivity == 'linear':
					for feature in xrange(conv_features):
						if feature != conv_features - 1:
							for this_n in xrange(n_e):
								for other_n in xrange(n_e):
									if is_lattice_connection(n_e_sqrt, this_n, other_n, lattice_structure):
										if test_mode:
											connections[conn_name][feature * n_e + this_n, (feature + 1) * n_e + other_n] = weight_matrix[feature * n_e + this_n, (feature + 1) * n_e + other_n]
										else:
											connections[conn_name][feature * n_e + this_n, (feature + 1) * n_e + other_n] = (b.random() + 0.01) * 0.3
						if feature != 0:
							for this_n in xrange(n_e):
								for other_n in xrange(n_e):
									if is_lattice_connection(n_e_sqrt, this_n, other_n, lattice_structure):
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
			lattice_locations[this_n] = [ other_n for other_n in xrange(conv_features * n_e) if is_lattice_connection(n_e_sqrt, this_n % n_e, other_n % n_e, lattice_structure) ]
	elif connectivity == 'pairs':
		lattice_locations = {}
		for this_n in xrange(conv_features * n_e):
			lattice_locations[this_n] = []
			for other_n in xrange(conv_features * n_e):
				if this_n // n_e % 2 == 0:
					if is_lattice_connection(n_e_sqrt, this_n % n_e, other_n % n_e, lattice_structure) and other_n // n_e == this_n // n_e + 1:
						lattice_locations[this_n].append(other_n)
				elif this_n // n_e % 2 == 1:
					if is_lattice_connection(n_e_sqrt, this_n % n_e, other_n % n_e, lattice_structure) and other_n // n_e == this_n // n_e - 1:
						lattice_locations[this_n].append(other_n)
	elif connectivity == 'linear':
		lattice_locations = {}
		for this_n in xrange(conv_features * n_e):
			lattice_locations[this_n] = []
			for other_n in xrange(conv_features * n_e):
				if this_n // n_e != conv_features - 1:
					if is_lattice_connection(n_e_sqrt, this_n % n_e, other_n % n_e, lattice_structure) and other_n // n_e == this_n // n_e + 1:
						lattice_locations[this_n].append(other_n)
				elif this_n // n_e != 0:
					if is_lattice_connection(n_e_sqrt, this_n % n_e, other_n % n_e, lattice_structure) and other_n // n_e == this_n // n_e - 1:
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
				weight_matrix = np.load(os.path.join(weights_dir, conn_name + '_' + ending + '.npy')) 
				weight_matrix[weight_matrix < 0.20] = 0

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

			if test_mode:
				normalize_weights()
				if do_plot:
					plot_2d_input_weights()
					fig_num += 1	

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
				kmeans, kmeans_assignments, simple_clusters, simple_cluster_assignments, index_matrix

	# plot input weights
	if not test_mode and do_plot:
		input_weight_monitor, fig_weights = plot_2d_input_weights()
		fig_num += 1
		if connectivity != 'none':
			patch_weight_monitor, fig2_weights = plot_patch_weights()
			fig_num += 1
		neuron_rects, fig_neuron_votes = plot_neuron_votes(assignments, result_monitor[:])
		fig_num += 1

	average_firing_rate = np.ones(10)
	if do_plot and not test_mode:
		cluster_monitor, cluster_fig = plot_cluster_centers([ np.zeros((conv_size, conv_size)) ] * 25)
		fig_num += 1

	# plot input intensities
	if do_plot:
		input_image_monitor, input_image = plot_input(rates)
		fig_num += 1

	# plot performance
	num_evaluations = int(num_examples / update_interval)
	performances = {}
	performances['all'], performances['most_spiked'], performances['top_percent'], performances['kmeans'], \
										performances['simple_clusters'], performances['spatial_clusters'] = ( np.zeros(num_evaluations) for _ in xrange(6) )
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
			assignments, kmeans, kmeans_assignments, simple_clusters, weights, average_firing_rate, index_matrix = assign_labels(result_monitor[:], input_numbers[j - update_interval : j])
			if do_plot and not test_mode:
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
			if connectivity != 'none':
				update_patch_weights(patch_weight_monitor, fig2_weights)
			
		if not test_mode and do_plot:
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
							output_numbers['kmeans'][j, :], output_numbers['simple_clusters'][j, :], output_numbers['spatial_clusters'][j, :] = \
							predict_label(assignments, kmeans_assignments, kmeans, simple_clusters, index_matrix, 
							input_numbers[j - update_interval - (j % update_interval) : j - (j % update_interval)], result_monitor[j % update_interval, :], average_firing_rate)
			
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
				target = open(os.path.join(performance_dir, ending + '.txt'), 'w')
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


def save_results():
	'''
	Save results of simulation (train or test)
	'''
	print '...Saving results'

	if not test_mode:
		save_connections(weights_dir, connections, input_connections, ending)
		save_theta(weights_dir, population_names, neuron_groups, ending)
	else:
		np.save(os.path.join(activity_dir, 'results_' + str(num_examples) + '_' + ending), result_monitor)
		np.save(os.path.join(activity_dir, 'input_numbers_' + str(num_examples) + '_' + ending), input_numbers)

	print '\n'


def evaluate_results():
	global update_interval

	start_time_training = start_time_testing = 0
	end_time_training = end_time_testing = num_examples

	update_interval = end_time_training

	training_result_monitor = testing_result_monitor = result_monitor
	training_input_numbers = testing_input_numbers = input_numbers

	print '...getting assignments'

	assignments, kmeans, kmeans_assignments, simple_clusters, weights, average_firing_rate, index_matrix = \
																assign_labels(training_result_monitor, training_input_numbers)

	voting_mechanisms = [ 'all', 'most-spiked (per patch)', 'most-spiked (overall)', 'KMeans patch weights clusters',
												'activity clusters', 'spatial correlation clusters' ]

	test_results = {}
	for mechanism in voting_mechanisms:
		# test_results[mechanism] = np.zeros((10, end_time_testing - start_time_testing))
		test_results[mechanism] = np.zeros((10, num_examples))

	print '\n...calculating accuracy per voting mechanism'

	# for idx in xrange(end_time_testing - end_time_training):
	for idx in xrange(num_examples):
		for (mechanism, label_ranking) in zip(voting_mechanisms, predict_label(assignments, kmeans_assignments, kmeans, simple_clusters, index_matrix,
														training_input_numbers, testing_result_monitor[idx, :], average_firing_rate)):
			test_results[mechanism][:, idx] = label_ranking

	differences = { mechanism : test_results[mechanism][0, :] - testing_input_numbers for mechanism in voting_mechanisms }
	correct = { mechanism : len(np.where(differences[mechanism] == 0)[0]) for mechanism in voting_mechanisms }
	incorrect = { mechanism : len(np.where(differences[mechanism] != 0)[0]) for mechanism in voting_mechanisms }
	accuracies = { mechanism : correct[mechanism] / float(end_time_testing - start_time_testing) * 100 for mechanism in voting_mechanisms }

	for mechanism in voting_mechanisms:
		print '\n-', mechanism, 'accuracy:', accuracies[mechanism]

	results = pd.DataFrame([ accuracies.values() ], index=[ str(num_examples) + '_' + ending ], columns=accuracies.keys())
	if not 'accuracy_results.csv' in os.listdir(results_path):
		results.to_csv(results_path + model_name + '.csv', )
	else:
		all_results = pd.read_csv(results_path + model_name + '.csv')
		all_results.append(results)
		all_results.to_csv(results_path + model_name + '.csv')

	print '\n'


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', default='train', help='Network operating mode: "train" mode learns the synaptic weights of the network, and \
														"test" mode holds the weights fixed and evaluates classification accuracy on the test dataset.')
	parser.add_argument('--connectivity', default='none', help='Between-patch connectivity: choose from "none", "pairs", "linear", and "full".')
	parser.add_argument('--weight_dependence', default='no_weight_dependence', help='Modifies the STDP rule to either use or not use the weight dependence mechanism.')
	parser.add_argument('--post_pre', default='postpre', help='Modifies the STDP rule to incorporate both post- and pre-synaptic weight updates, rather than just post-synaptic updates.')
	parser.add_argument('--conv_size', type=int, default=16, help='Side length of the square convolution window used by the input -> excitatory layer of the network.')
	parser.add_argument('--conv_stride', type=int, default=4, help='Horizontal, vertical stride of the convolution window used by the input -> excitatory layer of the network.')
	parser.add_argument('--conv_features', type=int, default=50, help='Number of excitatory convolutional features / filters / patches used in the network.')
	parser.add_argument('--weight_sharing', default='no_weight_sharing', help='Whether to use within-patch weight sharing (each neuron in an excitatory patch shares a single set of weights).')
	parser.add_argument('--lattice_structure', default='4', help='The lattice neighborhood to which connected patches project their connections: one of "none", "4", "8", or "all".')
	parser.add_argument('--random_lattice_prob', type=float, default=0.0, help='Probability with which a neuron from an excitatory patch connects to a neuron in a neighboring excitatory patch \
																												with which it is not already connected to via the between-patch wiring scheme.')
	parser.add_argument('--random_inhibition_prob', type=float, default=0.0, help='Probability with which a neuron from the inhibitory layer connects to any given excitatory neuron with which \
																															it is not already connected to via the inhibitory wiring scheme.')
	parser.add_argument('--top_percent', type=int, default=10, help='The percentage of neurons which are allowed to cast "votes" in the "top_percent" labeling scheme.')
	parser.add_argument('--do_plot', type=bool, default=False, help='Whether or not to display plots during network training / testing. Defaults to False, as this makes the network operation \
																																				speedier, and possible to run on HPC resources.')
	parser.add_argument('--sort_euclidean', type=bool, default=False, help='When plotting reshaped input -> excitatory weights, whether to plot each row (corresponding to locations in the input) \
																																				sorted by Euclidean distance from the 0 matrix.')
	parser.add_argument('--num_examples', type=int, default=10000, help='The number of examples for which to train or test the network on.')

	# parse arguments and place them in local scope
	args = parser.parse_args()
	args = vars(args)
	locals().update(args)

	print '\nOptional argument values:'
	for key, value in args.items():
		print '-', key, ':', value

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
		training = get_labeled_data(os.path.join(MNIST_data_path, 'training'), b_train=True)
		end = time.time()
		print 'time needed to load training set:', end - start

	else:
		start = time.time()
		testing = get_labeled_data(os.path.join(MNIST_data_path, 'testing'), b_train=False)
		end = time.time()
		print 'time needed to load test set:', end - start

	# set parameters for simulation based on train / test mode
	if test_mode:
		use_testing_set = True
		do_plot_performance = False
		record_spikes = True
		ee_STDP_on = False
	else:
		use_testing_set = False
		do_plot_performance = False
		record_spikes = True
		ee_STDP_on = True

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
	ending = connectivity + '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e) + '_' + \
					weight_dependence + '_' + post_pre + '_' + weight_sharing + '_' + lattice_structure + '_' + str(random_lattice_prob)

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
	index_matrix = np.empty((update_interval, n_e))
	index_matrix[:] = np.nan
	input_numbers = [0] * num_examples
	output_numbers['all'] = np.zeros((num_examples, 10))
	output_numbers['most_spiked'] = np.zeros((num_examples, 10))
	output_numbers['top_percent'] = np.zeros((num_examples, 10))
	output_numbers['kmeans'] = np.zeros((num_examples, 10))
	output_numbers['simple_clusters'] = np.zeros((num_examples, 10))
	output_numbers['spatial_clusters'] = np.zeros((num_examples, 10))
	rates = np.zeros((n_input_sqrt, n_input_sqrt))

	# run the simulation of the network
	run_simulation()

	# save and plot results
	save_results()

	# evaluate results
	if test_mode:
		evaluate_results()

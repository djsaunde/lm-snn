'''
Convolutional spiking neural network training, testing, and evaluation script. Evaluation can be done outside of this script; however, it is most straightforward to call this 
script with mode=train, then mode=test on HPC systems, where in the test mode, the network evaluation is written to disk.
'''

import warnings
warnings.filterwarnings('ignore')

import matplotlib.cm as cmap
import brian_no_units
import networkx as nx
import cPickle as p
import pandas as pd
import numpy as np
import brian as b
import argparse
import random
import timeit
import time
import math
import os

from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from struct import unpack
from brian import *

from util import *

np.set_printoptions(threshold=np.nan, linewidth=200)

# only show log messages of level ERROR or higher
b.log_level_error()

# set these appropriate to your directory structure
top_level_path = os.path.join('..', '..')
MNIST_data_path = os.path.join(top_level_path, 'data')
model_name = 'csnn_pc_inhibit_far'
results_path = os.path.join(top_level_path, 'results', model_name)
plots_path = os.path.join(top_level_path, 'plots', model_name)

performance_dir = os.path.join(top_level_path, 'performance', model_name)
activity_dir = os.path.join(top_level_path, 'activity', model_name)
weights_dir = os.path.join(top_level_path, 'weights', model_name)
deltas_dir = os.path.join(top_level_path, 'deltas', model_name)
random_dir = os.path.join(top_level_path, 'random', model_name)
best_weights_dir = os.path.join(weights_dir, 'best')
end_weights_dir = os.path.join(weights_dir, 'end')
misc_dir = os.path.join(top_level_path, 'misc', model_name)
best_misc_dir = os.path.join(misc_dir, 'best')
end_misc_dir = os.path.join(misc_dir, 'end')

for d in [ performance_dir, activity_dir, weights_dir, deltas_dir, random_dir, misc_dir, best_misc_dir, \
			MNIST_data_path, results_path, plots_path, best_weights_dir, end_weights_dir, end_misc_dir ]:
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
	Squash the input to excitatory synaptic weights to sum to a prespecified number.
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
		if 'AeAe' in conn_name and exc_stdp:
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
	fig = plt.figure(fig_num, figsize = (5, 5))
	im = plt.imshow(rates.reshape((28, 28)), interpolation = 'nearest', vmin=0, vmax=64, cmap='binary')
	plt.colorbar(im)
	plt.title('Current input example')
	fig.canvas.draw()
	return im, fig


def update_input(rates, im, fig):
	'''
	Update the input image to use for input plotting.
	'''
	im.set_array(rates.reshape((28, 28)))
	fig.canvas.draw()
	return im


def plot_assignments(assignments):
	cmap = plt.get_cmap('RdBu', 11)
	im = plt.matshow(assignments.reshape((int(np.sqrt(n_e_total)), int(np.sqrt(n_e_total)))).T, cmap=cmap, vmin=-1.5, vmax=9.5)
	plt.colorbar(im, ticks=np.arange(-1, 10))
	plt.title('Neuron labels')
	return im


def update_assignments_plot(assignments, im):
	im.set_array(assignments.reshape((int(np.sqrt(n_e_total)), int(np.sqrt(n_e_total)))).T)
	return im


def get_2d_input_weights():
	'''
	Get the weights from the input to excitatory layer and reshape it to be two
	dimensional and square.
	'''
	# specify the desired shape of the reshaped input -> excitatory weights
	rearranged_weights = np.zeros((conv_features * conv_size, conv_size * n_e))

	# get the input -> excitatory synaptic weights
	connection = input_connections['XeAe'][:]

	if sort_euclidean:
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

	else:
		for n in xrange(n_e):
			for feature in xrange(conv_features):
				temp = connection[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)].todense()
				rearranged_weights[ feature * conv_size : (feature + 1) * conv_size, n * conv_size : (n + 1) * conv_size ] = \
																temp[convolution_locations[n]].reshape((conv_size, conv_size))

	if n_e == 1:
		ceil_sqrt = int(math.ceil(math.sqrt(conv_features)))
		square_weights = np.zeros((28 * ceil_sqrt, 28 * ceil_sqrt))

		for n in xrange(conv_features):
			square_weights[(n // ceil_sqrt) * 28 : ((n // ceil_sqrt) + 1) * 28, 
							(n % ceil_sqrt) * 28 : ((n % ceil_sqrt) + 1) * 28] = rearranged_weights[n * 28 : (n + 1) * 28, :]

		return square_weights.T
	else:
		square_weights = np.zeros((conv_size * features_sqrt * n_e_sqrt, conv_size * features_sqrt * n_e_sqrt))

		for n_1 in xrange(n_e_sqrt):
			for n_2 in xrange(n_e_sqrt):
				for f_1 in xrange(features_sqrt):
					for f_2 in xrange(features_sqrt):
						square_weights[conv_size * (n_2 * features_sqrt + f_2) : conv_size * (n_2 * features_sqrt + f_2 + 1), \
								conv_size * (n_1 * features_sqrt + f_1) : conv_size * (n_1 * features_sqrt + f_1 + 1)] = \
						 		rearranged_weights[(f_1 * features_sqrt + f_2) * conv_size : (f_1 * features_sqrt + f_2 + 1) * conv_size, \
						 				(n_1 * n_e_sqrt + n_2) * conv_size : (n_1 * n_e_sqrt + n_2 + 1) * conv_size]

		return square_weights.T


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

	if n_e != 1:
		fig = plt.figure(fig_num, figsize=(9, 9))
	else:
		fig = plt.figure(fig_num, figsize=(9, 9))

	im = plt.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
	
	if n_e != 1:
		plt.colorbar(im, fraction=0.06)
	else:
		plt.colorbar(im, fraction=0.06)

	plt.title(ending.replace('_', ' '))

	if n_e != 1:
		plt.xticks(xrange(conv_size, conv_size * n_e_sqrt * features_sqrt + 1, conv_size), xrange(1, conv_size * n_e_sqrt * features_sqrt + 1))
		plt.yticks(xrange(conv_size, conv_size * n_e_sqrt * features_sqrt + 1, conv_size), xrange(1, conv_size * n_e_sqrt * features_sqrt + 1))
		for pos in xrange(conv_size * features_sqrt, conv_size * features_sqrt * n_e_sqrt, conv_size * features_sqrt):
			plt.axhline(pos)
			plt.axvline(pos)
	else:
		plt.xticks(xrange(conv_size, conv_size * (int(np.sqrt(conv_features)) + 1), conv_size), xrange(1, int(np.sqrt(conv_features)) + 1))
		plt.yticks(xrange(conv_size, conv_size * (int(np.sqrt(conv_features)) + 1), conv_size), xrange(1, int(np.sqrt(conv_features)) + 1))
	
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
			all_summed_rates[i] = np.sum(spike_rates[:, assignments == i]) / num_assignments[i]

	fig = plt.figure(fig_num, figsize=(6, 4))
	rects = plt.bar(xrange(10), [ 0.1 ] * 10, align='center')
	
	plt.ylim([0, 1])
	plt.xticks(xrange(10))
	plt.title('Percentage votes per label')
	
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
			all_summed_rates[i] = np.sum(spike_rates[:, assignments == i]) / num_assignments[i]

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
	global input_numbers

	current_evaluation = int(current_example_num / update_interval)
	if current_example_num == num_examples -1:
		current_evaluation+=1
	start_num = current_example_num - update_interval
	end_num = current_example_num

	wrong_idxs = {}
	wrong_labels = {}

	for scheme in performances.keys():
		difference = output_numbers[scheme][start_num : end_num, 0] - input_numbers[start_num : end_num]
		correct = len(np.where(difference == 0)[0])

		wrong_idxs[scheme] = np.where(difference != 0)[0]
		wrong_labels[scheme] = output_numbers[scheme][start_num : end_num, 0][np.where(difference != 0)[0]]
		performances[scheme][current_evaluation] = correct / float(update_interval) * 100


	return performances, wrong_idxs, wrong_labels


def plot_performance(fig_num, performances, num_evaluations):
	'''
	Set up the performance plot for the beginning of the simulation.
	'''
	time_steps = range(0, num_evaluations)

	fig = plt.figure(fig_num, figsize = (12, 4))
	fig_num += 1

	for performance in performances:
		plt.plot(time_steps[:np.size(np.nonzero(performances[performance]))], \
									np.extract(np.nonzero(performances[performance]), performances[performance]), label=performance)

	lines = plt.gca().lines

	plt.ylim(ymax=100)
	plt.xticks(xrange(0, num_evaluations + 10, 10), xrange(0, ((num_evaluations + 10) * update_interval), 10))
	plt.legend()
	plt.grid(True)
	plt.title('Classification performance per update interval')
	
	fig.canvas.draw()

	return lines, fig_num, fig


def update_performance_plot(lines, performances, current_example_num, fig):
	'''
	Update the plot of the performance based on results thus far.
	'''
	performances, wrong_idxs, wrong_labels = get_current_performance(performances, current_example_num)
	
	for line, performance in zip(lines, performances):
		line.set_xdata(range((current_example_num / update_interval) + 1))
		line.set_ydata(performances[performance][:(current_example_num / update_interval) + 1])

	fig.canvas.draw()

	return lines, performances, wrong_idxs, wrong_labels


def plot_deltas(fig_num, deltas, num_weight_updates):
	'''
	Set up the performance plot for the beginning of the simulation.
	'''
	time_steps = range(0, num_weight_updates)

	fig = plt.figure(fig_num, figsize = (12, 4))
	fig_num += 1

	plt.plot([], [], label='Absolute difference in weights')

	lines = plt.gca().lines

	plt.ylim(ymin=0, ymax=conv_size*n_e_total)
	plt.xticks(xrange(0, num_weight_updates + weight_update_interval, 100), \
			xrange(0, ((num_weight_updates + weight_update_interval) * weight_update_interval), 100))
	plt.legend()
	plt.grid(True)
	plt.title('Absolute difference in weights per weight update interval')
	
	fig.canvas.draw()

	return lines[0], fig_num, fig


def update_deltas_plot(line, deltas, current_example_num, fig):
	'''
	Update the plot of the performance based on results thus far.
	'''
	delta = deltas[int(current_example_num / weight_update_interval)]
	
	line.set_xdata(range(int(current_example_num / weight_update_interval) + 1))
	ydata = list(line.get_ydata())
	ydata.append(delta)
	line.set_ydata(ydata)

	fig.canvas.draw()

	return line, deltas


def plot_all_deltas(fig_num, all_deltas, num_weight_updates):
	'''
	Set up the performance plot for the beginning of the simulation.
	'''
	time_steps = range(0, num_weight_updates)

	fig = plt.figure(fig_num, figsize = (12, 4))
	fig_num += 1

	for idx in xrange(all_deltas.shape[1]):
		plt.plot([], [])

	lines = plt.gca().lines

	plt.ylim(ymin=-1, ymax=1)
	plt.xticks(xrange(0, num_weight_updates + weight_update_interval, 100), \
			xrange(0, ((num_weight_updates + weight_update_interval) * weight_update_interval), 100))
	plt.legend()
	plt.grid(True)
	plt.title('Differences per synapse weight per weight update interval')
	
	fig.canvas.draw()

	return lines, fig_num, fig


def update_all_deltas_plot(lines, all_deltas, current_example_num, fig):
	'''
	Update the plot of the performance based on results thus far.
	'''
	deltas = all_deltas[int(current_example_num / weight_update_interval)]
	
	for idx, line in enumerate(lines):
		line.set_xdata(range(int(current_example_num / weight_update_interval) + 1))
		ydata = list(line.get_ydata())
		ydata.append(deltas[idx])
		line.set_ydata(ydata)

	fig.canvas.draw()

	return lines, deltas


def predict_label(assignments, spike_rates, accumulated_rates, spike_proportions):
	'''
	Given the label assignments of the excitatory layer and their spike rates over
	the past 'update_interval', get the ranking of each of the categories of input.
	'''
	output_numbers = {}

	for scheme in voting_schemes:
		summed_rates = [0] * 10
		num_assignments = [0] * 10

		if scheme == 'all':
			for i in xrange(10):
				num_assignments[i] = len(np.where(assignments == i)[0])
				if num_assignments[i] > 0:
					summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]

		elif scheme == 'most_spiked_patch':
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
					summed_rates[i] = np.sum(spike_rates[np.where(np.logical_and(assignments == i,
													 most_spiked_array))]) / float(np.sum(spike_rates[most_spiked_array]))

		elif scheme == 'most_spiked_location':
			most_spiked_array = np.array(np.zeros((conv_features, n_e)), dtype=bool)

			for n in xrange(n_e):
				# find the excitatory neuron which spiked the most in this input location
				most_spiked_array[np.argmax(spike_rates[:, n : n + 1]), n] = True

			# for each label
			for i in xrange(10):
				# get the number of label assignments of this type
				num_assignments[i] = len(np.where(assignments[most_spiked_array] == i)[0])

				if len(spike_rates[np.where(assignments[most_spiked_array] == i)]) > 0:
					# sum the spike rates of all excitatory neurons with this label, which fired the most in its patch
					summed_rates[i] = np.sum(spike_rates[np.where(np.logical_and(assignments == i,
													 most_spiked_array))]) / float(np.sum(spike_rates[most_spiked_array]))

		elif scheme == 'top_percent':
			top_percents = np.array(np.zeros((conv_features, n_e)), dtype=bool)
			top_percents[np.where(spike_rates > np.percentile(spike_rates, 100 - top_percent))] = True

			# for each label
			for i in xrange(10):
				# get the number of label assignments of this type
				num_assignments[i] = len(np.where(assignments[top_percents] == i)[0])

				if len(np.where(assignments[top_percents] == i)) > 0:
					# sum the spike rates of all excitatory neurons with this label, which fired the most in its patch
					summed_rates[i] = len(spike_rates[np.where(np.logical_and(assignments == i, top_percents))])

		elif scheme == 'confidence_weighting':
			for i in xrange(10):
				num_assignments[i] = np.count_nonzero(assignments == i)
				if num_assignments[i] > 0:
					summed_rates[i] = np.sum(spike_rates[assignments == i] * spike_proportions[(assignments == i).ravel(), i]) / num_assignments[i]
		
		output_numbers[scheme] = np.argsort(summed_rates)[::-1]
	
	return output_numbers


def assign_labels(result_monitor, input_numbers, accumulated_rates, accumulated_inputs):
	'''
	Based on the results from the previous 'update_interval', assign labels to the
	excitatory neurons.
	'''
	for j in xrange(10):
		num_assignments = len(np.where(input_numbers == j)[0])
		if num_assignments > 0:
			accumulated_inputs[j] += num_assignments
			accumulated_rates[:, j] = accumulated_rates[:, j] * accumulation_decay + np.ravel(np.sum(result_monitor[input_numbers == j], axis=0) / num_assignments)
	
	assignments = np.argmax(accumulated_rates, axis=1).reshape((conv_features, n_e))

	spike_proportions = np.divide(accumulated_rates, np.sum(accumulated_rates, axis=0))

	return assignments, accumulated_rates, spike_proportions


def build_network():
	global fig_num

	neuron_groups['e'] = b.NeuronGroup(n_e_total, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=scr_e, compile=True, freeze=True)
	neuron_groups['i'] = b.NeuronGroup(n_e_total, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i, compile=True, freeze=True)

	for name in population_names:
		print '...Creating neuron group:', name

		# get a subgroup of size 'n_e' from all exc
		neuron_groups[name + 'e'] = neuron_groups['e'].subgroup(conv_features * n_e)
		# get a subgroup of size 'n_i' from the inhibitory layer
		neuron_groups[name + 'i'] = neuron_groups['i'].subgroup(conv_features * n_e)

		# start the membrane potentials of these groups 40mV below their resting potentials
		neuron_groups[name + 'e'].v = v_rest_e - 40. * b.mV
		neuron_groups[name + 'i'].v = v_rest_i - 40. * b.mV

	print '...Creating recurrent connections'

	for name in population_names:
		# if we're in test mode / using some stored weights
		if test_mode:
			# load up adaptive threshold parameters
			if save_best_model:
				neuron_groups['e'].theta = np.load(os.path.join(best_weights_dir, '_'.join(['theta_A', ending +'_best.npy'])))
			else:
				neuron_groups['e'].theta = np.load(os.path.join(end_weights_dir, '_'.join(['theta_A', ending +'_end.npy'])))
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

			elif conn_type == 'ie' and not test_remove_inhibition:
				# create connection name (composed of population and connection types)
				conn_name = name + conn_type[0] + name + conn_type[1]
				# create a connection from the first group in conn_name with the second group
				connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
				
				# define the actual synaptic connections and strengths
				for feature in xrange(conv_features):
					if inhib_scheme in ['far', 'strengthen']:
						for other_feature in set(range(conv_features)) - set(neighbor_mapping[feature]):
							if inhib_scheme == 'far':
								for n in xrange(n_e):
									connections[conn_name][feature * n_e + n, other_feature * n_e + n] = max_inhib

							elif inhib_scheme == 'strengthen':
								if n_e == 1:
									x, y = feature // np.sqrt(n_e_total), feature % np.sqrt(n_e_total)
									x_, y_ = other_feature // np.sqrt(n_e_total), other_feature % np.sqrt(n_e_total)
								else:
									x, y = feature // np.sqrt(conv_features), feature % np.sqrt(conv_features)
									x_, y_ = other_feature // np.sqrt(conv_features), other_feature % np.sqrt(conv_features)

								for n in xrange(n_e):
									connections[conn_name][feature * n_e + n, other_feature * n_e + n] = \
													min(max_inhib, inhib_const * np.sqrt(euclidean([x, y], [x_, y_])))

					elif inhib_scheme == 'increasing':
						for other_feature in xrange(conv_features):
							if n_e == 1:
								x, y = feature // np.sqrt(n_e_total), feature % np.sqrt(n_e_total)
								x_, y_ = other_feature // np.sqrt(n_e_total), other_feature % np.sqrt(n_e_total)
							else:
								x, y = feature // np.sqrt(conv_features), feature % np.sqrt(conv_features)
								x_, y_ = other_feature // np.sqrt(conv_features), other_feature % np.sqrt(conv_features)

							if feature != other_feature:
								for n in xrange(n_e):
									connections[conn_name][feature * n_e + n, other_feature * n_e + n] = \
													min(max_inhib, inhib_const * np.sqrt(euclidean([x, y], [x_, y_])))

					elif inhib_scheme == 'constant':
						for other_feature in xrange(conv_features):
							if feature != other_feature:
								for n in xrange(n_e):
									connections[conn_name][feature * n_e + n, other_feature * n_e + n] = max_inhib
					else:
						raise Exception('Expecting one of "far", "increasing", "strengthen", or "constant" for argument "inhib_scheme".')

			elif conn_type == 'ee' and exc_stdp:
				# create connection name (composed of population and connection types)
				conn_name = name + conn_type[0] + name + conn_type[1]

				# get weights from file if we are in test mode
				if test_mode:
					if save_best_model:
						weight_matrix = np.load(os.path.join(best_weights_dir, '_'.join([conn_name, ending + '_best.npy'])))
					else:
						weight_matrix = np.load(os.path.join(end_weights_dir, '_'.join([conn_name, ending + '_end.npy'])))

				# create a connection from the first group in conn_name with the second group
				connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
				
				# define the actual synaptic connections and strengths
				for feature in xrange(conv_features):
					if excite_scheme == 'near':
						for other_feature in neighbor_mapping[feature]:
							for n in xrange(n_e):
								if test_mode:
									connections[conn_name][feature * n_e + n, other_feature * n_e + n] = \
																weight_matrix[feature * n_e + n, other_feature * n_e + n]
								else:
									connections[conn_name][feature * n_e + n, other_feature * n_e + n] = b.random() + 0.01 * 0.1 * wmax_exc

					elif excite_scheme == 'all':
						for other_feature in xrange(conv_features):
							if feature != other_feature:
								for n in xrange(n_e):
									if test_mode:
										connections[conn_name][feature * n_e + n, other_feature * n_e + n] = \
																	weight_matrix[feature * n_e + n, other_feature * n_e + n]
									else:
										connections[conn_name][feature * n_e + n, other_feature * n_e + n] = b.random() + 0.01 * 0.1 * wmax_exc								

					elif excite_scheme == 'decreasing':
						for other_feature in xrange(conv_features):
							if n_e == 1:
								x, y = feature // np.sqrt(n_e_total), feature % np.sqrt(n_e_total)
								x_, y_ = other_feature // np.sqrt(n_e_total), other_feature % np.sqrt(n_e_total)
							else:
								x, y = feature // np.sqrt(conv_features), feature % np.sqrt(conv_features)
								x_, y_ = other_feature // np.sqrt(conv_features), other_feature % np.sqrt(conv_features)

							if feature != other_feature:
								for n in xrange(n_e):
									# needs rethinking!
									connections[conn_name][feature * n_e + n, other_feature * n_e + n] = \
													min(wmax_exc, np.sqrt(euclidean([x, y], [x_, y_])))

					else:
						raise Exception('Expecting one of "near" or "decreasing" for argument "excite_scheme".')

		# if STDP from excitatory -> excitatory is on and this connection is excitatory -> excitatory
		if exc_stdp and 'ee' in recurrent_conn_names and not test_mode:
			stdp_methods[name + 'e' + name + 'e'] = b.STDP(connections[name + 'e' + name + 'e'], \
							eqs=eqs_stdp_ee, pre='pre = 1.; w -= 0.001 * post', post='w += 0.1 * pre; post = 1.', wmin=0., wmax=wmax_exc)

		print '...Creating monitors for:', name

		# spike rate monitors for excitatory and inhibitory neuron populations
		rate_monitors[name + 'e'] = b.PopulationRateMonitor(neuron_groups[name + 'e'], bin=(single_example_time + resting_time) / b.second)
		rate_monitors[name + 'i'] = b.PopulationRateMonitor(neuron_groups[name + 'i'], bin=(single_example_time + resting_time) / b.second)
		spike_counters[name + 'e'] = b.SpikeCounter(neuron_groups[name + 'e'])

		# record neuron population spikes if specified
		if record_spikes:
			spike_monitors[name + 'e'] = b.SpikeMonitor(neuron_groups[name + 'e'])
			spike_monitors[name + 'i'] = b.SpikeMonitor(neuron_groups[name + 'i'])

	if record_spikes and do_plot:
		b.figure(fig_num, figsize=(8, 6))
		
		fig_num += 1
		
		b.ion()
		b.subplot(211)
		b.raster_plot(spike_monitors['Ae'], refresh=1000 * b.ms, showlast=1000 * b.ms, title='Excitatory spikes per neuron')
		b.subplot(212)
		b.raster_plot(spike_monitors['Ai'], refresh=1000 * b.ms, showlast=1000 * b.ms, title='Inhibitory spikes per neuron')
		b.tight_layout()

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

	if exc_stdp:
		weight['ee_recurr'] = (conv_features - 1) * 0.15 * wmax_exc

	# creating Poission spike train from input image (784 vector, 28x28 image)
	for name in input_population_names:
		input_groups[name + 'e'] = b.PoissonGroup(n_input, 0)
		rate_monitors[name + 'e'] = b.PopulationRateMonitor(input_groups[name + 'e'], bin=(single_example_time + resting_time) / b.second)

	# creating connections from input Poisson spike train to excitatory neuron population(s)
	for name in input_connection_names:
		print '\n...Creating connections between', name[0], 'and', name[1]
		
		# for each of the input connection types (in this case, excitatory -> excitatory)
		for conn_type in input_conn_names:
			# saved connection name
			conn_name = name[0] + conn_type[0] + name[1] + conn_type[1]

			# get weight matrix depending on training or test phase
			if test_mode:
				if save_best_model:
					weight_matrix = np.load(os.path.join(best_weights_dir, '_'.join([conn_name, ending + '_best.npy'])))
				else:
					weight_matrix = np.load(os.path.join(end_weights_dir, '_'.join([conn_name, ending + '_end.npy'])))

			# create connections from the windows of the input group to the neuron population
			input_connections[conn_name] = b.Connection(input_groups['Xe'], neuron_groups[name[1] + conn_type[1]], \
									structure='sparse', state='g' + conn_type[0], delay=True, max_delay=delay[conn_type][1])
			
			if test_mode:
				for feature in xrange(conv_features):
					for n in xrange(n_e):
						for idx in xrange(conv_size ** 2):
							input_connections[conn_name][convolution_locations[n][idx], feature * n_e + n] = \
															weight_matrix[convolution_locations[n][idx], feature * n_e + n]
			else:
				for feature in xrange(conv_features):
					for n in xrange(n_e):
						for idx in xrange(conv_size ** 2):
							input_connections[conn_name][convolution_locations[n][idx], feature * n_e + n] = (b.random() + 0.01) * 0.3

			if test_mode:
				if do_plot:
					plot_2d_input_weights()
					fig_num += 1	

		# if excitatory -> excitatory STDP is specified, add it here (input to excitatory populations)
		if not test_mode:
			print '...Creating STDP for connection', name
			
			# STDP connection name
			conn_name = name[0] + conn_type[0] + name[1] + conn_type[1]
			# create the STDP object
			stdp_methods[conn_name] = b.STDP(input_connections[conn_name], eqs=eqs_stdp_ee, \
							pre=eqs_stdp_pre_ee, post=eqs_stdp_post_ee, wmin=0., wmax=wmax_ee)

	print '\n'


def run_train():
	global fig_num, input_intensity, previous_spike_count, rates, assignments, clusters, cluster_assignments, \
				kmeans, kmeans_assignments, simple_clusters, simple_cluster_assignments, index_matrix, accumulated_rates, \
				accumulated_inputs, spike_proportions

	if do_plot:
		assignments_image = plot_assignments(assignments)
		fig_num += 1

		if not test_mode:
			input_weight_monitor, fig_weights = plot_2d_input_weights()
			fig_num += 1
		
			if connectivity != 'none':
				patch_weight_monitor, fig2_weights = plot_patch_weights()
				fig_num += 1

	# set up performance recording and plotting
	num_evaluations = int(num_examples / update_interval) + 1
	performances = { voting_scheme : np.zeros(num_evaluations) for voting_scheme in voting_schemes }
	num_weight_updates = int(num_examples / weight_update_interval)
	all_deltas = np.zeros((num_weight_updates, (conv_size ** 2) * n_e_total))
	deltas = np.zeros(num_weight_updates)

	if do_plot:
		performance_monitor, fig_num, fig_performance = plot_performance(fig_num, performances, num_evaluations)
		line, fig_num, deltas_figure = plot_deltas(fig_num, deltas, num_weight_updates)
		if plot_all_deltas:
			lines, fig_num, all_deltas_figure = plot_all_deltas(fig_num, all_deltas, num_weight_updates)
	else:
		performances, wrong_idxs, wrong_labels = get_current_performance(performances, 0)

	# initialize network
	j = 0
	num_retries = 0
	b.run(0)

	if save_best_model:
		best_performance = 0.0

	# start recording time
	start_time = timeit.default_timer()

	last_weights = input_connections['XeAe'][:].todense()

	if exc_stdp:
		last_ee_weights = connections['AeAe'][:].todense()

	while j < num_examples:
		# get the firing rates of the next input example
		if noise:
			rates = (data['x'][j % data_size, :, :] / 8.0) * input_intensity + np.random.normal(loc=32.0 * noise_const, scale=1.0, size=(28, 28))
		else:
			rates = (data['x'][j % data_size, :, :] / 8.0) * input_intensity
		
		# sets the input firing rates
		input_groups['Xe'].rate = rates.reshape(n_input)

		# get weights before running the network for a single iteration
		previous_weights = input_connections['XeAe'][:].todense()
		
		# run the network for a single example time
		b.run(single_example_time)

		# get difference between weights from before and after running a single iteration
		new_weights = input_connections['XeAe'][:].todense() - previous_weights

		if exc_stdp:
			if j == 0:
				exc_weights_image = plt.matshow(connections['AeAe'][:].todense().T, cmap='binary', vmin=0, vmax=wmax_exc)
				plt.colorbar()
				plt.title('Excitatory to excitatory weights')
			else:
				exc_weights_image.set_array(connections['AeAe'][:].todense().T)

		if exc_stdp:
			last_ee_weights = connections['AeAe'][:].todense()

		# add weight gain / loss to neighboring excitatory neuron weights by a constant factor of the weight gain / loss
		if inhib_scheme == 'strengthen':
			for feature in xrange(conv_features):
				for other_feature in neighbor_mapping[feature]:
					input_connections['XeAe'][:, feature] += strengthen_const * new_weights[:, other_feature]

		# get new neuron label assignments every 'update_interval'
		if j % update_interval == 0 and j > 0:
			assignments, accumulated_rates, spike_proportions = assign_labels(result_monitor, input_numbers[j - update_interval : j], accumulated_rates, accumulated_inputs)

		# get count of spikes over the past iteration
		current_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e)) - previous_spike_count
		previous_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e))

		# make sure synapse weights don't grow too large
		normalize_weights()

		if not j % weight_update_interval == 0 and save_weights:
			save_connections(weights_dir, connections, input_connections, ending, j)
			save_theta(weights_dir, population_names, neuron_groups, ending, j)

			np.save(os.path.join(misc_dir, '_'.join(['assignments', ending, str(j)])), assignments)
			np.save(os.path.join(misc_dir, '_'.join(['accumulated_rates', ending, str(j)])), accumulated_rates)
			np.save(os.path.join(misc_dir, '_'.join(['spike_proportions', ending, str(j)])), spike_proportions)
			
		if j % weight_update_interval == 0:
			deltas[j / weight_update_interval] = np.sum(np.abs((input_connections['XeAe'][:].todense() - last_weights)))
			if plot_all_deltas:
				all_deltas[j / weight_update_interval, :] = np.ravel(input_connections['XeAe'][:].todense() - last_weights)
			last_weights = input_connections['XeAe'][:].todense()

			# pickling performance recording and iteration number
			p.dump((j, deltas), open(os.path.join(deltas_dir, ending + '.p'), 'wb'))

		# update weights every 'weight_update_interval'
		if j % weight_update_interval == 0 and do_plot:
			update_2d_input_weights(input_weight_monitor, fig_weights)
			# save_connections(weights_dir, connections, input_connections, ending, j)
			if connectivity != 'none':
				update_patch_weights(patch_weight_monitor, fig2_weights)
		
		if do_plot:
			assignments_image = update_assignments_plot(assignments, assignments_image)

		# if the neurons in the network didn't spike more than four times
		if np.sum(current_spike_count) < 5 and num_retries < 3:
			# increase the intensity of input
			input_intensity += 2
			num_retries += 1
			
			# set all network firing rates to zero
			for name in input_population_names:
				input_groups[name + 'e'].rate = 0

			# let the network relax back to equilibrium
			if homeostasis and not reset_state_vars:
				b.run(resting_time)
			else:
				for neuron_group in neuron_groups:
					neuron_groups[neuron_group].v = v_reset_e
					neuron_groups[neuron_group].ge = 0
					neuron_groups[neuron_group].gi = 0

		# otherwise, record results and continue simulation
		else:			
			num_retries = 0

			# record the current number of spikes
			result_monitor[j % update_interval, :] = current_spike_count
			
			# get true label of last input example
			input_numbers[j] = data['y'][j % data_size][0]
			
			# get the output classifications of the network
			for scheme, outputs in predict_label(assignments, result_monitor[j % update_interval, :], accumulated_rates, spike_proportions).items():
				output_numbers[scheme][j, :] = outputs

			# print progress
			if j % print_progress_interval == 0 and j > 0:
				print 'runs done:', j, 'of', int(num_examples), '(time taken for past', print_progress_interval, 'runs:', str(timeit.default_timer() - start_time) + ')'
				start_time = timeit.default_timer()

			if j % weight_update_interval == 0 and do_plot:
				update_deltas_plot(line, deltas, j, deltas_figure)
				if plot_all_deltas:
					update_all_deltas_plot(lines, all_deltas, j, all_deltas_figure)
			
			# plot performance if appropriate
			if (j % update_interval == 0 or j == num_examples - 1) and j > 0:
				if do_plot:
					# updating the performance plot
					perf_plot, performances, wrong_idxs, wrong_labels = update_performance_plot(performance_monitor, performances, j, fig_performance)
				else:
					performances, wrong_idxs, wrong_labels = get_current_performance(performances, j)

				# pickling performance recording and iteration number
				p.dump((j, performances), open(os.path.join(performance_dir, ending + '.p'), 'wb'))

				# Save the best model's weights and theta parameters (if so specified)
				if save_best_model:
					for performance in performances:
						if performances[performance][int(j / float(update_interval))] > best_performance:
							print '\n', 'Best model thus far! Voting scheme:', performance, '\n'

							best_performance = performances[performance][int(j / float(update_interval))]
							save_connections(best_weights_dir, connections, input_connections, ending, 'best')
							save_theta(best_weights_dir, population_names, neuron_groups, ending, 'best')

							np.save(os.path.join(best_misc_dir, '_'.join(['assignments', ending, 'best'])), assignments)
							np.save(os.path.join(best_misc_dir, '_'.join(['accumulated_rates', ending, 'best'])), accumulated_rates)
							np.save(os.path.join(best_misc_dir, '_'.join(['spike_proportions', ending, 'best'])), spike_proportions)
							

				# Print out performance progress intermittently
				for performance in performances:
					print '\nClassification performance (' + performance + ')', performances[performance][1:int(j / float(update_interval)) + 1], \
								'\nAverage performance:', sum(performances[performance][1:int(j / float(update_interval)) + 1]) / \
									float(len(performances[performance][1:int(j / float(update_interval)) + 1])), \
									'\nBest performance:', max(performances[performance][1:int(j / float(update_interval)) + 1]), '\n'
						
			# set input firing rates back to zero
			for name in input_population_names:
				input_groups[name + 'e'].rate = 0
			
			# run the network for 'resting_time' to relax back to rest potentials
			if homeostasis and not reset_state_vars:
				b.run(resting_time)
			else:
				for neuron_group in neuron_groups:
					neuron_groups[neuron_group].v = v_reset_e
					neuron_groups[neuron_group].ge = 0
					neuron_groups[neuron_group].gi = 0

			# bookkeeping
			input_intensity = start_input_intensity
			j += 1

	# ensure weights don't grow without bound
	normalize_weights()

	print '\n'


def run_relabeling():
	global fig_num, input_intensity, previous_spike_count, rates, assignments, clusters, cluster_assignments, \
				kmeans, kmeans_assignments, simple_clusters, simple_cluster_assignments, index_matrix, accumulated_rates, \
				accumulated_inputs, spike_proportions

	if do_plot:
		assignments_image = plot_assignments(assignments)
		fig_num += 1

	# initialize network
	j = 0
	num_retries = 0
	b.run(0)

	# start recording time
	start_time = timeit.default_timer()

	while j < num_examples:
		# get the firing rates of the next input example
		if noise:
			rates = (data['x'][j % data_size, :, :] / 8.0) * input_intensity + \
					np.random.normal(loc=32.0 * noise_const, scale=1.0, size=(28, 28))
		else:
			rates = (data['x'][j % data_size, :, :] / 8.0) * input_intensity
		
		# sets the input firing rates
		input_groups['Xe'].rate = rates.reshape(n_input)

		# run the network for a single example time
		b.run(single_example_time)

		if do_plot and exc_stdp and j == 0:
			exc_weights_image = plt.matshow(connections['AeAe'][:].todense().T, cmap='binary', vmin=0, vmax=wmax_exc)
			plt.colorbar()
			plt.title('Excitatory to excitatory weights')

		# get count of spikes over the past iteration
		current_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e)) - previous_spike_count
		previous_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e))

		# if the neurons in the network didn't spike more than four times
		if np.sum(current_spike_count) < 5 and num_retries < 3:
			# increase the intensity of input
			input_intensity += 2
			num_retries += 1
			
			# set all network firing rates to zero
			for name in input_population_names:
				input_groups[name + 'e'].rate = 0

			# let the network relax back to equilibrium
			if homeostasis and not reset_state_vars:
				b.run(resting_time)
			else:
				for neuron_group in neuron_groups:
					neuron_groups[neuron_group].v = v_reset_e
					neuron_groups[neuron_group].ge = 0
					neuron_groups[neuron_group].gi = 0

		# otherwise, record results and continue simulation
		else:			
			num_retries = 0

			# record the current number of spikes
			result_monitor[j % update_interval, :] = current_spike_count
			
			# get true label of the past input example
			input_numbers[j] = data['y'][j % data_size][0]
			
			# get the output classifications of the network
			for scheme, outputs in predict_label(assignments, result_monitor[j % update_interval, :], accumulated_rates, spike_proportions).items():
				output_numbers[scheme][j, :] = outputs

			# print progress
			if j % print_progress_interval == 0 and j > 0:
				print 'runs done:', j, 'of', int(num_examples), '(time taken for past', print_progress_interval, 'runs:', str(timeit.default_timer() - start_time) + ')'
				start_time = timeit.default_timer()
						
			# set input firing rates back to zero
			for name in input_population_names:
				input_groups[name + 'e'].rate = 0
			
			# run the network for 'resting_time' to relax back to rest potentials
			if homeostasis and not reset_state_vars:
				b.run(resting_time)
			else:
				for neuron_group in neuron_groups:
					neuron_groups[neuron_group].v = v_reset_e
					neuron_groups[neuron_group].ge = 0
					neuron_groups[neuron_group].gi = 0

			# bookkeeping
			input_intensity = start_input_intensity
			j += 1

	# get new neuron label assignments every 'update_interval'
	assignments, accumulated_rates, spike_proportions = assign_labels(result_monitor, input_numbers[j - update_interval : j], accumulated_rates, accumulated_inputs)

	np.save(os.path.join(best_misc_dir, '_'.join(['assignments', ending, 'best'])), assignments)
	np.save(os.path.join(best_misc_dir, '_'.join(['accumulated_rates', ending, 'best'])), accumulated_rates)
	np.save(os.path.join(best_misc_dir, '_'.join(['spike_proportions', ending, 'best'])), spike_proportions)

	print '\n'


def run_test():
	global fig_num, input_intensity, previous_spike_count, rates, assignments, clusters, cluster_assignments, \
				kmeans, kmeans_assignments, simple_clusters, simple_cluster_assignments, index_matrix, accumulated_rates, \
				accumulated_inputs, spike_proportions

	if do_plot:
		assignments_image = plot_assignments(assignments)
		fig_num += 1

	# set up performance recording and plotting
	num_evaluations = int(num_examples / update_interval) + 1
	performances = { voting_scheme : np.zeros(num_evaluations) for voting_scheme in voting_schemes }
	num_weight_updates = int(num_examples / weight_update_interval)
	all_deltas = np.zeros((num_weight_updates, (conv_size ** 2) * n_e_total))
	deltas = np.zeros(num_weight_updates)

	# initialize network
	j = 0
	num_retries = 0
	b.run(0)

	# start recording time
	start_time = timeit.default_timer()

	while j < num_examples:
		# get the firing rates of the next input example
		if noise:
			rates = (data['x'][j % data_size, :, :] / 8.0) * input_intensity + np.random.normal(loc=32.0 * noise_const, scale=1.0, size=(28, 28))
		else:
			rates = (data['x'][j % data_size, :, :] / 8.0) * input_intensity
		
		# sets the input firing rates
		input_groups['Xe'].rate = rates.reshape(n_input)

		# run the network for a single example time
		b.run(single_example_time)

		if do_plot and exc_stdp and j == 0:
				exc_weights_image = plt.matshow(connections['AeAe'][:].todense().T, cmap='binary', vmin=0, vmax=wmax_exc)
				plt.colorbar()
				plt.title('Excitatory to excitatory weights')

		# get count of spikes over the past iteration
		current_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e)) - previous_spike_count
		previous_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e))

		# if the neurons in the network didn't spike more than four times
		if np.sum(current_spike_count) < 5 and num_retries < 3:
			# increase the intensity of input
			input_intensity += 2
			num_retries += 1
			
			# set all network firing rates to zero
			for name in input_population_names:
				input_groups[name + 'e'].rate = 0

			# let the network relax back to equilibrium
			if homeostasis and not reset_state_vars:
				b.run(resting_time)
			else:
				for neuron_group in neuron_groups:
					neuron_groups[neuron_group].v = v_reset_e
					neuron_groups[neuron_group].ge = 0
					neuron_groups[neuron_group].gi = 0

		# otherwise, record results and continue simulation
		else:			
			num_retries = 0

			# record the current number of spikes
			result_monitor[j % update_interval, :] = current_spike_count
			
			# get true label of the past input example
			input_numbers[j] = data['y'][j % data_size][0]
			
			# get the output classifications of the network
			for scheme, outputs in predict_label(assignments, result_monitor[j % update_interval, :], accumulated_rates, spike_proportions).items():
				output_numbers[scheme][j, :] = outputs

			# print progress
			if j % print_progress_interval == 0 and j > 0:
				print 'runs done:', j, 'of', int(num_examples), '(time taken for past', print_progress_interval, 'runs:', str(timeit.default_timer() - start_time) + ')'
				start_time = timeit.default_timer()
						
			# set input firing rates back to zero
			for name in input_population_names:
				input_groups[name + 'e'].rate = 0
			
			# run the network for 'resting_time' to relax back to rest potentials
			if homeostasis and not reset_state_vars:
				b.run(resting_time)
			else:
				for neuron_group in neuron_groups:
					neuron_groups[neuron_group].v = v_reset_e
					neuron_groups[neuron_group].ge = 0
					neuron_groups[neuron_group].gi = 0

			# bookkeeping
			input_intensity = start_input_intensity
			j += 1

	print '\n'


def save_results():
	'''
	Save results of simulation (train or test)
	'''
	print '...Saving results'

	if not test_mode:
		save_connections(end_weights_dir, connections, input_connections, ending, 'end')
		save_theta(end_weights_dir, population_names, neuron_groups, ending, 'end')

		np.save(os.path.join(end_misc_dir, '_'.join(['assignments', ending, 'end'])), assignments)
		np.save(os.path.join(end_misc_dir, '_'.join(['accumulated_rates', ending, 'end'])), accumulated_rates)
		np.save(os.path.join(end_misc_dir, '_'.join(['spike_proportions', ending, 'end'])), spike_proportions)
	else:
		np.save(os.path.join(activity_dir, '_'.join(['results', str(num_examples), ending])), result_monitor)
		np.save(os.path.join(activity_dir, '_'.join(['input_numbers', str(num_examples), ending])), input_numbers)

	print '\n'


def evaluate_results():
	'''
	Evalute the network using the various voting schemes in test mode
	'''
	global update_interval

	test_results = {}
	for scheme in voting_schemes:
		test_results[scheme] = np.zeros((10, num_examples))

	print '\n...Calculating accuracy per voting scheme'

	# for idx in xrange(end_time_testing - end_time_training):
	for idx in xrange(num_examples):
		label_rankings = predict_label(assignments, result_monitor[idx, :], accumulated_rates, spike_proportions)
		for scheme in voting_schemes:
			test_results[scheme][:, idx] = label_rankings[scheme]

	differences = { scheme : test_results[scheme][0, :] - input_numbers for scheme in voting_schemes }
	correct = { scheme : len(np.where(differences[scheme] == 0)[0]) for scheme in voting_schemes }
	incorrect = { scheme : len(np.where(differences[scheme] != 0)[0]) for scheme in voting_schemes }
	accuracies = { scheme : correct[scheme] / float(num_examples) * 100 for scheme in voting_schemes }

	conf_matrix = confusion_matrix(test_results[scheme][0, :], input_numbers)

	print '\nConfusion matrix:\n\n', conf_matrix

	if do_plot:
		fig = plt.figure(fig_num, figsize=(5, 5))
		im = plt.matshow(conf_matrix)
		plt.colorbar(im)
		plt.title('Confusion matrix')
		fig.canvas.draw()

		time.sleep(100)

	for scheme in voting_schemes:
		print '\n-', scheme, 'accuracy:', accuracies[scheme]

	results = pd.DataFrame([ [ ending ] + accuracies.values() ], columns=[ 'Model' ] + accuracies.keys())
	if not 'results.csv' in os.listdir(results_path):
		results.to_csv(os.path.join(results_path, 'results.csv'), index=False)
	else:
		all_results = pd.read_csv(os.path.join(results_path, 'results.csv'))
		all_results = pd.concat([all_results, results], ignore_index=True)
		all_results.to_csv(os.path.join(results_path, 'results.csv'), index=False)

	print '\n'


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', default='train', help='Network operating mode: "train" mode learns the synaptic weights of the network, and \
														"test" mode holds the weights fixed and evaluates classification accuracy on the test dataset.')
	parser.add_argument('--connectivity', default='none', help='Between-patch connectivity: choose from "none", "pairs", "linear", and "full".')
	parser.add_argument('--conv_size', type=int, default=28, help='Side length of the square convolution \
																			window used by the input -> excitatory layer of the network.')
	parser.add_argument('--conv_stride', type=int, default=0, help='Horizontal, vertical stride \
														of the convolution window used by the input -> excitatory layer of the network.')
	parser.add_argument('--conv_features', type=int, default=100, help='Number of excitatory convolutional features / filters / patches used in the network.')
	parser.add_argument('--lattice_structure', default='4', help='The lattice neighborhood to which connected \
															patches project their connections: one of "none", "4", "8", or "all".')
	parser.add_argument('--top_percent', type=int, default=10, help='The percentage of neurons which are allowed \
																to cast "votes" in the "top_percent" labeling scheme.')
	parser.add_argument('--do_plot', type=str, default='False', help='Whether or not to display plots during network \
																	training / testing. Defaults to False, as this makes the network operation \
																	speedier, and possible to run on HPC resources.')
	parser.add_argument('--sort_euclidean', type=str, default='False', help='When plotting reshaped \
														input -> excitatory weights, whether to plot each row (corresponding to locations \
																in the input) sorted by Euclidean distance from the 0 matrix.')
	parser.add_argument('--num_train', type=int, default=10000, help='The number of examples for which to train the network on.')
	parser.add_argument('--num_test', type=int, default=10000, help='The number of examples for which to test the network on.')
	parser.add_argument('--random_seed', type=int, default=42, help='The random seed (any integer) from which to generate random numbers.')
	parser.add_argument('--reduced_dataset', type=str, default='False', help='Whether or not to a reduced dataset.')
	parser.add_argument('--classes', type=int, default=range(10), nargs='+', help='List of classes to use in reduced dataset.')
	parser.add_argument('--examples_per_class', type=int, default=1000, help='Number of examples per class to use in reduced dataset.')
	parser.add_argument('--neighborhood', type=str, default='8', help='The structure of neighborhood not to inhibit on firing. One of "4", "8".')
	parser.add_argument('--inhib_scheme', type=str, default='strengthen', help='The scheme with which one excitatory neuron\'s firing activity \
																			inhibits others. One of "far", "increasing".')
	parser.add_argument('--inhib_const', type=float, default=2.5, help='A constant which controls how quickly inhibition strengthens \
																			between two neurons as their relative distance increases.')
	parser.add_argument('--strengthen_const', type=float, default=0.1, help='A constant which controls how much weights learned in \
																		one iteration are added to neighboring excitatory neurons\' weights.')
	parser.add_argument('--noise', type=str, default='False', help='Whether or not to add Gaussian noise to input images.')
	parser.add_argument('--noise_const', type=float, default=0.1, help='A constant which gives the mean of the Gaussian noise \
																			added to the input images (fraction of maximum firing rate.')
	parser.add_argument('--save_weights', type=str, default='False', help='Whether or not to save the weights of the model every `weight_update_interval`.')
	parser.add_argument('--homeostasis', type=str, default='True', help='Whether or not to use the homeostasis mechanism.')
	parser.add_argument('--weight_update_interval', type=int, default=10, help='How often to update the plot of network filter weights.')
	parser.add_argument('--save_best_model', type=str, default='True', help='Whether to save the current best version of the model.')
	parser.add_argument('--load_best_model', type=str, default='True', help='Whether to load the best version of the model from training.')
	parser.add_argument('--update_interval', type=int, default=250, help='How often to update neuron labels and classify new inputs.')
	parser.add_argument('--accumulate_votes', type=str, default='True', help='Whether to base neuron votes on all past spikes \
																					or only on the spikes from the last "update_interval"')
	parser.add_argument('--accumulation_decay', type=float, default=0.75, help='How much to decay the influence of past spikes \
																						on the labeling of the excitatory neurons.')
	parser.add_argument('--plot_all_deltas', type=str, default='False', help='Whether or not to plot weight changes for all \
																						synapses from input to excitatory layer.')
	parser.add_argument('--test_remove_inhibition', type=str, default='False', help='Whether or not to remove lateral inhibition during the test phase.')
	parser.add_argument('--exc_stdp', type=str, default='False', help='Whether to use STDP synapses between neurons in the excitatory layer.')
	parser.add_argument('--excite_scheme', type=str, default='all', help='The scheme with which one excitatory neuron excites other excitatory neurons.')
	parser.add_argument('--wmax_exc', type=float, default=10.0, help='The max weight on synapses between any two connected excitatory neurons.')
	parser.add_argument('--max_inhib', type=float, default=17.4, help='The maximum synapse weight for inhibitory to excitatory connections.')
	parser.add_argument('--reset_state_vars', type=str, default='False', help='Whether to reset neuron / synapse state variables or run a "reset" period.')

	# parse arguments and place them in local scope
	args = parser.parse_args()
	args = vars(args)
	locals().update(args)

	print '\nOptional argument values:'
	for key, value in args.items():
		print '-', key, ':', value

	print '\n'

	for var in [ 'do_plot', 'sort_euclidean', 'reduced_dataset', 'noise', 'plot_all_deltas', 'exc_stdp', 'reset_state_vars', \
					'save_weights', 'homeostasis', 'save_best_model', 'accumulate_votes', 'test_remove_inhibition', 'load_best_model' ]:
		if locals()[var] == 'True':
			locals()[var] = True
		elif locals()[var] == 'False':
			locals()[var] = False
		else:
			raise Exception('Expecting True or False-valued command line argument "' + var + '".')

	# test or training mode
	test_mode = mode == 'test'
	relabel_mode = mode == 'relabel'
	train_mode = mode == 'train'

	if test_mode:
		num_examples = num_test
	elif train_mode or relabel_mode:
		num_examples = num_train

	if reduced_dataset:
		data_size = len(classes) * examples_per_class
	elif test_mode:
		data_size = 10000
	elif train_mode or relabel_mode:
		data_size = 60000

	# set brian global preferences
	b.set_global_preferences(defaultclock = b.Clock(dt=0.5*b.ms), useweave = True, gcc_options = ['-ffast-math -march=native'], usecodegen = True,
		usecodegenweave = True, usecodegenstateupdate = True, usecodegenthreshold = False, usenewpropagate = True, usecstdp = True, openmp = False,
		magic_useframes = False, useweave_linear_diffeq = True)

	# for reproducibility's sake
	np.random.seed(random_seed)

	start = timeit.default_timer()
	data = get_labeled_data(os.path.join(MNIST_data_path, 'testing' if test_mode else 'training'), 
												not test_mode, reduced_dataset, classes, examples_per_class)
	
	print 'Time needed to load data:', timeit.default_timer() - start

	# set parameters for simulation based on train / test mode
	record_spikes = True

	# number of inputs to the network
	n_input = 784
	n_input_sqrt = int(math.sqrt(n_input))

	# number of neurons parameters
	if conv_size == 28 and conv_stride == 0:
		n_e = 1
	else:
		n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
	
	n_e_total = n_e * conv_features
	n_e_sqrt = int(math.sqrt(n_e))
	n_i = n_e
	features_sqrt = int(math.ceil(math.sqrt(conv_features)))

	# time (in seconds) per data example presentation and rest period in between
	single_example_time = 0.35 * b.second
	resting_time = 0.15 * b.second

	# set the update interval
	if test_mode:
		update_interval = num_examples

	# weight updates and progress printing intervals
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
	if conv_size == 28 and conv_stride == 0:
		weight['ee_input'] = (conv_size ** 2) * 0.15
	else:
		weight['ee_input'] = (conv_size ** 2) * 0.1625

	delay['ee_input'] = (0 * b.ms, 10 * b.ms)
	delay['ei_input'] = (0 * b.ms, 5 * b.ms)
	input_intensity = start_input_intensity = 2.0

	# time constants, learning rates, max weights, weight dependence, etc.
	tc_pre_ee, tc_post_ee = 20 * b.ms, 20 * b.ms
	nu_ee_pre, nu_ee_post = 0.0001, 0.01
	nu_AeAe_pre, nu_Ae_Ae_post = 0.1, 0.5
	wmax_ee = 1.0
	exp_ee_post = exp_ee_pre = 0.2
	w_mu_pre, w_mu_post = 0.2, 0.2

	# setting up differential equations (depending on train / test mode)
	if test_mode or not homeostasis:
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
			
	if test_mode or not homeostasis:
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

	# STDP synaptic traces
	eqs_stdp_ee = '''
				dpre/dt = -pre / tc_pre_ee : 1.0
				dpost/dt = -post / tc_post_ee : 1.0
				'''

	eqs_stdp_AeAe = '''
				dpre/dt = -pre / tc_pre_ee : 1.0
				dpost/dt = -post / tc_post_ee : 1.0
				'''

	# STDP rule (post-pre, no weight dependence)
	eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post'
	eqs_stdp_post_ee = 'w += nu_ee_post * pre; post = 1.'

	eqs_stdp_pre_AeAe = 'pre += 1.; w -= nu_AeAe_pre * post'
	eqs_stdp_post_AeAe = 'w += nu_AeAe_post * pre; post += 1.'

	print '\n'

	# set ending of filename saves
	ending = '_'.join([ connectivity, str(conv_size), str(conv_stride), str(conv_features), str(n_e), str(reduced_dataset), \
						'_'.join([ str(class_) for class_ in classes ]), str(examples_per_class), neighborhood, inhib_scheme, \
			str(inhib_const), str(strengthen_const), str(num_train), str(random_seed), str(accumulate_votes), str(accumulation_decay) ])

	b.ion()
	fig_num = 1
	
	# creating dictionaries for various objects
	neuron_groups, input_groups, connections, input_connections, stdp_methods, \
		rate_monitors, spike_monitors, spike_counters, output_numbers = {}, {}, {}, {}, {}, {}, {}, {}, {}

	# creating convolution locations inside the input image
	convolution_locations = {}
	for n in xrange(n_e):
		convolution_locations[n] = [ ((n % n_e_sqrt) * conv_stride + (n // n_e_sqrt) * n_input_sqrt * \
						conv_stride) + (x * n_input_sqrt) + y for y in xrange(conv_size) for x in xrange(conv_size) ]

	# instantiating neuron "vote" monitor
	result_monitor = np.zeros((update_interval, conv_features, n_e))

	neighbor_mapping = {}
	for feature in xrange(conv_features):
		neighbor_mapping[feature] = range(conv_features)
		for other_feature in xrange(conv_features):
			if n_e == 1:
				x, y = feature // np.sqrt(n_e_total), feature % np.sqrt(n_e_total)
				x_, y_ = other_feature // np.sqrt(n_e_total), other_feature % np.sqrt(n_e_total)
			else:
				x, y = feature // np.sqrt(conv_features), feature % np.sqrt(conv_features)
				x_, y_ = other_feature // np.sqrt(conv_features), other_feature % np.sqrt(conv_features)
				

			if inhib_scheme == 'far':
				if neighborhood == '8':
					if feature != other_feature and euclidean([x, y], [x_, y_]) >= 2.0:
						neighbor_mapping[feature].remove(other_feature)
				elif neighborhood == '4':
					if feature != other_feature and euclidean([x, y], [x_, y_]) > 1.0:
						neighbor_mapping[feature].remove(other_feature)
				else:
					raise Exception('Expecting one of "8" or "4" for argument "neighborhood".')

			elif inhib_scheme in [ 'increasing', 'constant' ]:
				pass

			elif inhib_scheme == 'strengthen':
				if neighborhood == '8':
					if feature != other_feature and euclidean([x, y], [x_, y_]) >= 2.0:
						neighbor_mapping[feature].remove(other_feature)
				elif neighborhood == '4':
					if feature != other_feature and euclidean([x, y], [x_, y_]) > 1.0:
						neighbor_mapping[feature].remove(other_feature)
				else:
					raise Exception('Expecting one of "8" or "4" for argument "neighborhood".')

			else:
				raise Exception('Expecting one of "far", "increasing", or "strengthen" for argument "inhib_scheme".')

	# build the spiking neural network
	build_network()

	# bookkeeping variables
	previous_spike_count = np.zeros((conv_features, n_e))
	input_numbers = np.zeros(num_examples)
	rates = np.zeros((n_input_sqrt, n_input_sqrt))

	if test_mode:
		assignments = np.load(os.path.join(end_misc_dir, '_'.join(['assignments', ending, 'relabel.npy'])))
		accumulated_rates = np.load(os.path.join(end_misc_dir, '_'.join(['accumulated_rates', ending, 'relabel.npy'])))
		spike_proportions = np.load(os.path.join(end_misc_dir, '_'.join(['spike_proportions', ending, 'relabel.npy'])))
	else:
		assignments = -1 * np.ones((conv_features, n_e))

	voting_schemes = ['all', 'most_spiked_patch', 'top_percent', 'most_spiked_location', 'confidence_weighting']

	for scheme in voting_schemes:
		output_numbers[scheme] = np.zeros((num_examples, 10))

	if not test_mode:
		accumulated_rates = np.zeros((conv_features * n_e, 10))
		accumulated_inputs = np.zeros(10)
		spike_proportions = np.zeros((conv_features * n_e, 10))
	
	# run the simulation of the network
	if test_mode:
		run_test()
	elif relabel_mode:
		run_relabeling()
	else:
		run_train()

	# save and plot results
	save_results()

	# evaluate results
	if test_mode:
		evaluate_results()

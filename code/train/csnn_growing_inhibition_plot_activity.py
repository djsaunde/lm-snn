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

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix
from struct import unpack
from brian import *

from util import *

np.set_printoptions(threshold=np.nan, linewidth=200)

# only show log messages of level ERROR or higher
b.log_level_error()

# set these appropriate to your directory structure
top_level_path = os.path.join('..', '..')
MNIST_data_path = os.path.join(top_level_path, 'data')
model_name = 'csnn_growing_inhibition'
results_path = os.path.join(top_level_path, 'results', model_name)

performance_dir = os.path.join(top_level_path, 'performance', model_name)
activity_dir = os.path.join(top_level_path, 'activity', model_name)
deltas_dir = os.path.join(top_level_path, 'deltas', model_name)
spikes_dir = os.path.join(top_level_path, 'spikes', model_name)

weights_dir = os.path.join(top_level_path, 'weights', model_name)
best_weights_dir = os.path.join(weights_dir, 'best')
end_weights_dir = os.path.join(weights_dir, 'end')

assignments_dir = os.path.join(top_level_path, 'assignments', model_name)
best_assignments_dir = os.path.join(assignments_dir, 'best')
end_assignments_dir = os.path.join(assignments_dir, 'end')

misc_dir = os.path.join(top_level_path, 'misc', model_name)
best_misc_dir = os.path.join(misc_dir, 'best')
end_misc_dir = os.path.join(misc_dir, 'end')

spike_activity_dir = os.path.join(top_level_path, 'spike_activity', model_name)

for d in [ performance_dir, activity_dir, weights_dir, deltas_dir, misc_dir, best_misc_dir,
				assignments_dir, best_assignments_dir, MNIST_data_path, results_path, spike_activity_dir,
			best_weights_dir, end_weights_dir, end_misc_dir, end_assignments_dir, spikes_dir ]:
	if not os.path.isdir(d):
		os.makedirs(d)


def plot_labels_and_spikes(assignments, spike_counts, ending, j, image, predictions, true):
	fig = plt.figure(15, figsize = (18, 12))
	plt.gcf().clear()
	plt.suptitle(', '.join([' : '.join([str(key), str(value)]) for (key, value) \
		in predictions.items()]) + ', ' + ' : '.join(['True', str(true)]) + '\n' + \
		'Inhibition strength : ' + str(np.max(connections['AiAe'][:].todense())), fontsize=22)
	
	ax1 = plt.subplot(131)	
	ax1.imshow(image, cmap='gray')

	ax2 = plt.subplot(132)
	im = ax2.matshow(assignments.reshape([features_sqrt, features_sqrt]).T, \
		cmap=plt.get_cmap('RdBu', 10), vmin=-0.5, vmax=9.5, alpha=0.65)
	ax2.set_title('Total Spike Counts: %d' % sum(spike_counts))

	divider = make_axes_locatable(ax2)
	cax = divider.append_axes("right", size="5%", pad=0.1)

	plt.colorbar(im, cax=cax, ticks=np.arange(0, 10))

	spike_counts_reshaped = spike_counts.reshape([features_sqrt, features_sqrt])
	for x in xrange(features_sqrt):
		for y in xrange(features_sqrt):
			c = spike_counts_reshaped[x, y]
			if c > 0:
				ax2.text(x, y, str(c), va='center', ha='center', weight='heavy')
			else:
				ax2.text(x, y, '', va='center', ha='center')

	ax3 = plt.subplot(133)
	ax3.matshow(get_2d_input_weights(), interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))

	plt.tight_layout()
	fig.canvas.draw()

	if interactive_mode:
		time.sleep(0.5)

	plt.savefig(os.path.join(spike_activity_dir, ending, str(j) + '.png'))
	

def plot_input(rates):
	'''
	Plot the current input example during the training procedure.
	'''
	fig = plt.figure(fig_num, figsize = (5, 5))
	im = plt.imshow(rates.reshape((28, 28)), interpolation='nearest', vmin=0, vmax=64, cmap='binary')
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

		elif scheme == 'confidence_weighting':
			for i in xrange(10):
				num_assignments[i] = np.count_nonzero(assignments == i)
				if num_assignments[i] > 0:
					summed_rates[i] = np.sum(spike_rates[assignments == i] * spike_proportions[(assignments == i).ravel(), i]) / num_assignments[i]

		output_numbers[scheme] = np.argsort(summed_rates)[::-1]

	return output_numbers


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


def build_network():
	global fig_num

	neuron_groups['e'] = b.NeuronGroup(n_e_total, neuron_eqs_e, threshold=v_thresh_e, \
							refractory=refrac_e, reset=scr_e, compile=True, freeze=True)
	neuron_groups['i'] = b.NeuronGroup(n_e_total, neuron_eqs_i, threshold=v_thresh_i, \
						refractory=refrac_i, reset=v_reset_i, compile=True, freeze=True)

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
		neuron_groups['e'].theta = np.load(os.path.join(best_weights_dir, '_'.join(['theta_A', ending +'_best.npy'])))
		
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

				# load weight matrix
				weight_matrix = np.load(os.path.join(best_weights_dir, '_'.join([conn_name, ending, 'best.npy'])))
				
				# create a connection from the first group in conn_name with the second group
				connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
				
				# define the actual synaptic connections and strengths
				for feature in xrange(conv_features):
					for other_feature in xrange(conv_features):
						if feature != other_feature:
							for n in xrange(n_e):
								connections[conn_name][feature * n_e + n, other_feature * n_e + n] = inhibition_level

		print '...Creating monitors for:', name

		# spike rate monitors for excitatory and inhibitory neuron populations
		rate_monitors[name + 'e'] = b.PopulationRateMonitor(neuron_groups[name + 'e'], bin=(single_example_time + resting_time) / b.second)
		rate_monitors[name + 'i'] = b.PopulationRateMonitor(neuron_groups[name + 'i'], bin=(single_example_time + resting_time) / b.second)
		spike_counters[name + 'e'] = b.SpikeCounter(neuron_groups[name + 'e'])

		# record neuron population spikes if specified
		if record_spikes and do_plot:
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
			weight_matrix = np.load(os.path.join(best_weights_dir, '_'.join([conn_name, ending + '_best.npy'])))

			# create connections from the windows of the input group to the neuron population
			input_connections[conn_name] = b.Connection(input_groups['Xe'], neuron_groups[name[1] + conn_type[1]], \
							structure='sparse', state='g' + conn_type[0], delay=True, max_delay=delay[conn_type][1])
			
			for feature in xrange(conv_features):
				for n in xrange(n_e):
					for idx in xrange(conv_size ** 2):
						input_connections[conn_name][convolution_locations[n][idx], feature * n_e + n] = \
														weight_matrix[convolution_locations[n][idx], feature * n_e + n]

			if do_plot:
				plot_2d_input_weights()
				fig_num += 1	

	print '\n'


def run_activity_plotting():
	global fig_num, input_intensity, previous_spike_count, rates, assignments, clusters, cluster_assignments, \
				simple_clusters, simple_cluster_assignments, index_matrix, accumulated_rates, \
				accumulated_inputs, spike_proportions

	ending = '_'.join([ str(conv_size), str(conv_stride), str(conv_features), str(n_e), \
						str(num_train), str(random_seed), str(normalize_inputs), 
						str(proportion_grow), str(noise_const), str(inhibition_level) ])

	if not os.path.isdir(os.path.join(spike_activity_dir, ending)):
		os.makedirs(os.path.join(spike_activity_dir, ending))

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

	# get network filter weights
	filters = input_connections['XeAe'][:].todense()

	# start recording time
	start_time = timeit.default_timer()

	while j < num_examples:
		# get the firing rates of the next input example
		rates = (data['x'][j % data_size, :, :] / 8.0) * input_intensity
		
		# sets the input firing rates
		input_groups['Xe'].rate = rates.reshape(n_input)

		# run the network for a single example time
		b.run(single_example_time)

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
			if not reset_state_vars:
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
				if scheme != 'distance':
					output_numbers[scheme][j, :] = outputs
				elif scheme == 'distance':
					current_input = (rates * (weight['ee_input'] / np.sum(rates))).ravel()
					output_numbers[scheme][j, 0] = assignments[np.argmin([ euclidean(current_input, \
													filters[:, i]) for i in xrange(conv_features) ])]

			# Get the predicted labels (from our voting schemes) as well as the ground truth label for this example.
			predicted_labels = { scheme : int(output_numbers[scheme][j, 0]) for scheme in voting_schemes }
			true_label = int(input_numbers[j])

			plot_labels_and_spikes(assignments, current_spike_count, ending, j, data['x']\
										[j % data_size, :, :], predicted_labels, true_label)

			# print progress
			if j % print_progress_interval == 0 and j > 0:
				print 'runs done:', j, 'of', int(num_examples), '(time taken for past', print_progress_interval, 'runs:', str(timeit.default_timer() - start_time) + ')'
				start_time = timeit.default_timer()
						
			# set input firing rates back to zero
			for name in input_population_names:
				input_groups[name + 'e'].rate = 0
			
			# run the network for 'resting_time' to relax back to rest potentials
			if not reset_state_vars:
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--conv_size', type=int, default=28, help='Side length of the square convolution \
											window used by the input -> excitatory layer of the network.')
	parser.add_argument('--conv_stride', type=int, default=0, help='Horizontal, vertical stride \
									of the convolution window used by input layer of the network.')
	parser.add_argument('--conv_features', type=int, default=100, help='Number of excitatory \
								convolutional features / filters / patches used in the network.')
	parser.add_argument('--do_plot', type=str, default='False', help='Whether or not to display plots during network \
													training / testing. Defaults to False, as this makes the \
												network operation speedier, and possible to run on HPC resources.')
	parser.add_argument('--num_train', type=int, default=10000, help='The number of \
										examples for which to train the network on.')
	parser.add_argument('--num_test', type=int, default=10000, help='The number of \
										examples for which to test the network on.')
	parser.add_argument('--random_seed', type=int, default=42, help='The random seed \
									(any integer) from which to generate random numbers.')
	parser.add_argument('--save_weights', type=str, default='False', help='Whether or not to \
									save the weights of the model every `weight_update_interval`.')
	parser.add_argument('--weight_update_interval', type=int, default=10, help='How often \
												to update the plot of network filter weights.')
	parser.add_argument('--save_best_model', type=str, default='True', help='Whether \
										to save the current best version of the model.')
	parser.add_argument('--update_interval', type=int, default=250, help='How often \
										to update neuron labels and classify new inputs.')
	parser.add_argument('--plot_all_deltas', type=str, default='False', help='Whether or not to \
								plot weight changes for all neurons from input to excitatory layer.')
	parser.add_argument('--train_remove_inhibition', type=str, default='False', help='Whether or not to \
														remove lateral inhibition during the training phase.')
	parser.add_argument('--test_no_inhibition', type=str, default='False', help='Whether or not to \
														remove lateral inhibition during the test phase.')
	parser.add_argument('--test_max_inhibition', type=str, default='False', help='Whether or not to \
														use ETH-style inhibition during the test phase.')
	parser.add_argument('--start_inhib', type=float, default=0.01, help='The beginning value \
														of inhibiton for the increasing scheme.')
	parser.add_argument('--max_inhib', type=float, default=17.4, help='The maximum synapse \
											weight for inhibitory to excitatory connections.')
	parser.add_argument('--reset_state_vars', type=str, default='False', help='Whether to \
							reset neuron / synapse state variables or run a "reset" period.')
	parser.add_argument('--inhib_update_interval', type=int, default=250, \
							help='How often to increase the inhibition strength.')
	parser.add_argument('--inhib_schedule', type=str, default='linear', help='How to \
							update the strength of inhibition as the training progresses.')
	parser.add_argument('--save_spikes', type=str, default='False', help='Whether or not to \
							save 2D graphs of spikes to later use to make an activity time-lapse.')
	parser.add_argument('--normalize_inputs', type=str, default='False', help='Whether or not \
											to ensure all inputs contain the same amount of "ink".')
	parser.add_argument('--proportion_grow', type=float, default=1.0, help='What proportion of \
								the training to grow the inhibition from "start_inhib" to "max_inhib".')
	parser.add_argument('--noise_const', type=float, default=0.0, help='The scale of the \
															noise added to input examples.')
	parser.add_argument('--inhib_scheme', type=str, default='increasing', help='How inhibition from \
															inhibitory to excitatory neurons is handled.')
	parser.add_argument('--weights_noise', type=str, default='False', help='Whether to use multiplicative \
														Gaussian noise on synapse weights on each iteration.')
	parser.add_argument('--weights_noise_constant', type=float, default=1e-2, help='The spread of the \
																Gaussian noise used on synapse weights ')
	parser.add_argument('--start_input_intensity', type=float, default=2.0, help='The intensity at which the \
																input is (default) presented to the network.')
	parser.add_argument('--test_adaptive_threshold', type=str, default='False', help='Whether or not to allow \
															neuron thresholds to adapt during the test phase.')
	parser.add_argument('--train_time', type=float, default=0.35, help='How long training \
														inputs are presented to the network.')
	parser.add_argument('--train_rest', type=float, default=0.15, help='How long the network is allowed \
												to settle back to equilibrium between training examples.')
	parser.add_argument('--test_time', type=float, default=0.35, help='How long test \
												inputs are presented to the network.')
	parser.add_argument('--test_rest', type=float, default=0.15, help='How long the network is allowed \
												to settle back to equilibrium between test examples.')
	parser.add_argument('--interactive_mode', type=str, default='False', help='Allows user to observe \
														activation plots as the testing phase goes on.')
	parser.add_argument('--inhibition_level', type=float, default=17.4, help='Specifies the weight on \
												the synapses from inhibitory to excitatory populations.')

	# parse arguments and place them in local scope
	args = parser.parse_args()
	args = vars(args)
	locals().update(args)

	print '\nOptional argument values:'
	for key, value in args.items():
		print '-', key, ':', value

	print '\n'

	for var in [ 'do_plot', 'plot_all_deltas', 'reset_state_vars', 'test_max_inhibition', \
					'normalize_inputs', 'save_weights', 'save_best_model', 'test_no_inhibition', \
					'save_spikes', 'weights_noise', 'test_adaptive_threshold', 'interactive_mode' ]:
		if locals()[var] == 'True':
			locals()[var] = True
		elif locals()[var] == 'False':
			locals()[var] = False
		else:
			raise Exception('Expecting True or False-valued command line argument "' + var + '".')

	num_examples = 10000
	data_size = 10000

	# set brian global preferences
	b.set_global_preferences(defaultclock=b.Clock(dt=0.5*b.ms), useweave=True, gcc_options=['-ffast-math -march=native'],
								usecodegen=True, usecodegenweave=True, usecodegenstateupdate=True,
								usecodegenthreshold=False, usenewpropagate=True, usecstdp=True,
								openmp=False, magic_useframes=False, useweave_linear_diffeq=True)

	# for reproducibility's sake
	np.random.seed(random_seed)

	start = timeit.default_timer()
	data = get_labeled_data(os.path.join(MNIST_data_path, 'testing'), False, False, xrange(10), 1000, normalize_inputs)
	
	print 'Time needed to load data:', timeit.default_timer() - start

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
	single_example_time = test_time * b.second
	resting_time = test_rest * b.second

	# set the update interval
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
	weight['ee_input'] = (conv_size ** 2) * 0.099489796
	delay['ee_input'] = (0 * b.ms, 10 * b.ms)
	delay['ei_input'] = (0 * b.ms, 5 * b.ms)
	input_intensity = start_input_intensity

	current_inhibition = 1.0

	# time constants, learning rates, max weights, weight dependence, etc.
	tc_pre_ee, tc_post_ee = 20 * b.ms, 20 * b.ms
	nu_ee_pre, nu_ee_post = 0.0001, 0.01
	nu_AeAe_pre, nu_Ae_Ae_post = 0.1, 0.5
	wmax_ee = 1.0
	exp_ee_post = exp_ee_pre = 0.2
	w_mu_pre, w_mu_post = 0.2, 0.2

	# setting up differential equations (depending on train / test mode)
	scr_e = 'v = v_reset_e; timer = 0*ms'

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
			
	neuron_eqs_e += '\n  theta      :volt'
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
	ending = '_'.join([ str(conv_size), str(conv_stride), str(conv_features), str(n_e), \
						str(num_train), str(random_seed), str(normalize_inputs), 
						str(proportion_grow), str(noise_const) ])

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

	# build the spiking neural network
	build_network()

	# bookkeeping variables
	previous_spike_count = np.zeros((conv_features, n_e))
	input_numbers = np.zeros(num_examples)
	rates = np.zeros((n_input_sqrt, n_input_sqrt))

	assignments = np.load(os.path.join(best_assignments_dir, '_'.join(['assignments', ending, 'best.npy'])))
	accumulated_rates = np.load(os.path.join(best_misc_dir, '_'.join(['accumulated_rates', ending, 'best.npy'])))
	spike_proportions = np.load(os.path.join(best_misc_dir, '_'.join(['spike_proportions', ending, 'best.npy'])))

	voting_schemes = ['all', 'most_spiked_patch', 'most_spiked_location', 'confidence_weighting', 'distance']

	for scheme in voting_schemes:
		output_numbers[scheme] = np.zeros((num_examples, 10))
	
	run_activity_plotting()

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
import math
import time
import os

from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from struct import unpack
from brian import *

from util import *

np.set_printoptions(threshold=np.nan, linewidth=200)

# only show log messages of level ERROR or higher
b.log_level_error()

model_name = 'csnn_pc_inhibit_far'
best_weights_dir = os.path.join('..', '..', 'weights', model_name, 'best')
best_misc_dir = os.path.join('..', '..', 'misc', model_name, 'best')

filenames = [ filename for filename in os.listdir(best_weights_dir) if 'XeAe' in filename and 'True_0.75' in filename ]

print '\n'.join([ str(idx + 1) + ' - ' + filename for idx, filename in enumerate(filenames) ])

model_idx = int(raw_input('\nEnter the index of the model to view max firing of: ')) - 1
model_title = '_'.join(filenames[model_idx].split('_')[1:])[:-9]

print model_title

model_params = model_title.split('_')

conv_size = int(model_params[1])
conv_stride = int(model_params[2])
conv_features = int(model_params[3])
n_e = int(model_params[4])
reduced_dataset = model_params[5]
examples_per_class = int(model_params[16])
neighborhood = model_params[17]
inhib_scheme = model_params[18]
inhib_const = model_params[19]
strengthen_const = model_params[20]
num_train = model_params[21]
random_seed = int(model_params[22])
accumulate_votes = model_params[23]
accumulation_decay = model_params[24]

n_e_total = n_e * conv_features
n_e_sqrt = int(math.sqrt(n_e))
n_i = n_e
features_sqrt = int(math.ceil(math.sqrt(conv_features)))

single_example_time = 0.35 * b.second
resting_time = 0.15 * b.second

neuron_groups = {}
connections = {}
input_connections = {}

top_percent = 10


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


def plot_assignments(assignments):
	cmap = plt.get_cmap('RdBu', 11)
	im = plt.matshow(assignments.reshape((int(np.sqrt(n_e_total)), int(np.sqrt(n_e_total)))).T, cmap=cmap, vmin=-1.5, vmax=9.5)
	plt.colorbar(im, ticks=np.arange(-1, 10))
	plt.title('Neuron labels')
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

	plt.title('Input to excitatory synpatic weights')

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


def build_network():
	global fig_num

	neuron_groups['e'] = b.NeuronGroup(n_e_total, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=scr_e, compile=True, freeze=True)
	neuron_groups['i'] = b.NeuronGroup(n_e_total, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i, compile=True, freeze=True)

	for name in [ 'A' ]:
		print '...Creating neuron group:', name

		# get a subgroup of size 'n_e' from all exc
		neuron_groups[name + 'e'] = neuron_groups['e'].subgroup(conv_features * n_e)
		# get a subgroup of size 'n_i' from the inhibitory layer
		neuron_groups[name + 'i'] = neuron_groups['i'].subgroup(conv_features * n_e)

		# start the membrane potentials of these groups 40mV below their resting potentials
		neuron_groups[name + 'e'].v = v_rest_e - 40. * b.mV
		neuron_groups[name + 'i'].v = v_rest_i - 40. * b.mV

	print '...Creating recurrent connections'

	for name in [ 'A' ]:
		neuron_groups['e'].theta = np.load(os.path.join(best_weights_dir, '_'.join(['theta_A', model_title +'_best.npy'])))
		
		for conn_type in ['ei', 'ie']:
			if conn_type == 'ei':
				# create connection name (composed of population and connection types)
				conn_name = name + conn_type[0] + name + conn_type[1]
				# create a connection from the first group in conn_name with the second group
				connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], \
								neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
				# instantiate the created connection
				for feature in xrange(conv_features):
					for n in xrange(n_e):
						connections[conn_name][feature * n_e + n, feature * n_e + n] = 10.4

			elif conn_type == 'ie':
				# create connection name (composed of population and connection types)
				conn_name = name + conn_type[0] + name + conn_type[1]
				# create a connection from the first group in conn_name with the second group
				connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], \
								neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
				# define the actual synaptic connections and strengths
				for feature in xrange(conv_features):
					if inhib_scheme in ['far', 'strengthen']:
						for other_feature in set(range(conv_features)) - set(neighbor_mapping[feature]):
							if inhib_scheme == 'far':
								for n in xrange(n_e):
									connections[conn_name][feature * n_e + n, other_feature * n_e + n] = 17.4

							elif inhib_scheme == 'strengthen':
								if n_e == 1:
									x, y = feature // np.sqrt(n_e_total), feature % np.sqrt(n_e_total)
									x_, y_ = other_feature // np.sqrt(n_e_total), other_feature % np.sqrt(n_e_total)
								else:
									x, y = feature // np.sqrt(conv_features), feature % np.sqrt(conv_features)
									x_, y_ = other_feature // np.sqrt(conv_features), other_feature % np.sqrt(conv_features)

								for n in xrange(n_e):
									connections[conn_name][feature * n_e + n, other_feature * n_e + n] = \
													min(17.4, inhib_const * np.sqrt(euclidean([x, y], [x_, y_])))

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
													min(17.4, inhib_const * np.sqrt(euclidean([x, y], [x_, y_])))

					else:
						raise Exception('Expecting one of "far", "increasing", or "strengthen" for argument "inhib_scheme".')

		# spike rate monitors for excitatory and inhibitory neuron populations
		rate_monitors[name + 'e'] = b.PopulationRateMonitor(neuron_groups[name + 'e'], bin=(single_example_time + resting_time) / b.second)
		rate_monitors[name + 'i'] = b.PopulationRateMonitor(neuron_groups[name + 'i'], bin=(single_example_time + resting_time) / b.second)
		spike_counters[name + 'e'] = b.SpikeCounter(neuron_groups[name + 'e'])

		# record neuron population spikes if specified
		if record_spikes:
			spike_monitors[name + 'e'] = b.SpikeMonitor(neuron_groups[name + 'e'])
			spike_monitors[name + 'i'] = b.SpikeMonitor(neuron_groups[name + 'i'])

	if record_spikes:
		b.figure(fig_num, figsize=(8, 6))
		
		fig_num += 1
		
		b.ion()
		b.subplot(211)
		b.raster_plot(spike_monitors['Ae'], refresh=1000 * b.ms, showlast=1000 * b.ms, title='Excitatory spikes per neuron')
		b.subplot(212)
		b.raster_plot(spike_monitors['Ai'], refresh=1000 * b.ms, showlast=1000 * b.ms, title='Inhibitory spikes per neuron')
		b.tight_layout()

	# creating Poission spike train from input image (784 vector, 28x28 image)
	for name in [ 'X' ]:
		input_groups[name + 'e'] = b.PoissonGroup(n_input, 0)
		rate_monitors[name + 'e'] = b.PopulationRateMonitor(input_groups[name + 'e'], bin=(single_example_time + resting_time) / b.second)

	# creating connections from input Poisson spike train to convolution patch populations
	for name in [ 'XA' ]:
		print '\n...Creating connections between', name[0], 'and', name[1]
		
		# for each of the input connection types (in this case, excitatory -> excitatory)
		for conn_type in [ 'ee_input' ]:
			# saved connection name
			conn_name = name[0] + conn_type[0] + name[1] + conn_type[1]

			# get weight matrix depending on training or test phase
			weight_matrix = np.load(os.path.join(best_weights_dir, '_'.join([conn_name, model_title + '_best.npy'])))

			# create connections from the windows of the input group to the neuron population
			input_connections[conn_name] = b.Connection(input_groups['Xe'], neuron_groups[name[1] + \
					conn_type[1]], structure='sparse', state='g' + conn_type[0], delay=True, max_delay=delay[conn_type][1])
			
			for feature in xrange(conv_features):
				for n in xrange(n_e):
					for idx in xrange(conv_size ** 2):
						input_connections[conn_name][convolution_locations[n][idx], feature * n_e + n] = \
											weight_matrix[convolution_locations[n][idx], feature * n_e + n]

			plot_2d_input_weights()
			fig_num += 1	

	print '\n'


def run_test():
	global fig_num, input_intensity, previous_spike_count, rates, assignments, clusters, cluster_assignments, \
				kmeans, kmeans_assignments, simple_clusters, simple_cluster_assignments, index_matrix, accumulated_rates, \
				accumulated_inputs, spike_proportions

	assignments_image = plot_assignments(assignments)
	fig_num += 1

	# initialize network
	j = 0
	num_retries = 0
	b.run(0)

	# start recording time
	start_time = timeit.default_timer()

	max_fired = None

	while j < num_examples:
		# get the firing rates of the next input example
		rates = (data['x'][j % data_size, :, :] / 8.0) * input_intensity
		
		# sets the input firing rates
		input_groups['Xe'].rate = rates.reshape(n_input)

		# run the network for a single example time
		b.run(single_example_time)

		# if exc_stdp and j == 0:
		# 		exc_weights_image = plt.matshow(connections['AeAe'][:].todense().T, cmap='binary', vmin=0, vmax=wmax_exc)
		# 		plt.colorbar()
		# 		plt.title('Excitatory to excitatory weights')

		# get count of spikes over the past iteration
		current_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e)) - previous_spike_count
		previous_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e))

		# if the neurons in the network didn't spike more than four times
		if np.sum(current_spike_count) < 5 and num_retries < 3:
			# increase the intensity of input
			input_intensity += 2
			num_retries += 1
			
			# set all network firing rates to zero
			for name in [ 'X' ]:
				input_groups[name + 'e'].rate = 0

			# let the network relax back to equilibrium
			b.run(resting_time)

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

			max_fired = np.argmax(result_monitor[j % update_interval, :])

			rates = (data['x'][j % data_size, :, :] / 8.0) * input_intensity
			fig = plt.figure(9, figsize = (8, 8))
			plt.imshow(rates.reshape((28, 28)), interpolation='nearest', vmin=0, vmax=64, cmap='binary')
			plt.title(str(data['y'][j % data_size][0]) + ' : ' + ', '.join([str(int(output_numbers[scheme][j, 0])) for scheme in voting_schemes]))
			# plt.title('Misclassified with ' + performance + ' as ' + str(int(wrong_label)) + \
			# 	' in location (' + str(int(max_fired // features_sqrt)) + ', ' + \
			# 			str(int(max_fired % features_sqrt)) + ')')

			max_fired_location = np.zeros((conv_features))
			max_fired_location[max_fired] = 1
			fig = plt.figure(10, figsize = (7, 7))
			plt.xticks(xrange(features_sqrt))
			plt.yticks(xrange(features_sqrt))
			plt.imshow(max_fired_location.reshape((features_sqrt, features_sqrt)).T, interpolation='nearest', cmap='binary')
			plt.grid(True)
			
			fig.canvas.draw()

			time.sleep(5.0)
						
			# set input firing rates back to zero
			input_groups['Xe'].rate = 0
			
			# run the network for 'resting_time' to relax back to rest potentials
			b.run(resting_time)
			
			# bookkeeping
			input_intensity = start_input_intensity
			j += 1

	print '\n'


if __name__ == '__main__':
	# weight updates and progress printing intervals
	print_progress_interval = 10
	update_interval = 100

	# rest potential parameters, reset potential parameters, threshold potential parameters, and refractory periods
	v_rest_e, v_rest_i = -65. * b.mV, -60. * b.mV
	v_reset_e, v_reset_i = -65. * b.mV, -45. * b.mV
	v_thresh_e, v_thresh_i = -52. * b.mV, -40. * b.mV
	refrac_e, refrac_i = 5. * b.ms, 2. * b.ms

	b.ion()
	fig_num = 1

	data_size = 10000

	# set brian global preferences
	b.set_global_preferences(defaultclock = b.Clock(dt=0.5*b.ms), useweave = True, gcc_options = ['-ffast-math -march=native'], usecodegen = True,
		usecodegenweave = True, usecodegenstateupdate = True, usecodegenthreshold = False, usenewpropagate = True, usecstdp = True, openmp = False,
		magic_useframes = False, useweave_linear_diffeq = True)

	# for reproducibility's sake
	np.random.seed(random_seed)

	start = timeit.default_timer()
	data = get_labeled_data(os.path.join(MNIST_data_path, 'testing'), False, reduced_dataset, range(10), examples_per_class)
	
	print 'Time needed to load data:', timeit.default_timer() - start

	# dictionaries for weights and delays
	weight, delay, rate_monitors, spike_monitors, spike_counters, output_numbers, \
			input_connections, connections, input_groups = {}, {}, {}, {}, {}, {}, {}, {}, {}

	record_spikes = True

	delay['ee_input'] = (0 * b.ms, 10 * b.ms)
	delay['ei_input'] = (0 * b.ms, 5 * b.ms)
	input_intensity = start_input_intensity = 2.0

	# number of inputs to the network
	n_input = 784
	n_input_sqrt = int(math.sqrt(n_input))

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

	build_network()

	num_examples = 10000

	voting_schemes = ['all', 'most_spiked_patch', 'top_percent', 'most_spiked_location', 'confidence_weighting']

	for scheme in voting_schemes:
		output_numbers[scheme] = np.zeros((num_examples, 10))

	previous_spike_count = np.zeros((conv_features, n_e))
	input_numbers = np.zeros(num_examples)
	rates = np.zeros((n_input_sqrt, n_input_sqrt))

	assignments = np.load(os.path.join(best_misc_dir, '_'.join(['assignments', model_title, 'best.npy'])))
	accumulated_rates = np.load(os.path.join(best_misc_dir, '_'.join(['accumulated_rates', model_title, 'best.npy'])))
	spike_proportions = np.load(os.path.join(best_misc_dir, '_'.join(['spike_proportions', model_title, 'best.npy'])))

	run_test()
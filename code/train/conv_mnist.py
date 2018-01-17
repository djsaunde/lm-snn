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
model_name = 'conv_mnist'
results_path = os.path.join(top_level_path, 'results', model_name)

performance_dir = os.path.join(top_level_path, 'performance', model_name)
activity_dir = os.path.join(top_level_path, 'activity', model_name)
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

for d in [ performance_dir, activity_dir, weights_dir, misc_dir, best_misc_dir,
				assignments_dir, best_assignments_dir, MNIST_data_path, results_path, 
			best_weights_dir, end_weights_dir, end_misc_dir, end_assignments_dir, spikes_dir ]:
	if not os.path.isdir(d):
		os.makedirs(d)


def normalize_weights():
	'''
	Squash the input to excitatory synaptic weights to sum to a prespecified number.
	'''
	for conn_name in input_connections:
		connection = input_connections[conn_name][:].todense()

		for n in xrange(n_neurons):
			n_connection = connection[:, n]
			column_sums = np.sum(np.asarray(n_connection), axis=0)
			column_factors = weight['ee_input'] / column_sums

			dense_weights = input_connections[conn_name][:, n].todense()
			dense_weights *= column_factors
			input_connections[conn_name][:, n] = dense_weights


def plot_input(rates):
	'''
	Plot the current input example during the training procedure.
	'''
	fig = plt.figure(fig_num, figsize = (5, 5))
	im = plt.imshow(rates.reshape([window, window]), interpolation='nearest', vmin=0, vmax=64, cmap='binary')
	plt.colorbar(im)
	plt.title('Current input example')
	fig.canvas.draw()
	return im, fig


def update_input(rates, im, fig):
	'''
	Update the input image to use for input plotting.
	'''
	im.set_array(rates.reshape([window, window]))
	fig.canvas.draw()
	return im


def update_assignments_plot(assignments, im):
	im.set_array(assignments.reshape((int(np.sqrt(n_e_total)), int(np.sqrt(n_e_total)))).T)
	return im


def get_2d_input_weights():
	'''
	Get the weights from the input to excitatory layer and reshape it to be two
	dimensional and square.
	'''
	connection = input_connections['XeAe'][:]
	square_weights = np.zeros([n_neurons_sqrt * window, n_neurons_sqrt * window])

	for n in xrange(n_neurons):
		temp = connection[:, n].todense()
		square_weights[(n // n_neurons_sqrt) * window : ((n // n_neurons_sqrt) + 1) * window, 
			(n % n_neurons_sqrt) * window : ((n % n_neurons_sqrt) + 1) * window] = temp.reshape([window, window])

	return square_weights.T


def plot_weights_and_assignments(assignments):
	'''
	Plot the weights from input to excitatory layer to view during training.
	'''
	weights = get_2d_input_weights()

	fig = plt.figure(fig_num, figsize=(18, 9))

	ax1 = plt.subplot(121)
	image1 = ax1.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
	ax1.set_title(ending.replace('_', ' '))

	ax2 = plt.subplot(122)
	color = plt.get_cmap('RdBu', 11)
	reshaped_assignments = assignments.reshape([n_neurons_sqrt, n_neurons_sqrt]).T
	image2 = ax2.matshow(reshaped_assignments, cmap=color, vmin=-1.5, vmax=9.5)
	ax2.set_title('Neuron labels')

	divider1 = make_axes_locatable(ax1)
	divider2 = make_axes_locatable(ax2)
	cax1 = divider1.append_axes("right", size="5%", pad=0.05)
	cax2 = divider2.append_axes("right", size="5%", pad=0.05)

	plt.colorbar(image1, cax=cax1)
	plt.colorbar(image2, cax=cax2, ticks=np.arange(-1, 10))

	ax1.set_xticks(xrange(window, window * (n_neurons_sqrt + 1), window), xrange(1, int(np.sqrt(n_neurons)) + 1))
	ax1.set_yticks(xrange(window, window * (n_neurons_sqrt + 1), window), xrange(1, int(np.sqrt(n_neurons)) + 1))
	
	plt.tight_layout()

	fig.canvas.draw()
	return fig, ax1, ax2, image1, image2


def update_weights_and_assignments(fig, ax1, ax2, im1, im2, assignments):
	'''
	Update the plot of the weights from input to excitatory layer to view during training.
	'''
	weights = get_2d_input_weights()
	im1.set_array(weights)
	
	reshaped_assignments = assignments.reshape([n_neurons_sqrt, n_neurons_sqrt]).T
	im2.set_array(reshaped_assignments)

	fig.canvas.draw()


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
			accumulated_rates[:, j] = accumulated_rates[:, j] * 0.9 + \
					np.ravel(np.sum(result_monitor[input_numbers == j], axis=0) / num_assignments)
	
	assignments = np.argmax(accumulated_rates, axis=1)

	spike_proportions = np.divide(accumulated_rates, np.sum(accumulated_rates, axis=0))

	return assignments, accumulated_rates, spike_proportions


def build_network():
	global fig_num, assignments

	neuron_groups['e'] = b.NeuronGroup(n_neurons, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=scr_e, compile=True, freeze=True)
	neuron_groups['i'] = b.NeuronGroup(n_neurons, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i, compile=True, freeze=True)

	for name in population_names:
		print '...Creating neuron group:', name

		# get a subgroup of size 'n_e' from all exc
		neuron_groups[name + 'e'] = neuron_groups['e'].subgroup(n_neurons)
		# get a subgroup of size 'n_i' from the inhibitory layer
		neuron_groups[name + 'i'] = neuron_groups['i'].subgroup(n_neurons)

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
			neuron_groups['e'].theta = np.ones((n_neurons)) * 20.0 * b.mV
		
		for conn_type in recurrent_conn_names:
			if conn_type == 'ei':
				# create connection name (composed of population and connection types)
				conn_name = name + conn_type[0] + name + conn_type[1]
				# create a connection from the first group in conn_name with the second group
				connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
				
				# instantiate the created connection
				for n in xrange(n_neurons):
					connections[conn_name][n, n] = 10.4

			elif conn_type == 'ie':
				# create connection name (composed of population and connection types)
				conn_name = name + conn_type[0] + name + conn_type[1]
				
				# get weight matrix depending on training or test phase
				if test_mode:
					if save_best_model:
						weights = np.load(os.path.join(best_weights_dir, '_'.join([conn_name, ending + '_best.npy'])))
					else:
						weights = np.load(os.path.join(end_weights_dir, '_'.join([conn_name, ending + '_end.npy'])))
				
				# create a connection from the first group in conn_name with the second group
				connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
				
				# define the synaptic connections and strengths
				for n in xrange(n_neurons):
					for other_n in xrange(n_neurons):
						if n != other_n:
							x, y = n // np.sqrt(n_neurons), n % np.sqrt(n_neurons)
							x_, y_ = other_n // np.sqrt(n_neurons), other_n % np.sqrt(n_neurons)

							if test_mode:
								connections[conn_name][n, other_n] = weights[n, other_n]
							else:
								connections[conn_name][n, other_n] = inhib

		print '...Creating monitors for:', name

		# spike rate monitors for excitatory and inhibitory neuron populations
		rate_monitors[name + 'e'] = b.PopulationRateMonitor(neuron_groups[name + 'e'], bin=(single_example_time + resting_time) / b.second)
		rate_monitors[name + 'i'] = b.PopulationRateMonitor(neuron_groups[name + 'i'], bin=(single_example_time + resting_time) / b.second)
		spike_counters[name + 'e'] = b.SpikeCounter(neuron_groups[name + 'e'])

		# record neuron population spikes if specified
		if record_spikes and plot:
			spike_monitors[name + 'e'] = b.SpikeMonitor(neuron_groups[name + 'e'])
			spike_monitors[name + 'i'] = b.SpikeMonitor(neuron_groups[name + 'i'])

	if record_spikes and plot:
		b.figure(fig_num, figsize=(8, 6))
		b.ion()
		b.subplot(211)
		b.raster_plot(spike_monitors['Ae'], refresh=1000 * b.ms, showlast=1000 * b.ms, title='Excitatory spikes per neuron')
		b.subplot(212)
		b.raster_plot(spike_monitors['Ai'], refresh=1000 * b.ms, showlast=1000 * b.ms, title='Inhibitory spikes per neuron')
		b.tight_layout()

		fig_num += 1

	# creating Poission spike train from input image (784 vector, 28x28 image)
	for name in input_population_names:
		input_groups[name + 'e'] = b.PoissonGroup(window ** 2, 0)
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
					weights = np.load(os.path.join(best_weights_dir, '_'.join([conn_name, ending + '_best.npy'])))
				else:
					weights = np.load(os.path.join(end_weights_dir, '_'.join([conn_name, ending + '_end.npy'])))
			else:
				weights = (b.random([window ** 2, n_neurons]) + 0.01) * 0.3

			# create connections from the windows of the input group to the neuron population
			input_connections[conn_name] = b.Connection(input_groups['Xe'], neuron_groups[name[1] + conn_type[1]],
							structure='dense', state='g' + conn_type[0], delay=True, max_delay=delay[conn_type][1])
			
			input_connections[conn_name].connect(input_groups[conn_name[0:2]],
					neuron_groups[conn_name[2:4]], weights, delay=delay[conn_type])
			
			if test_mode:
				if plot:
					plot_weights_and_assignments(assignments)
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
				simple_clusters, simple_cluster_assignments, index_matrix, accumulated_rates, \
				accumulated_inputs, spike_proportions

	if plot:
		input_image_monitor, input_image = plot_input(rates)
		fig_num += 1

		weights_assignments_figure, weights_axes, assignments_axes, weights_image, \
						assignments_image = plot_weights_and_assignments(assignments)
		fig_num += 1

	# set up performance recording and plotting
	num_evaluations = int(num_examples / update_interval) + 1
	performances = { voting_scheme : np.zeros(num_evaluations) for voting_scheme in voting_schemes }
	num_weight_updates = int(num_examples / weight_update_interval)

	if plot:
		performance_monitor, fig_num, fig_performance = plot_performance(fig_num, performances, num_evaluations)
	else:
		performances, wrong_idxs, wrong_labels = get_current_performance(performances, 0)

	# initialize network
	j = 0
	num_retries = 0
	b.run(0)

	if save_best_model:
		best_performance = 0.0

	start_time = timeit.default_timer()
	while j < num_examples:
		# get the firing rates of the next input example
		rates = ((data['x'][j % data_size, :, :] / 8.0) * input_intensity).ravel()
		for n in xrange(n_e):
			if rates[convolution_locations[n]].sum() < 1:
				continue

			# sets the input firing rates
			input_groups['Xe'].rate = rates[convolution_locations[n]]

			# plot the input at this step
			if plot:
				input_image_monitor = update_input(rates[convolution_locations[n]], input_image_monitor, input_image)

			# run the network for a single example time
			b.run(single_example_time)

			# make sure synapse weights don't grow too large
			normalize_weights()

			# let the network relax back to equilibrium
			b.run(resting_time)

		# get new neuron label assignments every 'update_interval'
		if j % update_interval == 0 and j > 0:
			if j % data_size == 0:
				assignments, accumulated_rates, spike_proportions = assign_labels(result_monitor, 
					input_numbers[data_size - update_interval : data_size], accumulated_rates, accumulated_inputs)
			else:
				assignments, accumulated_rates, spike_proportions = assign_labels(result_monitor, 
					input_numbers[(j % data_size) - update_interval : j % data_size], accumulated_rates, accumulated_inputs)

		# get count of spikes over the past iteration
		current_spike_count = np.copy(spike_counters['Ae'].count[:]) - previous_spike_count
		previous_spike_count = np.copy(spike_counters['Ae'].count[:])

		# update weights every 'weight_update_interval'
		if j % weight_update_interval == 0 and plot:
			update_weights_and_assignments(weights_assignments_figure, weights_axes, assignments_axes, \
										weights_image, assignments_image, assignments)
			
		# if the neurons in the network didn't spike more than four times
		if np.sum(current_spike_count) < 5 and num_retries < 3:
			# increase the intensity of input
			input_intensity += 2
			num_retries += 1
			
			# set all network firing rates to zero
			for name in input_population_names:
				input_groups[name + 'e'].rate = 0

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

			if j % print_progress_interval == 0 and j > 0:
				print 'Progress: (%d / %d) - Elapsed time: %.4f' % (j, num_examples, timeit.default_timer() - start_time)
				start_time = timeit.default_timer()

			# plot performance if appropriate
			if (j % update_interval == 0 or j == num_examples - 1) and j > 0:
				if plot:
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

							np.save(os.path.join(best_assignments_dir, '_'.join(['assignments', ending, 'best'])), assignments)
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
			b.run(resting_time)

			# bookkeeping
			input_intensity = start_input_intensity
			j += 1

	# ensure weights don't grow without bound
	normalize_weights()

	print '\n'


def run_test():
	global fig_num, input_intensity, previous_spike_count, rates, assignments, clusters, cluster_assignments, \
				simple_clusters, simple_cluster_assignments, index_matrix, accumulated_rates, \
				accumulated_inputs, spike_proportions

	# set up performance recording and plotting
	num_evaluations = int(num_examples / update_interval) + 1
	performances = { voting_scheme : np.zeros(num_evaluations) for voting_scheme in voting_schemes }
	num_weight_updates = int(num_examples / weight_update_interval)

	# initialize network
	j = 0
	num_retries = 0
	b.run(0)

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
		current_spike_count = np.copy(spike_counters['Ae'].count[:]) - previous_spike_count
		previous_spike_count = np.copy(spike_counters['Ae'].count[:])

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
			b.run(resting_time)

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

		np.save(os.path.join(end_assignments_dir, '_'.join(['assignments', ending, 'end'])), assignments)
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

	print test_results

	differences = { scheme : test_results[scheme][0, :] - input_numbers for scheme in voting_schemes }
	correct = { scheme : len(np.where(differences[scheme] == 0)[0]) for scheme in voting_schemes }
	incorrect = { scheme : len(np.where(differences[scheme] != 0)[0]) for scheme in voting_schemes }
	accuracies = { scheme : correct[scheme] / float(num_examples) * 100 for scheme in voting_schemes }

	conf_matrices = np.array([confusion_matrix(test_results[scheme][0, :], \
								input_numbers) for scheme in voting_schemes])
	np.save(os.path.join(results_path, '_'.join(['confusion_matrix', ending]) + '.npy'), conf_matrices)

	print '\nConfusion matrix:\n\n', conf_matrices

	for scheme in voting_schemes:
		print '\n-', scheme, 'accuracy:', accuracies[scheme]

	results = pd.DataFrame([ [ ending ] + accuracies.values() ], columns=[ 'Model' ] + accuracies.keys())
	filename = '_'.join([str(n_neurons), 'results.csv'])
	if not filename in os.listdir(results_path):
		results.to_csv(os.path.join(results_path, filename), index=False)
	else:
		all_results = pd.read_csv(os.path.join(results_path, filename))
		all_results = pd.concat([all_results, results], ignore_index=True)
		all_results.to_csv(os.path.join(results_path, filename), index=False)

	print '\n'


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', default='train')
	parser.add_argument('--window', type=int, default=14)
	parser.add_argument('--stride', type=int, default=14)
	parser.add_argument('--n_neurons', type=int, default=100)
	parser.add_argument('--plot', type=str, default='False')
	parser.add_argument('--n_train', type=int, default=10000)
	parser.add_argument('--n_test', type=int, default=10000)
	parser.add_argument('--random_seed', type=int, default=0)
	parser.add_argument('--weight_update_interval', type=int, default=1)
	parser.add_argument('--save_best_model', type=str, default='True')
	parser.add_argument('--update_interval', type=int, default=250)
	parser.add_argument('--inhib', type=float, default=17.5)
	parser.add_argument('--start_input_intensity', type=float, default=2.0)
	parser.add_argument('--dt', type=float, default=0.25)

	# parse arguments and place them in local scope
	args = parser.parse_args()
	args = vars(args)
	locals().update(args)

	print '\nOptional argument values:'
	for key, value in args.items():
		print '-', key, ':', value

	print '\n'

	for var in ['plot', 'save_best_model']:
		if locals()[var] == 'True':
			locals()[var] = True
		elif locals()[var] == 'False':
			locals()[var] = False
		else:
			raise Exception('Expecting True or False-valued command line argument "' + var + '".')

	# test or training mode
	test_mode = mode == 'test'

	if test_mode:
		num_examples = n_test
	else:
		num_examples = n_train

	if test_mode:
		data_size = 10000
	else:
		data_size = 60000

	b.set_global_preferences(defaultclock = b.Clock(dt=dt*b.ms), useweave=True, gcc_options=['-ffast-math -march=native'], usecodegen=True,
											usecodegenweave=True, usecodegenstateupdate=True, usecodegenthreshold=False, usenewpropagate=True,
															usecstdp=True, openmp=False, magic_useframes=False, useweave_linear_diffeq=True)

	np.random.seed(random_seed)

	start = timeit.default_timer()
	data = get_labeled_data(os.path.join(MNIST_data_path, 'testing' if test_mode else 'training'), 
												not test_mode, False, xrange(10), 1000, False)
	
	print 'Time needed to load data:', timeit.default_timer() - start

	record_spikes = True

	# number of inputs to the network
	n_input = 784
	n_input_sqrt = int(math.sqrt(n_input))

	# number of neurons parameters
	if window == 28 and stride == 0:
		n_e = 1
	else:
		n_e = ((n_input_sqrt - window) / stride + 1) ** 2
	
	n_e_total = n_e * n_neurons
	n_e_sqrt = int(math.sqrt(n_e))
	n_i = n_e
	n_neurons_sqrt = int(math.ceil(math.sqrt(n_neurons)))

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
	weight['ee_input'] = (window ** 2) * 0.15
	delay['ee_input'] = (0 * b.ms, 10 * b.ms)
	delay['ei_input'] = (0 * b.ms, 5 * b.ms)
	input_intensity = start_input_intensity

	# time constants, learning rates, max weights, weight dependence, etc.
	tc_pre_ee, tc_post_ee = 20 * b.ms, 20 * b.ms
	nu_ee_pre, nu_ee_post = 0.0001, 0.01
	nu_AeAe_pre, nu_Ae_Ae_post = 0.1, 0.5
	wmax_ee = 1.0
	exp_ee_post = exp_ee_pre = 0.2
	w_mu_pre, w_mu_post = 0.2, 0.2

	# setting up differential equations (depending on train / test mode)
	if test_mode and not test_adaptive_threshold:
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
	ending = '_'.join([ str(window), str(stride), str(n_neurons), str(n_e), \
									str(n_train), str(random_seed), str(inhib) ])

	b.ion()
	fig_num = 1
	
	# creating dictionaries for various objects
	neuron_groups, input_groups, connections, input_connections, stdp_methods, \
		rate_monitors, spike_monitors, spike_counters, output_numbers = {}, {}, {}, {}, {}, {}, {}, {}, {}

	# creating convolution locations inside the input image
	convolution_locations = {}
	for n in xrange(n_e):
		convolution_locations[n] = [ ((n % n_e_sqrt) * stride + (n // n_e_sqrt) * n_input_sqrt * \
						stride) + (x * n_input_sqrt) + y for y in xrange(window) for x in xrange(window) ]

	# instantiating neuron "vote" monitor
	result_monitor = np.zeros((update_interval, n_neurons))

	# bookkeeping variables
	previous_spike_count = np.zeros(n_neurons)
	input_numbers = np.zeros(num_examples)
	rates = np.zeros([window, window])

	if test_mode:
		assignments = np.load(os.path.join(best_assignments_dir, '_'.join(['assignments', ending, 'best.npy'])))
		accumulated_rates = np.load(os.path.join(best_misc_dir, '_'.join(['accumulated_rates', ending, 'best.npy'])))
		spike_proportions = np.load(os.path.join(best_misc_dir, '_'.join(['spike_proportions', ending, 'best.npy'])))
	else:
		assignments = -1 * np.ones(n_neurons)

	build_network()

	voting_schemes = ['all']

	for scheme in voting_schemes:
		output_numbers[scheme] = np.zeros([num_examples, 10])

	if not test_mode:
		accumulated_rates = np.zeros([n_neurons, 10])
		accumulated_inputs = np.zeros(10)
		spike_proportions = np.zeros([n_neurons, 10])
	
	# run the simulation of the network
	if test_mode:
		run_test()
	else:
		run_train()

	# save and plot results
	save_results()

	# evaluate results
	if test_mode:
		evaluate_results()

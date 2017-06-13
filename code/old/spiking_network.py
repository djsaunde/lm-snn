'''
This file is meant to contain various spiking neural network models and associated methods and attributes.

Experiments can be run using these network models with arbitrary parameter settings.

Work in progress!

@author: Dan Saunders (djsaunde.github.io)
'''

import numpy as np
import brian as b
import brian_no_units

import time, os.path, scipy, math, sys, timeit, random, argparse

sys.path.append('/home/dan/code/python_mcl/mcl')

from sklearn.cluster import KMeans
from mcl_clustering import networkx_mcl
from scipy.sparse import coo_matrix
from struct import unpack
from brian import *

MNIST_data_path = '../data/'
top_level_path = '../'
weight_path = top_level_path + 'weights/conv_patch_connectivity_weights/'

do_plot = True
n_input = 784


def get_MNIST_data(pickle_name, train):
	'''
	Read input-vector (image) and target class (label, 0-9) and return it as 
	a list of tuples.
	'''
	if os.path.isfile('%s.pickle' % pickle_name):
		data = p.load(open('%s.pickle' % pickle_name))
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


class SpikingETH(object):
	'''
	The original model implemented in "Unsupervised learning of digit recognition using spike-timing-dependent plasticity" (Diehl
	and Cook 2015).
	'''

	def __init__(n_excitatory=100):
		pass


	def __repr__():
		pass


	def train():
		pass


	def test():
		pass


	def step():
		pass


class SpikingCNN(object):
	'''
	Our proposed model with an added "convolutional" layer.
	'''

	def __init__(self, n_input=784, conv_size=16, conv_stride=4, conv_features=50, connectivity='all', weight_dependence=False, post_pre=True, 
					weight_sharing=False, lattice_structure='4', random_lattice_prob=0.0, random_inhibition_prob=0.0):
		'''
		Constructor for the spiking convolutional neural network model.

		n_input: (flattened) dimensionality of the input data
		conv_size: side length of convolution windows used
		conv_stride: stride (horizontal and vertical) of convolution windows used
		conv_features: number of convolution features (or patches) used
		connectivity: connection style between patches; one of 'none', 'pairs', all'; more to be added
		weight_dependence: whether to use weight STDP with weight dependence
		post_pre: whether to use STDP with both post- and pre-synpatic traces
		weight_sharing: whether to impose that all neurons within a convolution patch share a common set of weights
		lattice_structure: lattice connectivity pattern between patches; one of 'none', '4', '8', and 'all'
		random_lattice_prob: probability of adding random additional lattice connections between patches
		random_inhibition_prob: probability of adding random additional inhibition edges from the inhibitory to excitatory population
		'''
		self.n_input, self.conv_size, self.conv_stride, self.conv_features, self.connectivity, self.weight_dependence, \
			self.post_pre, self.weight_sharing, self.lattice_structure, self.random_lattice_prob, self.random_inhibition_prob = \
			n_input, conv_size, conv_stride, conv_features, connectivity, weight_dependence, post_pre, weight_sharing, lattice_structure, \
			random_lattice_prob, random_inhibition_prob

		# number of inputs to the network
		self.n_input_sqrt = int(math.sqrt(self.n_input))
		self.n_excitatory_patch = ((self.n_input_sqrt - self.conv_size) / self.conv_stride + 1) ** 2
		self.n_excitatory = self.n_excitatory_patch * self.conv_features
		self.n_excitatory_patch_sqrt = int(math.sqrt(self.n_excitatory_patch))
		self.n_inhibitory_patch = self.n_excitatory_patch
		self.n_inhibitory = self.n_excitatory
		self.conv_features_sqrt = int(math.ceil(math.sqrt(self.conv_features)))

		# time (in seconds) per data example presentation and rest period in between
		self.single_example_time = 0.35 * b.second
		self.resting_time = 0.15 * b.second

		# set update intervals
		self.update_interval = 100
		self.weight_update_interval = 10
		self.print_progress_interval = 10

		# rest potential parameters, reset potential parameters, threshold potential parameters, and refractory periods
		v_rest_e, v_rest_i = -65. * b.mV, -60. * b.mV
		v_reset_e, v_reset_i = -65. * b.mV, -45. * b.mV
		v_thresh_e, v_thresh_i = -52. * b.mV, -40. * b.mV
		refrac_e, refrac_i = 5. * b.ms, 2. * b.ms

		# time constants, learning rates, max weights, weight dependence, etc.
		tc_pre_ee, tc_post_ee = 20 * b.ms, 20 * b.ms
		nu_ee_pre, nu_ee_post = 0.0001, 0.01
		exp_ee_post = exp_ee_pre = 0.2
		w_mu_pre, w_mu_post = 0.2, 0.2

		# parameters for neuron equations
		tc_theta = 1e7 * b.ms
		theta_plus = 0.05 * b.mV
		scr_e = 'v = v_reset_e; theta += theta_plus; timer = 0*ms'
		offset = 20.0 * b.mV
		v_thresh_e = '(v>(theta - offset + ' + str(v_thresh_e) + ')) * (timer>refrac_e)'

		# equations for neurons
		neuron_eqs_e = '''
				dv / dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / (100 * ms)  : volt
				I_synE = ge * nS * - v  : amp
				I_synI = gi * nS * (-100. * mV - v)  : amp
				dge / dt = -ge / (1.0*ms)  : 1
				dgi / dt = -gi / (2.0*ms)  : 1
				dtheta / dt = -theta / (tc_theta)  : volt
				dtimer / dt = 100.0  : ms
			'''

		neuron_eqs_i = '''
				dv/dt = ((v_rest_i - v) + (I_synE + I_synI) / nS) / (10*ms)  : volt
				I_synE = ge * nS *         -v                           : amp
				I_synI = gi * nS * (-85.*mV-v)                          : amp
				dge/dt = -ge/(1.0*ms)                                   : 1
				dgi/dt = -gi/(2.0*ms)                                  : 1
			'''

		# STDP synaptic traces
		eqs_stdp_ee = '''
				dpre / dt = -pre / tc_pre_ee : 1.0
				dpost / dt = -post / tc_post_ee : 1.0
			'''

		# dictionaries for weights and delays
		self.weight, self.delay = {}, {}

		# setting weight, delay, and intensity parameters
		self.weight['ee_input'] = (conv_size ** 2) * 0.175
		self.delay['ee_input'] = (0 * b.ms, 10 * b.ms)
		self.delay['ei_input'] = (0 * b.ms, 5 * b.ms)
		self.input_intensity = self.start_input_intensity = 2.0
		self.wmax_ee = 1.0

		# populations, connections, saved connections, etc.
		self.input_population_names = [ 'X' ]
		self.population_names = [ 'A' ]
		self.input_connection_names = [ 'XA' ]
		self.save_connections = [ 'XeAe', 'AeAe' ]
		self.input_connection_names = [ 'ee_input' ]
		self.recurrent_connection_names = [ 'ei', 'ie', 'ee' ]

		# setting STDP update rule
		if weight_dependence:
			if post_pre:
				eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post * w ** exp_ee_pre'
				eqs_stdp_post_ee = 'w += nu_ee_post * pre * (wmax_ee - w) ** exp_ee_post; post = 1.'

			else:
				eqs_stdp_pre_ee = 'pre = 1.'
				eqs_stdp_post_ee = 'w += nu_ee_post * pre * (wmax_ee - w) ** exp_ee_post; post = 1.'

		else:
			if post_pre:
				eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post'
				eqs_stdp_post_ee = 'w += nu_ee_post * pre; post = 1.'

			else:
				eqs_stdp_pre_ee = 'pre = 1.'
				eqs_stdp_post_ee = 'w += nu_ee_post * pre; post = 1.'

		print '\n'

		# for filesaving purposes
		stdp_input = ''
		if self.weight_dependence:
			stdp_input += 'weight_dependence_'
		else:
			stdp_input += 'no_weight_dependence_'
		if self.post_pre:
			stdp_input += 'post_pre'
		else:
			stdp_input += 'no_post_pre'
		if self.weight_sharing:
			use_weight_sharing = 'weight_sharing'
		else:
			use_weight_sharing = 'no_weight_sharing'

		# set ending of filename saves
		self.ending = self.connectivity + '_' + str(self.conv_size) + '_' + str(self.conv_stride) + '_' + str(self.conv_features) + \
							'_' + str(self.n_excitatory_patch) + '_' + stdp_input + '_' + \
							use_weight_sharing + '_' + str(self.lattice_structure) + '_' + str(self.random_lattice_prob) + \
							'_' + str(self.random_inhibition_prob)

		self.fig_num = 1

		# creating dictionaries for various objects
		self.neuron_groups, self.input_groups, self.connections, self.input_connections, self.stdp_methods, self.rate_monitors, \
			self.spike_monitors, self.spike_counters, self.output_numbers = {}, {}, {}, {}, {}, {}, {}, {}, {}

		# creating convolution locations inside the input image
		self.convolution_locations = {}
		for n in xrange(self.n_excitatory_patch):
			self.convolution_locations[n] = [ ((n % self.n_excitatory_patch_sqrt) * self.conv_stride + (n // self.n_excitatory_patch_sqrt) \
													* self.n_input_sqrt * self.conv_stride) + (x * self.n_input_sqrt) + y \
													for y in xrange(self.conv_size) for x in xrange(self.conv_size) ]
		
		# instantiating neuron spike / votes monitor
		self.result_monitor = np.zeros((self.update_interval, self.conv_features, self.n_excitatory_patch))

		# creating overarching neuron populations
		self.neuron_groups['e'] = b.NeuronGroup(self.n_excitatory, neuron_eqs_e, threshold=v_thresh_e, \
															refractory=refrac_e, reset=scr_e, compile=True, freeze=True)
		self.neuron_groups['i'] = b.NeuronGroup(self.n_inhibitory, neuron_eqs_i, threshold=v_thresh_i, \
															refractory=refrac_i, reset=v_reset_i, compile=True, freeze=True)

		# create neuron subpopulations
		for name in self.population_names:
			print '...creating neuron group:', name

			# get a subgroup of size 'n_e' from all exc
			self.neuron_groups[name + 'e'] = self.neuron_groups['e'].subgroup(self.conv_features * self.n_excitatory_patch)
			# get a subgroup of size 'n_i' from the inhibitory layer
			self.neuron_groups[name + 'i'] = self.neuron_groups['i'].subgroup(self.conv_features * self.n_excitatory_patch)

			# start the membrane potentials of these groups 40mV below their resting potentials
			self.neuron_groups[name + 'e'].v = v_rest_e - 40. * b.mV
			self.neuron_groups[name + 'i'].v = v_rest_i - 40. * b.mV

		print '...creating recurrent connections'

		for name in self.population_names:
			# set the adaptive additive threshold parameter at 20mV
			self.neuron_groups['e'].theta = np.ones((self.n_excitatory)) * 20.0 * b.mV

			for connection_type in self.recurrent_connection_names:
				if connection_type == 'ei':
					# create connection name (composed of population and connection types)
					connection_name = name + connection_type[0] + name + connection_type[1]
					# create a connection from the first group in conn_name with the second group
					self.connections[connection_name] = b.Connection(self.neuron_groups[connection_name[0:2]], \
													self.neuron_groups[connection_name[2:4]], structure='sparse', state='g' + conn_type[0])
					# instantiate the created connection
					for feature in xrange(self.conv_features):
						for n in xrange(self.n_excitatory_patch):
							self.connections[conn_name][feature * self.n_excitatory_patch + n, \
															feature * self.n_excitatory_patch + n] = 10.4

				elif connection_type == 'ie':
					# create connection name (composed of population and connection types)
					connection_name = name + connection_type[0] + name + connection_type[1]
					# create a connection from the first group in conn_name with the second group
					self.connections[connection_name] = b.Connection(self.neuron_groups[connection_name[0:2]], \
													self.neuron_groups[connection_name[2:4]], structure='sparse', state='g' + conn_type[0])
					# instantiate the created connection
					for feature in xrange(self.conv_features):
						for other_feature in xrange(self.conv_features):
							if feature != other_feature:
								for n in xrange(self.n_excitatory_patch):
									self.connections[connection_name][feature * self.n_excitatory_patch + n, \
															other_feature * self.n_excitatory_patch + n] = 17.4

					# adding random inhibitory connections as specified
					if self.random_inhibition_prob != 0.0:
						for feature in xrange(self.conv_features):
							for other_feature in xrange(self.conv_features):
								for n_this in xrange(self.n_excitatory_patch):
									for n_other in xrange(self.n_excitatory_patch):
										if n_this != n_other:
											if b.random() < self.random_inhibition_prob:
												self.connections[connection_name][feature * self.n_excitatory_patch + n_this, \
														other_feature * self.n_excitatory_patch + n_other] = 17.4

				elif connection_type == 'ee':
					# create connection name (composed of population and connection types)
					connection_name = name + connection_type[0] + name + connection_type[1]
					# create a connection from the first group in conn_name with the second group
					self.connections[connection_name] = b.Connection(self.neuron_groups[connection_name[0:2]], \
								self.neuron_groups[connection_name[2:4]], structure='sparse', state='g' + connection_type[0])
					# instantiate the created connection
					if self.connectivity == 'all':
						for feature in xrange(self.conv_features):
							for other_feature in xrange(self.conv_features):
								if feature != other_feature:
									for this_n in xrange(self.n_excitatory_patch):
										for other_n in xrange(self.n_excitatory_patch):
											if is_lattice_connection(self.n_excitatory_patch_sqrt, this_n, other_n):
												self.connections[connection_name][feature * self.n_excitatory_patch + this_n, \
														other_feature * self.n_excitatory_patch + other_n] = \
																(b.random() + 0.01) * 0.3

					elif self.connectivity == 'pairs':
						for feature in xrange(self.conv_features):
							if feature % 2 == 0:
								for this_n in xrange(self.n_excitatory_patch):
									for other_n in xrange(self.n_excitatory_patch):
										if is_lattice_connection(self.n_excitatory_patch_sqrt, this_n, other_n):
											self.connections[connection_name][feature * self.n_excitatory_patch + this_n, \
													(feature + 1) * self.n_excitatory_patch + other_n] = (b.random() + 0.01) * 0.3
							elif feature % 2 == 1:
								for this_n in xrange(self.n_excitatory_patch):
									for other_n in xrange(self.n_excitatory_patch):
										if is_lattice_connection(self.n_excitatory_patch_patch, this_n, other_n):
											self.connections[connection_name][feature * self.n_excitatory_patch + this_n, \
													(feature - 1) * self.n_excitatory_patch + other_n] = (b.random() + 0.01) * 0.3

					elif connectivity == 'linear':
						for feature in xrange(self.conv_features):
							if feature != self.conv_features - 1:
								for this_n in xrange(self.n_excitatory_patch):
									for other_n in xrange(self.n_excitatory_patch):
										if is_lattice_connection(self.n_excitatory_patch_sqrt, this_n, other_n):
											self.connections[connection_name][feature * self.n_excitatory_patch + this_n, \
													(feature + 1) * self.n_excitatory_patch + other_n] = \
																(b.random() + 0.01) * 0.3
							if feature != 0:
								for this_n in xrange(self.n_excitatory_patch):
									for other_n in xrange(self.n_excitatory_patch):
										if is_lattice_connection(self.n_excitatory_patch_sqrt, this_n, other_n):
											self.connections[connection_name][feature * self.n_excitatory_patch + this_n, \
													(feature - 1) * self.n_excitatory_patch + other_n] = \
																(b.random() + 0.01) * 0.3

					elif self.connectivity == 'none':
						pass

			# if STDP from excitatory -> excitatory is on and this connection is excitatory -> excitatory
			if 'ee' in self.recurrent_conn_names:
				self.stdp_methods[name + 'e' + name + 'e'] = b.STDP(self.connections[name + 'e' + name + 'e'], \
																eqs=eqs_stdp_ee, pre=eqs_stdp_pre_ee, \
																post=eqs_stdp_post_ee, wmin=0., wmax=self.wmax_ee)

			print '...creating monitors for:', name

			# spike rate monitors for excitatory and inhibitory neuron populations
			self.rate_monitors[name + 'e'] = b.PopulationRateMonitor(self.neuron_groups[name + 'e'], \
													bin=(self.single_example_time + self.resting_time) / b.second)
			self.rate_monitors[name + 'i'] = b.PopulationRateMonitor(self.neuron_groups[name + 'i'], \
													bin=(self.single_example_time + self.resting_time) / b.second)
			self.spike_counters[name + 'e'] = b.SpikeCounter(self.neuron_groups[name + 'e'])

			# record neuron population spikes
			self.spike_monitors[name + 'e'] = b.SpikeMonitor(self.neuron_groups[name + 'e'])
			self.spike_monitors[name + 'i'] = b.SpikeMonitor(self.neuron_groups[name + 'i'])

		if do_plot:
			b.figure(self.fig_num)
			fig_num += 1
			b.ion()
			b.subplot(211)
			b.raster_plot(self.spike_monitors['Ae'], refresh=1000 * b.ms, showlast=1000 * b.ms)
			b.subplot(212)
			b.raster_plot(self.spike_monitors['Ai'], refresh=1000 * b.ms, showlast=1000 * b.ms)

		# specifying locations of lattice connections
		self.lattice_locations = {}
		if self.connectivity == 'all':
			for this_n in xrange(self.conv_features * self.n_excitatory_patch):
				self.lattice_locations[this_n] = [ other_n for other_n in xrange(self.conv_features * self.n_excitatory_patch) \
												if is_lattice_connection(self.n_excitatory_patch_sqrt, \
												this_n % self.n_excitatory_patch, other_n % self.n_excitatory_patch) ]
		elif self.connectivity == 'pairs':
			for this_n in xrange(self.conv_features * self.n_excitatory_patch):
				self.lattice_locations[this_n] = []
				for other_n in xrange(self.conv_features * self.n_excitatory_patch):
					if this_n // self.n_excitatory_patch % 2 == 0:
						if is_lattice_connection(self.n_excitatory_patch_sqrt, this_n % self.n_excitatory_patch, \
													other_n % self.n_excitatory_patch) and \
													other_n // self.n_excitatory_patch == this_n // self.n_excitatory_patch + 1:
							self.lattice_locations[this_n].append(other_n)
					elif this_n // self.n_excitatory_patch % 2 == 1:
						if is_lattice_connection(self.n_excitatory_patch_sqrt, this_n % self.n_excitatory_patch, \
													other_n % self.n_excitatory_patch) and \
													other_n // self.n_excitatory_patch == this_n // self.n_excitatory_patch - 1:
							self.lattice_locations[this_n].append(other_n)
		elif self.connectivity == 'linear':
			for this_n in xrange(self.conv_features * self.n_excitatory_patch):
				self.lattice_locations[this_n] = []
				for other_n in xrange(conv_features * self.n_excitatory_patch):
					if this_n // self.n_excitatory_patch != self.conv_features - 1:
						if is_lattice_connection(self.n_excitatory_patch_sqrt, this_n % self.n_excitatory_patch, \
													other_n % self.n_excitatory_patch) and \
													other_n // self.n_excitatory_patch == this_n // self.n_excitatory_patch + 1:
							self.lattice_locations[this_n].append(other_n)
					elif this_n // self.n_excitatory_patch != 0:
						if is_lattice_connection(self.n_excitatory_patch_sqrt, this_n % self.n_excitatory_patch, \
													other_n % self.n_excitatory_patch) and \
													other_n // self.n_excitatory_patch == this_n // self.n_excitatory_patch - 1:
							self.lattice_locations[this_n].append(other_n)

		# setting up parameters for weight normalization between patches
		num_lattice_connections = sum([ len(value) for value in lattice_locations.values() ])
		self.weight['ee_recurr'] = (num_lattice_connections / self.conv_features) * 0.15

		# creating Poission spike train from input image (784 vector, 28x28 image)
		for name in self.input_population_names:
			self.input_groups[name + 'e'] = b.PoissonGroup(self.n_input, 0)
			self.rate_monitors[name + 'e'] = b.PopulationRateMonitor(self.input_groups[name + 'e'], \
														bin=(self.single_example_time + self.resting_time) / b.second)

		# creating connections from input Poisson spike train to convolution patch populations
		for name in self.input_connection_names:
			print '\n...creating connections between', name[0], 'and', name[1]
			
			# for each of the input connection types (in this case, excitatory -> excitatory)
			for connection_type in self.input_conn_names:
				# saved connection name
				connection_name = name[0] + connection_type[0] + name[1] + connection_type[1]

				# create connections from the windows of the input group to the neuron population
				self.input_connections[connection_name] = b.Connection(self.input_groups['Xe'], \
								self.neuron_groups[name[1] + connection_type[1]], structure='sparse', \
								state='g' + connection_type[0], delay=True, max_delay=self.delay[connection_type][1])

				for feature in xrange(self.conv_features):
					for n in xrange(self.n_excitatory_patch):
						for idx in xrange(self.conv_size ** 2):
							self.input_connections[connection_name][self.convolution_locations[n][idx], \
												feature * self.n_excitatory_patch + n] = (b.random() + 0.01) * 0.3

			# if excitatory -> excitatory STDP is specified, add it here (input to excitatory populations)
			print '...creating STDP for connection', name
			
			# STDP connection name
			connection_name = name[0] + connection_type[0] + name[1] + connection_type[1]
			# create the STDP object
			self.stdp_methods[connection_name] = b.STDP(self.input_connections[connection_name], \
					eqs=eqs_stdp_ee, pre=eqs_stdp_pre_ee, post=eqs_stdp_post_ee, wmin=0., wmax=self.wmax_ee)

		print '\n'

	@classmethod
	def from_file(ending):
		'''
		For the test phase, load the necessary network parameters from disk.

		ending: 
		'''
		this.ending = ending


	def train(self):
		'''
		Main loop for training phase.
		'''
		start = timeit.default_timer()
		data = get_labeled_data(MNIST_data_path + 'training', b_train=True)
		print 'time needed to load training dataset:', timeit.default_timer() - start

		runtime = len(data[0]) * (self.single_example_time + self.resting_time)


	def test(self):
		'''
		Main loop for test phase.
		'''
		start = timeit.default_timer()
		data = get_labeled_data(MNIST_data_path + 'test', b_train=False)
		print 'time needed to load test dataset:', timeit.default_timer() - start

		runtime = len(data[0]) * (self.single_example_time + self.resting_time)

	def step(self):
		'''
		Do one iteration (corresponding to a single input example) of the algorithm.
		'''
		pass


	def is_lattice_connection(self, i, j):
		'''
		Boolean method which checks if two indices in a network correspond to neighboring nodes in a 4-, 8-, or all-lattice.

		self: this SpikingCNN object
		i: First neuron's index
		j: Second neuron's index
		'''
		sqrt = int(math.sqrt(this.n_excitatory_patch))

		if this.lattice_structure == 'none':
			return False
		if this.lattice_structure == '4':
			return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j
		if this.lattice_structure == '8':
			return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j or i + sqrt == j + 1 and j % sqrt != 0 or i + sqrt == j - 1 and i % sqrt != 0 or i - sqrt == j + 1 and i % sqrt != 0 or i - sqrt == j - 1 and j % sqrt != 0
		if this.lattice_structure == 'all':
			return True


	def save_connections(self):
		'''
		Save all connections in 'save_conns'; ending may be set to the index of the last
		example run through the network

		self: this SpikingCNN object
		'''

		# iterate over all connections to save
		for connection_name in self.save_connections:
			save_name = connection_name + '_' + self.ending
			print '...saving connection: weights/conv_patch_connectivity_weights/' + save_name

			if connection_name == 'AeAe':
				connection_matrix = this.connections[connection_name][:]
			elif connection_name == 'XeAe':
				connection_matrix = this.input_connections[connection_name][:]

			# sparsify it into (row, column, entry) tuples and save it to disk
			sparse_connection = ([(i, j, connection_matrix[i, j]) for i in xrange(connection_matrix.shape[0]) for j in xrange(connection_matrix.shape[1]) ])
			np.save(top_level_path + 'weights/conv_patch_connectivity_weights/' + save_name, sparse_connection)


	def save_theta(self):
		'''
		Save the adaptive threshold parameters to a file.
		'''

		# iterate over population for which to save theta parameters
		for population_name in self.population_names:
			save_name = 'theta_' + population_name + '_' + self.ending
			# print out saved theta populations
			print '...saving theta: weights/conv_patch_connectivity_weights/' + save_name

			# save out the theta parameters to file
			np.save(top_level_path + 'weights/conv_patch_connectivity_weights/' + save_name, neuron_groups[population_name + 'e'].theta)


	def set_weights_most_fired(self):
		'''
		For each convolutional patch, set the weights to those of the neuron which
		fired the most in the last iteration.
		'''

		for connection_name in self.input_connections:
			for feature in xrange(self.conv_features):
				# count up the spikes for the neurons in this convolution patch
				column_sums = np.sum(self.current_spike_count[feature : feature + 1, :], axis=0)

				# find the excitatory neuron which spiked the most
				most_spiked = np.argmax(column_sums)

				# create a "dense" version of the most spiked excitatory neuron's weight
				most_spiked_dense = self.input_connections[connection_name][:, feature * self.n_excitatory_patch + most_spiked].todense()

				# set all other neurons' (in the same convolution patch) weights the same as the most-spiked neuron in the patch
				for n in xrange(n_excitatory_patch):
					if n != most_spiked:
						other_dense = self.input_connections[connection_name][:, feature * n_e + n].todense()
						other_dense[convolution_locations[n]] = most_spiked_dense[self.convolution_locations[most_spiked]]
						self.input_connections[connection_name][:, feature * n_e + n] = other_dense


	def normalize_weights(self):
		'''
		Squash the input -> excitatory weights to sum to a prespecified number.
		'''
		for connection_name in self.input_connections:
			connection = self.input_connections[connection_name][:].todense()
			for feature in xrange(self.conv_features):
				feature_connection = connection[:, feature * self.n_excitatory_patch : (feature + 1) * n_excitatory_patch]
				column_sums = np.sum(feature_connection, axis=0)
				column_factors = self.weight['ee_input'] / column_sums

				for n in xrange(n_e):
					dense_weights = self.input_connections[connection_name][:, feature * n_e + n].todense()
					dense_weights[convolution_locations[n]] *= column_factors[n]
					self.input_connections[connection_name][:, feature * n_e + n] = dense_weights

		for connection_name in connections:
			if 'AeAe' in connection_name and lattice_structure != 'none':
				connection = connections[connection_name][:].todense()
				for feature in xrange(conv_features):
					feature_connection = connection[feature * n_e : (feature + 1) * n_e, :]
					column_sums = np.sum(feature_connection)
					column_factors = weight['ee_recurr'] / column_sums

					for idx in xrange(feature * n_e, (feature + 1) * n_e):
						connections[connection_name][idx, :] *= column_factors


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', default='train')
	parser.add_argument('--conv_size', type=int, default=16)
	parser.add_argument('--conv_stride', type=int, default=4)
	parser.add_argument('--conv_features', type=int, default=50)
	parser.add_argument('--connectivity', default='all')
	parser.add_argument('--weight_dependence', default=False)
	parser.add_argument('--post_pre', default=True)
	parser.add_argument('--weight_sharing', default=False)
	parser.add_argument('--lattice_structure', default='4')
	parser.add_argument('--random_lattice_prob', type=float, default=0.0)
	parser.add_argument('--random_inhibition_prob', type=float, default=0.0)
	parser.add_argument('--top_percent', type=int, default=10)
	
	args = parser.parse_args()
	mode, connectivity, weight_dependence, post_pre, conv_size, conv_stride, conv_features, weight_sharing, lattice_structure, random_lattice_prob, random_inhibition_prob, top_percent = \
		args.mode, args.connectivity, args.weight_dependence, args.post_pre, args.conv_size, args.conv_stride, args.conv_features, args.weight_sharing, \
		args.lattice_structure, args.random_lattice_prob, args.random_inhibition_prob, args.top_percent

	print '\n'

	for arg in sorted(args.__dict__):
		print arg, ':', args.__dict__[arg]

	print '\n'

	# set global preferences
	b.set_global_preferences(defaultclock = b.Clock(dt=0.5*b.ms), useweave = True, gcc_options = ['-ffast-math -march=native'], usecodegen = True,
		usecodegenweave = True, usecodegenstateupdate = True, usecodegenthreshold = False, usenewpropagate = True, usecstdp = True, openmp = False,
		magic_useframes = False, useweave_linear_diffeq = True)

	np.random.seed(0)
	b.ion()

	if mode == 'train':
		network = SpikingCNN(n_input, conv_size, conv_stride, conv_features, connectivity, weight_dependence, post_pre, weight_sharing, lattice_structure, random_lattice_prob, random_inhibition_prob)
		network.build_network(mode)
		network.train()
	elif mode == 'test':
		network = SpikingCNN.from_file(ending)
		network.build_network(mode)
		network.test()
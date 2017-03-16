'''
Much the same as 'spiking_MNIST.py', but we instead use a number of convolutional
windows to map the input to a reduced space.
'''

import numpy as np
import matplotlib.cm as cmap
import time, os.path, scipy, math, sys, timeit
import cPickle as p
import brian_no_units
import brian as b

from scipy.sparse import coo_matrix
from struct import unpack
from brian import *

np.set_printoptions(threshold=np.nan)

# only show log messages of level ERROR or higher
b.log_level_error()

# specify the location of the MNIST data
MNIST_data_path = '../data/'


def get_labeled_data(picklename, bTrain = True):
    '''
    Read input-vector (image) and target class (label, 0-9) and return it as 
    a list of tuples.
    '''
    if os.path.isfile('%s.pickle' % picklename):
        data = p.load(open('%s.pickle' % picklename))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
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


def save_connections():
    '''
    Save all connections in 'save_conns'; ending may be set to the index of the last
    example run through the network
    '''

    # print out saved connections
    print '...saving connections: weights/' + save_conns[0] + '_' + stdp_input

    # iterate over all connections to save
    for conn_name in save_conns:
        # get the connection matrix for this connection
        conn_matrix = input_connections[conn_name][:]
        # sparsify it into (row, column, entry) tuples
        conn_list_sparse = ([(i, j, conn_matrix[i, j]) for i in xrange(conn_matrix.shape[0]) for j in xrange(conn_matrix.shape[1]) ])
        # save it out to disk
        np.save(data_path + 'weights/' + conn_name + '_' + stdp_input, conn_list_sparse)


def save_theta():
    '''
    Save the adaptive threshold parameters to a file.
    '''

    # iterate over population for which to save theta parameters
    for pop_name in population_names:
    	# print out saved theta populations
        print '...saving theta: weights/theta_' + pop_name + ending + '_' + stdp_input

        # save out the theta parameters to file
        np.save(data_path + 'weights/theta_' + pop_name + ending + '_' + stdp_input, neuron_groups[pop_name + 'e'].theta)


def set_weights_most_fired():
    '''
    For each convolutional patch, set the weights to those of the neuron which
    fired the most in the last iteration.
    '''
    for conn_name in input_connections:
    	for feature in xrange(conv_features):
            column_sums = np.sum(current_spike_count[feature : feature + 1, :], axis=0)
            most_spiked = np.argmax(column_sums)

            for n in xrange(n_e):
                if n != most_spiked:
                    temp1 = input_connections[conn_name][:, feature * n_e + n].todense()
                    temp2 = input_connections[conn_name][:, feature * n_e + most_spiked].todense()
                    temp1[convolution_locations[n]] = temp2[convolution_locations[most_spiked]]
                    input_connections[conn_name][:, feature * n_e + n] = temp1


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
                temp = input_connections[conn_name][:, feature * n_e + n].todense()
                temp[convolution_locations[n]] *= column_factors[n]
                input_connections[conn_name][:, feature * n_e + n] = temp


def plot_input():
    '''
    Plot the current input example during the training procedure.
    '''
    fig = b.figure(fig_num, figsize = (5, 5))
    im3 = b.imshow(rates.reshape((28, 28)), interpolation = 'nearest', vmin=0, vmax=64, cmap=cmap.get_cmap('gray'))
    b.colorbar(im3)
    b.title('Current input example')
    fig.canvas.draw()
    return im3, fig


def update_input(im3, fig):
    '''
    Update the input image to use for input plotting.
    '''
    im3.set_array(rates.reshape((28, 28)))
    fig.canvas.draw()
    return im3


def get_2d_input_weights():
    '''
    Get the weights from the input to excitatory layer and reshape it to be two
    dimensional and square.
    '''
    rearranged_weights = np.zeros(( conv_features * conv_size, conv_size * n_e ))
    
    # counts number of input -> excitatory weights displayed so far
    connection = input_connections['XeAe' + ending][:]

    # for each convolution feature
    for feature in xrange(conv_features):
        # for each excitatory neuron in this convolution feature
        for n in xrange(n_e):
            # get the connection weights from the input to this neuron
            temp = connection[:, feature * n_e + n].todense()
            # add it to the rearranged weights for displaying to the user
            rearranged_weights[feature * conv_size : (feature + 1) * conv_size, n * conv_size : (n + 1) * conv_size] = temp[convolution_locations[n]].reshape((conv_size, conv_size))

    # return the rearranged weights to display to the user
    return rearranged_weights.T


def plot_2d_input_weights():
    '''
    Plot the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_input_weights()
    fig = b.figure(fig_num, figsize=(18, 18))
    im2 = b.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    b.title('Convolutional Connection Weights')
    fig.canvas.draw()
    return im2, fig


def update_2d_input_weights(im, fig):
    '''
    Update the plot of the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im


def get_current_performance(performance, current_example_num):
    '''
    Evaluate the performance of the network on the past 'update_interval' training
    examples.
    '''
    current_evaluation = int(current_example_num / update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance


def plot_performance(fig_num):
    '''
    Set up the performance plot for the beginning of the simulation.
    '''
    num_evaluations = int(num_examples / update_interval)
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = b.figure(fig_num, figsize = (5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #my_cmap
    b.ylim(ymax = 100)
    b.title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig_num, fig


def update_performance_plot(im, performance, current_example_num, fig):
    '''
    Update the plot of the performance based on results thus far.
    '''
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance


def get_recognized_number_ranking(assignments, spike_rates):
    '''
    Given the label assignments of the excitatory layer and their spike rates over
    the past 'update_interval', get the ranking of each of the categories of input.
    '''
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in xrange(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]


def get_new_assignments(result_monitor, input_numbers):
    '''
    Based on the results from the previous 'update_interval', assign labels to the
    excitatory neurons.
    '''
    assignments = np.zeros((conv_features, n_e))
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
    
    return assignments

##############
# LOAD MNIST #
##############

if raw_input('Enter "test" for testing mode, "train" for training mode (default training mode): ') == 'test':
    test_mode = True
else:
    test_mode = False

if not test_mode:
    start = time.time()
    training = get_labeled_data(MNIST_data_path + 'training')
    end = time.time()
    print 'time needed to load training set:', end - start

else:
    start = time.time()
    testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
    end = time.time()
    print 'time needed to load test set:', end - start

################################
# SET PARAMETERS AND EQUATIONS #
################################

b.set_global_preferences(
                        defaultclock = b.Clock(dt=0.5*b.ms), # The default clock to use if none is provided or defined in any enclosing scope.
                        useweave = True, # Defines whether or not functions should use inlined compiled C code where defined.
                        gcc_options = ['-ffast-math -march=native'],  # Defines the compiler switches passed to the gcc compiler. 
                        #For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimizations are turned on 
                        usecodegen = True,  # Whether or not to use experimental code generation support.
                        usecodegenweave = True,  # Whether or not to use C with experimental code generation support.
                        usecodegenstateupdate = True,  # Whether or not to use experimental code generation support on state updaters.
                        usecodegenthreshold = False,  # Whether or not to use experimental code generation support on thresholds.
                        usenewpropagate = True,  # Whether or not to use experimental new C propagation functions.
                        usecstdp = True,  # Whether or not to use experimental new C STDP.
                        openmp = False, # whether or not to use OpenMP pragmas in generated C code.
                        magic_useframes = False, # defines whether or not the magic functions should serach for objects defined only in the calling frame,
                                                # or if they should find all objects defined in any frame. Set to "True" if not in an interactive shell.
                        useweave_linear_diffeq = True, # Whether to use weave C++ acceleration for the solution of linear differential equations.
                       )

# for reproducibility's sake
np.random.seed(0)

# where the MNIST data files are stored
data_path = '../'

# set parameters for simulation based on train / test mode
if test_mode:
    weight_path = data_path + 'weights/'
    num_examples = 100 * 1
    use_testing_set = True
    do_plot_performance = False
    record_spikes = True
    ee_STDP_on = False
    update_interval = num_examples
else:
    weight_path = data_path + 'random/'
    num_examples = 100 * 1
    use_testing_set = False
    do_plot_performance = True
    record_spikes = True
    ee_STDP_on = True

# plotting or not
do_plot = False

# number of inputs to the network
n_input = 784
n_input_sqrt = int(math.sqrt(n_input))

# number of classes to learn
classes_input = raw_input('Enter classes to learn as comma-separated list (e.g, 0,1,2,3,...) (default all 10 classes): ')
if classes_input == '':
    classes = range(10)
else:
    classes = set([ int(token) for token in classes_input.split(',') ])

# reduce size of dataset if necessary
if not test_mode and classes_input != '':
    new_training = {'x' : [], 'y' : [], 'rows' : training['rows'], 'cols' : training['cols']}
    for idx in xrange(len(training['x'])):
        if training['y'][idx][0] in classes:
            new_training['y'].append(training['y'][idx])
            new_training['x'].append(training['x'][idx])
    new_training['x'], new_training['y'] = np.asarray(new_training['x']), np.asarray(new_training['y'])
    training = new_training

elif test_mode and classes_input != '':
    new_testing = {'x' : [], 'y' : [], 'rows' : testing['rows'], 'cols' : testing['cols']}
    for idx in xrange(len(testing['x'])):
        if testing['y'][idx][0] in classes:
            new_testing['y'].append(testing['y'][idx])
            new_testing['x'].append(testing['x'][idx])
    new_testing['x'], new_testing['y'] = np.asarray(new_testing['x']), np.asarray(new_testing['y'])
    testing = new_testing

# size of convolution windows
conv_size = raw_input('Enter size of square side length of convolution window (default 27): ')
if conv_size == '':
    conv_size = 27
else:
    conv_size = int(conv_size)

# stride of convolution windows
conv_stride = raw_input('Enter stride size of convolution window (default 1): ')
if conv_stride == '':
    conv_stride = 1
else:
    conv_stride = int(conv_stride)

# number of convolution features
conv_features = raw_input('Enter number of convolution features to learn (default 10): ')
if conv_features == '':
    conv_features = 10
else:
    conv_features = int(conv_features)

# number of excitatory neurons (number output from convolutional layer)
n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2

n_e_total = n_e * conv_features
n_e_sqrt = int(math.sqrt(n_e))

# number of inhibitory neurons (number of convolutational features (for now))
n_i = n_e

# set ending of filename saves
ending = '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e)

# time (in seconds) per data example presentation
single_example_time = 0.35 * b.second

# time (in seconds) per rest period between data examples
resting_time = 0.15 * b.second

# total runtime (number of examples times (presentation time plus rest period))
runtime = num_examples * (single_example_time + resting_time)

# set the update interval
if num_examples <= 10000:
    update_interval = num_examples
else:
    update_interval = 100

# set weight update interval (plotting)
weight_update_interval = 1

# set progress printing interval
print_progress_interval = 10

# rest potential parameters, reset potential parameters, threshold potential parameters, and refractory periods
v_rest_e = -65. * b.mV
v_rest_i = -60. * b.mV
v_reset_e = -65. * b.mV
v_reset_i = -45. * b.mV
v_thresh_e = -52. * b.mV
v_thresh_i = -40. * b.mV
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms

# dictionaries for weights and delays
weight = {}
delay = {}

# naming neuron populations (X for input, A for population, XA for input -> connection, etc...
input_population_names = [ 'X' ]
population_names = [ 'A' ]
input_connection_names = [ 'XA' ]
save_conns = [ 'XeAe' + ending ]
input_conn_names = [ 'ee_input' ]
recurrent_conn_names = [ 'ei', 'ie' ]
weight['ee_input'] = (conv_size ** 2) / 7.0
delay['ee_input'] = (0 * b.ms, 10 * b.ms)
delay['ei_input'] = (0 * b.ms, 5 * b.ms)
input_intensity = start_input_intensity = 2.0

# time constants, learning rates, max weights, weight dependence, etc.
tc_pre_ee = 20 * b.ms
tc_post_ee = 20 * b.ms
nu_ee_pre =  0.0001
nu_ee_post = 0.01
wmax_ee = 1.0
exp_ee_post = exp_ee_pre = 0.2
w_mu_pre = 0.2
w_mu_post = 0.2

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

# determine STDP rule to use
stdp_input = ''

if raw_input('Use weight dependence (default no)?: ') in [ 'no', '' ]:
	use_weight_dependence = False
	stdp_input += 'weight_dependence_'
else:
	use_weight_dependence = True
	stdp_input += 'no_weight_dependence_'

if raw_input('Enter (yes / no) for post-pre (default yes): ') in [ 'yes', '' ]:
	post_pre = True
	stdp_input += 'postpre'
else:
	post_pre = False
	stdp_input += 'no_postpre'

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
    if post_pre:
        eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post'
        eqs_stdp_post_ee = 'w += nu_ee_post * pre; post = 1.'

    else:
        eqs_stdp_pre_ee = 'pre = 1.'
        eqs_stdp_post_ee = 'w += nu_ee_post * pre; post = 1.'


b.ion()

fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
input_connections = {}
stdp_methods = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}

result_monitor = np.zeros((update_interval, conv_features, n_e))

neuron_groups['e'] = b.NeuronGroup(n_e_total, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=scr_e, compile=True, freeze=True)
neuron_groups['i'] = b.NeuronGroup(n_e_total / n_e, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i, compile=True, freeze=True)

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
    if test_mode or weight_path[-8:] == 'weights/':
        # load up adaptive threshold parameters
        neuron_groups['e'].theta = np.load(weight_path + 'theta_A' + ending + '_' + stdp_input + '.npy')
    else:
        # otherwise, set the adaptive additive threshold parameter at 20mV
        neuron_groups['e'].theta = np.ones((n_e_total)) * 20.0 * b.mV

    for conn_type in recurrent_conn_names:
        if conn_type == 'ei':
            # create connection name (composed of population and connections types)
            conn_name = name + conn_type[0] + name + conn_type[1] + ending
            # get the corresponding stored weights from file
            weight_matrix = get_matrix_from_file(data_path + 'random/' + conn_name + '.npy', n_src=conv_features * n_e, n_tgt=conv_features)
            # create a connection from the first group in conn_name with the second group
            connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
            # instantiate the created connection with the 'weightMatrix' loaded from file
            for feature in xrange(conv_features):
                for n in xrange(n_e):
                    connections[conn_name][feature * n_e + n, feature] = weight_matrix[feature * n_e + n, feature]

        elif conn_type == 'ie':
            # create connection name (composed of population and connections types)
            conn_name = name + conn_type[0] + name + conn_type[1] + ending
            # get the corresponding stored weights from file
            weight_matrix = get_matrix_from_file(data_path + 'random/' + conn_name + '.npy', n_src=conv_features, n_tgt=(conv_features ** 2) * n_e)
            # create a connection from the first group in conn_name with the second group
            connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
            # instantiate the created connection with the 'weightMatrix' loaded from file
            for feature in xrange(conv_features):
                for other_feature in xrange(conv_features):
                    if feature != other_feature:
                        for n in xrange(n_e):
                            connections[conn_name][feature, other_feature * n_e + n] = weight_matrix[feature, other_feature * n_e + n]


    # if STDP from excitatory -> excitatory is on and this connection is excitatory -> excitatory
    if ee_STDP_on and 'ee' in recurrent_conn_names:
        stdp_methods[name + 'e' + name + 'e'] = b.STDP(connections[name + 'e' + name + 'e' + ending], eqs=eqs_stdp_ee, pre=eqs_stdp_pre_ee, post=eqs_stdp_post_ee, wmin=0., wmax=wmax_ee)

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
    
################################################################# 
# CREATE INPUT POPULATION AND CONNECTIONS FROM INPUT POPULATION #
#################################################################

# creating convolution locations inside the input image
convolution_locations = {}
for n in xrange(n_e):
    convolution_locations[n] = [ ((n % n_e_sqrt) * conv_stride + (n // n_e_sqrt) * n_input_sqrt * conv_stride) + (x * n_input_sqrt) + y for y in xrange(conv_size) for x in xrange(conv_size) ]

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
        conn_name = name[0] + conn_type[0] + name[1] + conn_type[1] + ending
        
        # get weight matrix depending on training or test phase
        if test_mode:
            weight_matrix = get_matrix_from_file(weight_path + conn_name + '_' + stdp_input + '.npy', n_input, conv_features * n_e)
        else:
            weight_matrix = get_matrix_from_file(weight_path + conn_name + '.npy', n_input, conv_features * n_e)	

        # create connections from the windows of the input group to the neuron population
        input_connections[conn_name] = b.Connection(input_groups['Xe'], neuron_groups[name[1] + conn_type[1]], structure='sparse', state='g' + conn_type[0], delay=True, max_delay=delay[conn_type][1])
        
        for feature in xrange(conv_features):
            for n in xrange(n_e):
                for idx in xrange(conv_size ** 2):
                    input_connections[conn_name][convolution_locations[n][idx], feature * n_e + n] = weight_matrix[convolution_locations[n][idx], feature * n_e + n]

    # if excitatory -> excitatory STDP is specified, add it here (input to excitatory populations)
    if ee_STDP_on:
        print '...creating STDP for connection', name
        
        # STDP connection name
        conn_name = name[0] + conn_type[0] + name[1] + conn_type[1] + ending
        # create the STDP object
        stdp_methods[conn_name] = b.STDP(input_connections[conn_name], eqs=eqs_stdp_ee, pre=eqs_stdp_pre_ee, post=eqs_stdp_post_ee, wmin=0., wmax=wmax_ee)

print '\n'

#################################
# RUN SIMULATION AND SET INPUTS #
#################################

# bookkeeping variables
previous_spike_count = np.zeros((conv_features, n_e))
assignments = np.zeros((conv_features, n_e))
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))

# plot input weights
if not test_mode and do_plot:
    input_weight_monitor, fig_weights = plot_2d_input_weights()
    fig_num += 1

# plot input intensities
if do_plot:
    rates = np.zeros((n_input_sqrt, n_input_sqrt))
    input_image_monitor, input_image = plot_input()
    fig_num += 1

# plot performance
if do_plot_performance and do_plot:
    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)
else:
    performance = get_current_performance(np.zeros(int(num_examples / update_interval)), 0)

# set firing rates to zero initially
for name in input_population_names:
    input_groups[name + 'e'].rate = 0

# initialize network
j = 0
b.run(0)

# start recording time
start_time = timeit.default_timer()

while j < num_examples:
    # fetched rates depend on training / test phase, and whether we use the 
    # testing dataset for the test phase
    if test_mode:
        if use_testing_set:
            rates = testing['x'][j % 10000, :, :] / 8. * input_intensity
        else:
            rates = training['x'][j % 60000, :, :] / 8. * input_intensity
    
    else:
    	# ensure weights don't grow without bound
        normalize_weights()
        # get the firing rates of the next input example
        rates = training['x'][j % 60000, :, :] / 8. * input_intensity
    
    # plot the input at this step
    if do_plot:
        input_image_monitor = update_input(input_image_monitor, input_image)
    
    # sets the input firing rates
    input_groups['Xe'].rate = rates.reshape(n_input)
    
    # run the network for a single example time
    b.run(single_example_time)
    
    # get new neuron label assignments every 'update_interval'
    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j - update_interval : j])
    
    # get count of spikes over the past iteration
    current_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e)) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e))
    
    # set weights to those of the most-fired neuron
    set_weights_most_fired()

    # update weights every 'weight_update_interval'
    if j % weight_update_interval == 0 and not test_mode and do_plot:
        update_2d_input_weights(input_weight_monitor, fig_weights)
    
    # if the neurons in the network didn't spike more than four times
    if np.sum(current_spike_count) < 5:
        # increase the intensity of input
        input_intensity += 2
        
        # set all network firing rates to zero
        for name in input_population_names:
            input_groups[name + 'e'].rate = 0

        # let the network relax back to equilibrium
        b.run(resting_time)
    # otherwise, record results and confinue simulation
    else:
    	# record the current number of spikes
        result_monitor[j % update_interval, :] = current_spike_count
        
        # decide whether to evaluate on test or training set
        if test_mode and use_testing_set:
            input_numbers[j] = testing['y'][j % 10000][0]
        else:
            input_numbers[j] = training['y'][j % 60000][0]
        
        # get the output classifications of the network
        outputNumbers[j, :] = get_recognized_number_ranking(assignments, result_monitor[j % update_interval, :])
        
        # print progress
        if j % print_progress_interval == 0 and j > 0:
            print 'runs done:', j, 'of', int(num_examples), '(time taken for past', print_progress_interval, 'runs:', str(timeit.default_timer() - start_time) + ')'
            start_time = timeit.default_timer()
        
        # plot performance if appropriate
        if j % update_interval == 0 and j > 0:
            if do_plot_performance and do_plot:
                # updating the performance plot
                perf_plot, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
            else:
                performance = get_current_performance(performance, j)
            # printing out classification performance results so far
            print '\nClassification performance', performance[:int(j / float(update_interval)) + 1], '\n'
            target = open('../performance/' + conn_name + '_' + stdp_input + '_' + ending + '_iter_' + str(j), 'w')
            target.truncate()
            target.write(str(performance[:int(j / float(update_interval)) + 1]))
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

################ 
# SAVE RESULTS #
################ 

print '...saving results'

if not test_mode:
    save_theta()
if not test_mode:
    save_connections()
else:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples) + '_' + stdp_input + ending, result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples) + '_' + stdp_input + ending, input_numbers)

################ 
# PLOT RESULTS #
################

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

    b.ioff()
    b.show()
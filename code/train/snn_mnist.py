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

np.set_printoptions(threshold=np.nan)

# only show log messages of level ERROR or higher
b.log_level_error()

# set these appropriate to your directory structure
top_level_path = os.path.join('..', '..')
MNIST_data_path = os.path.join(top_level_path, 'data')
model_name = 'snn'

assignments_dir = os.path.join(top_level_path, 'assignments', model_name)
performance_dir = os.path.join(top_level_path, 'performance', model_name)
activity_dir = os.path.join(top_level_path, 'activity', model_name)
weights_dir = os.path.join(top_level_path, 'weights', model_name)
random_dir = os.path.join(top_level_path, 'random', model_name)

results_path = os.path.join(top_level_path, 'results', model_name)

for d in [ assignments_dir, performance_dir, activity_dir, weights_dir, random_dir, results_path ]:
    if not os.path.isdir(d):
        os.makedirs(d)


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


def save_connections():
    '''
    Save all connections in 'save_conns'; ending may be set to the index of the last
    example run through the network
    '''
    # iterate over all connections to save
    for conn_name in save_conns:
        # get the connection matrix for this connection
        conn_matrix = input_connections[conn_name][:].todense()
        # save it out to disk
        np.save(os.path.join(weights_dir, '_'.join([conn_name, ending])), conn_matrix)


def save_theta():
    '''
    Save the adaptive threshold parameters to a file.
    '''
    # iterate over population for which to save theta parameters
    for pop_name in population_names:
        # save out the theta parameters to file
        np.save(os.path.join(weights_dir, '_'.join(['theta', pop_name, ending])), neuron_groups[pop_name + 'e'].theta)


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


def plot_2d_input_weights():
    '''
    Plot the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_input_weights()
    fig = b.figure(fig_num, figsize=(18, 18))
    im = b.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
    b.colorbar(im)
    b.title('Reshaped weights from input -> excitatory layer')
    fig.canvas.draw()
    return im, fig


def update_2d_input_weights(im, fig):
    '''
    Update the plot of the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()


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

    most_spiked_array = np.array(np.zeros((conv_features, n_e)), dtype=bool)

    for feature in xrange(conv_features):
        # count up the spikes for the neurons in this convolution patch
        column_sums = np.sum(spike_rates[feature : feature + 1, :], axis=0)

        # find the excitatory neuron which spiked the most
        most_spiked_array[feature, np.argmax(column_sums)] = True

    # for each label
    for i in xrange(10):
        # get the number of label assignments of this type
        num_assignments[i] = len(np.where(assignments[most_spiked_array] == i))
        if num_assignments[i] > 0:
            # sum the spike rates of all excitatory neurons with this label, which fired the most in its patch
            summed_rates[i] = np.sum(spike_rates[np.where(assignments[most_spiked_array] == i)]) / num_assignments[i]

    return np.argsort(summed_rates)[::-1]


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
    
    return assignments

def evaluate_results():
    '''
    Evalute the network using the various voting schemes in test mode
    '''
    test_results = np.zeros((10, num_examples))

    for i in xrange(num_examples):
        test_results[:, i] = get_recognized_number_ranking(assignments, result_monitor[i, :])

    difference = test_results[0, :] - input_numbers
    correct = len(np.where(difference == 0)[0])
    incorrect = np.where(difference != 0)[0]
    accuracy = correct / float(num_examples) * 100

    print 'Sum response - accuracy: ', accuracy, ' num incorrect: ', len(incorrect)

    results = pd.DataFrame([[ending] + accuracies.values()], columns=['Model'] + accuracies.keys())
    filename = '_'.join([str(conv_features), 'results.csv'])
    if not filename in os.listdir(results_path):
        results.to_csv(os.path.join(results_path, filename), index=False)
    else:
        all_results = pd.read_csv(os.path.join(results_path, filename))
        all_results = pd.concat([all_results, results], ignore_index=True)
        all_results.to_csv(os.path.join(results_path, filename), index=False)

    print '\n'

################################
# SET PARAMETERS AND EQUATIONS #
################################

b.set_global_preferences( 
                        defaultclock = b.Clock(dt=0.1*b.ms), # The default clock to use if none is provided or defined in any enclosing scope.
                        useweave = True, # Defines whether or not functions should use inlined compiled C code where defined.
                        gcc_options = ['-ffast-math -march=native'],  # Defines the compiler switches passed to the gcc compiler. 
                        #For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimizations are turned on 
                        usecodegen = True,  # Whether or not to use experimental code generation support.
                        usecodegenweave = True,  # Whether or not to use C with experimental code generation support.
                        usecodegenstateupdate = True,  # Whether or not to use experimental code generation support on state updaters.
                        usecodegenthreshold = False,  # Whether or not to use experimental code generation support on thresholds.
                        usenewpropagate = True,  # Whether or not to use experimental new C propagation functions.
                        usecstdp = True,  # Whether or not to use experimental new C STDP.
                       )

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', default=0, type=int, help='Random seed: sets initialization of weights and Poisson random firing.')
parser.add_argument('--mode', default='train', type=str, help='Training or testing phase of network.')
parser.add_argument('--num_train', default=60000, type=int, help='Number of examples to train on.')
parser.add_argument('--num_test', default=10000, type=int, help='Number of examples to test on.')
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--conv_features', default=400, type=int, help='Number of neurons in excitatory, inhibitory population (in this case).')
parser.set_defaults(plot=False)

# parse arguments and place them in local scope
args = parser.parse_args()
args = vars(args)
locals().update(args)

print '\nOptional argument values:'
for key, value in args.items():
    print '-', key, ':', value

print '\n'

# for reproducibility's sake
np.random.seed(random_seed)

# test or training mode
test_mode = mode == 'test'

if test_mode:
    num_examples = num_test
else:
    num_examples = num_train

if test_mode:
    data_size = 10000
else:
    data_size = 60000

##############
# LOAD MNIST #
##############

if not test_mode:
    start = time.time()
    training = get_labeled_data(os.path.join(MNIST_data_path, 'training'))
    end = time.time()
    print 'time needed to load training set:', end - start

else:
    start = time.time()
    testing = get_labeled_data(os.path.join(MNIST_data_path, 'testing'), bTrain = False)
    end = time.time()
    print 'time needed to load test set:', end - start

# set parameters for simulation based on train / test mode
if test_mode:
    use_testing_set = True
    record_spikes = True
else:
    use_testing_set = False
    record_spikes = True

# number of inputs to the network
n_input = 784
n_input_sqrt = int(math.sqrt(n_input))

# size of convolution windows, stride, and features per patch
conv_size = 28
conv_stride = 0

# number of excitatory neurons (number output from convolutional layer)
n_e = 1
n_e_total = n_e * conv_features
n_e_sqrt = int(math.sqrt(n_e))

# number of inhibitory neurons (number of convolutional features (for now))
n_i = n_e

# set ending of filename saves
ending = '_'.join([str(conv_size), str(conv_stride), str(conv_features), str(n_e), str(random_seed), str(num_train)]) 

# time (in seconds) per data example presentation
single_example_time = 0.35 * b.second

# time (in seconds) per rest period between data examples
resting_time = 0.15 * b.second

# total runtime (number of examples times (presentation time plus rest period))
runtime = num_examples * (single_example_time + resting_time)

# set the update interval
if test_mode:
    update_interval = num_examples
else:
    update_interval = 250

# set weight update interval (plotting)
weight_update_interval = 10

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
weight, delay = {}, {}

# populations, connections, saved connections, etc.
input_population_names = [ 'X' ]
population_names = [ 'A' ]
input_connection_names = [ 'XA' ]
save_conns = [ 'XeAe' ]

# weird and bad names for variables, I think
input_conn_names = [ 'ee_input' ]
recurrent_conn_names = [ 'ei', 'ie' ]

# setting weight, delay, and intensity parameters
weight['ee_input'] = 78.
delay['ee_input'] = (0 * b.ms, 10 * b.ms)
delay['ei_input'] = (0 * b.ms, 5 * b.ms)
input_intensity = start_input_intensity = 2.0

# time constants, learning rates, max weights, weight dependence, etc.
tc_pre_ee = 20*b.ms
tc_post_1_ee = 20*b.ms
tc_post_2_ee = 40*b.ms
nu_ee_pre, nu_ee_post = 0.0001, 0.01
nu_AeAe_pre, nu_Ae_Ae_post = 0.1, 0.5
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

# STDP synaptic traces
eqs_stdp_ee = '''
                post2before                            : 1.0
                dpre/dt   =   -pre/(tc_pre_ee)         : 1.0
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1.0
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1.0
            '''
eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1'
eqs_stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2before; post1 = 1.; post2 = 1.'

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
neuron_groups['i'] = b.NeuronGroup(n_e_total, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i, compile=True, freeze=True)

########################################################
# CREATE NETWORK POPULATIONS AND RECURRENT CONNECTIONS #
########################################################

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
        neuron_groups['e'].theta = np.load(os.path.join(weights_dir, '_'.join(['theta_A', ending + '.npy'])))
    else:
        # otherwise, set the adaptive additive threshold parameter at 20mV
        neuron_groups['e'].theta = np.ones((n_e_total)) * 20.0 * b.mV

    for conn_type in recurrent_conn_names:
        if conn_type == 'ei':
            # create connection name (composed of population and connections types)
            conn_name = name + conn_type[0] + name + conn_type[1]
            # create a connection from the first group in conn_name with the second group
            connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
            # instantiate the created connection with the 'weightMatrix' loaded from file
            for feature in xrange(conv_features):
                for n in xrange(n_e):
                    connections[conn_name][feature * n_e + n, feature * n_e + n] = 10.4

        elif conn_type == 'ie':
            # create connection name (composed of population and connections types)
            conn_name = name + conn_type[0] + name + conn_type[1]
            # create a connection from the first group in conn_name with the second group
            connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure='sparse', state='g' + conn_type[0])
            # instantiate the created connection with the 'weightMatrix' loaded from file
            for feature in xrange(conv_features):
                for other_feature in xrange(conv_features):
                    if feature != other_feature:
                        for n in xrange(n_e):
                            connections[conn_name][feature * n_e + n, other_feature * n_e + n] = 17.4


    print '...creating monitors for:', name

    # spike rate monitors for excitatory and inhibitory neuron populations
    rate_monitors[name + 'e'] = b.PopulationRateMonitor(neuron_groups[name + 'e'], bin=(single_example_time + resting_time) / b.second)
    rate_monitors[name + 'i'] = b.PopulationRateMonitor(neuron_groups[name + 'i'], bin=(single_example_time + resting_time) / b.second)
    spike_counters[name + 'e'] = b.SpikeCounter(neuron_groups[name + 'e'])

    # record neuron population spikes if specified
    if record_spikes:
        spike_monitors[name + 'e'] = b.SpikeMonitor(neuron_groups[name + 'e'])
        spike_monitors[name + 'i'] = b.SpikeMonitor(neuron_groups[name + 'i'])

if record_spikes and plot:
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
        conn_name = name[0] + conn_type[0] + name[1] + conn_type[1]
        
        # get weight matrix depending on training or test phase
        if test_mode:
            weight_matrix = np.load(os.path.join(weights_dir, '_'.join([conn_name, ending + '.npy'])))
        else:
            weight_matrix = (b.random([n_input, conv_features]) + 0.01) * 0.3

        # create connections from the windows of the input group to the neuron population
        input_connections[conn_name] = b.Connection(input_groups['Xe'], neuron_groups[name[1] + conn_type[1]], \
                        structure='sparse', state='g' + conn_type[0], delay=True, max_delay=delay[conn_type][1])
        input_connections[conn_name].connect(input_groups[conn_name[0:2]], \
            neuron_groups[conn_name[2:4]], weight_matrix, delay=delay[conn_type])        

    # if excitatory -> excitatory STDP is specified, add it here (input to excitatory populations)
        if not test_mode:
            print '...Creating STDP for connection', name
            
            # STDP connection name
            conn_name = name[0] + conn_type[0] + name[1] + conn_type[1]
            # create the STDP object
            stdp_methods[conn_name] = b.STDP(input_connections[conn_name], eqs=eqs_stdp_ee, \
                            pre=eqs_stdp_pre_ee, post=eqs_stdp_post_ee, wmin=0., wmax=wmax_ee)

print '\n'

#################################
# RUN SIMULATION AND SET INPUTS #
#################################

# bookkeeping variables
previous_spike_count = np.zeros((conv_features, n_e))
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))

if test_mode:
    assignments = np.load(os.path.join(assignments_dir, '_'.join(['assignments', ending + '.npy'])))
else:
    assignments = -1 * np.ones((conv_features, n_e))

# plot input weights
if not test_mode and plot:
    input_weight_monitor, fig_weights = plot_2d_input_weights()
    fig_num += 1

# plot input intensities
if plot:
    rates = np.zeros((n_input_sqrt, n_input_sqrt))
    input_image_monitor, input_image = plot_input()
    fig_num += 1

# plot performance
if plot:
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
    if plot:
        input_image_monitor = update_input(input_image_monitor, input_image)
    
    # sets the input firing rates
    input_groups['Xe'].rate = rates.reshape(n_input)
    
    # run the network for a single example time
    b.run(single_example_time)
    
    # get new neuron label assignments every 'update_interval'
        if j % update_interval == 0 and j > 0:
            if j % data_size == 0:
                assignments = get_new_assignments(result_monitor[:], input_numbers[j - update_interval : j])
            else:
                assignments = get_new_assignments(result_monitor[:], input_numbers[(j % data_size) - update_interval : j % data_size])

    # get count of spikes over the past iteration
    current_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e)) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e))
    
    # update weights every 'weight_update_interval'
    if j % weight_update_interval == 0 and not test_mode and plot:
        update_2d_input_weights(input_weight_monitor, fig_weights)
    
    # if the neurons in the network didn't spike more than four times
    if np.sum(current_spike_count) < 5:
        # increase the intensity of input
        input_intensity += 1

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
            # pickling performance recording and iteration number
            p.dump((j, performance), open(os.path.join(performance_dir, ending + '.p'), 'wb'))

            if plot:
                # updating the performance plot
                perf_plot, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
            else:
                performance = get_current_performance(performance, j)
            
            # printing out classification performance results so far
            print '\nClassification performance', performance[:int(j / float(update_interval)) + 1], '\n'
                
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
    save_connections()

    np.save(os.path.join(assignments_dir, '_'.join(['assignments', ending])), assignments)
else:
    np.save(os.path.join(activity_dir, '_'.join(['resultPopVecs', ending])), result_monitor)
    np.save(os.path.join(activity_dir, '_'.join(['inputNumbers', ending])), input_numbers)

# evaluate results
if test_mode:
    evaluate_results()
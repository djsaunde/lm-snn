'''
Much the same as 'spiking_MNIST.py', but we instead use a number of convolutional
windows to map the input to a reduced space.
'''

import numpy as np
import matplotlib.cm as cmap
import time, os.path, scipy, math, sys
import cPickle as pickle
import brian_no_units
import brian as b
import cPickle as p

from struct import unpack
from brian import *

# specify the location of the MNIST data
MNIST_data_path = './'


def get_labeled_data(picklename, bTrain = True):
    '''
    Read input-vector (image) and target class (label, 0-9) and return it as 
    a list of tuples.
    '''
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
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
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data


def get_matrix_from_file(file_name):
    '''
    Given the name of a file pointing to a .npy ndarray object, load it into
    'weight_matrix' and return it
    '''

    offset = len(weight_path)

    # connection comes from input
    if file_name[offset] == 'X':
        n_src = n_input
    else:
        # connection comes from excitatory layer
        if file_name[offset + 1] == 'e':
            n_src = n_e
        # connection comes from inhibitory layer
        else:
            n_src = n_i

    # connection goes to excitatory layer
    if file_name[offset + 3] == 'e':
        n_tgt = n_e
    # connection goes to inhibitory layer
    else:
        n_tgt = n_i

    # load the stored ndarray into 'readout', instantiate 'weight_matrix' as 
    # correctly-shaped zeros matrix
    readout = np.load(file_name)
    weight_matrix = np.zeros((n_src, n_tgt))

    # read the 'readout' ndarray values into weight_matrix by (row, column) indices
    weight_matrix[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]

    # return the weight matrix read from file
    return weight_matrix


def save_connections(ending = ''):
    '''
    Save all connections in 'save_conns'; ending may be set to the index of the last
    example run through the network
    '''

    # print out saved connections
    print '...saving connections: ' + ', '.join(save_conns)

    # iterate over all connections to save
    for connName in save_conns:
        # get the connection matrix for this connection
        connMatrix = connections[connName][:]
        # sparsify it into (row, column, entry) tuples
        connListSparse = ([(i,j,connMatrix[i,j]) for i in xrange(connMatrix.shape[0]) for j in xrange(connMatrix.shape[1]) ])
        # save it out to disk
        np.save(data_path + 'weights/' + connName + '_' + stdp_input + '_' + ending, connListSparse)


def save_theta(ending = ''):
    '''
    Save the adaptive threshold parameters to a file.
    '''

    # print out saved theta populations
    print '...saving theta:  ' + ', '.join(population_names)

    # iterate over population for which to save theta parameters
    for pop_name in population_names:
    	# save out the theta parameters to file
        np.save(data_path + 'weights/theta_' + pop_name + '_' + stdp_input + '_' + ending, neuron_groups[pop_name + 'e'].theta)


def normalize_weights():
    '''
    Squash the weights to sum to a prespecified number.
    '''
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            connection = connections[connName][:]
            temp_conn = np.copy(connection)
            colSums = np.sum(temp_conn, axis = 0)
            colFactors = weight['ee_input'] / colSums
            for j in xrange(n_e):
                connection[:,j] *= colFactors[j]


def is_lattice_connection(n, i, j):
    '''
    Boolean method which checks if two indices in a network correspond to neighboring nodes in a lattice.

    n: number of nodes in lattice
    i: First neuron's index
    k: Second neuron's index
    '''
    sqrt = int(math.sqrt(n))
    return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j


def get_2d_input_weights():
    '''
    Get the weights from the input to excitatory layer and reshape it to be two
    dimensional and square.
    '''
    name = 'XeAe' + str(n_e)
    weight_matrix = np.zeros((n_input, n_e))

    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))

    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = connections[name][:]
    weight_matrix = np.copy(connMatrix)

    for i in xrange(n_e_sqrt):
        for j in xrange(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))

    return rearranged_weights


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
    b.title('Current input example')
    fig.canvas.draw()
    return im3


def plot_2d_input_weights():
    '''
    Plot the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_input_weights()
    fig = b.figure(fig_num, figsize = (18, 18))
    im2 = b.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    b.title('weights of connection ' + name)
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
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    for j in xrange(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
            for i in xrange(n_e):
                if rate[i] > maximum_rate[i]:
                    maximum_rate[i] = rate[i]
                    assignments[i] = j
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
                        magic_useframes = True, # defines whether or not the magic functions should serach for objects defined only in the calling frame,
                                                # or if they should find all objects defined in any frame. Set to "True" if not in an interactive shell.
                        useweave_linear_diffeq = True, # Whether to use weave C++ acceleration for the solution of linear differential equations.
                       )

# for reproducibility's sake
np.random.seed(0)

# where the MNIST data files are stored
data_path = './'

# set parameters for simulation based on train / test mode
if test_mode:
    weight_path = data_path + 'weights/'
    num_examples = 10000 * 1
    use_testing_set = True
    do_plot_performance = False
    record_spikes = True
    ee_STDP_on = False
    update_interval = num_examples
else:
    weight_path = data_path + 'random/'
    num_examples = 60000 * 1
    use_testing_set = False
    do_plot_performance = True
    record_spikes = False
    ee_STDP_on = True


# number of inputs to the network
n_input = 784
sqrt = int(math.sqrt(n_input))

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
conv_size = raw_input('Enter size of square side length of convolution window (default 20): ')
if conv_size == '':
    conv_size = 20
else:
    conv_size = int(conv_size)

# stride of convolution windows
conv_stride = raw_input('Enter stride size of convolution window (default 8): ')
if conv_stride == '':
    conv_stride = 8
else:
    conv_stride = int(conv_stride)

# number of convolution features
conv_features = raw_input('Enter number of convolution features to learn (default 10): ')
if conv_features == '':
    conv_features = 10
else:
    conv_features = int(conv_features)

# number of excitatory neurons (number output from convolutional layer)
n_e = ((sqrt - conv_stride) / conv_stride + 1) ** 2

# number of inhibitory neurons (number of convolutational features (for now))
n_i = conv_features

# number of excitatory neurons per convolution map
n_e_conv = n_e / conv_features

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
weight_update_interval = 25

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
input_population_names = ['X']
population_names = [ 'A' + str(i) for i in range(conv_features) ]
input_connection_names = [ 'XA' + str(i) for i in range(conv_features) ]
save_conns = ['XeAe' + ending]
input_conn_names = ['ee_input']
recurrent_conn_names = [ 'ei', 'ie']
weight['ee_input'] = 78.0
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
stdp_methods = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}

result_monitor = np.zeros((update_interval,n_e))

neuron_groups['e'] = b.NeuronGroup(n_e, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=scr_e, compile=True, freeze=True)
neuron_groups['i'] = b.NeuronGroup(n_i, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i, compile=True, freeze=True)

########################################################
# CREATE NETWORK POPULATIONS AND RECURRENT CONNECTIONS #
########################################################

for name in population_names:
    print '...creating neuron group:', name

    # get a subgroup of size 'n_e' from all exc
    neuron_groups[name + 'e'] = neuron_groups['e'].subgroup(n_e_conv)
    # get a subgroup of size 'n_i' from the inhibitory layer
    neuron_groups[name + 'i'] = neuron_groups['i'].subgroup(1)

    # start the membrane potentials of these groups 40mV below their resting potentials
    neuron_groups[name + 'e'].v = v_rest_e - 40. * b.mV
    neuron_groups[name + 'i'].v = v_rest_i - 40. * b.mV

    # if we're in test mode / using some stored weights
    if test_mode or weight_path[-8:] == 'weights/':
        # load up adaptive threshold parameters
        neuron_groups['e'].theta = np.load(weight_path + 'theta_A_' + stdp_input + '.npy')
    else:
        # otherwise, set the adaptive additive threshold parameter at 20mV
        neuron_groups['e'].theta = np.ones((n_e)) * 20.0 * b.mV

    print '...creating recurrent connections'

    for conn_type in recurrent_conn_names:
        # create connection name (composed of population and connections types)
        conn_name = name + conn_type[0] + name + conn_type[1] + ending
        # get the corresponding stored weights from file
        weight_matrix = get_matrix_from_file('random/' + conn_name + '.npy')
        # create a connection from the first group in conn_name with the second group
        connections[conn_name] = b.Connection(neuron_groups[conn_name[0:3]], neuron_groups[conn_name[3:6]], structure='dense', state='g' + conn_type[0])
        # instantiate the created connection with the 'weightMatrix' loaded from file
        print weight_matrix
        connections[conn_name].connect(neuron_groups[conn_name[0:3]], neuron_groups[conn_name[3:6]], weight_matrix)

    # if STDP from excitatory neurons to exctatory neurons is on and this connection is excitatory -> excitatory
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

if record_spikes:
    b.figure(fig_num)
    fig_num += 1
    b.ion()
    b.subplot(211)
    b.raster_plot(spike_monitors['Ae'], refresh=1000 * b.ms, showlast=1000 * b.ms)
    b.subplot(212)
    b.raster_plot(spike_monitors['Ai'], refresh=1000 * b.ms, showlast=1000 * b.ms)







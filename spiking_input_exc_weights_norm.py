'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

 
import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy
import math
import cPickle as pickle
import brian_no_units  #import it to deactivate unit checking --> This should NOT be done for testing/debugging 
import brian as b
from struct import unpack
from brian import *

# specify the location of the MNIST data
MNIST_data_path = './'

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------     
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
            images = open(MNIST_data_path + 'train-images-idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels-idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images-idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels-idx1-ubyte','rb')
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


def get_matrix_from_file(fileName):
    offset = len(str(n_e)) + 4
    if fileName[-offset-4] == 'X':
        n_src = n_input                
    else:
        if fileName[-offset-3]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-offset-1]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        print readout.shape
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr


def save_connections(ending = ''):
    print 'save connections'
    for connName in save_conns:
        connMatrix = connections[connName][:]
        connListSparse = ([(i,j,connMatrix[i,j]) for i in xrange(connMatrix.shape[0]) for j in xrange(connMatrix.shape[1]) ])
        np.save(data_path + 'weights/' + connName + '_' + ending + '_input_exc_weights_norm', connListSparse)


def save_theta(ending = ''):
    print 'save theta'
    for pop_name in population_names:
        np.save(data_path + 'weights/theta_' + pop_name + '_' + ending + '_input_exc_weights_norm', neuron_groups[pop_name + 'e'].theta)


def normalize_weights():
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            connection = connections[connName][:]
            temp_conn = np.copy(connection)
            colSums = np.sum(temp_conn, axis = 0)
            colFactors = weight['ee_input']/colSums
            for j in xrange(n_e):
                connection[:,j] *= colFactors[j]


def is_lattice_connection(n, i, j):
    '''
    Boolean method which checks if two indices in a network correspond to neighboring nodes in a lattice.
    
    Args:
        n: number of nodes in lattice
        i: First neuron's index
        k: Second neuron's index
    '''
    sqrt = int(math.sqrt(n))
    return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j


def smooth_weights():
	'''
	Updates the weights of the input to excitatory layer connections by taking a 
	weighted average of each cell and its neighbors
	'''
	# get input to excitatory weights
	input_to_exc_conn = connections['XeAe' + str(n_e)][:]
	
	# copy and reshape it for convenient computation
	temp_conn = np.copy(input_to_exc_conn).reshape((int(math.sqrt(n_input)), int(math.sqrt(n_input)), n_e))
	
	# for each neuron, smooth its weights according to its neighbors' values in the lattice
	for exc in range(temp_conn.shape[2]):
		for i in range(temp_conn.shape[0]):
			for j in range(temp_conn.shape[1]):
				# get neighboring weights
				neighbors = []
				for x, y in zip([i - 1, i + 1, i, i], [j, j, j - 1, j + 1]):
					if is_lattice_connection(n_input, i * int(math.sqrt(n_input)) + j, x * int(math.sqrt(n_input)) + y):
						neighbors.append(temp_conn[i, j, exc])
				input_to_exc_conn[i * temp_conn.shape[0] + j, exc] = 0.5 * temp_conn[i, j, exc] + 0.5 * ( sum(neighbors) / float(len(neighbors)) )

            
def get_2d_input_weights():
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


def plot_2d_input_weights():
    weights = get_2d_input_weights()
    fig = b.figure(fig_num, figsize = (18, 18))
    im2 = b.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    b.title('weights of connection ' + name)
    fig.canvas.draw()
    return im2, fig
    

def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im


def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance


def plot_performance(fig_num):
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
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance
    

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in xrange(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]


def get_new_assignments(result_monitor, input_numbers):
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
    
    
#------------------------------------------------------------------------------ 
# load MNIST
#------------------------------------------------------------------------------
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


#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------

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
                        openmp = True, # whether or not to use OpenMP pragmas in generated C code.
                        magic_useframes = True, # defines whether or not the magic functions should serach for objects defined only in the calling frame,
                                                # or if they should find all objects defined in any frame. Set to "True" if not in an interactive shell.
                        useweave_linear_diffeq = True, # Whether to use weave C++ acceleration for the solution of linear differential equations.
                       )


np.random.seed(0)

data_path = './'

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
    record_spikes = True
    ee_STDP_on = True


# number of inputs to the network
n_input = 784
# number of classes to learn
classes_input = raw_input('Enter classes to learn as comma-separated list (e.g, 0,1,2,3,...) (default all 10 classes): ')
if classes_input == '':
	classes = range(10)
else:
	classes = set([ int(token) for token in classes_input.split(',') ])

# reduce size of dataset if necessary
if not test_mode and classes_input != 0:
	new_training = {'x' : [], 'y' : [], 'rows' : training['rows'], 'cols' : training['cols']}
	for idx in xrange(len(training['x'])):
		if training['y'][idx][0] in classes:
			new_training['y'].append(training['y'][idx])
			new_training['x'].append(training['x'][idx])
	new_training['x'], new_training['y'] = np.asarray(new_training['x']), np.asarray(new_training['y'])
	training = new_training
	
elif test_mode and clases_input != 0:
	new_testing = {'x' : [], 'y' : [], 'rows' : testing['rows'], 'cols' : testing['cols']}
	for idx in xrange(len(testing['x'])):
		if testing['y'][idx][0] in classes:
			new_testing['y'].append(testing['y'][idx])
			new_testing['x'].append(testing['x'][idx])
	new_testing['x'], new_testing['y'] = np.asarray(new_testing['x']), np.asarray(new_testing['y'])
	testing = new_testing
	
# number of excitatory neurons
n_e_input = raw_input('Enter number of excitatory / inhibitory neurons (default 100): ')
if n_e_input == '':
	n_e = 100
else:
	n_e = int(n_e_input)

# number of inhibitory neurons
n_i = n_e
# set ending of filename saves
ending = str(n_e)
# time (in seconds) per data example presentation
single_example_time = 0.35 * b.second
# time (in seconds) per rest period between data examples
resting_time = 0.15 * b.second
# total runtime (number of examples times (presentation time plus rest period))
runtime = num_examples * (single_example_time + resting_time)

# set the update interval and weight update interval (for network weights?) 
if num_examples <= 10000:
    update_interval = num_examples
    weight_update_interval = 5
else:
    update_interval = 100
    weight_update_interval = 50
# setting save connections to file parameters
if num_examples <= 60000:
    save_connections_interval = 500
else:
    save_connections_interval = 500
    update_interval = 10000

weight_smoothing_interval = 20


# rest potential parameters, reset potential parameters, threshold potential parameters, and refractory periods
v_rest_e = -65. * b.mV 
v_rest_i = -60. * b.mV 
v_reset_e = -65. * b.mV
v_reset_i = -45. * b.mV
v_thresh_e = -52. * b.mV
v_thresh_i = -40. * b.mV
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms

# connection structure
conn_structure = 'dense'

# dictionaries for weights and delays
weight = {}
delay = {}

# naming neuron populations (X for input, A for population, XA for input -> connection, etc...
input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
save_conns = ['XeAe' + str(n_e)]
input_conn_names = ['ee_input'] 
recurrent_conn_names = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0*b.ms,10*b.ms)
delay['ei_input'] = (0*b.ms,5*b.ms)
input_intensity = 2.
start_input_intensity = input_intensity

# time constants, learning rates, max weights, weight dependence, etc.
tc_pre_ee = 20 * b.ms
tc_post_ee = 20 * b.ms
tc_post_1_ee = 20 * b.ms
tc_post_2_ee = 40 * b.ms
nu_ee_pre =  0.0001
nu_ee_post = 0.01
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4
w_mu_pre = 0.2
w_mu_post = 0.2

# setting up differential equations (depending on train / test mode)
if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
    tc_theta = 1e7 * b.ms
    theta_plus_e = 0.05 * b.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*b.mV
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
neuron_eqs_e += '\n  dtimer/dt = 100.0  : ms'

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
            dpost/dt = -post / tc_post_1_ee : 1.0
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

neuron_groups['e'] = b.NeuronGroup(n_e * len(population_names), neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=scr_e, 
                                   compile = True, freeze = True)

neuron_groups['i'] = b.NeuronGroup(n_i * len(population_names), neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i, 
                                   compile = True, freeze = True)


#------------------------------------------------------------------------------ 
# create network population and recurrent connections
#------------------------------------------------------------------------------ 
for name in population_names:
    print 'create neuron group', name
    
    neuron_groups[name + 'e'] = neuron_groups['e'].subgroup(n_e)
    neuron_groups[name + 'i'] = neuron_groups['i'].subgroup(n_i)
    
    neuron_groups[name+'e'].v = v_rest_e - 40. * b.mV
    neuron_groups[name+'i'].v = v_rest_i - 40. * b.mV
    if test_mode or weight_path[-8:] == 'weights/':
        neuron_groups['e'].theta = np.load(weight_path + 'theta_A' + '.npy')
    else:
        neuron_groups['e'].theta = np.ones((n_e)) * 20.0*b.mV
    
    print 'create recurrent connections'
    for conn_type in recurrent_conn_names:
        connName = name + conn_type[0] + name + conn_type[1] + ending
        weightMatrix = get_matrix_from_file(weight_path + '../random/' + connName + '.npy')
        connections[connName] = b.Connection(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], structure= conn_structure, 
                                                    state = 'g'+conn_type[0])
        connections[connName].connect(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix)
                
    if ee_STDP_on:
        if 'ee' in recurrent_conn_names:
            stdp_methods[name + 'e' + name + 'e'] = b.STDP(connections[name + 'e' + name + 'e' + ending], eqs=eqs_stdp_ee, pre = eqs_stdp_pre_ee, 
                                                           post = eqs_stdp_post_ee, wmin=0., wmax= wmax_ee)

    print 'create monitors for', name
    rate_monitors[name+'e'] = b.PopulationRateMonitor(neuron_groups[name+'e'], bin = (single_example_time+resting_time)/b.second)
    rate_monitors[name+'i'] = b.PopulationRateMonitor(neuron_groups[name+'i'], bin = (single_example_time+resting_time)/b.second)
    spike_counters[name+'e'] = b.SpikeCounter(neuron_groups[name+'e'])
    
    if record_spikes:
        spike_monitors[name+'e'] = b.SpikeMonitor(neuron_groups[name+'e'])
        spike_monitors[name+'i'] = b.SpikeMonitor(neuron_groups[name+'i'])

if record_spikes:
    b.figure(fig_num)
    fig_num += 1
    b.ion()
    b.subplot(211)
    b.raster_plot(spike_monitors['Ae'], refresh=1000*b.ms, showlast=1000*b.ms)
    b.subplot(212)
    b.raster_plot(spike_monitors['Ai'], refresh=1000*b.ms, showlast=1000*b.ms)


#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------ 
pop_values = [0,0,0]
for i,name in enumerate(input_population_names):
    input_groups[name+'e'] = b.PoissonGroup(n_input, 0)
    rate_monitors[name+'e'] = b.PopulationRateMonitor(input_groups[name+'e'], bin = (single_example_time+resting_time)/b.second)

for name in input_connection_names:
    print 'create connections between', name[0], 'and', name[1]
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1] + ending
        weightMatrix = get_matrix_from_file(weight_path + connName + '.npy')
        connections[connName] = b.Connection(input_groups[connName[0:2]], neuron_groups[connName[2:4]], structure= conn_structure, 
                                                    state = 'g'+connType[0], delay=True, max_delay=delay[connType][1])
        connections[connName].connect(input_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix, delay=delay[connType])
     
    if ee_STDP_on:
        print 'create STDP for connection', name[0]+'e'+name[1]+'e'
        stdp_methods[name[0]+'e'+name[1]+'e'] = b.STDP(connections[name[0] + 'e' + name[1] + 'e' + ending], eqs=eqs_stdp_ee, pre = eqs_stdp_pre_ee, 
                                                       post = eqs_stdp_post_ee, wmin=0., wmax= wmax_ee)


#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 

# bookkeeping variables
previous_spike_count = np.zeros(n_e)
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))

# plot input weights
if not test_mode:
    input_weight_monitor, fig_weights = plot_2d_input_weights()
    fig_num += 1

# plot performance
if do_plot_performance:
    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)

# set firing rates to zero initially
for name in input_population_names:
    input_groups[name + 'e'].rate = 0

# initialize network and set current example to zero
b.run(0)
j = 0

while j < (int(num_examples)):

    if test_mode:
        if use_testing_set:
            rates = testing['x'][j % 10000, :, :].reshape((n_input)) / 8. * input_intensity
        else:
            rates = training['x'][j % 60000, :, :].reshape((n_input)) / 8. * input_intensity
    
    else:
    	# ensure weights don't grow without bound
        normalize_weights()
        # averages weights from neighbors in lattice
        if j > 0 and j % weight_smoothing_interval == 0:
        	smooth_weights()
        # get the firing rates of the next input example
        rates = training['x'][j % 60000, :, :].reshape((n_input)) / 8. * input_intensity
    
    # sets the input firing rates
    input_groups['Xe'].rate = rates
    
    # run the network for a single example time
    b.run(single_example_time)
    
    # get new neuron label assignments every 'update_interval'
    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
    
    # update weights every 'weight_update_interval'
    if j % weight_update_interval == 0 and not test_mode:
        update_2d_input_weights(input_weight_monitor, fig_weights)
    
    # save the weights to file every 'save_connection_interval'
    if j % save_connections_interval == 0 and j > 0 and not test_mode:
        save_connections(str(j))
        save_theta(str(j))
    
    # get count of spikes over the past iteration
    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])
    
    # if there weren't a certain number of spikes
    if np.sum(current_spike_count) < 5:
        # increase the intesity of input
        input_intensity += 1
        
        # set the input firing rates back to zero
        for name in input_population_names:
            input_groups[name + 'e'].rate = 0
        
        # run the simulation for 'resting_time' to relax back to rest potentials
        b.run(resting_time)
    # if there were enough spikes
    else:
    	# record the current number of spikes
        result_monitor[j % update_interval, :] = current_spike_count
        
        # decide whether to evaluate on test or training set
        if test_mode and use_testing_set:
            input_numbers[j] = testing['y'][j % 10000][0]
        else:
            input_numbers[j] = training['y'][j % 60000][0]
        
        # get the output classifications of the network
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])
        
        # print progress
        if j % 100 == 0 and j > 0:
            print 'runs done:', j, 'of', int(num_examples)
        
        # plot performance if appropriate
        if j % update_interval == 0 and j > 0:
            if do_plot_performance:
                # updating the performance plot
                perf_plot, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
                # printing out classification performance results so far
                print 'Classification performance', performance[:(j / float(update_interval)) + 1]
                
        # set input firing rates back to zero
        for name in input_population_names:
            input_groups[name + 'e'].rate = 0
        
        # run the network for 'resting_time' to relax back to rest potentials
        b.run(resting_time)
        # reset the input firing intensity
        input_intensity = start_input_intensity
        # increment the example counter
        j += 1


#------------------------------------------------------------------------------ 
# save results
#------------------------------------------------------------------------------ 
print 'save results'
if not test_mode:
    save_theta()
if not test_mode:
    save_connections()
else:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)
    

#------------------------------------------------------------------------------ 
# plot results
#------------------------------------------------------------------------------ 
if rate_monitors:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(rate_monitors):
        b.subplot(len(rate_monitors), 1, i + 1)
        b.plot(rate_monitors[name].times/b.second, rate_monitors[name].rate, '.')
        b.title('Rates of population ' + name)
    b.savefig('../plots/rate_monitors_' + str(n_e) + '_' + stdp_input + '_input_exc_weights_norm.png')
    
if spike_monitors:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_monitors):
        b.subplot(len(spike_monitors), 1, i + 1)
        b.raster_plot(spike_monitors[name])
        b.title('Spikes of population ' + name)
    b.savefig('../plots/spike_monitors_' + str(n_e) + '_' + stdp_input + '_input_exc_weights_norm.png')
        
if spike_counters:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_counters):
        b.subplot(len(spike_counters), 1, i + 1)
        b.plot(spike_counters['Ae'].count[:])
        b.title('Spike count of population ' + name)
    b.savefig('../plots/rate_monitors_' + str(n_e) + '_' + stdp_input + '_input_exc_weights_norm.png')

plot_2d_input_weights()

b.savefig('../plots/input_to_exc_weights' + str(n_e) + '_' + stdp_input + '_input_exc_weights_norm.png')

b.ioff()
b.show()




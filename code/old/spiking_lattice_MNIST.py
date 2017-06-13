'''
Extending the 'spiking_MNIST.py' script written by Peter Diehl to utilize a lattice
connectivity between neurons in the excitatory layer.

@author: Dan Saunders
'''

 
import numpy as np
import matplotlib.cm as cmap
import time, sys, os.path, scipy 
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
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input                
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    print readout.shape, fileName
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr


def save_connections(ending = ''):
    print 'save connections'
    for connName in save_conns:
        connMatrix = connections[connName][:]
        connListSparse = ([(i,j,connMatrix[i,j]) for i in xrange(connMatrix.shape[0]) for j in xrange(connMatrix.shape[1]) ])
        np.save(data_path + 'weights/' + connName + ending, connListSparse)


def save_theta(ending = ''):
    print 'save theta'
    for pop_name in population_names:
        np.save(data_path + 'weights/theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)


def normalize_weights():
	for connName in connections:
		if connName == 'XeAe' + str(n_e):
			connection = connections[connName][:]
			temp_conn = np.copy(connection)
			colSums = np.sum(temp_conn, axis = 0)
			colFactors = weight['ee_input'] / colSums
			for j in xrange(n_e):
				connection[:, j] *= colFactors[j]

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
    name = 'XeAe' + str(n_e)
    weights = get_2d_input_weights()
    fig = b.figure(fig_num, figsize = (18, 18))
    im2 = b.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    b.title('weights of connection ' + name)
    fig.canvas.draw()
    return im2, fig


def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im


def get_2d_excitatory_weights():
    name = 'AeAe' + str(n_e)
    num_values_col = num_values_row = n_e
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    
    for i in xrange(n_e):
    	for j in xrange(n_e):
    		rearranged_weights[i, j] = connections[name].W[i, j]
    
    return rearranged_weights


def plot_2d_excitatory_weights():
	name = 'AeAe' + str(n_e)
	weights = get_2d_excitatory_weights()
	fig = b.figure(fig_num, figsize=(10, 10))
	im2 = b.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
	b.colorbar(im2)
	b.title('weights of connection ' + name)
	fig.canvas.draw()
	return im2, fig


def update_2d_excitatory_weights(im, fig):
    weights = get_2d_excitatory_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im


def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num / update_interval)
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
    im2, = ax.plot(time_steps, performance)
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
start = time.time()
training = get_labeled_data(MNIST_data_path + 'training')
end = time.time()
print 'time needed to load training set:', end - start

start = time.time()
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
end = time.time()
print 'time needed to load test set:', end - start


#------------------------------------------------------------------------------ 
# set parameters and equations
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

b.set_global_preferences( 
                        defaultclock = b.Clock(dt=0.5*b.ms), # The default clock to use if none is provided or defined in any enclosing scope.
                        useweave = True, # Defines whether or not functions should use inlined compiled C code where defined.
                        gcc_options = ['-ffast-math -march=native'],  # Defines the compiler switches passed to the gcc compiler. 
                        # For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimizations are turned on 
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
    weight_update_interval = 20
else:
    update_interval = 10000
    weight_update_interval = 5

# setting save connections to file parameter and update interval
if num_examples <= 60000:
    save_connections_interval = 10000
else:
    save_connections_interval = 10000
    
update_interval = 100

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
population_names = ['A']
input_connection_names = ['XA']
save_conns = ['XeAe' + str(n_e)], ['AeAe' + str(n_e)]
input_conn_names = ['ee_input'] 
recurrent_conn_names = ['ei', 'ie', 'ee']
weight['ee_input'] = 78.
weight['ee'] = int(n_e / 10.)
delay['ee_input'] = (0 * b.ms, 10 * b.ms)
delay['ei_input'] = (0 * b.ms, 5 * b.ms)
input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20 * b.ms
tc_post_ee = 20 * b.ms
tc_post_1_ee = 20 * b.ms
tc_post_2_ee = 40 * b.ms
nu_ee_pre = 0.0001
nu_ee_post = 0.01
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4


# setting up differential equations (depending on train / test mode)
if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
    tc_theta = 1e7 * b.ms
    theta_plus_e = 0.05 * b.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0 * ms'
offset = 20.0 * b.mV
v_thresh_e = '(v>(theta - offset + ' + str(v_thresh_e) + ')) * (timer>refrac_e)'

# equations for neurons
neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / (100 * ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100. * mV-v)                          : amp
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
        I_synI = gi * nS * (-85. * mV-v)                          : amp
        dge/dt = -ge / (1.0*ms)                                   : 1
        dgi/dt = -gi / (2.0*ms)                                  : 1
        '''

stdp_input = raw_input('Enter STDP learning rule to use (standard / hebbian / exp_weight_depend) (default standard): ')

if raw_input('Enter (yes / no) for post-pre (default yes): ') in [ 'yes', '' ]:
	post_pre = True
else:
	post_pre = False

if stdp_input in [ 'standard', '' ]:
	if post_pre:
		eqs_stdp_ee = '''
		            dpre / dt = -pre / tc_pre_ee : 1.0
		            dpost / dt = -post / tc_post_ee : 1.0
		        '''
		
		eqs_stdp_pre_ee = 'pre = 1.0; w -= nu_ee_pre * post'
		eqs_stdp_post_ee = 'post = 1.0; w += nu_ee_post * pre'
	else:
		eqs_stdp_ee = '''
		            dpre / dt = -pre / tc_pre_ee : 1.0
		        '''
		
		eqs_stdp_pre_ee = 'pre = 1.0'
		eqs_stdp_post_ee = 'w += nu_ee_post * pre'

elif stdp_input == 'hebbian':
	if post_pre:
		eqs_stdp_ee = '''
                dpre / dt = -pre / tc_pre_ee : 1.0
                dpost / dt = -post / tc_post_ee : 1.0
            '''
    
		eqs_stdp_pre_ee = 'pre = 1.0; w += nu_ee_pre * post'
		eqs_stdp_post_ee = 'post = 1.0; w += nu_ee_post * pre'
	else:
		eqs_stdp_ee = '''
                dpre / dt = -pre / tc_pre_ee : 1.0
            '''
    
		eqs_stdp_pre_ee = 'pre = 1.0'
		eqs_stdp_post_ee = 'w += nu_ee_post * pre'
		
elif stdp_input == 'exp_weight_depend':
	if post_pre:
		eqs_stdp_ee = '''
                dpre/dt = -pre/tc_pre_ee : 1.0
                dpost/dt = -post/tc_post_ee : 1.0
            '''
    
		eqs_stdp_pre_ee = 'pre = 1.0; w -= (nu_ee_pre * post) * ((wmax_ee - w) ** w_mu_pre)'
		eqs_stdp_post_ee = 'post = 1.0; w += (nu_ee_post * pre) * ((wmax_ee - w) ** w_mu_post)'
	else:
		eqs_stdp_ee = '''
                dpre/dt = -pre/tc_pre_ee : 1.0
            '''
    
		eqs_stdp_pre_ee = 'pre = 1.0'
		eqs_stdp_post_ee = 'w += (nu_ee_post * pre) * ((wmax_ee - w) ** w_mu_post)'

else:
    raise NotImplementedError


b.ion()
fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
stdp_methods = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((update_interval, n_e))

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
    
    neuron_groups[name + 'e'].v = v_rest_e - 40. * b.mV
    neuron_groups[name + 'i'].v = v_rest_i - 40. * b.mV
    if test_mode or weight_path[-8:] == 'weights/':
        neuron_groups['e'].theta = np.load(weight_path + 'theta_' + name + ending + '.npy')
    else:
        neuron_groups['e'].theta = np.ones((n_e)) * 20.0*b.mV
    
    print 'create recurrent connections'
    
    for conn_type in recurrent_conn_names:
        connName = name + conn_type[0] + name + conn_type[1] + ending
        weightMatrix = get_matrix_from_file(weight_path + '../random/' + connName + '.npy')
        
        if connName == 'AeAe' + str(n_e):
        	connections[connName] = b.Connection(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], structure='sparse', state='g' + conn_type[0])
        	for i in xrange(n_e):
        		for j in xrange(n_e):
        			if weightMatrix[i, j] != 0:
        				connections[connName][i, j] = weightMatrix[i, j]
        else:
        	connections[connName] = b.Connection(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], structure='dense', state='g' + conn_type[0])
    		connections[connName].connect(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix)
                
    if ee_STDP_on:
        if 'ee' in recurrent_conn_names:
            print 'create STDP for connection' + name + 'e' + name + 'e'
            stdp_methods[name + 'e' + name + 'e'] = b.STDP(connections[name + 'e' + name + 'e' + ending], eqs=eqs_stdp_ee, pre='pre = 1.0; w -= 0.001 * post', post='post = 1.0; w += 0.01 * pre', wmin=0., wmax=wmax_ee)

    print 'create monitors for', name
    
    rate_monitors[name + 'e'] = b.PopulationRateMonitor(neuron_groups[name + 'e'], bin=(single_example_time + resting_time) / b.second)
    rate_monitors[name + 'i'] = b.PopulationRateMonitor(neuron_groups[name + 'i'], bin=(single_example_time + resting_time) / b.second)
    spike_counters[name + 'e'] = b.SpikeCounter(neuron_groups[name + 'e'])
    
    if record_spikes:
        spike_monitors[name + 'e'] = b.SpikeMonitor(neuron_groups[name + 'e'])
        spike_monitors[name + 'i'] = b.SpikeMonitor(neuron_groups[name + 'i'])


if record_spikes:
    b.figure(fig_num)
    fig_num += 1
    b.ion()
    b.subplot(211)
    b.raster_plot(spike_monitors['Ae'], refresh=1000 * b.ms, showlast=1000 * b.ms)
    b.title('Excitatory Spikes')
    b.subplot(212)
    b.raster_plot(spike_monitors['Ai'], refresh=1000 * b.ms, showlast=1000 * b.ms)
    b.title('Inhibitory Spikes')


#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------ 
pop_values = [0, 0, 0]
for i,name in enumerate(input_population_names):
    input_groups[name + 'e'] = b.PoissonGroup(n_input, 0)
    rate_monitors[name + 'e'] = b.PopulationRateMonitor(input_groups[name + 'e'], bin=(single_example_time+resting_time) / b.second)

for name in input_connection_names:
    print 'create connections between', name[0], 'and', name[1]
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1] + ending
        weightMatrix = get_matrix_from_file(weight_path + connName + '.npy')
        connections[connName] = b.Connection(input_groups[connName[0:2]], neuron_groups[connName[2:4]], structure='dense', 
                                                    state='g' + connType[0], delay=True, max_delay=delay[connType][1])
        connections[connName].connect(input_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix, delay=delay[connType])
     
    if ee_STDP_on:
        print 'create STDP for connection', name[0] + 'e' + name[1] + 'e'
        stdp_methods[name[0] + 'e' + name[1] + 'e'] = b.STDP(connections[name[0] + 'e' + name[1] + 'e' + ending], eqs=eqs_stdp_ee, pre=eqs_stdp_pre_ee, 
                                                       post=eqs_stdp_post_ee, wmin=0., wmax=wmax_ee)


#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 
previous_spike_count = np.zeros(n_e)
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))

# create input and excitatory weight monitors
if not test_mode:
    input_weight_monitor, fig_weights = plot_2d_input_weights()
    fig_num += 1
    excitatory_weight_monitor, exc_fig_weights = plot_2d_excitatory_weights()
    fig_num += 1

# create performance monitors and initialize performance values
if do_plot_performance:
    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)

# set input firing rates to zero
for name in input_population_names:
    input_groups[name + 'e'].rate = 0

b.run(0)
j = 0

while j < (int(num_examples)):
    if test_mode:
        if use_testing_set:
            rates = testing['x'][j % 10000,:,:].reshape((n_input)) / 8. *  input_intensity
        else:
            rates = training['x'][j % 60000,:,:].reshape((n_input)) / 8. *  input_intensity
    else:
        normalize_weights()
        rates = training['x'][j % 60000,:,:].reshape((n_input)) / 8. *  input_intensity
    
    # set the input neuron firing rates (by way of Poisson spike trains)
    input_groups['Xe'].rate = rates

    # run the simulation for 350ms (or whatever 'single_example_time' is set to)
    b.run(single_example_time)
    
    # assign labels to neurons
    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
    
    # update input to excitatory and excitatory to input weight monitoring
    if j % weight_update_interval == 0 and not test_mode:
        update_2d_input_weights(input_weight_monitor, fig_weights)
        update_2d_excitatory_weights(excitatory_weight_monitor, exc_fig_weights)
    
    # save connection weights if the interval is a certain way
    if j % save_connections_interval == 0 and j > 0 and not test_mode:
        save_connections(str(j))
        save_theta(str(j))
    
    # keep track of the number of spikes over the iteration
    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])
    
    # if the number of spikes over the last iteration was less than five, run the 
    # sample again with increased input intensity
    if np.sum(current_spike_count) < 5:
        input_intensity += 1
        
        for name in input_population_names:
            input_groups[name + 'e'].rate = 0
        
        b.run(resting_time)
    # otherwise, collect results of this run, and do classification as needed
    else:
        result_monitor[j % update_interval,:] = current_spike_count
        
        # get correct label for this example
        if test_mode and use_testing_set:
            input_numbers[j] = testing['y'][j%10000][0]
        else:
            input_numbers[j] = training['y'][j%60000][0]
        
        # get our network's predicted label based on current neuron labels and
        # this past iteration's spike results
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j % update_interval,:])
        
        # print progress to console
        if j % 100 == 0 and j > 0:
            print 'runs done:', j, 'of', int(num_examples)
        
        # plot performance if 'do_plot_performance'
        if j % update_interval == 0 and j > 0:
            if do_plot_performance:
                perf_plot, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
                print 'Classification performance', performance[:int((j / float(update_interval)) + 1)]
        
        # set input firing rates back to zero
        for name in input_population_names:
            input_groups[name + 'e'].rate = 0
        
        # run the resting / relaxing period of the network
        b.run(resting_time)
        # reset input intensity back to normal
        input_intensity = start_input_intensity
        # increment iteration counter
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
        b.plot(spike_counters['Ae'].count[:])
        b.title('Spike count of population ' + name)

plot_2d_input_weights()
b.ioff()
b.show()




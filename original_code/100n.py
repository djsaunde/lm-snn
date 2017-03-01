'''
Created on 05.11.2012

@author: peter
'''

import matplotlib
# matplotlib.use('Agg')
 
import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import os
import scipy 
import cPickle as pickle
from struct import unpack
from matplotlib.pyplot import savefig

MNIST_data_path = os.getcwd() + '/..'

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------     
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + '/train-images-idx3-ubyte','rb')
            labels = open(MNIST_data_path + '/train-labels-idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + '/t10k-images-idx3-ubyte','rb')
            labels = open(MNIST_data_path + '/t10k-labels-idx1-ubyte','rb')
    
        # Read the binary data
    
        # We have to get big endian unsigned int. So we need '>I'
    
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
    elif fileName[-4-offset] == 'Y':
        n_src = n_label                        
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
    
#     print value_arr
#     print fileName, n_src, n_tgt
#     figure()
#     im2 = imshow(value_arr, interpolation="nearest", vmin = 0, cmap=cm.get_cmap('gist_ncar')) #my_cmap
#     cbar2 = colorbar(im2)
#     title(fileName)
#     show()
    return value_arr


def save_connections(ending = ''):
    print 'save connections'
    for connName in save_conns:
        connMatrix = connections[connName][:]
        connListSparse = ([(i,j[0],j[1]) for i in xrange(connMatrix.shape[0]) for j in zip(connMatrix.rowj[i],connMatrix.rowdata[i])])
    #     print len(connListSparse)
        np.save(data_path + '/weights/' + connName + ending, connListSparse)

def save_theta(ending = ''):
    print 'save theta'
    for pop_name in population_names:
        np.save(data_path + '/weights/theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)


def normalize_weights(plot_results = False):
#     print 'normalize weights'
    for connName in connections:
        
        if connName[1] == 'e' and connName[3] == 'e':
            if plot_results:
                w_pre = np.zeros((n_input, n_e))
                w_end = np.zeros((n_input, n_e))
                
            if connName[0] == 'X':
                w_post = np.zeros((n_input, n_e))
                max_range = min(n_input, n_e)
                factor = weight['ee_input']
            else:
                if connName[0] == connName[2]:
                    w_post = np.zeros((n_e, n_e))
                    factor = weight['ee']   
                    max_range = n_e
                else:
                    w_post = np.zeros((n_label, n_e))
                    factor = weight['ee_input_label']  
                    max_range = min(n_label, n_e) 
                    max_range = n_label
            connection = connections[connName][:]
            
            for i in xrange(max_range):#
                rowi = connection.rowdata[i]
#                 rowMean = np.mean(rowi)
                if plot_results:
                    w_pre[i, connection.rowj[i]] = rowi
#                 connection.rowdata[i] *= factor/rowMean
                w_post[i, connection.rowj[i]] = connection.rowdata[i]
                
            colSums = np.sum(w_post, axis = 0)
#             colDataEntries = [len(connection.coldataindices[j]) for j in xrange(n_e)]
#             colMeans = colSums/colDataEntries
            colFactors = factor/colSums
            
            for j in xrange(n_e):#
                connection[:,j] *= colFactors[j]
                
            if plot_results:
                print factor
                print connName, colFactors, colSums#, colDataEntries, colMeans
                for i in xrange(n_input):#
                    w_end[i, connection.rowj[i]] = connection.rowdata[i]
                
                if connName == 'XeAe':
                    b.figure()
                    im2 = b.imshow(w_pre, interpolation="nearest", vmin = 0, cmap=cmap.get_cmap('gist_rainbow')) #my_cmap
                    b.colorbar(im2)
                    b.title(connName + ' pre')
                    b.figure()
                    im2 = b.imshow(w_post, interpolation="nearest", vmin = 0, cmap=cmap.get_cmap('gist_rainbow')) #my_cmap
                    b.colorbar(im2)
                    b.title(connName + ' post')
                    b.figure()
                    im2 = b.imshow(w_end, interpolation="nearest", vmin = 0, cmap=cmap.get_cmap('gist_rainbow')) #my_cmap
                    b.colorbar(im2)
                    b.title(connName + ' end')
                    b.show()

def get_2d_input_weights():
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, n_e))
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = connections[name][:]
    for i in xrange(n_input):
        weight_matrix[i, connMatrix.rowj[i]] = connMatrix.rowdata[i]
        
    for i in xrange(n_e_sqrt):
        for j in xrange(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = b.figure(fig_num, figsize = (18, 18))
    im2 = b.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot')) #my_cmap
    b.colorbar(im2)
    b.title('weights of connection' + name)
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
#     time_steps, performance = get_current_performance()
    num_evaluations = int(num_examples/update_interval)
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


def update_performance_plot(im, performance, current_example_num, fig, use_remote_access = False):
    performance = get_current_performance(performance, current_example_num)
#     print performance
    if not use_remote_access:
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
        
#     print 'summed_rates:', summed_rates, ', num_assignments: ', num_assignments
    return np.argsort(summed_rates)[::-1]


def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
#     print 'input_numbers', input_nums
    for j in xrange(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
#             print 'j', j, ', num_assignments', num_assignments, ', np.where(input_numbers == j)[0]', np.where(input_nums == j)[0], \
#                 'rate', rate, 'result_monitor[input_nums == j]', result_monitor[input_nums == j]
        for i in xrange(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j 
#     print 'new assignments:', assignments
    return assignments
    
    

#------------------------------------------------------------------------------ 
# load MNIST
#------------------------------------------------------------------------------


start = time.time()
training = get_labeled_data(MNIST_data_path + '/training')
end = time.time()
print 'time needed to load training set:', end - start
 
 
start = time.time()
testing = get_labeled_data(MNIST_data_path + '/testing', bTrain = False)
end = time.time()
print 'time needed to load test set:', end - start



#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------

import brian_no_units  #import it to deactivate unit checking --> This should NOT be done for testing/debugging 
import brian as b
from brian import *
b.set_global_preferences( 
                        defaultclock = b.Clock(dt=0.5*b.ms), # The default clock to use if none is provided or defined in any enclosing scope.
                        useweave = True, # Defines whether or not functions should use inlined compiled C code where defined.
                        gcc_options = ['-ffast-math -march=native'],  # Defines the compiler switches passed to the gcc compiler. 
                        #For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimisations are turned on 
                        #- if you need IEEE guaranteed results, turn this switch off.
                        useweave_linear_diffeq = False,  # Whether to use weave C++ acceleration for the solution of linear differential 
                        #equations. Note that on some platforms, typically older ones, this is faster and on some platforms, 
                        #typically new ones, this is actually slower.
                        usecodegen = True,  # Whether or not to use experimental code generation support.
                        usecodegenweave = True,  # Whether or not to use C with experimental code generation support.
                        usecodegenstateupdate = True,  # Whether or not to use experimental code generation support on state updaters.
                        usecodegenreset = False,  # Whether or not to use experimental code generation support on resets. 
                        #Typically slower due to weave overheads, so usually leave this off.
                        usecodegenthreshold = True,  # Whether or not to use experimental code generation support on thresholds.
                        usenewpropagate = True,  # Whether or not to use experimental new C propagation functions.
                        usecstdp = True,  # Whether or not to use experimental new C STDP.
                        openmp = False,  # Whether or not to use OpenMP pragmas in generated C code. 
                        #If supported on your compiler (gcc 4.2+) it will use multiple CPUs and can run substantially faster.
                        magic_useframes = True,  # Defines whether or not the magic functions should search for objects 
                        #defined only in the calling frame or if they should find all objects defined in any frame. 
                        #This should be set to False if you are using Brian from an interactive shell like IDLE or IPython 
                        #where each command has its own frame, otherwise set it to True.
                       ) 
  
# import brian.experimental.cuda.gpucodegen as gpu


data_path = os.getcwd()
weight_path = data_path +  '/../weights/' # '/random/' #    
ending = ''
use_remote_access = False
n_input = 784
n_label = 4000
n_e = 100
n_i = n_e #/ 4
single_example_time =   0.35 * b.second #
num_examples = 60000 * 1
resting_time = 0.15 * b.second
runtime = num_examples * (single_example_time + resting_time)
if num_examples <= 10000:    
    update_interval = 1000
    weight_update_interval = 20
else:
    update_interval = 10000
    weight_update_interval = 10
if num_examples <= 60000:    
    save_connections_interval = 10000
else:
    save_connections_interval = 10000
    update_interval = 10000
# defaultclock.dt = 0.2*ms
use_testing_set = False
use_adaptive_threshold = True
use_weight_dependence = False
use_classic_STDP = True
use_plastic_supervision = False
normalize_input = False
test_mode = True
if test_mode:
    do_plot_performance = True
    record_spikes = False
    ee_STDP_on = False # True # 
    input_intensity = 2.
    label_intensity = 0.
    noise_level = 0.
    update_interval = num_examples
else:
    do_plot_performance = False
    if num_examples <= 60000:    
        record_spikes = True
    else:
        record_spikes = True
    ee_STDP_on = True
    input_intensity = 2.
    label_intensity = 0.
    noise_level = 10.0
if use_weight_dependence:
    weight_normalization_interval = 2000000
else:
    weight_normalization_interval = 20
record_states = False
do_plot_performance = True
start_input_intensity = input_intensity
save_conns = ['XeAe']

b.ion()
# b.show()

v_rest_e = -65. * b.mV 
v_rest_i = -60. * b.mV 
v_reset_e = -65. * b.mV
v_reset_i = -45. * b.mV
v_thresh_e = -52. * b.mV
v_thresh_i = -40. * b.mV
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms

conn_structure = 'sparse' # 'dense' 
weight = {}
delay = {}
input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
if label_intensity > 0:
    input_population_names.append('Y')
    input_connection_names.append('YA')
input_conn_names = ['ee_input'] # 
recurrent_conn_names = ['ei', 'ie'] # 'ee', , 'ii'
weight['ee_input_label'] = 5.0 # 25.0 * 2
weight['ee_input'] = 10. # 25.0 * 2
weight['ee'] = 5.0 # 
delay['ee_input'] = (0*b.ms,10*b.ms)
delay['ei_input'] = (0*b.ms,5*b.ms)
delay['ee'] = (0*b.ms,10*b.ms)
delay['ei'] = (0*b.ms,0*b.ms)
delay['ie'] = (0*b.ms,0*b.ms)
delay['ii'] = (0*b.ms,2*b.ms)

if use_adaptive_threshold:
    b.set_global_preferences( 
                        usecodegenreset = False,  # Whether or not to use experimental code generation support on resets. 
                        #Typically slower due to weave overheads, so usually leave this off.
                        usecodegenthreshold = False,  # Whether or not to use experimental code generation support on thresholds.
                        )
#     theta_plus_i = 2*mV
    if test_mode:
        scr_e = 'v = v_reset_e; timer = 0*ms'
    else:
        tc_theta = 1*1e7 * b.ms
        theta_plus_e = 0.025 * b.mV
        scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
            
    offset = 10.0*b.mV
    v_thresh_e = '(v>(theta - offset + ' + str(v_thresh_e) + ')) * (timer>refrac_e)'
#     v_reset_i = 'v = v_reset_i; theta += theta_plus_i'
#     scr_i = 'v>(theta - offset) + ' + str(v_thresh_i)

tc_pre_ee = 20*b.ms
tc_post_1_ee = 20*b.ms
tc_post_2_ee = 40*b.ms
tc_pre_ie = 20*b.ms
tc_post_ie = 20*b.ms
nu_ee_pre =  0.000002
nu_ee_post = 0.0015
nu_ie =      0.001
alpha_ie = 0.5*b.Hz*tc_post_ie*2    # controls the firing rate
wmax_ee = 1.0
wmax_ie = 1000.
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
if use_adaptive_threshold:
    if test_mode:
        neuron_eqs_e += '\n  theta      :volt'
    else:
        neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
    neuron_eqs_e += '\n  dtimer/dt = 100.0  : ms'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
# if use_adaptive_threshold:
#     neuron_eqs_i += 'dtheta/dt = -theta / (tc_theta)  : volt'

eqs_stdp_ie = '''
            dpre/dt   =  -pre/(tc_pre_ie)        : 1.0
            dpost/dt  = -post/(tc_post_ie)       : 1.0
            '''

if use_classic_STDP:
    eqs_stdp_ee = '''
                post2before                            : 1.0
                dpre/dt   =   -pre/(tc_pre_ee)         : 1.0
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1.0
                dpost2/dt  = -post1/(tc_post_2_ee)     : 1.0
                '''
#                 dpost/dt  = -post/(tc_post_1_ee)     : 1.0
    if use_weight_dependence:
#         euler = np.e
        eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1 * w**exp_ee_pre'
        eqs_stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2before * (wmax_ee - w)**exp_ee_post; post1 = 1.; post2 = 1.'
#         eqs_stdp_pre_ee = 'pre += 1.; w -= nu_ee_pre * post * w**exp_ee_pre' #
#         eqs_stdp_post_ee = 'w += nu_ee_post * (pre - STDP_offset) * (wmax_ee - w)**exp_ee_post; post += 1.'
#         eqs_stdp_pre_ee = 'pre += 1.; w -= nu_ee_pre * post * euler**(-1*exp_ee_pre*(wmax_ee - w))' #
#         eqs_stdp_post_ee = 'w += nu_ee_post * (pre - STDP_offset) * euler**(-1*exp_ee_post*(w)); post += 1.'
#         eqs_stdp_pre_ee = 'pre += 1.' #; w -= nu_ee_pre * post * w**exp_ee_pre'
#         eqs_stdp_post_ee = 'w += nu_ee_post * (pre - STDP_offset) * (wmax_ee - w)**exp_ee_post; post += 1.'        
#         eqs_stdp_pre_ee = 'pre += 1.' #
#         eqs_stdp_post_ee = 'w += nu_ee_post * (pre * euler**(exp_ee_post*(w)) - STDP_offset * euler**(exp_ee_pre*(wmax_ee - w))); post += 1.'
        
        
    else:
        eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1'
        eqs_stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2before; post1 = 1.; post2 = 1.'
#         eqs_stdp_pre_ee = 'pre += 1.' #; w -= nu_ee_pre * post'
#         eqs_stdp_post_ee = 'w += nu_ee_post * (pre - STDP_offset); post += 1.'
    
else:
    eqs_stdp_ee = '''
                post2before                          : 1.0
                dpre/dt   =   -pre/(tc_pre_ee)       : 1.0
                dpost1/dt = -post1/(tc_post_1_ee)     : 1.0
                dpost2/dt = -post2/(tc_post_2_ee)     : 1.0
                '''
    if use_weight_dependence:
        eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1 * w**exp_ee_pre'
        eqs_stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2 * (wmax_ee - w)**exp_ee_post; post1 = 1.; post2 = 1.'
    else:
        eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1'
        eqs_stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2; post1 = 1.; post2 = 1.'


eqs_stdp_pre_ie = 'pre += 1.; w += nu_ie * (post-alpha_ie)'
eqs_stdp_post_ie = 'post += 1.; w += nu_ie * pre'


fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
stdp_methods = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
state_monitors = {}
result_monitor = np.zeros((update_interval,n_e))



    
if use_adaptive_threshold:
    neuron_groups['e'] = b.NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, 
                                       compile = True, freeze = True)
#     neuron_groups['i'] = b.NeuronGroup(n_i*len(population_names), neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= scr_i, 
#                      compile = True, freeze = True)
else:
    neuron_groups['e'] = b.NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= v_reset_e, 
                                       compile = True, freeze = True)
neuron_groups['i'] = b.NeuronGroup(n_i*len(population_names), neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, 
                                   compile = True, freeze = True)


#------------------------------------------------------------------------------ 
# create network population and recurrent connections
#------------------------------------------------------------------------------ 
for name in population_names:
    print 'create neuron group', name
    
    neuron_groups[name+'e'] = neuron_groups['e'].subgroup(n_e)
    neuron_groups[name+'i'] = neuron_groups['i'].subgroup(n_i)
    
    neuron_groups[name+'e'].v = v_rest_e - 40. * b.mV
    neuron_groups[name+'i'].v = v_rest_i - 40. * b.mV
    if test_mode or weight_path[-8:] == 'weights/':
        neuron_groups['e'].theta = np.load(weight_path + 'theta_' + name + ending + '.npy')
    else:
        neuron_groups['e'].theta = np.ones((n_e)) * 20.0*b.mV
    
    print 'create recurrent connections'
    for conn_type in recurrent_conn_names:
        connName = name + conn_type[0] + name + conn_type[1]
        weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')
#         delayMatrix = np.load(data_path +'threeWayConnectionMatrix_d'+connName+'.npy')
#         print weightMatrix.shape, delayMatrix.shape
        weightMatrix = scipy.sparse.lil_matrix(weightMatrix)
#         delayMatrix = scipy.sparse.lil_matrix(delayMatrix)
        if delay[conn_type] == (0*b.ms,0*b.ms):
            connections[connName] = b.Connection(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], structure= conn_structure, 
                                                        state = 'g'+conn_type[0])
            connections[connName].connect(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix)
        else:
            connections[connName] = b.Connection(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], structure= conn_structure, 
                                                        state = 'g'+conn_type[0], delay=True, max_delay=delay[conn_type][1])#, delay=delay[conn_type])
            connections[connName].connect(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix, delay=delay[conn_type])
#         connections[connName].connect_from_sparse(weightMatrix)#, delay = delayMatrix)
#         for i in xrange(len(nonZeroDelays[0])):
#             connections[connName].delay[nonZeroDelays[0][i],nonZeroDelays[1][i]] = delayMatrix[nonZeroDelays[0][i],nonZeroDelays[1][i]]
#         nonZeroDelays = np.nonzero(delayMatrix)
#         connections[connName].delay[nonZeroDelays] = delayMatrix[nonZeroDelays]
            
            
    
    if ee_STDP_on:
        if 'ee' in recurrent_conn_names:
            stdp_methods[name+'e'+name+'e'] = b.STDP(connections[name+'e'+name+'e'], eqs=eqs_stdp_ee, pre = eqs_stdp_pre_ee, 
                                                           post = eqs_stdp_post_ee, wmin=0., wmax= wmax_ee)
    if not use_adaptive_threshold:
        stdp_methods[name+'i'+name+'e'] = b.STDP(connections[name+'i'+name+'e'], eqs=eqs_stdp_ie, pre = eqs_stdp_pre_ie, 
                                                 post = eqs_stdp_post_ie, wmin=0., wmax= wmax_ie)

    print 'create monitors for', name
    rate_monitors[name+'e'] = b.PopulationRateMonitor(neuron_groups[name+'e'], bin = (single_example_time+resting_time)/b.second)
    rate_monitors[name+'i'] = b.PopulationRateMonitor(neuron_groups[name+'i'], bin = (single_example_time+resting_time)/b.second)
    spike_counters[name+'e'] = b.SpikeCounter(neuron_groups[name+'e'])
    
    if record_spikes:
        spike_monitors[name+'e'] = b.SpikeMonitor(neuron_groups[name+'e'])
        spike_monitors[name+'i'] = b.SpikeMonitor(neuron_groups[name+'i'])
    if record_states:
        state_monitors[name+'e'] = b.MultiStateMonitor(neuron_groups[name+'e'], ['v', 'ge', 'gi'], record=[0])
        state_monitors[name+'i'] = b.MultiStateMonitor(neuron_groups[name+'i'], ['v', 'ge', 'gi'], record=[0])

if record_spikes:
#     if not use_remote_access:
    b.figure(fig_num)
    fig_num += 1
    b.ion()
    b.subplot(211)
    b.raster_plot(spike_monitors['Ae'], refresh=1000*b.ms, showlast=1000*b.ms)
    b.subplot(212)
    b.raster_plot(spike_monitors['Ai'], refresh=1000*b.ms, showlast=1000*b.ms)



#------------------------------------------------------------------------------ 
# create input population
#------------------------------------------------------------------------------ 
pop_values = [0,0,0]
for i,name in enumerate(input_population_names):
    if name == 'Y':
        input_groups[name+'e'] = b.PoissonGroup(n_label, 0)
    else:
        input_groups[name+'e'] = b.PoissonGroup(n_input, 0)
    rate_monitors[name+'e'] = b.PopulationRateMonitor(input_groups[name+'e'], bin = (single_example_time+resting_time)/b.second)
#     spike_monitors[name+'e'] = SpikeMonitor(input_groups[name+'e'])


#------------------------------------------------------------------------------ 
# create connections from input populations to network populations
#------------------------------------------------------------------------------ 
for name in input_connection_names:
    print 'create connections between', name[0], 'and', name[1]
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1]
        weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')
        weightMatrix = scipy.sparse.lil_matrix(weightMatrix)
        connections[connName] = b.Connection(input_groups[connName[0:2]], neuron_groups[connName[2:4]], structure= conn_structure, 
                                                    state = 'g'+connType[0], delay=True, max_delay=delay[connType][1])
        connections[connName].connect(input_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix, delay=delay[connType])
     
    if ee_STDP_on and (not connName[0] == 'Y' or use_plastic_supervision):
        print 'create STDP for connection', name[0]+'e'+name[1]+'e'
        stdp_methods[name[0]+'e'+name[1]+'e'] = b.STDP(connections[name[0]+'e'+name[1]+'e'], eqs=eqs_stdp_ee, pre = eqs_stdp_pre_ee, 
                                                       post = eqs_stdp_post_ee, wmin=0., wmax= wmax_ee)

real_time_monitor = None
# import brian.experimental.realtime_monitor as rltmMon
# real_time_monitor = rltmMon.RealtimeConnectionMonitor(connections['AeAe'], cmap=cmap.get_cmap('gist_ncar'), wmin=0, wmax=wmax_ee, clock = b.Clock(5000*b.ms))

#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 
previous_spike_count = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))
if not test_mode:
    input_weight_monitor, fig_weights = plot_2d_input_weights()
    fig_num += 1
if do_plot_performance:# and not use_remote_access:
    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)
assignments = np.zeros(n_e)
start = time.time()
for i,name in enumerate(input_population_names):
    input_groups[name+'e'].rate = 0
b.run(0)#, report='text')

j = 0
while j < (int(num_examples)):
    remaining_time = runtime - j*(single_example_time+resting_time)
    
    if test_mode:
        if use_testing_set:
            rates = testing['x'][j%10000,:,:].reshape((n_input)) / 8. *  input_intensity
        else:
            rates = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    else:
        rates = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
        
    if normalize_input:
        rates *= input_intensity / np.mean(rates)
    rates += noise_level
    input_groups['Xe'].rate = rates
#     print (rates)
    if label_intensity > 0:
        if test_mode:
            rates_label = np.zeros(n_label)
        else:
            rates_label = np.zeros(n_label)
            rates_label[(n_label/10)*training['y'][j%60000][0] : (n_label/10)*(training['y'][j%60000][0] + 1)] = label_intensity
        rates_label += noise_level
        input_groups['Ye'].rate = rates_label
#     print np.mean(rates)
#     print training['x'][j,:,:] / 10
#     print rates_label, rates_label.shape
#     imshow(rates.reshape((28,28)), interpolation = 'nearest', cmap = cmap.gray)
#     colorbar()
#     show()
            
#     if remaining_time <= 5 and record_spikes:
#         record_spikes = False
#         for name in neuron_groups:
#             if not name == 'Ce':
#                 spike_monitors[name] = SpikeMonitor(neuron_groups[name])

    if (j%weight_normalization_interval == 0) and not test_mode and weight_path[-8:] != 'weights/' and not use_weight_dependence:
        print weight_path[-8:]
        normalize_weights()
            
    print 'run number:', j+1, 'of', int(num_examples), ', remaining time:', remaining_time, 's'
    b.run(single_example_time)#, report='text')
#     neuron_groups['e'].v = v_rest_e
#     neuron_groups['i'].v = v_rest_i
            
    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
        
    if j % weight_update_interval == 0 and not test_mode:
        update_2d_input_weights(input_weight_monitor, fig_weights)
#         print assignments
    
    if j % save_connections_interval == 0 and j > 0 and not test_mode:
        save_connections(str(j))
        if use_adaptive_threshold:
            save_theta(str(j))
    
    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])
    if np.sum(current_spike_count) < 5:
        input_intensity += 1
#         print 'input increased at example ', j
        if resting_time:
            for i,name in enumerate(input_population_names):
                input_groups[name+'e'].rate = 0
            b.run(resting_time)#, report='text')
    else:
    #     print current_spike_count,  np.asarray(spike_counters['Ce'].count[:]), previous_spike_count
        result_monitor[j%update_interval,:] = current_spike_count
        if test_mode and use_testing_set:
            input_numbers[j] = testing['y'][j%10000][0]
        else:
            input_numbers[j] = training['y'][j%60000][0]
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])
        
        if j % update_interval == 0 and j > 0:
            print 'run number:', j+1, 'of', int(num_examples), ', remaining time:', remaining_time, 's'
            if do_plot_performance:
                unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
                b.figure(fig_performance.number)
                savefig(data_path + '/weights/training_accuracy')
                np.savetxt(data_path + '/weights/training_accuracy', performance[:(j/float(update_interval))+1])
                print 'Classification performance', performance[:(j/float(update_interval))+1]
        
        if resting_time:
            for i,name in enumerate(input_population_names):
                input_groups[name+'e'].rate = 0
            b.run(resting_time)#, report='text')
        input_intensity = start_input_intensity
    #     print "Presented number:", input_numbers[j], ', recognized numbers:', outputNumbers[j,:]#, ', theta avg:', np.mean(neuron_groups['e'].theta)
        j += 1
    
end = time.time()
print 'time needed to simulate:', end - start




#------------------------------------------------------------------------------ 
# save results
#------------------------------------------------------------------------------ 
print 'save results'
if use_adaptive_threshold and not test_mode:
    save_theta()

if not test_mode:
    save_connections(str(j))
    if not use_weight_dependence:
        normalize_weights()
    save_connections()
    if do_plot_performance:# and not use_remote_access:
        b.figure(fig_performance.number)
        savefig(data_path + '/weights/training_accuracy')
else:
    np.save(data_path + '/activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + '/activity/inputNumbers' + str(num_examples), input_numbers)
    


#------------------------------------------------------------------------------ 
# plot results
#------------------------------------------------------------------------------ 
# if not use_remote_access:
if rate_monitors:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(rate_monitors):
        b.subplot(len(rate_monitors), 1, i)
        b.plot(rate_monitors[name].times/b.second, rate_monitors[name].rate, '.')
        b.title('rates of population ' + name)
    
if spike_monitors:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_monitors):
        b.subplot(len(spike_monitors), 1, i)
        b.raster_plot(spike_monitors[name])
        b.title('spikes of population ' + name)
        
if spike_counters:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_counters):
        b.subplot(len(spike_counters), 1, i)
        b.plot(spike_counters['Ae'].count[:])
        b.title('spike count of population ' + name)

if state_monitors:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(state_monitors):
        b.plot(state_monitors[name].times/b.second, state_monitors[name]['v'][0], label = name + ' v 0')
    #     plot(state_monitors[name].times/second, state_monitors[name]['v'][5], label = name + ' 5')
        b.legend()
        b.title('membrane voltages of population ' + name)
    

    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(state_monitors):
        b.plot(state_monitors[name].times/b.second, state_monitors[name]['ge'][0], label = name + ' ge 0')
        b.plot(state_monitors[name].times/b.second, state_monitors[name]['gi'][0], label = name + ' gi 0')
    #     plot(state_monitors[name].times/second, state_monitors[name]['v'][5], label = name + ' 5')
        b.legend()
        b.title('conductances of population ' + name)

plot_weights = [
#                 'XeAe', 
#                 'XeAi', 
#                 'AeAe', 
#                 'AeAi', 
                'AiAe', 
#                 'AiAi', 
               ]

for name in plot_weights:
    b.figure(fig_num)
    fig_num += 1
    if name[1]=='e':
        n_src = n_input
    else:
        n_src = n_i
    if name[3]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
        
    w_post = np.zeros((n_src, n_tgt))
    connMatrix = connections[name][:]
    for i in xrange(n_src):
        w_post[i, connMatrix.rowj[i]] = connMatrix.rowdata[i]
    im2 = b.imshow(w_post, interpolation="nearest", vmin = 0, cmap=cmap.get_cmap('gist_ncar')) #my_cmap
    b.colorbar(im2)
    b.title('weights of connection' + name)
    
    
    
plot_2d_input_weights()


# error = np.abs(result_monitor[:,1] - result_monitor[:,0])
# correctionIdxs = np.where(error > 0.5)[0]
# correctedError = [1 - error[i] if (i in correctionIdxs) else error[i] for i in xrange(len(error))]
# correctedErrorSum = np.average(correctedError)
#     
# figure()
# scatter(result_monitor[:,1], result_monitor[:,0], c=range(len(error)), cmap=cm.gray)
# title('Error: ' + str(correctedErrorSum))
# xlabel('Desired activity')
# ylabel('Population activity')
# 
# figure()
# error = np.abs(result_monitor[:,1] - result_monitor[:,0])
# correctionIdxs = np.where(error > 0.5)[0]
# correctedError = [1 - error[i] if (i in correctionIdxs) else error[i] for i in xrange(len(error))]
# correctedErrorSum = np.average(correctedError)
# scatter(result_monitor[:,1], result_monitor[:,0], c=result_monitor[:,2], cmap=cm.gray)
# title('Error: ' + str(correctedErrorSum))
# xlabel('Desired activity')
# ylabel('Population activity')

b.ioff()
b.show()



























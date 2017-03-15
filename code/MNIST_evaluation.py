'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

import brian as b

from brian import *

import numpy as np
import matplotlib, time, scipy, math
import matplotlib.cm as cmap
import os.path
import cPickle as pickle

from struct import unpack

import brian.experimental.realtime_monitor as rltmMon


#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------
def get_labeled_data(picklename, b_train=True):
    '''
    Read input-vector (image) and target class (label, 0-9) and return
    it as list of tuples.
    '''
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
    else:
        # Open the images with gzip in read binary mode
        if b_train:
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


def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in xrange(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]


def get_new_assignments(result_monitor, input_numbers):    
    assignments = np.ones((conv_features, n_e)) * -1
    input_nums = np.asarray(input_numbers)#.reshape((conv_features, n_e))
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


MNIST_data_path = '../data/'
data_path = '../activity/'
training_ending = '10000'
testing_ending = '10000'
start_time_training = 0
end_time_training = int(training_ending)
start_time_testing = 0
end_time_testing = int(testing_ending)


# input and square root of input
n_input = 784
n_input_sqrt = int(math.sqrt(n_input))

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

print '\n'

# set ending of filename saves
ending = '_' + stdp_input + '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e)


print '...loading MNIST'
training = get_labeled_data(MNIST_data_path + 'training', b_train=True)
testing = get_labeled_data(MNIST_data_path + 'testing', b_train=False)


print '...loading results'
training_result_monitor = np.load(data_path + 'resultPopVecs' + training_ending + ending + '.npy')
training_input_numbers = np.load(data_path + 'inputNumbers' + training_ending + ending + '.npy')
testing_result_monitor = np.load(data_path + 'resultPopVecs' + testing_ending + ending + '.npy')
testing_input_numbers = np.load(data_path + 'inputNumbers' + testing_ending + ending + '.npy')


print '...getting assignments'
test_results = np.zeros((10, end_time_testing - start_time_testing))
test_results_max = np.zeros((10, end_time_testing - start_time_testing))
test_results_top = np.zeros((10, end_time_testing - start_time_testing))
test_results_fixed = np.zeros((10, end_time_testing - start_time_testing))
assignments = get_new_assignments(training_result_monitor[start_time_training : end_time_training], 
                                  training_input_numbers[start_time_training : end_time_training])

counter = 0 
num_tests = end_time_testing / 10000
sum_accurracy = [0] * num_tests


while (counter < num_tests):
    end_time = min(end_time_testing, 10000 * (counter + 1))
    start_time = 10000*counter
    test_results = np.zeros((10, end_time-start_time))
    
    print '...calculating accuracy for sum'
    
    for i in xrange(end_time - start_time):
        test_results[:,i] = get_recognized_number_ranking(assignments, testing_result_monitor[i + start_time, :])
    
    difference = test_results[0,:] - testing_input_numbers[start_time:end_time]
    correct = len(np.where(difference == 0)[0])
    incorrect = np.where(difference != 0)[0]
    sum_accurracy[counter] = correct/float(end_time-start_time) * 100
    
    print 'Sum response - accuracy: ', sum_accurracy[counter], ' num incorrect: ', len(incorrect)
    
    counter += 1

print 'Sum response - accuracy --> mean: ', np.mean(sum_accurracy), '\n'

b.show()
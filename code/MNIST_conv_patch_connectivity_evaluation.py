'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

import brian as b

from brian import *

import numpy as np
import matplotlib, time, scipy, math, sys, argparse
import matplotlib.cm as cmap
import os.path
import cPickle as pickle

from struct import unpack

import brian.experimental.realtime_monitor as rltmMon

np.set_printoptions(threshold=np.nan)

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
    '''
    Given the label assignments of the excitatory layer and their spike rates over
    the past 'update_interval', get the ranking of each of the categories of input.
    '''
    if voting_mechanism == 'most-spiked':
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

    elif voting_mechanism == 'all':
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


MNIST_data_path = '../data/'
data_path = '../activity/conv_patch_connectivity_activity/'

training_ending = '10000'
testing_ending = '10000'

start_time_training = 0
end_time_training = int(training_ending)
start_time_testing = 0
end_time_testing = int(testing_ending)

parser = argparse.ArgumentParser()

parser.add_argument('-m',  '--mode', default='train')
parser.add_argument('-c',  '--connectivity', default='all')
parser.add_argument('-wd', '--weight_dependence', default='no_weight_dependence')
parser.add_argument('-pp', '--post_pre', default='postpre')
parser.add_argument('--conv_size', type=int, default=20)
parser.add_argument('--conv_stride', type=int, default=2)
parser.add_argument('--conv_features', type=int, default=25)
parser.add_argument('--weight_sharing', default='no_weight_sharing')
parser.add_argument('--lattice_structure', default='4')
parser.add_argument('--random_inhibition_prob', type=float, default=0.0)
parser.add_argument('--top_percent', type=int, default=10)

args = parser.parse_args()
mode, connectivity, weight_dependence, post_pre, conv_size, conv_stride, conv_features, weight_sharing, lattice_structure, random_inhibition_prob, top_percent = \
    args.mode, args.connectivity, args.weight_dependence, args.post_pre, args.conv_size, args.conv_stride, args.conv_features, args.weight_sharing, \
    args.lattice_structure, args.random_inhibition_prob, args.top_percent

print '\n'

print args.mode, args.connectivity, args.weight_dependence, args.post_pre, args.conv_size, args.conv_stride, args.conv_features, args.weight_sharing, \
    args.lattice_structure, args.random_inhibition_prob, args.top_percent

print '\n'

# input and square root of input
n_input = 784
n_input_sqrt = int(math.sqrt(n_input))

# number of excitatory neurons (number output from convolutional layer)
n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
n_e_total = n_e * conv_features
n_e_sqrt = int(math.sqrt(n_e))

# number of inhibitory neurons (number of convolutational features (for now))
n_i = n_e

# STDP rule
stdp_input = weight_dependence + '_' + post_pre
if weight_dependence == 'weight_dependence':
    use_weight_dependence = True
else:
    use_weight_dependence = False
if post_pre == 'postpre':
    use_post_pre = True
else:
    use_post_pre = False

# set ending of filename saves
ending = connectivity + '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e) + '_' + stdp_input + '_' + weight_sharing + '_' + lattice_structure + '_' + str(random_inhibition_prob)

print '...loading MNIST'
training = get_labeled_data(MNIST_data_path + 'training', b_train=True)
testing = get_labeled_data(MNIST_data_path + 'testing', b_train=False)

print '...loading results'
training_result_monitor = np.load(data_path + 'results_' + training_ending + '_' + ending + '.npy')
training_input_numbers = np.load(data_path + 'input_numbers_' + training_ending + '_' + ending + '.npy')
testing_result_monitor = np.load(data_path + 'results_' + testing_ending + '_' + ending + '.npy')
testing_input_numbers = np.load(data_path + 'input_numbers_' + testing_ending + '_' + ending + '.npy')

print '...getting assignments'
test_results = np.zeros((10, end_time_testing - start_time_testing))
test_results_max = np.zeros((10, end_time_testing - start_time_testing))
test_results_top = np.zeros((10, end_time_testing - start_time_testing))
test_results_fixed = np.zeros((10, end_time_testing - start_time_testing))
assignments = get_new_assignments(training_result_monitor[start_time_training : end_time_training], training_input_numbers[start_time_training : end_time_training])

counter = 0
num_tests = end_time_testing / 10000
sum_accurracy = [0] * num_tests


while (counter < num_tests):
    end_time = min(end_time_testing, 10000 * (counter + 1))
    start_time = 10000 * counter
    test_results = np.zeros((10, end_time - start_time))

    print '...calculating accuracy for sum'

    for i in xrange(end_time - start_time):
        test_results[:, i] = get_recognized_number_ranking(assignments, testing_result_monitor[i + start_time, :])

    difference = test_results[0,:] - testing_input_numbers[start_time:end_time]
    correct = len(np.where(difference == 0)[0])
    incorrect = np.where(difference != 0)[0]
    sum_accurracy[counter] = correct / float(end_time-start_time) * 100

    print 'Sum response - accuracy: ', sum_accurracy[counter], ' num incorrect: ', len(incorrect)

    counter += 1

print 'Sum response - accuracy --> mean: ', np.mean(sum_accurracy), '\n'

b.show()

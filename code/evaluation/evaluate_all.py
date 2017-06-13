import brian as b
import numpy as np
import pandas as pd
import cPickle as p
import networkx as nx
import matplotlib.cm as cmap
import brian.experimental.realtime_monitor as rltmMon
import matplotlib, time, scipy, math, sys, argparse, os

from brian import *
from struct import unpack

np.set_printoptions(threshold=np.nan)


def get_labeled_data(picklename, b_train=True):
    '''
    Read input-vector (image) and target class (label, 0-9) and return it as 
    a list of tuples.
    '''
    if os.path.isfile('%s.pickle' % picklename):
        data = p.load(open('%s.pickle' % picklename))
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


def get_recognized_number_ranking(assignments, simple_clusters, spike_rates, average_firing_rate):
    '''
    Given the label assignments of the excitatory layer and their spike rates over
    the past 'update_interval', get the ranking of each of the categories of input.
    '''

    most_spiked_summed_rates = [0] * 10
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
        num_assignments[i] = len(np.where(assignments[most_spiked_array] == i)[0])

        if len(spike_rates[np.where(assignments[most_spiked_array] == i)]) > 0:
            # sum the spike rates of all excitatory neurons with this label, which fired the most in its patch
            most_spiked_summed_rates[i] = np.sum(spike_rates[np.where(np.logical_and(assignments == i, most_spiked_array))]) / float(np.sum(spike_rates[most_spiked_array]))

    all_summed_rates = [0] * 10
    num_assignments = [0] * 10

    for i in xrange(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            all_summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]

    top_percent_summed_rates = [0] * 10
    num_assignments = [0] * 10
    
    top_percent_array = np.array(np.zeros((conv_features, n_e)), dtype=bool)
    top_percent_array[np.where(spike_rates > np.percentile(spike_rates, 100 - top_percent))] = True

    # for each label
    for i in xrange(10):
        # get the number of label assignments of this type
        num_assignments[i] = len(np.where(assignments[top_percent_array] == i)[0])

        if len(np.where(assignments[top_percent_array] == i)) > 0:
            # sum the spike rates of all excitatory neurons with this label, which fired the most in its patch
            top_percent_summed_rates[i] = len(spike_rates[np.where(np.logical_and(assignments == i, top_percent_array))])

    spike_rates_flat = np.copy(np.ravel(spike_rates))

    simple_cluster_summed_rates = [0] * 10
    num_assignments = [0] * 10

    for i in xrange(10):
        if i in simple_clusters.keys() and len(simple_clusters[i]) > 1:
            # simple_cluster_summed_rates[i] = np.sum(spike_rates_flat[simple_clusters[i]]) / float(len(simple_clusters[i]))
            this_spike_rates = spike_rates_flat[simple_clusters[i]]
            simple_cluster_summed_rates[i] = np.sum(this_spike_rates[np.argpartition(this_spike_rates, -10)][-10:])

    return [ np.argsort(summed_rates)[::-1] for summed_rates in (all_summed_rates, most_spiked_summed_rates, top_percent_summed_rates, simple_cluster_summed_rates) ]


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

    simple_clusters = {}
    votes_vector = {}

    for cluster in simple_clusters.keys():
        votes_vector[cluster] = np.zeros(10)

    average_firing_rate = np.zeros(10)

    for j in xrange(10):
        this_result_monitor = result_monitor[input_nums == j]
        average_firing_rate[j] = np.sum(this_result_monitor[np.nonzero(this_result_monitor)]) \
                            / float(np.size(this_result_monitor[np.nonzero(this_result_monitor)]))

    for j in xrange(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis=0) / float(num_assignments)
            this_result_monitor = result_monitor[input_nums == j]
            simple_clusters[j] = np.argsort(np.ravel(np.sum(this_result_monitor, axis=0)))[::-1][:int(0.05 * (np.size(result_monitor) / 10000))]

    return assignments, simple_clusters, average_firing_rate


top_level_path = '../../'
MNIST_data_path = top_level_path + 'data/'
data_path = top_level_path + 'activity/conv_patch_connectivity_activity/'

print '\n...Loading MNIST'
training = get_labeled_data(MNIST_data_path + 'training', b_train=True)
testing = get_labeled_data(MNIST_data_path + 'testing', b_train=False)

# input and square root of input
n_input = 784
n_input_sqrt = int(math.sqrt(n_input))

file_names = [ file_name.split('results')[1] for file_name in sorted(os.listdir(data_path)) if 'results' in file_name and '10000' in file_name ]

results = {}

for file_name in file_names:
    print '\n...Evaluating', file_name

    training_result_monitor = np.load(data_path + 'results' + file_name)
    training_input_numbers = np.load(data_path + 'input_numbers' + file_name)
    testing_result_monitor = np.load(data_path + 'results' + file_name)
    testing_input_numbers = np.load(data_path + 'input_numbers' + file_name)

    training_ending = testing_ending = int(file_name.split('_')[1])

    conv_size = int(file_name.split('_')[3])
    conv_stride = int(file_name.split('_')[4])
    conv_features = int(file_name.split('_')[5])

    n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
    n_e_total = n_e * conv_features
    n_e_sqrt = int(math.sqrt(n_e))
    n_i = n_e

    top_percent = 10

    print '\n...Getting assignments'
    assignments, simple_clusters, average_firing_rate = get_new_assignments(training_result_monitor, training_input_numbers)

    print '\n...Calculating accuracy for sum'

    test_results = np.zeros((4, 10, testing_ending))
    for j in xrange(testing_ending):
        temp =  get_recognized_number_ranking(assignments, simple_clusters, testing_result_monitor[j, :], average_firing_rate)
        for i in xrange(4):
            test_results[i, :, j] = temp[i]

    differences = [ test_results[i, 0, :] - testing_input_numbers for i in xrange(test_results.shape[0]) ]
    corrects = [ len(np.where(difference == 0)[0]) for difference in differences ]
    incorrects = [ np.where(difference != 0)[0] for difference in differences ]
    accuracies = [ correct / float(testing_ending) * 100 for correct in corrects ]

    results[file_name] = accuracies

    print '\n'
    print 'All neurons response - accuracy:', accuracies[0], 'num incorrect:', len(incorrects[0])
    print 'Most-spiked (per patch) neurons vote - accuracy:', accuracies[1], 'num incorrect:', len(incorrects[1])
    print 'Most-spiked (overall) neurons vote - accuracy:', accuracies[2], 'num incorrect:', len(incorrects[2])
    print 'Simple clusters vote - accuracy:', accuracies[3], 'num incorrect:', len(incorrects[3])
    print '\n'

results_csv = pd.DataFrame([ [ accuracy for accuracy in result ] for result in results.values() ], index=results.keys(), columns=['All', 'Most-spiked per patch', 'Most-spiked overall', 'Correlation clustering']) # , columns=['All', 'Most-Spiked Per Patch', 'Most-Spiked Overall', 'Activity Correlation Clustering'])
results_csv.to_csv(top_level_path + 'data/all_accuracy_results.csv')

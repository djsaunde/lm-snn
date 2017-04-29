'''
@author: Dan Saunders
'''

import brian as b
import numpy as np
import networkx as nx
import cPickle as p
import matplotlib.cm as cmap
import brian.experimental.realtime_monitor as rltmMon
import matplotlib, time, scipy, math, sys, argparse, os

from brian import *
from struct import unpack

np.set_printoptions(threshold=np.nan, linewidth=200)


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


def predict_label(assignments, simple_clusters, index_matrix, input_numbers, spike_rates, average_firing_rate):
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

    # cluster_summed_rates = [0] * 10
    # num_assignments = [0] * 10

    # spike_rates_flat = np.copy(np.ravel(spike_rates))

    # for i in xrange(10):
    #   num_assignments[i] = 0
    #   for assignment in cluster_assignments.keys():
    #       if cluster_assignments[assignment] == i and len(clusters[assignment]) > 1:
    #           num_assignments[i] += 1
    #   if num_assignments[i] > 0:
    #       for assignment in cluster_assignments.keys():
    #           if cluster_assignments[assignment] == i and len(clusters[assignment]) > 1:
    #               cluster_summed_rates[i] += np.sum(spike_rates_flat[clusters[assignment]]) / float(len(clusters[assignment]))

    # spike_rates_flat = np.copy(np.ravel(spike_rates))

    # kmeans_summed_rates = [0] * 10
    # num_assignments = [0] * 10

    # for i in xrange(10):
    #     num_assignments[i] = 0
    #     for assignment in kmeans_assignments.keys():
    #         if kmeans_assignments[assignment] == i:
    #             num_assignments[i] += 1
    #     if num_assignments[i] > 0:
    #         for cluster, assignment in enumerate(kmeans_assignments.keys()):
    #             if kmeans_assignments[assignment] == i:
    #                 kmeans_summed_rates[i] += sum([ spike_rates_flat[idx] for idx, label in enumerate(kmeans.labels_) if label == cluster ]) / float(len([ label for label in kmeans.labels_ if label == i ]))

    spike_rates_flat = np.copy(np.ravel(spike_rates))

    simple_cluster_summed_rates = [0] * 10
    num_assignments = [0] * 10

    for i in xrange(10):
        if i in simple_clusters.keys() and len(simple_clusters[i]) > 1:
            # simple_cluster_summed_rates[i] = np.sum(spike_rates_flat[simple_clusters[i]]) / float(len(simple_clusters[i]))
            this_spike_rates = spike_rates_flat[simple_clusters[i]]
            simple_cluster_summed_rates[i] = np.sum(this_spike_rates[np.argpartition(this_spike_rates, -5)][-10:])

    # print simple_cluster_summed_rates

    # simple_cluster_summed_rates = simple_cluster_summed_rates / average_firing_rate

    spatial_cluster_index_vector = np.empty(n_e)
    spatial_cluster_index_vector[:] = np.nan

    for idx in xrange(n_e):
        this_spatial_location = spike_rates_flat[idx::n_e]
        if np.size(np.where(this_spatial_location > 0.9 * np.max(spike_rates_flat))) > 0:
            spatial_cluster_index_vector[idx] = np.argmax(this_spatial_location)

    spatial_cluster_summed_rates = [0] * 10
    if input_numbers != []:
        if np.count_nonzero([[ x == y for (x, y) in zip(spatial_cluster_index_vector, index_matrix[idx]) ] for idx in xrange(update_interval) ]) > 0:
            # best_col_idx = np.argmax([ sum([ 1.0 if x == y else 0.0 for (x, y) in zip(spatial_cluster_index_vector, index_matrix[idx]) ]) for idx in xrange(update_interval) ])
            # spatial_cluster_summed_rates[input_numbers[best_col_idx]] = 1.0
            for idx in xrange(update_interval):
              # print spatial_cluster_index_vector == index_matrix[idx]
              spatial_cluster_summed_rates[input_numbers[idx]] += np.count_nonzero([ x == y for (x, y) in zip(spatial_cluster_index_vector, index_matrix[idx]) ])

            # print '->', [ input_numbers.count(i) for i in xrange(10) ]
            spatial_cluster_summed_rates = [ x / float(y) if y != 0 else x for (x, y) in zip(spatial_cluster_summed_rates, [ np.count_nonzero(input_numbers == i)for i in xrange(10) ]) ]

    # if spatial_cluster_summed_rates == [0] * 10:
    #   print '>', spatial_cluster_index_vector

    # print spatial_cluster_summed_rates

    return ( np.argsort(summed_rates)[::-1] for summed_rates in (all_summed_rates, most_spiked_summed_rates, top_percent_summed_rates, \
                                                                     simple_cluster_summed_rates, spatial_cluster_summed_rates) )


def assign_labels(result_monitor, input_numbers):
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

    # weight_matrix = np.copy(np.array(connections['AeAe'][:].todense()))
    
    # print '\n'
    # print 'Maximum between-patch edge weight:', np.max(weight_matrix)
    # print '\n'
    
    # print '99-th percentile:', np.percentile(weight_matrix[np.where(weight_matrix != 0)], 99)
    # print '99.5-th percentile:', np.percentile(weight_matrix[np.where(weight_matrix != 0)], 99.5)
    # print '99.9-th percentile:', np.percentile(weight_matrix[np.where(weight_matrix != 0)], 99.9)

    # weight_matrix[weight_matrix < np.percentile(weight_matrix[np.where(weight_matrix != 0)], 99)] = 0.0
    # weight_matrix[weight_matrix > 0.0] = 1

    # recurrent_graph = nx.Graph(weight_matrix)

    # plt.figure(figsize=(18.5, 10))
    # nx.draw_circular(recurrent_graph, node_color='g', edge_color='#909090', edge_size=1, node_size=10)
    # plt.axis('equal')

    # plt.show()

    # _, temp = networkx_mcl(recurrent_graph, expand_factor=2, inflate_factor=2, mult_factor=2)

    # clusters = {}
    # for key, value in temp.items():
    #   if value not in clusters.values():
    #       clusters[key] = value

    # # print '\n'
    # # print 'Number of qualifying clusters:', len([ cluster for cluster in clusters.values() if len(cluster) > 1 ])
    # # print 'Average size of qualifying clusters:', sum([ len(cluster) for cluster in clusters.values() if len(cluster) > 1 ]) / float(len(clusters))
    # # print 'Nodes per cluster:', sorted([ len(cluster) for cluster in clusters.values() ], reverse=True)

    # cluster_assignments = {}
    # votes_vector = {}

    # for cluster in clusters.keys():
    #   cluster_assignments[cluster] = -1
    #   votes_vector[cluster] = np.zeros(10)

    # for j in xrange(10):
    #   num_assignments = len(np.where(input_nums == j)[0])
    #   if num_assignments > 0:
    #       rate = np.sum(result_monitor[input_nums == j], axis=0) / float(num_assignments)
    #       rate = np.ravel(rate)
    #       for cluster in clusters.keys():
    #           if len(clusters[cluster]) > 1:
    #               votes_vector[cluster][j] += np.sum(rate[clusters[cluster]]) / float(rate[clusters[cluster]].size)
    #       if j in cluster_assignments.values():
    #           votes_vector[j] / float(len([ value for value in cluster_assignments.values() if value == j ]))
    
    # for cluster in clusters.keys():
    #   cluster_assignments[cluster] = np.argmax(votes_vector[cluster])

    # print 'Qualifying cluster assignments (in order of label):', sorted([ value for key, value in cluster_assignments.items() if value != -1 and len(clusters[key]) > 1 ]), '\n'
    # for idx in xrange(10):
    #   print 'There are', len([ value for key, value in cluster_assignments.items() if value == idx and len(clusters[key]) > 1 ]), str(idx) + '-labeled qualifying clusters'
    # print '\n'

    # kmeans_assignments = {}
    # votes_vector = {}

    # # get the list of flattened input weights per neuron per feature
    # weights = get_input_weights(np.copy(input_connections['XeAe'][:].todense()))

    # # create and fit a KMeans model
    # kmeans = KMeans(n_clusters=25).fit(weights)

    # for cluster in xrange(kmeans.n_clusters):
    #     kmeans_assignments[cluster] = -1
    #     votes_vector[cluster] = np.zeros(10)

    # for j in xrange(10):
    #     num_assignments = len(np.where(input_nums == j)[0])
    #     if num_assignments > 0:
    #         rate = np.sum(result_monitor[input_nums == j], axis=0) / float(num_assignments)
    #         rate = np.ravel(rate)
    #         for cluster in xrange(kmeans.n_clusters):
    #             votes_vector[cluster][j] += sum([ rate[idx] for idx, label in enumerate(kmeans.labels_) if label == cluster ]) / float(len([ label for label in kmeans.labels_ if label == j ]))

    # for cluster in xrange(kmeans.n_clusters):
    #     kmeans_assignments[cluster] = np.argmax(votes_vector[cluster])

    # print 'kmeans cluster assignments (in order of label):', sorted([ value for key, value in kmeans_assignments.items() if value != -1 ]), '\n'
    # for idx in xrange(10):
    #   print 'There are', len([ value for key, value in kmeans_assignments.items() if value == idx ]), str(idx) + '-labeled KMeans clusters'
    # print '\n'

    simple_clusters = {}
    votes_vector = {}

    for cluster in simple_clusters.keys():
        votes_vector[cluster] = np.zeros(10)

    average_firing_rate = np.zeros(10)

    for j in xrange(10):
        this_result_monitor = result_monitor[input_nums == j]
        average_firing_rate[j] = np.sum(this_result_monitor[np.nonzero(this_result_monitor)]) \
                            / float(np.size(this_result_monitor[np.nonzero(this_result_monitor)]))

    # print '\n', average_firing_rate, '\n'

    for j in xrange(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis=0) / float(num_assignments)
            this_result_monitor = result_monitor[input_nums == j]
            # simple_clusters[j] = np.argwhere(np.sum(result_monitor[input_nums == j], axis=0) > np.percentile(this_result_monitor[np.nonzero(this_result_monitor)], 99))
            # simple_clusters[j] = np.array([ node[0] * n_e + node[1] for node in simple_clusters[j] ])
            # print '99-th percentile for cluster', j, ':', np.percentile(this_result_monitor[np.nonzero(this_result_monitor)], 99)
            simple_clusters[j] = np.argsort(np.ravel(np.sum(this_result_monitor, axis=0)))[::-1][:40]
            # simple_clusters[j] = np.array([ node[0] * n_e + node[1] for node in simple_clusters[j] ])
            # print simple_clusters[j]

    # np.savetxt('activity.txt', result_monitor[j])

    # print '\n'
    # for j in xrange(10):
    #     if j in simple_clusters.keys():
    #         print 'There are', len(simple_clusters[j]), 'neurons in the cluster for digit', j, '\n'
    # print '\n'

    index_matrix = np.empty((update_interval, n_e))
    index_matrix[:] = np.nan

    for idx in xrange(update_interval):
        this_result_monitor_flat = np.ravel(result_monitor[idx, :])
        for n in xrange(n_e):
            this_spatial_result_monitor_flat = this_result_monitor_flat[n::n_e]
            if np.size(np.where(this_spatial_result_monitor_flat > 0.9 * np.max(this_result_monitor_flat))) > 0:
                index_matrix[idx, n] = np.argmax(this_spatial_result_monitor_flat)

    # print index_matrix

    return assignments, simple_clusters, average_firing_rate, index_matrix


MNIST_data_path = '../data/'
data_path = '../activity/conv_patch_connectivity_activity/'

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train')
parser.add_argument('--connectivity', default='none')
parser.add_argument('--weight_dependence', default='no_weight_dependence')
parser.add_argument('--post_pre', default='postpre')
parser.add_argument('--conv_size', type=int, default=16)
parser.add_argument('--conv_stride', type=int, default=4)
parser.add_argument('--conv_features', type=int, default=50)
parser.add_argument('--weight_sharing', default='no_weight_sharing')
parser.add_argument('--lattice_structure', default='4')
parser.add_argument('--random_lattice_prob', type=float, default=0.0)
parser.add_argument('--random_inhibition_prob', type=float, default=0.0)
parser.add_argument('--top_percent', type=int, default=10)
parser.add_argument('--training_ending', type=int, default=10000)
parser.add_argument('--testing_ending', type=int, default=10000)
parser.add_argument('--training_partition', type=int, default=1000)
parser.add_argument('--testing_partition', type=int, default=1000)
parser.add_argument('--filename', type=bool, default=False)

args = parser.parse_args()

# input and square root of input
n_input = 784
n_input_sqrt = int(math.sqrt(n_input))

update_interval = args.training_partition
training_partition, testing_partition = args.training_partition, args.testing_partition

if args.filename == True:
    print '\n'
    print '\n'.join([ str(idx) + ' | ' + file_name for idx, file_name in enumerate([ file_name for file_name in sorted(os.listdir(data_path)) if 'results' in file_name ]) ])
    print '\n'

    to_evaluate = raw_input('Enter the index of the file from above which you\'d like to plot: ')
    file_name = [ file_name for file_name in sorted(os.listdir(data_path)) if 'results' in file_name ][int(to_evaluate)].split('results')[1]
    
    training_result_monitor = np.load(data_path + 'results' + file_name)
    training_input_numbers = np.load(data_path + 'input_numbers' + file_name)
    testing_result_monitor = np.load(data_path + 'results' + file_name)
    testing_input_numbers = np.load(data_path + 'input_numbers' + file_name)

    training_ending = args.training_ending
    testing_ending = args.testing_ending

    conv_size = int(file_name.split('_')[3])
    conv_stride = int(file_name.split('_')[4])
    conv_features = int(file_name.split('_')[5])

    # number of excitatory neurons (number output from convolutional layer)
    n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
    n_e_total = n_e * conv_features
    n_e_sqrt = int(math.sqrt(n_e))

    # number of inhibitory neurons
    n_i = n_e

    top_percent = 10
else:

    mode, connectivity, weight_dependence, post_pre, conv_size, conv_stride, conv_features, weight_sharing, lattice_structure, \
        random_lattice_prob, random_inhibition_prob, top_percent, training_ending, testing_ending = args.mode, args.connectivity, \
        args.weight_dependence, args.post_pre, args.conv_size, args.conv_stride, args.conv_features, args.weight_sharing, \
        args.lattice_structure, args.random_lattice_prob, args.random_inhibition_prob, args.top_percent, args.training_ending, args.testing_ending

    print '\n'

    print args.mode, args.connectivity, args.weight_dependence, args.post_pre, args.conv_size, args.conv_stride, args.conv_features, args.weight_sharing, \
        args.lattice_structure, args.random_lattice_prob, args.random_inhibition_prob, args.top_percent, args.training_ending, args.testing_ending

    print '\n'

    # number of excitatory neurons (number output from convolutional layer)
    n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
    n_e_total = n_e * conv_features
    n_e_sqrt = int(math.sqrt(n_e))

    # number of inhibitory neurons
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
    ending = connectivity + '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e) + '_' + weight_dependence + '_' + post_pre + '_' + weight_sharing + '_' + lattice_structure + '_' + str(random_lattice_prob) # + '_' + str(random_inhibition_prob)

    print '...loading results'
    training_result_monitor = np.load(data_path + 'results_' + str(training_ending) + '_' + ending + '.npy')[:training_partition]
    training_input_numbers = np.load(data_path + 'input_numbers_' + str(training_ending) + '_' + ending + '.npy')[:training_partition]
    testing_result_monitor = np.load(data_path + 'results_' + str(testing_ending) + '_' + ending + '.npy')[training_partition:training_partition + testing_partition]
    testing_input_numbers = np.load(data_path + 'input_numbers_' + str(testing_ending) + '_' + ending + '.npy')[training_partition:training_partition + testing_partition]


start_time_training = 0
end_time_training = int(training_partition)
start_time_testing = training_partition
end_time_testing = int(training_partition + testing_partition)

print '...loading MNIST'
training = get_labeled_data(MNIST_data_path + 'training', b_train=True)
testing = get_labeled_data(MNIST_data_path + 'testing', b_train=False)

print '...getting assignments'
test_results = np.zeros((10, end_time_testing - start_time_testing))
test_results_max = np.zeros((10, end_time_testing - start_time_testing))
test_results_top = np.zeros((10, end_time_testing - start_time_testing))
test_results_fixed = np.zeros((10, end_time_testing - start_time_testing))

assignments, simple_clusters, average_firing_rate, index_matrix = assign_labels(training_result_monitor, training_input_numbers)

num_tests = end_time_testing / testing_ending
sum_accuracy = [0, 0, 0, 0, 0]

end_time = end_time_testing
start_time = start_time_testing

test_results1 = np.zeros((10, end_time - start_time))
test_results2 = np.zeros((10, end_time - start_time))
test_results3 = np.zeros((10, end_time - start_time))
test_results4 = np.zeros((10, end_time - start_time))
test_results5 = np.zeros((10, end_time - start_time))

print '...calculating accuracy for sum'

# index_matrix[np.isnan(index_matrix)] = 0
# index_matrix[np.logical_not(index_matrix == 0)] = 1
# print '\n', index_matrix.shape
# print '\n', training_input_numbers.shape

# temp = np.zeros((index_matrix.shape[0], index_matrix.shape[1] + 1))
# for idx in range(250):
#     temp[idx, :] = [ item for item in index_matrix[idx] ] + [ training_input_numbers[idx] ]

# temp = temp[temp[:, 16].argsort()]
# temp = np.array([ temp[idx, :] for idx in xrange(250) if np.sum(temp[idx, 0:16]) == 1 ])

# print temp

# temp = temp[np.where(np.sum(temp[:, 0:16], axis=1))]

# np.savetxt('sorted_vectors3.txt', temp)


for i in xrange(training_partition):
    test_results1[:, i], test_results2[:, i], test_results3[:, i], test_results4[:, i], test_results5[:, i] = predict_label(assignments, simple_clusters, index_matrix, training_input_numbers, testing_result_monitor[i, :], average_firing_rate)

difference1 = test_results1[0, :] - testing_input_numbers
difference2 = test_results2[0, :] - testing_input_numbers
difference3 = test_results3[0, :] - testing_input_numbers
difference4 = test_results4[0, :] - testing_input_numbers
difference5 = test_results5[0, :] - testing_input_numbers

correct1 = len(np.where(difference1 == 0)[0])
correct2 = len(np.where(difference2 == 0)[0])
correct3 = len(np.where(difference3 == 0)[0])
correct4 = len(np.where(difference4 == 0)[0])
correct5 = len(np.where(difference5 == 0)[0])

incorrect1 = np.where(difference1 != 0)[0]
incorrect2 = np.where(difference2 != 0)[0]
incorrect3 = np.where(difference3 != 0)[0]
incorrect4 = np.where(difference4 != 0)[0]
incorrect5 = np.where(difference5 != 0)[0]

sum_accuracy[0] = correct1 / float(end_time - start_time) * 100
sum_accuracy[1] = correct2 / float(end_time - start_time) * 100
sum_accuracy[2] = correct3 / float(end_time - start_time) * 100
sum_accuracy[3] = correct4 / float(end_time - start_time) * 100
sum_accuracy[4] = correct5 / float(end_time - start_time) * 100

print 'All neurons response - accuracy:', sum_accuracy[0], 'num incorrect:', len(incorrect1)
print 'Most-spiked (per patch) neurons vote - accuracy:', sum_accuracy[1], 'num incorrect:', len(incorrect2)
print 'Most-spiked (overall) neurons vote - accuracy:', sum_accuracy[2], 'num incorrect:', len(incorrect3)
print 'Simple clusters vote - accuracy:', sum_accuracy[3], 'num incorrect:', len(incorrect4)
print 'Spatial correlation clusters vote - accuracy:', sum_accuracy [4], 'num incorrect:', len(incorrect5)

b.show()

print '\n'
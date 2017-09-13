from __future__ import division
# from __future__ import print_function
import numpy as np
import matplotlib.cm as cmap
import time, os, scipy, math, sys, timeit
import cPickle as p
import brian_no_units
import brian as b
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import os
from scipy.sparse import coo_matrix
from struct import unpack
from brian import *
from matplotlib import gridspec

fig_num = 0
wmax_ee = 1

top_level_path = os.path.join('..', '..')

perf_model_name = 'csnn_pc_inhibit_far'
performance_dir = os.path.join(top_level_path, 'performance', perf_model_name)
model_name = 'csnn_pc_inhibit_far'
weight_dir = os.path.join(top_level_path, 'weights', model_name)
plots_dir = os.path.join(top_level_path, 'plots', model_name)

if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)


def onclick(event):
    # clear subplots
    # print "***********************************"
    if event.inaxes:
        ax = event.inaxes  # the axes instance
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))
        print('data coords %f %f' % (event.xdata, event.ydata))
        x = event.xdata * 100
        t = 0
        if x % 10 > 5:
            t = 1
        # indx = np.searchsorted(ax.get_lines()[1].get_data()[0], [event.xdata])[0]
        iteration = ((int(x/10)+t)*10)
        print "iteration = " + str(iteration)
        new_file_name = 'XeAe_' + '_'.join(file_name.split('_')[1:-1]) + '_' + str(iteration) + '.npy'
        weight_matrix = np.load(os.path.join(weight_dir, new_file_name))
        plt.subplot(211).clear()

        '''
            Plot the weights from input to excitatory layer to view what happened during training.
            '''
        weights = get_2d_input_weights(weight_matrix)

        if n_e != 1:
            fig = plt.figure(1)
            ax1 = plt.subplot(211)
            fig.clf()
            plt.figure(figsize=(18, 9))

        else:
            fig = plt.figure(1)  # , figsize=(9, 9))
            ax1 = plt.subplot(211)

        im = ax1.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))

        plt.title('Reshaped input -> convolution weights')

        if n_e != 1:
            plt.xticks(xrange(conv_size, conv_size * (conv_features + 1), conv_size), xrange(1, conv_features + 1))
            plt.yticks(xrange(conv_size, conv_size * (n_e + 1), conv_size), xrange(1, n_e + 1))
            plt.xlabel('Convolution patch')
            plt.ylabel('Location in input (from top left to bottom right')

        fig.canvas.draw()



def is_lattice_connection(sqrt, i, j):
    '''
    Boolean method which checks if two indices in a network correspond to neighboring nodes in a 4-, 8-, or all-lattice.

    sqrt: square root of the number of nodes in population
    i: First neuron's index
    k: Second neuron's index
    '''
    if lattice_structure == 'none':
        return False
    if lattice_structure == '4':
        return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j
    if lattice_structure == '8':
        return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j or i + sqrt == j + 1 and j % sqrt != 0 or i + sqrt == j - 1 and i % sqrt != 0 or i - sqrt == j + 1 and i % sqrt != 0 or i - sqrt == j - 1 and j % sqrt != 0
    if lattice_structure == 'all':
        return True


def normalize_weights():
    '''
    Squash the input -> excitatory weights to sum to a prespecified number.
    '''
    for feature in xrange(conv_features):
        feature_connection = weight_matrix[:, feature * n_e: (feature + 1) * n_e]
        column_sums = np.sum(feature_connection, axis=0)
        column_factors = weight['ee_input'] / column_sums

        for n in xrange(n_e):
            dense_weights = weight_matrix[:, feature * n_e + n]
            dense_weights[convolution_locations[n]] *= column_factors[n]
            weight_matrix[:, feature * n_e + n] = dense_weights


def get_2d_input_weights(weight_matrix):
    '''
    Get the weights from the input to excitatory layer and reshape it to be two
    dimensional and square.
    '''
    # specify the desired shape of the reshaped input -> excitatory weights
    rearranged_weights = np.zeros((conv_features * conv_size, conv_size * n_e))

    # get the input -> excitatory synaptic weights
    connection = weight_matrix

    if sort_euclidean:
        # for each excitatory neuron in this convolution feature
        euclid_dists = np.zeros((n_e, conv_features))
        temps = np.zeros((n_e, conv_features, n_input))
        for n in xrange(n_e):
            # for each convolution feature
            for feature in xrange(conv_features):
                temp = connection[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)]
                if feature == 0:
                    if n == 0:
                        euclid_dists[n, feature] = 0.0
                    else:
                        euclid_dists[n, feature] = np.linalg.norm(
                            temps[0, 0, convolution_locations[n]] - temp[convolution_locations[n]])
                else:
                    euclid_dists[n, feature] = np.linalg.norm(
                        temps[n, 0, convolution_locations[n]] - temp[convolution_locations[n]])

                temps[n, feature, :] = temp.ravel()

            for idx, feature in enumerate(np.argsort(euclid_dists[n])):
                temp = temps[n, feature]
                rearranged_weights[idx * conv_size: (idx + 1) * conv_size, n * conv_size: (n + 1) * conv_size] = \
                    temp[convolution_locations[n]].reshape((conv_size, conv_size))

    else:
        for n in xrange(n_e):
            for feature in xrange(conv_features):
                temp = connection[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)]
                rearranged_weights[feature * conv_size: (feature + 1) * conv_size, n * conv_size: (n + 1) * conv_size] = \
                    temp[convolution_locations[n]].reshape((conv_size, conv_size))

    # return the rearranged weights to display to the user
    if n_e == 1:
        ceil_sqrt = int(math.ceil(math.sqrt(conv_features)))
        square_weights = np.zeros((28 * ceil_sqrt, 28 * ceil_sqrt))
        for n in xrange(conv_features):
            square_weights[(n // ceil_sqrt) * 28: ((n // ceil_sqrt) + 1) * 28,
            (n % ceil_sqrt) * 28: ((n % ceil_sqrt) + 1) * 28] = rearranged_weights[n * 28: (n + 1) * 28, :]

        return square_weights.T
    else:
        return rearranged_weights.T


def plot_2d_input_weights(weight_matrix):
    '''
    Plot the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_input_weights(weight_matrix)

    if n_e != 1:
        fig = plt.figure(1)
        ax1= plt.subplot(211)
        fig.clf()
        plt.figure(figsize=(18, 9))

    else:
        fig = plt.figure(1)  # , figsize=(9, 9))
        ax1 = plt.subplot(211)

    im = plt.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))

    if n_e != 1:
        plt.colorbar(im, fraction=0.016)
    else:
        plt.colorbar(im, fraction=0.06)

    plt.title('Reshaped input -> convolution weights')

    if n_e != 1:
        plt.xticks(xrange(conv_size, conv_size * (conv_features + 1), conv_size), xrange(1, conv_features + 1))
        plt.yticks(xrange(conv_size, conv_size * (n_e + 1), conv_size), xrange(1, n_e + 1))
        plt.xlabel('Convolution patch')
        plt.ylabel('Location in input (from top left to bottom right')

    fig.canvas.draw()
    return im, fig


def get_patch_weights():
    '''
    Get the weights from the input to excitatory layer and reshape them.
    '''
    rearranged_weights = np.zeros((conv_features * n_e, conv_features * n_e))
    connection = patch_weight_matrix

    for feature in xrange(conv_features):
        for other_feature in xrange(conv_features):
            if feature != other_feature:
                for this_n in xrange(n_e):
                    for other_n in xrange(n_e):
                        if is_lattice_connection(n_e_sqrt, this_n, other_n):
                            rearranged_weights[feature * n_e + this_n, other_feature * n_e + other_n] = connection[
                                feature * n_e + this_n, other_feature * n_e + other_n]

    return rearranged_weights


def plot_patch_weights():
    '''
    Plot the weights between convolution patches to view during training.
    '''
    weights = get_patch_weights()
    fig, ax = b.subplots(figsize=(8, 8))
    im = ax.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
    b.colorbar(im)
    b.title('Between-patch connectivity')
    fig.canvas.draw()
    return im, fig




# ax1 = None
# ax2 = None
def on_move(event):
    # get the x and y pixel coords
    x, y = event.x, event.y
    if not event.inaxes:
        return
    if event.inaxes:
        ax = event.inaxes  # the axes instance
        ax2 = ax
        print('data coords %f %f' % (event.xdata, event.ydata))
        txt = ax.get_children()[5]
        x, y = event.xdata, event.ydata
        # update the line positions
        print [x]
        # indx = np.searchsorted(ax.get_lines()[1].get_data()[0], [x])[0]
        # # print ax.get_lines()[0].get_data()
        # x =  ax.get_lines()[1].get_data()[0][indx]
        # y =  ax.get_lines()[1].get_data()[1][indx]

        lx = ax.get_lines()[3]
        ly = ax.get_lines()[4]
        lx.set_ydata(y)
        ly.set_xdata(x)
        txt.set_text('iteration=%1.2f, accuracy=%1.2f' % (x*100, y))
        # for l in ax.get_children():
        #     print ax.get_xlim()
        plt.draw()

def plot_performance(perf_file_name):
    # perf_file_name = sorted([file_name for file_name in os.listdir(performance_dir) if '.p' in file_name])[int(to_plot) - 1]

    # get pickled performances dictionary (voting mechanism, performance recordings over training)
    print "\nPlotting accuracy for the same experiment using file:" + perf_file_name
    _, performances = p.load(open(os.path.join(performance_dir, perf_file_name), 'rb'))

    print '\n'
    fig = plt.figure(1)
    ax = plt.subplot(212)
    performance_plots = []
    for voting_mechanism in sorted(performances.keys()):
        if voting_mechanism in ['all', 'most_spiked', 'top_percent', 'spatial_clusters']:
            performance_plots.append(plt.plot(performances[voting_mechanism], label=voting_mechanism)[0])
    print str(performances)
    print str(performances[performances.keys()[0]])
    plt.xlabel('Iteration number (1 through ' + str(len(performances[performances.keys()[0]]) * 100) + ')')
    print str(len(performances[performances.keys()[0]]))

    locs, labs = plt.xticks([x for x in xrange(0, len(performances[performances.keys()[0]]))],
                            [x * 100 for x in xrange(0, len(performances[performances.keys()[0]]))])
    print locs
    print labs
    # xtickslocs = (np.arange(0, len(performances[performances.keys()[0]])),
    #            np.arange(0, len(performances[performances.keys()[0]])*100, 100))
    #
    # ax.xaxis.set_major_locator(mticker.FixedLocator(xtickslocs))
    # ax.set_xlim(left=0, right=1000)

    # cursor = Cursor(ax)
    plt.connect('motion_notify_event', on_move)
    # sc = SomeClass(plt.subplot(211), plt.subplot(212))
    plt.connect('button_press_event', onclick)
    lx = ax.axhline(color='k')  # the horiz line
    ly = ax.axvline(color='k')
    txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)
    for l in ax.get_lines():
        print l

    print ax==plt.subplot(212)
    plt.ylabel('Classification accuracy (out of 100%)')

    title_strs = file_name[:file_name.index('weight') - 1].split('_')
    if 'no_weight_sharing' in file_name:
        weight_sharing = 'no_weight_sharing'
    else:
        weight_sharing = 'weight_sharing'

    plt.title(str(conv_size) + 'x' + str(conv_size) + ' convolutions, stride ' + str(conv_stride) + ', ' + str(
        conv_features) + \
              ' convolution patches, ' + ' '.join(weight_sharing.split('_')))

    # plt.tight_layout()

    # plt.savefig(os.path.join(performance_dir, 'performance_plots', file_name[:-2].replace('.', '_')))
    return fig

# --------------------------
print '\n'
print '\n'.join([str(idx) + ' | ' + file_name for idx, file_name in
                 enumerate([file_name for file_name in sorted(os.listdir(weight_dir)) if ('XeAe' in file_name) and file_name.split('npy')[0].split('_')[-1][0:-1] ==  file_name.split('npy')[0].split('_')[-2]])])
print '\n'

to_plot = raw_input('Enter the index of the file from above which you\'d like to plot: ')
if to_plot == '':
    file_name = [file_name for file_name in sorted(os.listdir(weight_dir)) if ('XeAe' in file_name) and file_name.split('npy')[0].split('_')[-1][0:-1] ==  file_name.split('npy')[0].split('_')[-2]][0]
else:
    file_name = [file_name for file_name in sorted(os.listdir(weight_dir)) if ('XeAe' in file_name) and file_name.split('npy')[0].split('_')[-1][0:-1] ==  file_name.split('npy')[0].split('_')[-2]][int(to_plot)]
print file_name
sort_euclidean = raw_input('Sort plot by Euclidean distance? (y / n, default no): ')
if sort_euclidean in ['', 'n']:
    sort_euclidean = False
elif sort_euclidean == 'y':
    sort_euclidean = True
else:
    raise Exception('Expecting one of "", "y", or "n".')

# number of inputs to the network
n_input = 784
n_input_sqrt = int(math.sqrt(n_input))

conv_size = int(file_name.split('_')[2])
conv_stride = int(file_name.split('_')[3])
conv_features = int(file_name.split('_')[4])
reduced_data = file_name.split('_')[13]

lattice_structure = file_name[-9:-8]
iteration = file_name[-1]#end

performance_file = '_'.join(file_name.split('_')[1:-1]) + '.p'
# number of excitatory neurons (number output from convolutional layer)
if conv_size == 28 and conv_stride == 0:
    n_e = n_e_sqrt = 1
    n_e_total = conv_features
else:
    n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
    n_e_total = n_e * conv_features
    n_e_sqrt = int(math.sqrt(n_e))

# number of inhibitory neurons (number of convolutational features (for now))
n_i = n_e

weight = {}

if conv_size == 28 and conv_stride == 0:
    weight['ee_input'] = n_e_total * 0.15
else:
    weight['ee_input'] = (conv_size ** 2) * 0.1625

conv_features_sqrt = int(math.ceil(math.sqrt(conv_features)))

print '\n'

# creating convolution locations inside the input image
convolution_locations = {}
for n in xrange(n_e):
    convolution_locations[n] = [
        ((n % n_e_sqrt) * conv_stride + (n // n_e_sqrt) * n_input_sqrt * conv_stride) + (x * n_input_sqrt) + y for y in
        xrange(conv_size) for x in xrange(conv_size)]

weight_matrix = np.load(os.path.join(weight_dir, file_name))


patch_weight_matrix = np.load(os.path.join(weight_dir, file_name.replace('XeAe', 'AeAe')))
patch_weight_matrix[patch_weight_matrix < np.percentile(patch_weight_matrix, 99.9)] = 0
patch_weight_matrix[np.nonzero(patch_weight_matrix)] = 1

fig_num += 1
plot_performance(performance_file)
fig_num += 1

input_weight_monitor, fig_weights = plot_2d_input_weights(weight_matrix)

plt.savefig(os.path.join(plots_dir, file_name[:-4] + '_patch_connectivity.png'))
plt.show()

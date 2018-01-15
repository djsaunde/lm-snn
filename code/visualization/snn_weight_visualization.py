import os
import sys
import math
import numpy as np
import cPickle as p
import matplotlib.cm as cmap
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

top_level_path = os.path.join('..', '..')
model_name = 'snn'
weights_dir = os.path.join(top_level_path, 'weights', model_name)
plots_dir = os.path.join(top_level_path, 'plots', model_name)

if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)

fig_num = 1

def get_2d_weights(weights):
    '''
    Get the weights from the input to excitatory layer and reshape it to be two
    dimensional and square.
    '''
    # specify the desired shape of the reshaped input -> excitatory weights
    rearranged_weights = np.zeros((conv_features * conv_size, conv_size * n_e))

    for n in xrange(n_e):
        for feature in xrange(conv_features):
            temp = weights[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)]
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


def plot_2d_weights(weights):
    '''
    Plot the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_weights(weights)
    fig = plt.figure(fig_num, figsize=(18, 18))
    im = plt.imshow(weights, interpolation='nearest', vmin=0, vmax=1, cmap=cmap.get_cmap('hot_r'))
    plt.colorbar(im)
    plt.title('Reshaped weights from input -> excitatory layer')
    fig.canvas.draw()
    plt.savefig(os.path.join(plots_dir, 'weights' + '_'.join(file_name.split('_')[1:])[:-4] + '.png'))
    plt.show()


print '\n'
print '\n'.join([ str(idx) + ' | ' + file_name for idx, file_name in \
    enumerate([ file_name for file_name in sorted(os.listdir(weights_dir))]) ])
print '\n'

to_plot = raw_input('Enter the index of the file from above which you\'d like to visualize: ')
if to_plot == '':
    file_name = [ file_name for file_name in sorted(os.listdir(weights_dir))][0]
else:
    file_name = [ file_name for file_name in sorted(os.listdir(weights_dir))][int(to_plot)]

conv_size = int(file_name.split('_')[1])
conv_stride = int(file_name.split('_')[2])
conv_features = int(file_name.split('_')[3])

n_input_sqrt = 28

if conv_size == 28 and conv_stride == 0:
    n_e = n_e_sqrt = 1
    n_e_total = conv_features
else:
    n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
    n_e_total = n_e * conv_features
    n_e_sqrt = int(math.sqrt(n_e))

# creating convolution locations inside the input image
convolution_locations = {}
for n in xrange(n_e):
    convolution_locations[n] = [ ((n % n_e_sqrt) * conv_stride + (n // n_e_sqrt) * n_input_sqrt * conv_stride) + (x * n_input_sqrt) + y for y in xrange(conv_size) for x in xrange(conv_size) ]


weights = np.load(os.path.join(weights_dir, file_name))
print weights

plot_2d_weights(weights)

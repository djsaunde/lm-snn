import numpy as np
import matplotlib.cm as cmap
import time, os.path, scipy, math, sys, timeit
import cPickle as p
import brian_no_units
import brian as b

from scipy.sparse import coo_matrix
from struct import unpack
from brian import *

weight_path = '../weights/conv_patch_connectivity_weights/'

fig_num = 0
wmax_ee = 1.0

def get_matrix_from_file(file_name, n_src, n_tgt):
    '''
    Given the name of a file pointing to a .npy ndarray object, load it into
    'weight_matrix' and return it
    '''

    # load the stored ndarray into 'readout', instantiate 'weight_matrix' as
    # correctly-shaped zeros matrix
    readout = np.load(file_name)
    weight_matrix = np.zeros((n_src, n_tgt))

    # read the 'readout' ndarray values into weight_matrix by (row, column) indices
    weight_matrix[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]

    # return the weight matrix read from file
    return weight_matrix


def get_2d_input_weights():
    '''
    Get the weights from the input to excitatory layer and reshape it to be two
    dimensional and square.
    '''
    rearranged_weights = np.zeros(( conv_features * conv_size, conv_size * n_e ))

    # counts number of input -> excitatory weights displayed so far
    connection = weight_matrix

    # for each convolution feature
    for feature in xrange(conv_features):
        # for each excitatory neuron in this convolution feature
        for n in xrange(n_e):
            # get the connection weights from the input to this neuron
            temp = connection[:, feature * n_e + n]
            # add it to the rearranged weights for displaying to the user
            rearranged_weights[feature * conv_size : (feature + 1) * conv_size, n * conv_size : (n + 1) * conv_size] = temp[convolution_locations[n]].reshape((conv_size, conv_size))

    # return the rearranged weights to display to the user
    return rearranged_weights.T


def plot_2d_input_weights():
    '''
    Plot the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_input_weights()
    fig = b.figure(fig_num, figsize=(18, 18))
    im2 = b.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    b.title('Convolutional Connection Weights')
    fig.canvas.draw()
    return im2, fig

# number of inputs to the network
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

# creating convolution locations inside the input image
convolution_locations = {}
for n in xrange(n_e):
    convolution_locations[n] = [ ((n % n_e_sqrt) * conv_stride + (n // n_e_sqrt) * n_input_sqrt * conv_stride) + (x * n_input_sqrt) + y for y in xrange(conv_size) for x in xrange(conv_size) ]

# determine STDP rule to use
stdp_input = ''

if raw_input('Use weight dependence (default no)?: ') in [ 'no', '' ]:
    use_weight_dependence = False
    stdp_input += 'weight_dependence_'
else:
    use_weight_dependence = True
    stdp_input += 'weight_dependence_'

if raw_input('Enter (yes / no) for post-pre (default yes): ') in [ 'yes', '' ]:
    post_pre = True
    stdp_input += 'postpre'
else:
    post_pre = False
    stdp_input += 'no_postpre'

# set ending of filename saves
ending = str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e)

weight_matrix = get_matrix_from_file(weight_path + 'XeAe_' + ending + '_' + stdp_input + '.npy', n_input, conv_features * n_e)

plot_2d_input_weights()

b.show()

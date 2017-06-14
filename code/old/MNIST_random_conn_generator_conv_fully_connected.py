'''
Much the same as 'MNIST_random_conn_generator.py', but with added logic for weights
for each of the convolution patches.

@author: Dan Saunders
'''

import scipy.ndimage as sp
import numpy as np
import pylab, math
        
    
def create_weights():
    '''
    Run from the main method. Creates the weights for all the network's synapses,
    for the original ETH model.
    '''
    
    # number of inputs
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

    # size of the fully-connected laye (used for "voting" / classification)
    full_size = raw_input('Enter number of neurons in the fully-connected layer (default 25): ')
    if full_size == '':
        full_size = 25
    else:
        full_size = int(full_size)

    # number of excitatory neurons (number output from convolutional layer)
    n_e_patch = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
    n_e_conv = n_e_patch * conv_features + full_size
    n_e_patch_sqrt = int(math.sqrt(n_e_patch))

    # number of inhibitory neurons in convolutional layer (number of convolutational patches (for now))
    n_i_conv = n_e_patch

    # number of excitatory, inhibitory neurons in fully-connected layer
    n_e_full = n_i_full = full_size

    # total number of excitatory, inhibitory neurons
    n_e_total = n_e_conv + n_e_full
    n_i_total = n_i_conv + n_i_full

    # file identifier (?)
    ending = str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e_patch) + '_' + str(n_e_full)
    
    # where to store the created weights
    data_path = '../random/conv_full_conn_random/'

    weight = {}
    
    # creating weights
    weight['XeCONV1e'] = 0.3
    weight['CONV1eFULL1e'] = 0.3
    weight['ei'] = 10.4
    weight['ie'] = 17.4
    
    # keeping track of indices of convolution windows within in the input space
    conv_indices = []
    for n in xrange(n_e_patch):
        conv_indices.append([ ((n % n_e_patch_sqrt) * conv_stride + (n // n_e_patch_sqrt) * n_input_sqrt * conv_stride) + (x * n_input_sqrt) + y for y in xrange(conv_size) for x in xrange(conv_size) ])

    print '\n'
    print '...creating random connection matrix from input -> convolutional excitatory layer'
    
    weight_matrix = (np.random.random((conv_size ** 2, conv_features * n_e_patch)) + 0.01) * weight['XeCONV1e']
    
    weight_list = []
    for feature in xrange(conv_features):
        for j in xrange(n_e_patch):
            weight_list.extend([ (i, feature * n_e_patch + j, weight_matrix[idx, feature * n_e_patch + j]) for idx, i in enumerate(conv_indices[j]) ])
    
    print '...saving connection matrix:', 'XeCONV1e_' + ending
    
    np.save(data_path + 'XeCONV1e_' + ending, weight_list)


    print '...creating random connection matrix from convolutional excitatory layer -> fully-connected excitatory layers'
    
    weight_matrix = (np.random.random((conv_features * n_e_patch, n_e_full)) + 0.01) * weight['CONV1eFULL1e']
    weight_list = [ (i, j, weight_matrix[i, j]) for j in xrange(n_e_full) for i in xrange(conv_features * n_e_patch) ]
    
    print '...saving connection matrix:', 'CONV1eFULL1e_' + ending
    
    np.save(data_path + 'CONV1eFULL1e_' + ending, weight_list)

    
    print '...creating connection matrix from convolutional excitatory layer -> convolutional inhibitory layer'
    
    weight_list = []
    for feature in xrange(conv_features):    
        weight_list.extend([ (feature * n_e_patch + i, feature, weight['ei']) for i in xrange(n_e_patch) ])
            
    print '...saving connection matrix:', 'CONV1eCONV1i_' + ending
    
    np.save(data_path + 'CONV1eCONV1i_' + ending, weight_list)


    print '...creating connection matrix from fully-connected excitatory layer -> fully_connected inhibitory layer'

    weight_list = [ (i, i, weight['ei']) for i in xrange(n_e_full) ]

    print '...saving connection matrix', 'FULL1eFULL1i_' + ending

    np.save(data_path + 'FULL1eFULL1i_' + ending, weight_list)
        
              
    print '...creating connection matrix from convolutional inhibitory layer -> convolutional excitatory layer'
    
    weight_list = []
    for feature in xrange(conv_features):
        weight_matrix = np.ones((1, conv_features * n_e_patch)) * weight['ie']
        weight_list.extend([ (feature, i, weight_matrix[0, i]) for i in xrange(conv_features * n_e_patch) ])

    print '...saving connection matrix:', 'CONV1iCONV1e_' + ending

    np.save(data_path + 'CONV1iCONV1e_' + ending, weight_list)


    print '...creating connection matrix from fully-connected inhibitory layer -> fully-connected excitatory layer'
    
    weight_list = [ (i, j, weight['ie']) for j in xrange(n_e_full) for i in xrange(n_e_full) if i != j ]

    print '...saving connection matrix:', 'FULL1iFULL1e_' + ending

    np.save(data_path + 'FULL1iFULL1e_' + ending, weight_list)

    
    print '\n'


if __name__ == '__main__':
    create_weights()
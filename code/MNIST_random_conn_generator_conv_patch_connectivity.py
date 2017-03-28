'''
Much the same as 'MNIST_random_conn_generator.py', but with added logic for weights
for each of the convolution patches.

@author: Dan Saunders
'''

import scipy.ndimage as sp
import numpy as np
import pylab, math, random


def is_lattice_connection(sqrt, i, j):
    '''
    Boolean method which checks if two indices in a network correspond to neighboring nodes in a lattice.

    sqrt: square root of the number of nodes in population
    i: First neuron's index
    k: Second neuron's index
    '''
    return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j
        
    
def create_weights():
    '''
    Run from the main method. Creates the weights for all the network's synapses,
    for the original ETH model.
    '''
    
    # number of inputs
    n_input = 784
    n_input_sqrt = int(math.sqrt(n_input))

    print '\n'
    
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
    n_e_sqrt = int(math.sqrt(n_e))

    # number of inhibitory neurons (number of convolutational features (for now))
    n_i = n_e
    
    ending = '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e)
    
    # where to store the created weights
    data_path = '../random/conv_patch_connectivity_random/'
    
    # creating weights
    weight = {}
    weight['ee_input'] = 0.3
    weight['ee_patch'] = 0.3
    weight['ei'] = 10.4
    weight['ie'] = 17.4
    
    print '\n'
    print '...creating random connection matrix from input -> excitatory layer'
    
    conv_indices = []
    for n in xrange(n_e):
        conv_indices.append([ ((n % n_e_sqrt) * conv_stride + (n // n_e_sqrt) * n_input_sqrt * conv_stride) + (x * n_input_sqrt) + y for y in xrange(conv_size) for x in xrange(conv_size) ])
    
    weight_matrix = (np.random.random((conv_size ** 2, conv_features * n_e)) + 0.01) * weight['ee_input']
    
    weight_list = []
    for feature in xrange(conv_features):
        for j in xrange(n_e):
            weight_list.extend([ (i, feature * n_e + j, weight_matrix[idx, feature * n_e + j]) for idx, i in enumerate(conv_indices[j]) ])
    
    print '...saving connection matrix:', 'XeAe' + ending
    
    np.save(data_path + 'XeAe' + ending, weight_list)
    
    
    print '...creating connection matrix from excitatory layer -> inbitory layer'
    
    weight_list = []
    for feature in xrange(conv_features):    
        weight_list.extend([ (feature * n_e + i, feature, weight['ei']) for i in xrange(n_e) ])
            
    print '...saving connection matrix:', 'AeAi' + ending
    
    np.save(data_path + 'AeAi' + ending, weight_list)
        
              
    print '...creating connection matrix from inhibitory layer -> excitatory layer'
    
    weight_list = []
    for feature in xrange(conv_features):
        weight_matrix = np.ones((conv_features * n_e, conv_features * n_e)) * weight['ie']
        weight_list.extend([ (i, j, weight_matrix[i, j]) for i in xrange(conv_features * n_e) for j in xrange(conv_features * n_e) ])
    
    print '...saving connection matrix:', 'AiAe' + ending

    np.save(data_path + 'AiAe' + ending, weight_list)


    print '...creating connection matrix between patches'

    weight_list = []
    for this_feature in xrange(conv_features):
        for other_feature in xrange(conv_features):
            for n_this in xrange(n_e):
                for n_other in xrange(n_e):
                    weight_list.append((this_feature * n_e + n_this, other_feature * n_e + n_other, random.random() * weight['ee_patch'] + 0.01))

    print '...saving connection matrix:', 'AeAe' + ending

    np.save(data_path + 'AeAe' + ending, weight_list)

    print '\n'


if __name__ == '__main__':
    create_weights()
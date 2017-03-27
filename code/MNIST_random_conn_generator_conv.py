'''
Much the same as 'MNIST_random_conn_generator.py', but with added logic for weights
for each of the convolution patches.

@author: Dan Saunders
'''

import scipy.ndimage as sp
import numpy as np
import pylab, math

n_input = 784
n_input_sqrt = int(math.sqrt(n_input))
        
    
def create_weights():
    '''
    Run from the main method. Creates the weights for all the network's synapses,
    for the original ETH model.
    '''
    
    # size of convolution windows
    conv_size = raw_input('Enter size of square side length of convolution window (default 20): ')
    if conv_size == '':
        conv_size = 20
    else:
        conv_size = int(conv_size)

    # stride of convolution windows
    conv_stride = raw_input('Enter stride size of convolution window (default 8): ')
    if conv_stride == '':
        conv_stride = 8
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
    dataPath = '../random/conv_random/'
    
    # creating weights
    weight = {}
    weight['ee_input'] = 0.30
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
    
    np.save(dataPath + 'XeAe' + ending, weight_list)
    
    
    print '...creating connection matrix from excitatory layer -> inbitory layer'
    
    weight_list = []
    for feature in xrange(conv_features):    
        weight_list.extend([ (feature * n_e + i, feature, weight['ei']) for i in xrange(n_e) ])
            
    print '...saving connection matrix:', 'AeAi' + ending
    
    np.save(dataPath + 'AeAi' + ending, weight_list)
        
              
    print '...creating connection matrix from inhibitory layer -> excitatory layer'
    
    weight_list = []
    for feature in xrange(conv_features):
        weight_matrix = np.ones((1, conv_features * n_e)) * weight['ie']
        weight_list.extend([ (feature, i, weight_matrix[0, i]) for i in xrange(conv_features * n_e) ])
    
        # weight_matrix = np.ones((conv_features, conv_features * n_e)) * weight['ie']
        # weight_list.extend([ (feature * n_e + j, feature * n_e + i, weight_matrix[0, i]) for i in xrange(conv_features * n_e) for j in xrange(conv_features) ])

    print '...saving connection matrix:', 'AiAe' + ending

    np.save(dataPath + 'AiAe' + ending, weight_list)
    
    print '\n'
         
if __name__ == '__main__':
    create_weights()
    











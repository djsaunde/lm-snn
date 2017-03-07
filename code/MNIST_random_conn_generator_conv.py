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
    conv_features = raw_input('Enter number of convolution features to learn (default 9): ')
    if conv_features == '':
        conv_features = 9
    else:
        conv_features = int(conv_features)

    # number of excitatory neurons (number output from convolutional layer)
    n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
    n_e_sqrt = int(math.sqrt(n_e))

    # number of inhibitory neurons (number of convolutational features (for now))
    n_i = conv_features
    
    ending = '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e)
    
    # where to store the created weights
    dataPath = '../random/'
    
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
        
    print len(conv_indices[0])
    
    connNameList = [ 'XeA' + str(i) + 'e' for i in xrange(conv_features) ]
    for name in connNameList:
        weight_matrix = (np.random.random((conv_size ** 2, 1)) + 0.01) * weight['ee_input']
        
        weight_list = []
        for j in xrange(n_e):
            weight_list.extend([ (i, j, weight_matrix[idx, 0]) for idx, i in enumerate(conv_indices[j]) ])
        
        print '...saving connection matrix:', name + ending
        
        np.save(dataPath + name + ending, weight_list)
    
    
    print '...creating connection matrix from excitatory layer -> inbitory layer'
    
    connNameList = [ 'A' + str(i) + 'eA' + str(i) + 'i' for i in xrange(conv_features) ]
    for name in connNameList:
        weight_list = [(i, 0, weight['ei']) for i in xrange(n_e)]
        
        print '...saving connection matrix:', name + ending
        
        np.save(dataPath + name + ending, weight_list)
        
              
    print '...creating connection matrix from inhbitory layer -> excitatory layer'
    
    connNameList = [ 'A' + str(i) + 'iA' + str(j) + 'e' for i in xrange(conv_features) for j in xrange(conv_features) if i != j ]
    for name in connNameList:
        weight_matrix = np.ones((1, n_e)) * weight['ie']
        weight_list = [(0, j, weight_matrix[0, j]) for j in xrange(n_e)]
        
        print '...saving connection matrix:', name + ending

        np.save(dataPath + name + ending, weight_list)
    
    print '\n'
         
if __name__ == "__main__":
    create_weights()
    











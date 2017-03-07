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
    sqrt = int(math.sqrt(n_input))
    
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
    conv_features = raw_input('Enter number of convolution features to learn (default 9 ): ')
    if conv_features == '':
        conv_features = 9
    else:
        conv_features = int(conv_features)

    # number of excitatory neurons (number output from convolutional layer)
    n_e = ((sqrt - conv_size) / conv_stride + 1) ** 2

    # number of inhibitory neurons (number of convolutational features (for now))
    n_i = conv_features
    
    # where to store the created weights
    dataPath = './random/'
    
    # creating weights
    weight = {}
    weight['ee_input'] = 0.25
    weight['ei'] = 10.4
    weight['ie'] = 17.4
    
    
    print '...creating random connection matrix from input -> excitatory layer'
    
    connNameList = [ 'XeA' + str(i) + 'e' for i in xrange(conv_features) ]
    for name in connNameList:
        weight_matrix = (np.random.random((conv_size ** 2, 1)) + 0.01) * weight['ee_input']
        weight_list = [(i, 0, weight_matrix[i, 0]) for i in xrange(conv_size ** 2)]
        
        print '...saving connection matrix:', name + '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e)
        
        np.save(dataPath + name + '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e), weight_list)
    
    
    print '...creating connection matrix from excitatory layer -> inbitory layer'
    
    connNameList = [ 'A' + str(i) + 'eA' + str(i) + 'i' for i in xrange(conv_features) ]
    for name in connNameList:
        weight_list = [(i, 0, weight['ei']) for i in xrange(n_e)]
        
        print '...saving connection matrix:', name + '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e)
        
        np.save(dataPath + name + '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e), weight_list)
        
              
    print '...creating connection matrix from inhbitory layer -> excitatory layer'
    
    connNameList = [ 'A' + str(i) + 'iA' + str(j) + 'e' for i in xrange(conv_features) for j in xrange(conv_features) if i != j ]
    for name in connNameList:
        weight_matrix = np.ones((1, n_e)) * weight['ie']
        weight_list = [(0, j, weight_matrix[0, j]) for j in xrange(n_e)]
        
        print '...saving connection matrix:', name + '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e)

        np.save(dataPath + name + '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(n_e), weight_list)
    
         
if __name__ == "__main__":
    create_weights()
    











'''
Updated for use in the BINDS lab, spring 2017.

@author: Dan Saunders
'''

import scipy.ndimage as sp
import numpy as np
import pylab
        
    
def create_weights():
    '''
    Run from the main method. Creates the weights for all the n_etwork's synapses,
    for the original ETH model.
    '''
    
    # number of inputs and exc / inhib neurons
    n_input = 784
    n_e = input('Enter number of excitatory / inhibitory neurons: ')
    n_i = n_e
    
    # where to store the created weights
    data_path = '../random/eth_model_random/'
    
    # creating weights
    weight = {}
    weight['ee_input'] = 0.3 
    weight['ei'] = 10.4
    weight['ie'] = 17.0
    
    print '\n'
    
    print '...creating random connection matrix from input -> excitatory layer'
    
    conn_name_list = ['XeAe']
    for name in conn_name_list:
        weight_matrix = (np.random.random((n_input, n_e)) + 0.01) * weight['ee_input']
        weight_list = [(i, j, weight_matrix[i,j]) for j in xrange(n_e) for i in xrange(n_input)]
        
        print '...saving connection matrix:', name + str(n_e)
        
        np.save(data_path + name + str(n_e), weight_list)
    
    
    print '...creating connection matrix from excitatory layer -> inbitory layer'
    
    conn_name_list = ['AeAi']
    for name in conn_name_list:
        weight_list = [(i, i, weight['ei']) for i in xrange(n_e)]
        
        print '...saving connection matrix:', name + str(n_e)
        
        np.save(data_path + name + str(n_e), weight_list)
        
              
    print '...creating connection matrix from inhbitory layer -> excitatory layer'
    
    conn_name_list = ['AiAe']
    for name in conn_name_list:
        weight_matrix = np.ones((n_i, n_e)) * weight['ie']
        np.fill_diagonal(weight_matrix, 0)
        weight_list = [(i, j, weight_matrix[i,j]) for i in xrange(n_i) for j in xrange(n_e)]
        
        print '...saving connection matrix:', name + str(n_i)

        np.save(data_path + name + str(n_e), weight_list)

    print '\n'
    
         
if __name__ == "__main__":
    create_weights()
    











'''
Updated for use in the BINDS lab, spring 2017.

@author: Dan Saunders
'''

import scipy.ndimage as sp
import numpy as np
import pylab
        
    
def create_weights():
    '''
    Run from the main method. Creates the weights for all the network's synapses,
    for the original ETH model.
    '''
    
    # number of inputs and exc / inhib neurons
    nInput = 784
    nE = input('Enter number of excitatory / inhibitory neurons: ')
    nI = nE
    
    # where to store the created weights
    dataPath = './random/'
    
    # creating weights
    weight = {}
    weight['ee_input'] = 0.3 
    weight['ei'] = 10.4
    weight['ie'] = 17.0
    
    
    print '...creating random connection matrix from input -> excitatory layer'
    
    connNameList = ['XeAe']
    for name in connNameList:
        weight_matrix = (np.random.random((nInput, nE)) + 0.01) * weight['ee_input']
        weight_list = [(i, j, weight_matrix[i,j]) for j in xrange(nE) for i in xrange(nInput)]
        
        print '...saving connection matrix:', name + str(nE)
        
        np.save(dataPath + name + str(nE), weight_list)
    
    
    print '...creating connection matrix from excitatory layer -> inbitory layer'
    
    connNameList = ['AeAi']
    for name in connNameList:
        weight_list = [(i, i, weight['ei']) for i in xrange(nE)]
        
        print '...saving connection matrix:', name + str(nE)
        
        np.save(dataPath + name + str(nE), weight_list)
        
              
    print '...creating connection matrix from inhbitory layer -> excitatory layer'
    
    connNameList = ['AiAe']
    for name in connNameList:
        weight_matrix = np.ones((nI, nE)) * weight['ie']
        np.fill_diagonal(weight_matrix, 0)
        weight_list = [(i, j, weight_matrix[i,j]) for i in xrange(nI) for j in xrange(nE)]
        
        print '...saving connection matrix:', name + str(nI)

        np.save(dataPath + name + str(nE), weight_list)
    
         
if __name__ == "__main__":
    create_weights()
    











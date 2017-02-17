'''
Creating one-to-one connections from input to excitatory layer. Allowing lattice
connections for 

@author: Dan Saunders (djsaunde.github.io)
'''

import scipy.ndimage as sp
import numpy as np
import pylab, math


def randomDelay(minDelay, maxDelay):
    return np.random.rand()*(maxDelay-minDelay) + minDelay
        
        
def computePopVector(popArray):
    size = len(popArray)
    complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in xrange(size)])
    cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
    return cur_pos

        
def sparsenMatrix(baseMatrix, pConn):
    weightMatrix = np.zeros(baseMatrix.shape)
    numWeights = 0
    numTargetWeights = baseMatrix.shape[0] * baseMatrix.shape[1] * pConn
    print numTargetWeights, pConn
    weightList = [0] * int(numTargetWeights)
    while numWeights < numTargetWeights:
        idx = (np.int32(np.random.rand()*baseMatrix.shape[0]), np.int32(np.random.rand()*baseMatrix.shape[1]))
        if not (weightMatrix[idx]):
            weightMatrix[idx] = baseMatrix[idx]
            weightList[numWeights] = (idx[0], idx[1], baseMatrix[idx])
            numWeights += 1
    return weightMatrix, weightList


def is_lattice_connection(n, i, k):
    """
    Boolean method which checks if two indices in a network correspond to neighboring neurons in a lattice.
    
    Args:
        n: number of neurons in lattice
        i: First neuron's index
        k: Second neuron's index
    """
    sqrt = math.sqrt(n)
    return i + 1 == k and k % sqrt != 0 or i - 1 == k and i % sqrt != 0 or i + sqrt == k or i - sqrt == k
        
    
def create_weights():
    '''
    Creates the weight matrices for all specified layer-to-layer interactions.
    '''
    
    n_input = 784
    n_e = input('Enter number of excitatory / inhibitory neurons: ')
    n_i = n_e 
    dataPath = './random/'
    weight = {}
    weight['ee_input'] = 0.3 
    weight['ei_input'] = 0.2 
    weight['ee'] = 0.001
    weight['ei'] = 10.4
    weight['ie'] = 17.0
    weight['ii'] = 0.4
    pConn = {}
    pConn['ee_input'] = 1.0 
    pConn['ei_input'] = 0.1
    pConn['ee'] = 1.0
    pConn['ei'] = 1.0
    pConn['ie'] = 1.0
    pConn['ii'] = 0.1
    
    
    print '...creating random connection matrices from input to E'
    connNameList = ['XeAe' + str(n_e) + 'onetoone']
    for name in connNameList:
        weightMatrix = np.random.random((n_input, n_e)) + 0.01
        weightMatrix *= weight['ee_input']
        if pConn['ee_input'] < 1.0:
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ee_input'])
        else:
            # creating one to one connections
            weightList = [(i, j, weightMatrix[i,j]) if i == j else (i, j, 0.0) for j in xrange(n_e) for i in xrange(n_input)]
        print 'save connection matrix', name
        np.save(dataPath + name, weightList)
        
    
    print '...creating connection matrices from E to I'
    connNameList = ['AeAi' + str(n_e) + 'onetoone']
    for name in connNameList:
        if n_e == n_i:
            weightList = [(i, i, weight['ei']) for i in xrange(n_e)]
        else:
            weightMatrix = np.random.random((n_e, n_i))
            weightMatrix *= weight['ei']
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ei'])
        print 'save connection matrix', name
        np.save(dataPath + name, weightList)
        
            
    print '...creating connection matrices from I to E'
    connNameList = ['AiAe' + str(n_e) + 'onetoone']
    for name in connNameList:
        if n_e == n_i:
            weightMatrix = np.ones((n_i, n_e))
            weightMatrix *= weight['ie']
            for i in xrange(n_i):
                weightMatrix[i,i] = 0
            weightList = [(i, j, weightMatrix[i, j]) for i in xrange(n_i) for j in xrange(n_e)]
        else:
            weightMatrix = np.random.random((n_i, n_e))
            weightMatrix *= weight['ie']
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ie'])
        print 'save connection matrix', name
        np.save(dataPath + name, weightList)
    
    
    print '...creating connection matrices from E to E'
    connNameList = ['AeAe' + str(n_e) + 'onetoone']
    if raw_input('Enter (none / lattice / random) for connectivity pattern in excitatory -> excitatory layer: ') == 'lattice':
        # add lattice connectivity in excitatory -> excitatory layer
        for name in connNameList:
            weightMatrix = np.array( [ 0.0 if not is_lattice_connection(n_e, i, j) else 1.0 for i in xrange(n_e) for j in xrange(n_e) ] )
            weightMatrix = weightMatrix.reshape((n_e, n_e))
            weightMatrix *= weight['ee']
                
            weightList = [(i, j, weightMatrix[i,j]) for i in xrange(n_e) for j in xrange(n_e)]
            print 'save connection matrix', name
            np.save(dataPath + name, weightList)
    
         
if __name__ == "__main__":
    create_weights()
    











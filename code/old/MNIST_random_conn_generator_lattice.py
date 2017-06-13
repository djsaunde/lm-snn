'''
Script to generator random connectivity to use in 'spiking_lattice_MNIST.py'.

@author: Dan Saunders
'''

import scipy.ndimage as sp
import numpy as np
import pylab, math

from scipy.sparse import csr_matrix


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
    weightList = [0]*int(numTargetWeights)
    while numWeights < numTargetWeights:
        idx = (np.int32(np.random.rand()*baseMatrix.shape[0]), np.int32(np.random.rand()*baseMatrix.shape[1]))
        if not (weightMatrix[idx]):
            weightMatrix[idx] = baseMatrix[idx]
            weightList[numWeights] = (idx[0], idx[1], baseMatrix[idx])
            numWeights += 1
    return weightMatrix, weightList


def is_lattice_connection(n, i, j):
        '''
        Boolean method which checks if two indices in a network correspond to neighboring neurons in a lattice.
        
        Args:
            n: number of neurons in lattice
            i: First neuron's index
            k: Second neuron's index
        '''
        sqrt = math.sqrt(n)
        return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j
        
    
def create_weights():
    '''
    Creates the weight matrices for all specified layer-to-layer interactions.
    '''
    
    nInput = 784
    nE = input('Enter number of excitatory / inhibitory neurons: ')
    nI = nE 
    dataPath = './random/'
    weight = {}
    weight['ee_input'] = 0.3 
    weight['ei_input'] = 0.2 
    weight['ee'] = 0.2
    weight['ei'] = 10.4
    weight['ie'] = 17.0
    weight['ii'] = 0.4
    pConn = {}
    pConn['ee_input'] = 1.0 
    pConn['ei_input'] = 0.1 
    pConn['ee'] = 1.0
    pConn['ei'] = 0.0025
    pConn['ie'] = 0.9
    pConn['ii'] = 0.1
    
    
    print '...creating random connection matrices from input to E'
    connNameList = ['XeAe']
    for name in connNameList:
        weightMatrix = np.random.random((nInput, nE)) + 0.01
        weightMatrix *= weight['ee_input']
        if pConn['ee_input'] < 1.0:
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ee_input'])
        else:
            weightList = [(i, j, weightMatrix[i,j]) for j in xrange(nE) for i in xrange(nInput)]
        print 'save connection matrix', name
        np.save(dataPath + name + str(nE), weightList)
        np.savetxt(dataPath + name + str(nE), weightList)
    
    
    print '...creating connection matrices from E to I'
    connNameList = ['AeAi']
    for name in connNameList:
        if nE == nI:
            weightList = [(i, i, weight['ei']) for i in xrange(nE)]
        else:
            weightMatrix = np.random.random((nE, nI))
            weightMatrix *= weight['ei']
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ei'])
        print 'save connection matrix', name
        np.save(dataPath + name + str(nE), weightList)
        np.savetxt(dataPath + name + str(nE), weightList)
    
        
    print '...creating connection matrices from I to E'
    connNameList = ['AiAe']
    for name in connNameList:
        if nE == nI:
            weightMatrix = np.ones((nI, nE))
            weightMatrix *= weight['ie']
            for i in xrange(nI):
                weightMatrix[i,i] = 0
            weightList = [(i, j, weightMatrix[i,j]) for i in xrange(nI) for j in xrange(nE)]
        else:
            weightMatrix = np.random.random((nI, nE))
            weightMatrix *= weight['ie']
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ie'])
        print 'save connection matrix', name
        np.save(dataPath + name + str(nE), weightList)
        np.savetxt(dataPath + name + str(nE), weightList)
    
    
    print '...creating connection matrices from E to E'
    connNameList = ['AeAe']
    for name in connNameList:
        weightMatrix = np.array( [ 0 if not is_lattice_connection(nE, i, j) else weight['ee'] + np.random.random() * 0.2 for i in xrange(nE) for j in xrange(nE) ] )
        weightMatrix = weightMatrix.reshape((nE, nE))
            
        weightList = [(i, j, weightMatrix[i,j]) for i in xrange(nE) for j in xrange(nE)]
        print 'save connection matrix', name
        np.save(dataPath + name + str(nE), weightList)
        np.savetxt(dataPath + name + str(nE), weightList)
    
         
if __name__ == "__main__":
    create_weights()
    











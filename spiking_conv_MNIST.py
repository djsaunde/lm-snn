'''
Much the same as 'spiking_MNIST.py', but we instead use a number of convolutional
windows to map the input to a reduced space.
'''

import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy
import math
import cPickle as pickle
import brian_no_units
import brian as b
import cPickle as p
import sys
from struct import unpack
from brian import *

# specify the location of the MNIST data
MNIST_data_path = './'



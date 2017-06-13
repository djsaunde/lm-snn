'''
Helper functions for the stdp-mnist project.

@author: Dan Saunders (djsaunde.github.io)
'''

from __main__ import *

data_path = '../'


def get_labeled_data(pname, b_train=True):
    '''
    Read input-vector (image) and target class (label, 0-9) and return it as 
    a list of tuples.

    pname: name of file to write pickled Python object to.
    b_train: whether to load the training or test dataset.
    '''
    if os.path.isfile('%s.pickle' % pname):
        data = p.load(open('%s.pickle' % pname))
    else:
        # Open the images with gzip in read binary mode
        if b_train:
            images = open('../data/train-images-idx3-ubyte', 'rb')
            labels = open('../data/train-labels-idx1-ubyte', 'rb')
        else:
            images = open('../data/t10k-images-idx3-ubyte', 'rb')
            labels = open('../data/t10k-labels-idx1-ubyte', 'rb')

        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]

        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')

        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)
        y = np.zeros((N, 1), dtype=np.uint8)

        for i in xrange(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in xrange(cols)]  for unused_row in xrange(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        p.dump(data, open("%s.pickle" % pname, "wb"))

    return data


def is_lattice_connection(sqrt, i, j):
    '''
    Boolean method which checks if two indices in a neuron population correspond to neighboring nodes in a lattice.

    sqrt: square root of the number of nodes in population
    i: First neuron's index
    k: Second neuron's index
    '''
    return i + 1 == j and j % sqrt != 0 or i - 1 == j and i % sqrt != 0 or i + sqrt == j or i - sqrt == j


def get_matrix_from_file(filename, n_src, n_tgt):
    '''
    Given the name of a file pointing to a .npy ndarray object, load it into
    'weight_matrix' and return it

    filename: name of file from which to load matrix of parameters.
    n_src: the number of neurons from which the synapses originate.
    n_tgt: the number of neuruons to which the synapses connect.
    '''
    # load the stored ndarray into 'readout', instantiate 'weight_matrix' as 
    # correctly-shaped zeros matrix
    readout = np.load(filename)
    weight_matrix = np.zeros((n_src, n_tgt))

    # read the 'readout' ndarray values into weight_matrix by (row, column) indices
    weight_matrix[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]

    # return the weight matrix read from file
    return weight_matrix


def save_connection(directory, filename, connection):
    '''
    Save connection parameters to file.

    directory: the directory to which to write the connection weights.
    filename: the file in the above directory to which to write the connection weights.
    connection: the BRIAN Connection object whose weights we wish to save.
    '''
    # print out saved connections
    print '...saving connections: weights/' + directory + filename
    
    # get matrix of connection weights
    conn_matrix = connection[:]
    # sparsify it into (row, column, entry) tuples
    conn_list_sparse = ([(i, j, conn_matrix[i, j]) for i in xrange(conn_matrix.shape[0]) for j in xrange(conn_matrix.shape[1]) ])
    # save it out to disk
    np.save(data_path + 'weights/' + directory + filename, conn_list_sparse)


def save_theta(directory, filename, population):
    '''
    Save the adaptive threshold parameters to a file.

    directory: the directory to which to write the theta parameters.
    filename: the file in the above directory to which to write the theta parameters.
    population: the BRIAN NeuronGroup object whose theta parameters we wish to save.
    '''
	# print out saved theta populations
    print '...saving theta: weights/' + directory + filename
    # save out the theta parameters to file
    np.save(data_path + 'weights/' + directory + filename, population.theta)


def set_weights_most_fired(connection):
    '''
    For each convolutional patch, set the weights to those of the neuron which
    fired the most in the last iteration.

    connection: the BRIAN Connection object whose weights we wish to set the most-spiked
    weights to.
    '''
    for feature in xrange(conv_features):
        # count up the spikes for the neurons in this convolution patch
        column_sums = np.sum(current_spike_count[feature : feature + 1, :], axis=0)

        # find the excitatory neuron which spiked the most
        most_spiked = np.argmax(column_sums)

        # create a "dense" version of the most spiked excitatory neuron's weight
        most_spiked_dense = connection[:, feature * n_e + most_spiked].todense()

        # set all other neurons' (in the same convolution patch) weights the same as the most-spiked neuron in the patch
        for n in xrange(n_e):
            if n != most_spiked:
                other_dense = connection[:, feature * n_e + n].todense()
                other_dense[convolution_locations[n]] = most_spiked_dense[convolution_locations[most_spiked]]
                connection[:, feature * n_e + n] = other_dense


def normalize_conv_weights(connection):
    '''
    Squash the input -> excitatory weights to sum to a specified number.
    '''
    for feature in xrange(conv_features):
        feature_connection = connection[:, feature * n_e : (feature + 1) * n_e]
        column_sums = np.sum(feature_connection, axis=0)
        column_factors = weight['ee_input'] / column_sums

        for n in xrange(n_e):
            dense_weights = connections[conn_name][:, feature * n_e + n].todense()
            dense_weights[convolution_locations[n]] *= column_factors[n]
            connections[conn_name][:, feature * n_e + n] = dense_weights


def normalize_conv_lattice_weights(connection):
    '''
    Squash the between-patch convolution weights to sum to a specified number.
	'''
    for feature in xrange(conv_features):
        feature_connection = connection[feature * n_e : (feature + 1) * n_e, :]
        column_sums = np.sum(feature_connection)
        column_factors = weight['ee_lattice'] / column_sums

        for idx in xrange(feature * n_e, (feature + 1) * n_e):
            connections[conn_name][idx, :] *= column_factors


def plot_input():
    '''
    Plot the current input example during the training procedure.
    '''
    fig = b.figure(fig_num, figsize = (5, 5))
    im = b.imshow(rates.reshape((28, 28)), interpolation = 'nearest', vmin=0, vmax=64, cmap=cmap.get_cmap('gray'))
    b.colorbar(im)
    b.title('Current input example')
    fig.canvas.draw()
    return im, fig


def update_input(im, fig):
    '''
    Update the input image to use for input plotting.
    '''
    im.set_array(rates.reshape((28, 28)))
    fig.canvas.draw()
    return im


def get_input_conv_weights(connection):
    '''
    Get the weights from the input to excitatory layer and reshape it to be two
    dimensional and square.
    '''
    rearranged_weights = np.zeros((conv_features * conv_size, conv_size * n_e))

    # for each convolution feature
    for feature in xrange(conv_features):
        # for each excitatory neuron in this convolution feature
        for n in xrange(n_e):
            # get the connection weights from the input to this neuron
            temp = connection[:, feature * n_e + n].todense()
            # add it to the rearranged weights for displaying to the user
            rearranged_weights[feature * conv_size : (feature + 1) * conv_size, n * conv_size : (n + 1) * conv_size] = temp[convolution_locations[n]].reshape((conv_size, conv_size))

    # return the rearranged weights to display to the user
    return rearranged_weights.T


def plot_input_conv_weights(connection):
    '''
    Plot the weights from input to excitatory layer to view during training.
    '''
    global conv_features, conv_size, n_e, fig_num, wmax_ee

    weights = get_input_conv_weights(connection, conv_features, conv_size, n_e)
    fig = b.figure(fig_num, figsize=(18, 18))
    im = b.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
    b.colorbar(im)
    b.title('Reshaped input -> convolution weights')
    b.xlabel('convolution patch')
    b.ylabel('neurons per patch')
    fig.canvas.draw()
    return im, fig


def update_2d_input_weights(im, fig):
    '''
    Update the plot of the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im


def get_patch_weights():
    '''
    Get the weights from the input to excitatory layer and reshape them.
    '''
    rearranged_weights = np.zeros((conv_features * n_e, conv_features * n_e))
    connection = connections['AeAe_' + ending][:]

    for feature in xrange(conv_features):
        for other_feature in xrange(conv_features):
            if feature != other_feature:
                for this_n in xrange(n_e):
                    for other_n in xrange(n_e):
                        if is_lattice_connection(n_e_sqrt, this_n, other_n):
                            rearranged_weights[feature * n_e + this_n, other_feature * n_e + other_n] = connection[feature * n_e + this_n, other_feature * n_e + other_n]

    return rearranged_weights


def plot_patch_weights():
    '''
    Plot the weights between convolution patches to view during training.
    '''
    weights = get_patch_weights()
    fig = b.figure(fig_num, figsize=(8,8))
    im = b.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
    b.colorbar(im)
    b.title('Between patch connectivity')
    fig.canvas.draw()
    return im, fig

def update_patch_weights(im, fig):
    '''
    Update the plot of the weights between convolution patches to view during training.
    '''
    weights = get_patch_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im
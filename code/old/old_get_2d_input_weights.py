def get_2d_input_weights():
		'''
  		Get the weights from the input to excitatory layer and reshape it to be two
    	dimensional and square.
    	'''
		# rearranged_weights = np.zeros((conv_features_sqrt * conv_size * n_e_sqrt, conv_features_sqrt * conv_size * n_e_sqrt))
        # connection = input_connections['XeAe'][:]

        # # for each convolution feature
        # for feature in xrange(conv_features):
        #       # for each excitatory neuron in this convolution feature
        #       for n in xrange(n_e):
        #               temp = connection[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)].todense()

        #               # print ((feature // conv_features_sqrt) * conv_size * n_e_sqrt) + ((n // n_e_sqrt) * conv_size), ((feature // conv_features_sqrt) * conv_size * n_e_sqrt) + ((n // n_e_sqrt) * conv_size) + conv_size, ((feature % conv_features_sqrt) * conv_size * n_e_sqrt) + ((n % n_e_sqrt) * (conv_size)), ((feature % conv_features_sqrt) * conv_size * n_e_sqrt) + ((n % n_e_sqrt) * (conv_size)) + conv_size

        #               rearranged_weights[ ((feature % conv_features_sqrt) * conv_size * n_e_sqrt) + ((n % n_e_sqrt) * conv_size) : ((feature % conv_features_sqrt) * conv_size * n_e_sqrt) + ((n % n_e_sqrt) * conv_size) + conv_size, ((feature // conv_features_sqrt) * conv_size * n_e_sqrt) + ((n // n_e_sqrt) * (conv_size)) : ((feature // conv_features_sqrt) * conv_size * n_e_sqrt) + ((n // n_e_sqrt) * (conv_size)) + conv_size ] = temp[convolution_locations[n]].reshape((conv_size, conv_size))

        # # return the rearranged weights to display to the user
        # return rearranged_weights.T

        # rearranged_weights = np.zeros((conv_features * conv_size, conv_size * n_e))

        # # counts number of input -> excitatory weights displayed so far
        # connection = input_connections['XeAe'][:]

        # # for each excitatory neuron in this convolution feature
        # for n in xrange(n_e):
        #       # for each convolution feature
        #       for feature in xrange(conv_features):
        #               temp = connection[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)].todense()
        #               rearranged_weights[ feature * conv_size : (feature + 1) * conv_size, n * conv_size : (n + 1) * conv_size ] = \
        #                                                                                                                                       temp[convolution_locations[n]].reshape((conv_size, conv_size))

        # # return the rearranged weights to display to the user
        # return rearranged_weights.T


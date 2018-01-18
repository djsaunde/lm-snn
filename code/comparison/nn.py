from __future__ import print_function

import os
import timeit
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata

plots_path = os.path.join('..', '..', 'plots')

print('\nLoading MNIST data.'); start = timeit.default_timer()
mnist = fetch_mldata('MNIST original', data_home='data')
print('Time: %.4f\n' % (timeit.default_timer() - start))

s = np.arange(mnist.data.shape[0])
np.random.shuffle(s)
mnist.data = mnist.data[s]
mnist.target = mnist.target[s]

expected = mnist.target[60000:]

interval = 250
maximum = 30000

for hidden in [128, 256, 512]:
	accuracies = []
	for examples in xrange(interval, maximum + interval, interval):
		print('Fitting neural network classification model with %d examples.' % examples); start = timeit.default_timer()
		model = MLPClassifier(verbose=True, hidden_layer_sizes=(hidden,), max_iter=1).fit(mnist.data[:examples], mnist.target[:examples])
		print('Time: %.4f\n' % (timeit.default_timer() - start))

		predictions = model.predict(mnist.data[60000:])
		accuracies.append((predictions == expected).mean() * 100)

	plt.plot(accuracies, label='2-layer NN (%d hidden units)' % hidden)

plt.legend()
plt.savefig(os.path.join(plots_path, 'nn_convergence.png'))
plt.show()
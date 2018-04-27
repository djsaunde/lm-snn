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

for hidden in [225, 400, 625]:
	accuracies = []
	model = MLPClassifier(verbose=True, hidden_layer_sizes=(hidden,), max_iter=1)
	
	for examples in xrange(interval, maximum + interval, interval):
		print('Fitting neural network classification model with %d examples.' % examples); start = timeit.default_timer()
		model = model.partial_fit(mnist.data[examples - interval:examples], mnist.target[examples - interval:examples], range(10))
		print('Time: %.4f\n' % (timeit.default_timer() - start))

		predictions = model.predict(mnist.data[60000:])
		accuracies.append((predictions == expected).mean() * 100)

	plt.plot(accuracies, label='2-layer NN (%d hidden units)' % hidden)

	np.save('nn_%d.npy' % hidden, accuracies)

plt.legend()
plt.savefig(os.path.join(plots_path, 'nn_convergence.png'))
plt.show()

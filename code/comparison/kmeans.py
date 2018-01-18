from __future__ import print_function, division

import os
import timeit
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import fetch_mldata

plots_path = os.path.join('..', '..', 'plots')

print('\nLoading MNIST data.'); start = timeit.default_timer()
mnist = fetch_mldata('MNIST original', data_home='data')
print('Time: %.4f\n' % (timeit.default_timer() - start))

s = np.arange(mnist.data.shape[0])
np.random.shuffle(s)
mnist.data = mnist.data[s]
mnist.target = mnist.target[s]

n_test = 10000

interval = 500
maximum = 30000

for n_clusters in [50, 100]:
	accuracies = []
	for n_train in xrange(interval, maximum + interval, interval):
		print('*** Fitting minibatch KMeans model with %d training examples and %d clusters. ***' % (n_train, n_clusters))
		start = timeit.default_timer()
		model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000).fit(mnist.data[:n_train])
		print('Time: %.4f\n' % (timeit.default_timer() - start))

		print('Labeling cluster centroids.'); start = timeit.default_timer()
		counts = np.zeros([n_clusters, 10])
		for x, y in zip(mnist.data[:max(n_train, 10000)], mnist.target[:max(n_train, 10000)]):
			distances = [np.linalg.norm(x - centroid) for centroid in model.cluster_centers_]
			counts[np.argmin(distances), int(y)] += 1

		proportions = counts / max(n_train, 10000)
		cluster_assignments = np.argmax(proportions, axis=1)
		print('Time: %.4f\n' % (timeit.default_timer() - start))

		print('Calculating classification accuracy of clustering.'); start = timeit.default_timer()
		correct = 0
		for x, y in zip(mnist.data[60000:], mnist.target[60000:]):
			distances = [np.linalg.norm(x - centroid) for centroid in model.cluster_centers_]
			if cluster_assignments[np.argmin(distances)] == int(y):
				correct += 1

		accuracy = (correct / n_test) * 100
		accuracies.append(accuracy)

		print('Time: %.4f\n' % (timeit.default_timer() - start))
		print('Correct: %d / %d' % (correct, n_test))
		print('Accuracy: %.4f\n' % accuracy)

	plt.plot(accuracies, label='K-Means with %d clusters' % n_clusters)

plt.legend()
plt.savefig(os.path.join(plots_path, 'kmeans_convergence.png'))
plt.show()
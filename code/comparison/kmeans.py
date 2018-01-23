from __future__ import print_function, division

import os
import timeit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

interval = 1000
maximum = 30000

for n_clusters in [400, 625]:
	accuracies = []
	model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=interval)

	for n_train in xrange(interval, maximum + interval, interval):
		print('*** Fitting minibatch KMeans model with %d training examples and %d clusters. ***' % (n_train, n_clusters))
		start = timeit.default_timer()
		model.fit(mnist.data[:n_train])
		print('Time: %.4f\n' % (timeit.default_timer() - start))

		print('Labeling cluster centroids.'); start = timeit.default_timer()
		distances = np.array([np.linalg.norm(mnist.data[:n_train] - centroid, axis=1) for centroid in model.cluster_centers_])
		clusters = np.argmin(distances, axis=0)
		proportions = np.zeros([n_clusters, 10])
		
		for cluster, target in zip(clusters, mnist.target[:n_train]):
			proportions[int(cluster), int(target)] += 1 / n_train

		assignments = np.argmax(proportions, axis=1)
		print('Time: %.4f\n' % (timeit.default_timer() - start))

		print('Calculating classification accuracy of clustering.'); start = timeit.default_timer()
		distances = np.array([np.linalg.norm(mnist.data[-n_test:] - centroid, axis=1) for centroid in model.cluster_centers_])
		correct = np.sum(assignments[np.argmin(distances, axis=0)] == mnist.target[-n_test:])
		accuracy = (correct / n_test) * 100
		accuracies.append(accuracy)

		print('Time: %.4f\n' % (timeit.default_timer() - start))
		print('Correct: %d / %d' % (correct, n_test))
		print('Accuracy: %.4f\n' % accuracy)

	plt.plot(accuracies, label='K-Means with %d clusters' % n_clusters)

	np.save('kmeans_%d.npy' % n_clusters, accuracies)

plt.legend()
plt.savefig(os.path.join(plots_path, 'kmeans_convergence.png'))
plt.show()
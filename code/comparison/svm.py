from __future__ import print_function

import timeit
import numpy as np

from sklearn.svm import SVC
from sklearn.datasets import fetch_mldata

print('\nLoading MNIST data.'); start = timeit.default_timer()
mnist = fetch_mldata('MNIST original', data_home='data')
print('Time: %.4f\n' % (timeit.default_timer() - start))

s = np.arange(mnist.data.shape[0])
np.random.shuffle(s)
mnist.data = mnist.data[s]
mnist.target = mnist.target[s]

print('Fitting support vector machine classification model.'); start = timeit.default_timer()
model = SVC(verbose=True, kernel='linear').fit(mnist.data[:50000], mnist.target[:50000])
print('Time: %.4f\n' % (timeit.default_timer() - start))

print('Predicting targets of test data.'); start = timeit.default_timer()
predictions = model.predict(mnist.data[60000:])
print('Time: %.4f\n' % (timeit.default_timer() - start))

accuracy = (predictions == mnist.target[60000:]).mean() * 100
print('SVM classification accuracy: %.4f\n' % accuracy)
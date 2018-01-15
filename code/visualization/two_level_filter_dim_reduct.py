import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

weights_path = os.path.join('..', '..', 'weights', 'csnn_two_level_inhibition', 'best')
assignments_path = os.path.join('..', '..', 'assignments', 'csnn_two_level_inhibition', 'best')
plots_path = os.path.join('..', '..', 'plots', 'two_level_pca')

if not os.path.isdir(plots_path):
	os.makedirs(plots_path)

parser = argparse.ArgumentParser()

parser.add_argument('--conv_size', type=int, default=28, help='Side length of the square convolution \
											window used by the input -> excitatory layer of the network.')
parser.add_argument('--conv_stride', type=int, default=0, help='Horizontal, vertical stride \
								of the convolution window used by input layer of the network.')
parser.add_argument('--conv_features', type=int, default=625, help='Number of excitatory \
							convolutional features / filters / patches used in the network.')
parser.add_argument('--num_train', type=int, default=60000, help='The number of \
										examples for which to train the network on.')
parser.add_argument('--random_seed', type=int, default=1, help='The random seed \
								(any integer) from which to generate random numbers.')
parser.add_argument('--start_inhib', type=float, default=1.0, help='The beginning value \
													of inhibiton for the increasing scheme.')
parser.add_argument('--max_inhib', type=float, default=20.0, help='The maximum synapse \
										weight for inhibitory to excitatory connections.')
parser.add_argument('--proportion_low', type=float, default=0.1, help='What proportion of \
							the training to grow the inhibition from "start_inhib" to "max_inhib".')

# Parse arguments and place them in local scope.
args = parser.parse_args()
args = vars(args)
locals().update(args)

print '\nOptional argument values:'
for key, value in args.items():
	print '-', key, ':', value

print '\n'

n_input_sqrt = 28

# Calculate number of neurons per convolution patch.
if conv_size == 28 and conv_stride == 0:
	n_e = 1
else:
	n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2

# Set ending of filename saves.
ending = '_'.join([ str(conv_size), str(conv_stride), str(conv_features), str(n_e), \
								str(num_train), str(random_seed), str(proportion_low), \
												str(start_inhib), str(max_inhib), 'best' ])

# Get filter weights.
weights = np.load(os.path.join(weights_path, '_'.join(['XeAe', ending]) + '.npy')).T

# Get neuron assignments.
assignments = np.load(os.path.join(assignments_path, '_'.join(['assignments', ending]) + '.npy')).ravel()

# Fit PCA model and transform weights to 2D projection.
X = PCA(n_components=2).fit_transform(weights)

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(19, 14))

cmap = plt.cm.get_cmap('hsv', 11)

for i in xrange(10):
	idxs = np.where(assignments == i)
	ax1.scatter(X[:, 0][idxs], X[:, 1][idxs], c=cmap(i), label=i)

ax1.set_xticks(())
ax1.set_yticks(())

ax1.legend()

ax2.set_xlim([min(X[:, 0]) - 0.15, max(X[:, 0]) + 0.15])
ax2.set_ylim([min(X[:, 1]) - 0.15, max(X[:, 1]) + 0.15])

for x, fltr in zip(X, weights):
	ax2.imshow(fltr.reshape([28, 28]), extent=(x[0]-0.1, x[0]+0.1, x[1]-0.1, x[1]+0.1), cmap='hot_r', alpha=0.5)

ax2.set_xticks(())
ax2.set_yticks(())

plt.savefig(os.path.join(plots_path, '2D_' + ending + '.png'))

plt.show()

fig2 = plt.figure(figsize=(16, 12))
ax3 = fig2.add_subplot(111, projection='3d')

# Fit PCA model and transform weights to 3D projection.
X = PCA(n_components=3).fit_transform(weights)

cmap = plt.cm.get_cmap('hsv', 11)

for i in xrange(10):
	idxs = np.where(assignments == i)
	ax3.scatter(X[:, 0][idxs], X[:, 1][idxs], X[:, 2][idxs], c=cmap(i), label=i)

ax3.set_xticks(())
ax3.set_yticks(())
ax3.set_zticks(())

ax3.legend()

plt.savefig(os.path.join(plots_path, '3D_' + ending + '.png'))

plt.show()

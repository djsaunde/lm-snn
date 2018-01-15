import os
import sys
import math
import numpy as np
import cPickle as p
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

top_level_path = os.path.join('..', '..')
model_name = 'snn'
performance_dir = os.path.join(top_level_path, 'performance', model_name)
plots_dir = os.path.join(top_level_path, 'plots', model_name)

if not os.path.isdir(plots_dir):
	os.makedirs(plots_dir)

fig_num = 1

def plot_performances(performances, labels):
	fig = plt.figure(fig_num, figsize = (8, 8))
	ax = plt.gca()

	for performance, label in zip(performances, labels):
		if label in ['all', 'confidence_weighting']:
			ax.plot(performance, label=label)

	plt.legend()

	fig.canvas.draw()

	plt.savefig(os.path.join(plots_dir, 'performances' + '_'.join(file_name.split('_')[1:])[:-4] + '.png'))

	plt.show()

print '\n'
print '\n'.join([ str(idx) + ' | ' + file_name for idx, file_name in \
	enumerate([ file_name for file_name in sorted(os.listdir(performance_dir))]) ])
print '\n'

to_plot = raw_input('Enter the index of the file from above which you\'d like to visualize: ')
if to_plot == '':
	file_name = [ file_name for file_name in sorted(os.listdir(performance_dir))][0]
else:
	file_name = [ file_name for file_name in sorted(os.listdir(performance_dir))][int(to_plot)]

conv_size = int(file_name.split('_')[1])
conv_stride = int(file_name.split('_')[2])
conv_features = int(file_name.split('_')[3])

n_input_sqrt = 28

if conv_size == 28 and conv_stride == 0:
	n_e = n_e_sqrt = 1
	n_e_total = conv_features
else:
	n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
	n_e_total = n_e * conv_features
	n_e_sqrt = int(math.sqrt(n_e))

performances = open(os.path.join(performance_dir, file_name), 'r').read()
print performances

performances, labels = np.array(performances.values()), performances.keys()

plot_performances(performances, labels)

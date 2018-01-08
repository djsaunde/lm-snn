import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

top_level_path = os.path.join('..', '..')
model_name = 'csnn_two_level_inhibition'
best_misc_dir = os.path.join(top_level_path, 'assignments', model_name, 'best')
plots_dir = os.path.join(top_level_path, 'plots', model_name)

if not os.path.isdir(plots_dir):
	os.makedirs(plots_dir)

fig_num = 1

def plot_labels(labels):
	fig = plt.figure(fig_num, figsize = (5, 5))
	ax = plt.gca()

	cmap = plt.get_cmap('RdBu', 10)
	labels = labels.reshape((int(np.sqrt(n_e_total)), int(np.sqrt(n_e_total)))).T
	
	im = ax.matshow(labels, cmap=cmap, vmin=-0.5, vmax=9.5)

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.1)
	
	plt.colorbar(im, cax=cax, ticks=np.arange(0, 10))

	fig.canvas.draw()

	plt.savefig(os.path.join(plots_dir, '_'.join(file_name.split('_')[1:])[:-4] + '.png'))

	plt.show()

	return im, fig

print '\n'
print '\n'.join([ str(idx) + ' | ' + file_name for idx, file_name in \
	enumerate([ file_name for file_name in sorted(os.listdir(best_misc_dir))]) ])
print '\n'

to_plot = raw_input('Enter the index of the file from above which you\'d like to visualize: ')
if to_plot == '':
	file_name = [ file_name for file_name in sorted(os.listdir(best_misc_dir))][0]
else:
	file_name = [ file_name for file_name in sorted(os.listdir(best_misc_dir))][int(to_plot)]

conv_size = int(file_name.split('_')[1])
conv_stride = int(file_name.split('_')[2])
conv_features = int(file_name.split('_')[3])

if conv_size == 28 and conv_stride == 0:
	n_e = n_e_sqrt = 1
	n_e_total = conv_features
else:
	n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
	n_e_total = n_e * conv_features
	n_e_sqrt = int(math.sqrt(n_e))

assignments = np.load(os.path.join(best_misc_dir, file_name))

plot_labels(assignments)

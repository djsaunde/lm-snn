import os
import numpy as np
import matplotlib.pyplot as plt

top_level_path = os.path.join('..', '..')
model_name = 'csnn_pc_inhibit_far'
best_misc_dir = os.path.join(top_level_path, 'misc', model_name, 'best')
plots_dir = os.path.join(top_level_path, 'plots', model_name)

fig_num = 1

plt.ion()

def plot_labels(labels):
	fig = plt.figure(fig_num, figsize = (5, 5))
	cmap = plt.get_cmap('RdBu', 11)
	labels = labels.reshape((int(np.sqrt(n_e_total)), int(np.sqrt(n_e_total)))).T
	im = plt.matshow(labels, cmap=cmap, vmin=-1.5, vmax=9.5)
	plt.colorbar(im, ticks=np.arange(-1, 10))
	plt.title('Neuron labels')
	fig.canvas.draw()
	plt.show()
	return im, fig

print '\n'
print '\n'.join([ str(idx) + ' | ' + file_name for idx, file_name in enumerate([ file_name for file_name in sorted(os.listdir(best_misc_dir))]) if 'accumulated_rates' in file_name ])
print '\n'

to_plot = raw_input('Enter the index of the file from above which you\'d like to visualize: ')
if to_plot == '':
	file_name = [ file_name for file_name in sorted(os.listdir(best_misc_dir))][0]
else:
	file_name = [ file_name for file_name in sorted(os.listdir(best_misc_dir))][int(to_plot)]

conv_size = int(file_name.split('_')[3])
conv_stride = int(file_name.split('_')[4])
conv_features = int(file_name.split('_')[5])

if conv_size == 28 and conv_stride == 0:
	n_e = n_e_sqrt = 1
	n_e_total = conv_features
else:
	n_e = ((n_input_sqrt - conv_size) / conv_stride + 1) ** 2
	n_e_total = n_e * conv_features
	n_e_sqrt = int(math.sqrt(n_e))

accumulated_rates = np.load(os.path.join(best_misc_dir, file_name))

for i in xrange(10):
	plot_labels(np.argsort(accumulated_rates, axis=1)[:, i].reshape((conv_features, n_e)))
	fig_num += 1

x = raw_input('Quit? ')
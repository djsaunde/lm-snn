import os
import numpy as np
import cPickle as p
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 5)

plots_path = os.path.join('..', '..', 'plots')
two_level_performance_dir = os.path.join('..', '..', 'performance', 'csnn_two_level_inhibition')

window_size = 2  # parameter controlling smoothness of training curves

def window(window_size=window_size):
	return np.ones(window_size) / float(window_size)

kmeans_curves = {}
for fname in os.listdir('.'):
	if '.npy' in fname:
		curve = np.load(fname)
		if 'kmeans' in fname and ('225' in fname or '400' in fname or '625' in fname):
			n_clusters = int(fname.split('.')[0].split('_')[1])
			kmeans_curves['k-Means (%d clusters)' % n_clusters] = [0] + list(curve)

nn_curves = {}
for fname in os.listdir('.'):
	if '.npy' in fname:
		curve = np.load(fname)
		if 'nn' in fname and ('225' in fname or '400' in fname or '625' in fname):
			n_hidden = int(fname.split('.')[0].split('_')[1])
			nn_curves['2-layer NN: (%d hidden units)' % n_hidden] = [0] + list(curve)

two_level_fnames = ['28_0_%d_1_60000_0_0.1_1.0_20.0.p' % n_neurons for n_neurons in [225, 400, 625]]
two_level_list = [p.load(open(os.path.join(two_level_performance_dir, two_level_fname), 'r'))[1]['confidence_weighting'] for two_level_fname in two_level_fnames]

two_level_curves = {}
for idx, n_neurons in enumerate([225, 400, 625]):
	two_level_curves[r'LM-SNN ($n_e = %d$)' % n_neurons] = two_level_list[idx][:30000 / 250 + 1]

colors = ['r', 'g', 'b']

for curve, color in zip(sorted(two_level_curves), colors):
	plot = two_level_curves[curve][0::4]
	plot = list(np.convolve(plot, window(), 'same')[:-window_size])
	plt.plot(plot + plot[-2:], linestyle='-', label=curve, color=color)

for curve, color in zip(sorted(kmeans_curves), colors):
	plot = kmeans_curves[curve]
	plot = list(np.convolve(plot, window(), 'same')[:-window_size])
	plt.plot(plot + plot[-2:], linestyle='--', label=curve, color=color)

for curve, color in zip(sorted(nn_curves), colors):
	plot = nn_curves[curve][0::4]
	plot = list(np.convolve(plot, window(), 'same')[:-window_size])
	plt.plot(plot + plot[-2:], linestyle=':', label=curve, color=color)

plt.ylim([0, 100]); plt.xlim([0, 30])
plt.xlabel('No. of training examples', fontsize=14); plt.ylabel('Test dataset accuracy', fontsize=14)
plt.xticks(xrange(0, 31, 2), xrange(0, 30000 + 2000, 2000))
plt.grid(); plt.legend(); plt.tight_layout()

plt.savefig(os.path.join(plots_path, 'nn_kmeans_lm-snn.png'))

plt.show()
import os
import numpy as np
import cPickle as p
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 5)

plots_path = os.path.join('..', '..', 'plots')
two_level_performance_dir = os.path.join('..', '..', 'performance', 'csnn_two_level_inhibition')

window_size = 10  # parameter controlling smoothness of training curves

def window(window_size=window_size):
	return np.ones(window_size) / float(window_size)

curves = {}
for fname in os.listdir('.'):
	if '.npy' in fname:
		curve = np.load(fname)
		if 'nn' in fname:
			n_hidden = int(fname.split('.')[0].split('_')[1])
			curves['2-layer NN: (%d hidden units)' % n_hidden] = [0] + list(curve)
		if 'kmeans' in fname:
			n_clusters = int(fname.split('.')[0].split('_')[1])
			curves['k-Means (%d clusters)' % n_clusters] = [0] + list(curve)

two_level_fname = '28_0_625_1_60000_0_0.1_1.0_20.0.p'
two_level_curve = p.load(open(os.path.join(two_level_performance_dir, two_level_fname), 'r'))[1]['confidence_weighting']
curves[r'LM-SNN ($n_e = 625$)'] = two_level_curve[:30000 / 250 + 1]

for curve in sorted(curves):
	if 'LM-SNN' in curve:
		plt.plot(curves[curve], linestyle='-', label=curve)

for curve in sorted(curves):
	if '2-layer NN' in curve:
		plt.plot(curves[curve], linestyle='--', label=curve)

for curve in sorted(curves):
	if 'k-Means' in curve:
		if '50' in curve and not '150' in curve:
			plt.plot(curves[curve], linestyle=':', label=curve)

for curve in sorted(curves):
	if 'k-Means' in curve:
		if '100' in curve:
			plt.plot(curves[curve], linestyle=':', label=curve)

for curve in sorted(curves):
	if 'k-Means' in curve:
		if '150' in curve:
			plt.plot(curves[curve], linestyle=':', label=curve)

plt.ylim([0, 100]); plt.xlim([0, 120])
plt.xlabel('No. of training examples', fontsize=14); plt.ylabel('Test dataset accuracy', fontsize=14)
plt.xticks(xrange(0, len(curves[curve]), 10), xrange(0, 30000 + 2500, 2500))
plt.grid(); plt.legend(); plt.tight_layout()

plt.savefig(os.path.join(plots_path, 'nn_kmeans_lm-snn.png'))

plt.show()
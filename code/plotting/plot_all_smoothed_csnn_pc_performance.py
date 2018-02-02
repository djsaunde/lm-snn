from __future__ import division

import matplotlib.pyplot as plt
import cPickle as p
import numpy as np
import sys
import os


top_level_path = os.path.join('..', '..')
snn_performance_dir = os.path.join('..', '..', 'performance', 'snn')
csnn_pc_performance_dir = os.path.join(top_level_path, 'performance', 'csnn_pc')

window_size = 3

sizes = ['100', '225', '400', '625', '900']
colors = ['r', 'b', 'g', 'y', 'k']

def window(size=window_size):
    return np.ones(window_size) / float(window_size)

def n_neurons(fname, sizes=sizes):
	return any([size in fname for size in sizes])

strings = ['none_16_2_20', 'none_16_2_40', 'none_16_2_60', 'none_16_2_80', 'none_16_2_100']

performances = {}
for f in os.listdir(csnn_pc_performance_dir):
	if any([string in f for string in strings]) and '_4_0.0.p' in f:
		performances[f] = p.load(open(os.path.join(csnn_pc_performance_dir, f), 'rb'))[1]['all']


handles = []
for idx, (fname, color) in enumerate(zip(sorted(performances, key=lambda x : int(x.split('_')[3])), colors)):
	performance = np.convolve(performances[fname], window(), 'same')
	handles.append(plt.plot(performance[:-window_size], color=color, label=r'C-SNN: $k = 16, s = 2, n_\mathrm{patches} =$ ' + fname.split('_')[3]))

snn_files = [ f for f in sorted(os.listdir(snn_performance_dir)) if '.p' in f and '0_60000' in f and '1225' not in f and n_neurons(f) ]
snn_performances = np.array([ p.load(open(os.path.join(snn_performance_dir, f), 'r'))[1] for f in snn_files ])

for perf, label, color in zip(snn_performances, sizes, colors):
	plt.plot(np.convolve(perf[:84 + window_size], window(), 'same')[:-window_size], '--', label=r'SNN: $n_e =$ ' + label, color=color)

fig = plt.gcf()
fig.set_size_inches(14, 8)

plt.xlabel('No. of iterations', fontsize=15)
plt.ylabel('Classification accuracy', fontsize=15)
plt.xticks(xrange(0, 84, 4), xrange(0, 20500, 1000))
plt.ylim([0, 100]); plt.xlim([0, 60])
plt.tight_layout(); plt.grid(); plt.legend()
plt.show()

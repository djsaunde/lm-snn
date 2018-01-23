import os
import sys
import argparse
import numpy as np
import cPickle as p
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 5)

snn_performance_dir = os.path.join('..', '..', 'performance', 'snn')
two_level_performance_dir = os.path.join('..', '..', 'performance', 'csnn_two_level_inhibition')
plots_dir = os.path.join('..', '..', 'plots')

parser = argparse.ArgumentParser()

parser.add_argument('--num_train', type=str, default='60000')
parser.add_argument('--random_seed', type=str, default='0')
parser.add_argument('--start_inhib', type=str, default='1.0')
parser.add_argument('--max_inhib', type=str, default='20.0')
parser.add_argument('--proportion_low', type=str, default='0.1')
parser.add_argument('--sizes', type=str, nargs='+', default=['225', '400', '625', '900'])
parser.add_argument('--window_size', type=int, default=10)

# parse arguments and place them in local scope
args = parser.parse_args()
args = vars(args)
locals().update(args)

filestring = '_'.join([num_train, random_seed, proportion_low, start_inhib, max_inhib])

print '\nOptional argument values:'
for key, value in args.items():
	print '-', key, ':', value

print '\n'

def n_neurons(fname, sizes=sizes):
	return any([ size in fname for size in sizes ])

def window(window_size=window_size):
	return np.ones(window_size) / float(window_size)

snn_performances = np.array([ p.load(open(os.path.join(snn_performance_dir, f), 'r'))[1] \
			for f in sorted(os.listdir(snn_performance_dir)) if '.p' in f and '4_60000' in f and '1225' not in f and n_neurons(f) ])

two_level_performances = np.array([ p.load(open(os.path.join(two_level_performance_dir, f), 'r'))[1]['confidence_weighting'] \
								for f in sorted(os.listdir(two_level_performance_dir)) if (filestring in f and n_neurons(f) and '1225' not in f) ])

colors = ['r', 'b', 'g', 'k']

for perf, label, color in zip(snn_performances, sizes, colors):
	plt.plot(np.convolve(perf, window(), 'same')[:-window_size], '--', label=r'SNN: $n_e =$ ' + label, color=color)

for perf, label, color in zip(two_level_performances, sizes, colors):
	plt.plot(np.convolve(perf, window(), 'same')[:-window_size], '-', label=r'LM-SNN: $n_e =$ ' + label, color=color)

plt.xticks(xrange(0, 200, 10), xrange(0, 47500, 2500)); plt.yticks(xrange(0, 110, 10))
plt.xlim([0, 180]); plt.ylim([0, 100])
plt.xlabel('No. of training examples', fontsize=14); plt.ylabel('Estimated test accuracy (smoothed)', fontsize=14)

plt.legend(); plt.grid(); plt.tight_layout()

plt.savefig(os.path.join(plots_dir, 'convergence_' + filestring + '.png'))

plt.show()
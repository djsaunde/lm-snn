from __future__ import division

import matplotlib.pyplot as plt
import cPickle as p
import numpy as np
import os

model_name = 'csnn_two_level_inhibition'

top_level_path = os.path.join('..', '..')
performance_dir = os.path.join(top_level_path, 'performance', model_name)
plots_dir = os.path.join(top_level_path, 'plots', model_name)

if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)

print '\n'
print '\n'.join([ str(idx + 1) + ' | ' + file_name for idx, file_name in enumerate(sorted(os.listdir(performance_dir))) if '.p' in file_name ]), '\n'

to_plot = raw_input('Enter the index of the file from above which you\'d like to plot: ')
file_name = sorted([ file_name for file_name in os.listdir(performance_dir) if '.p' in file_name ])[int(to_plot) - 1]

_, performances = p.load(open(os.path.join(performance_dir, file_name), 'rb'))

print '\n'

performance_plots = []
for voting_mechanism in sorted(performances.keys()):
    if voting_mechanism in ['all', 'confidence_weighting']:
        plt.plot(performances[voting_mechanism], label=voting_mechanism)[0]

fig = plt.gcf()
fig.set_size_inches(16, 12)

num_train = int(file_name.split('_')[4])
print num_train

plt.xlabel('No. of training samples')
plt.ylabel('Estimated classification accuracy')
plt.xticks(xrange(0, int(num_train / 250) + 20, 40), xrange(0, num_train + 10000, 10000));
plt.yticks(xrange(0, 110, 10))

plt.title(file_name.split('.')[0])
plt.tight_layout(); plt.grid(); plt.legend()

plt.savefig(os.path.join(plots_dir, 'performance_' + file_name[:file_name.index('.')] + ''))
plt.show()

print '\n'
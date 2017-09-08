from __future__ import division

import matplotlib.pyplot as plt
import cPickle as p
import numpy as np
import os


model_name = 'csnn_pc_inhibit_far'

top_level_path = os.path.join('..', '..')
performance_dir = os.path.join(top_level_path, 'performance', model_name)

if not os.path.isdir(os.path.join(performance_dir, 'performance_plots')):
    os.makedirs(os.path.join(performance_dir, 'performance_plots'))

print '\n'
print '\n'.join([ str(idx + 1) + ' | ' + file_name for idx, file_name in enumerate(sorted(os.listdir(performance_dir))) if '.p' in file_name ]), '\n'

to_plot = raw_input('Enter the index of the file from above which you\'d like to plot: ')
file_name = sorted([ file_name for file_name in os.listdir(performance_dir) if '.p' in file_name ])[int(to_plot) - 1]

# get pickled performances dictionary (voting mechanism, performance recordings over training)
_, performances = p.load(open(os.path.join(performance_dir, file_name), 'rb'))

print '\n'

performance_plots = []
for voting_mechanism in sorted(performances.keys()):
    if voting_mechanism in [ 'all', 'most_spiked', 'top_percent', 'spatial_clusters' ]:
        performance_plots.append(plt.plot(performances[voting_mechanism], label=voting_mechanism)[0])
    
plt.legend(handles=performance_plots)

fig = plt.gcf()
fig.set_size_inches(16, 12)

plt.xlabel('Iteration number (1 through ' + str(len(performances[performances.keys()[0]]) * 100) + ')')
plt.xticks([ x for x in xrange(0, len(performances[performances.keys()[0]]) + 25, 25) ], [ x * 100 for x in xrange(0, len(performances[performances.keys()[0]]) + 25, 25) ])
plt.ylabel('Classification accuracy (out of 100%)')

plt.title(file_name.split('_'))
plt.tight_layout()

plt.savefig(os.path.join(performance_dir, 'performance_plots', file_name[:-2].replace('.', '_')))
plt.show()

print '\n'

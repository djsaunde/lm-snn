from __future__ import division

import matplotlib.pyplot as plt
import cPickle as p
import numpy as np
import os


performance_dir = '../performance/conv_patch_connectivity_performance/'

print '\n'
print '\n'.join([ str(idx + 1) + ' | ' + file_name for idx, file_name in enumerate(sorted(os.listdir(performance_dir))) if '.p' in file_name ])
print '\n'

to_plot = raw_input('Enter the index of the file from above which you\'d like to plot: ')
file_name = sorted([ file_name for file_name in os.listdir(performance_dir) if '.p' in file_name ])[int(to_plot) - 1]

# get pickled performances dictionary (voting mechanism, performance recordings over training)
_, performances = p.load(open(performance_dir + file_name, 'rb'))

print '\n'

performance_plots = []
for voting_mechanism in sorted(performances.keys()):
    performance_plots.append(plt.plot(performances[voting_mechanism], label=voting_mechanism)[0])
    
plt.legend(handles=performance_plots)

fig = plt.gcf()
fig.set_size_inches(16, 12)

plt.xlabel('Iteration number (1 through ' + str(len(performances[performances.keys()[0]])) + ')')
plt.ylabel('Classification accuracy (out of 100%)')

title_strs = file_name[:file_name.index('weight') - 1].split('_')

plt.title(file_name[:file_name.index('.')])
plt.savefig(performance_dir + 'performance_plots/' + file_name[:file_name.index('.')])
plt.show()

print '\n'

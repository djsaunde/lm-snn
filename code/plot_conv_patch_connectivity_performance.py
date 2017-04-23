from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import os


performance_dir = '../performance/conv_patch_connectivity_performance/'

print '\n'
print '\n'.join([ str(idx + 1) + ' | ' + file_name for idx, file_name in enumerate(sorted(os.listdir(performance_dir))) if '.txt' in file_name ])
print '\n'

to_plot = raw_input('Enter the index of the file from above which you\'d like to plot: ')
file_name = sorted([ file_name for file_name in os.listdir(performance_dir) if '.txt' in file_name ])[int(to_plot) - 1]

print file_name

performances = {}
performance_lines = open(performance_dir + file_name, 'r').readlines()[1:]
for idx, line in enumerate(performance_lines):
    voting_mechanism, performance = [ token.strip() for token in line.split(':') ]
    performances[voting_mechanism] = [ float(token) for token in performance.split() ]

print '\n'

performance_plots = []
for voting_mechanism in sorted(performances.keys()):
    performance_plots.append(plt.plot(performances[voting_mechanism], label=voting_mechanism)[0])
    # average_plot, = plt.plot([ np.mean(performances[voting_mechanism]) ] * len(performances[voting_mechanism]),
    #                                             label='average: ' + str(np.mean(performances[voting_mechanism])))
    # upper_std_plot, = plt.plot([ np.mean(performances[voting_mechanism]) + np.std(performances[voting_mechanism]) ] * 
    #                                             len(performances[voting_mechanism]), label='plus one standard deviation: ' + 
    #                                             str(np.mean(performances[voting_mechanism]) + np.std(performances[voting_mechanism])))
    # lower_std_plot, = plt.plot([ np.mean(performances[voting_mechanism]) - np.std(performances[voting_mechanism]) ] * 
    #                                             len(performances[voting_mechanism]), label='minus one standard deviation: ' + 
    #                                             str(np.mean(performances[voting_mechanism]) - np.std(performances[voting_mechanism])))
    
plt.legend(handles=performance_plots)

fig = plt.gcf()
fig.set_size_inches(16, 12)

plt.xlabel('Iteration number (1 through ' + str(len(performances[performances.keys()[0]])) + ')')
plt.ylabel('Classification accuracy (out of 100%)')

title_strs = file_name[:file_name.index('weight') - 1].split('_')

# plt.title('Classification accuracy by iteration number (' + title_strs[0] + 'x' + title_strs[0] + ' convolution, stride ' + title_strs[1] + ', ' + title_strs[2] + ' convolution features, giving ' + title_strs[3] + ' excitatory neurons per convolutional patch')
plt.title(file_name[:file_name.index('.')])
plt.savefig(performance_dir + 'performance_plots/' + file_name[:file_name.index('.')])
plt.show()

print '\n'

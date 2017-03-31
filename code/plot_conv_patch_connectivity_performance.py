from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import os


perf_dir = '../performance/conv_patch_connectivity_performance/'


print '\n'
print '\n'.join([ str(idx) + ' | ' + file_name for idx, file_name in enumerate(os.listdir(perf_dir)) if '.txt' in file_name ])
print '\n'

to_plot = raw_input('Enter the index of the file from above which you\'d like to plot, or hit Enter to plot all: ')
if to_plot == '':
    file_names = [ file_name for file_name in os.listdir(perf_dir) if '.txt' in file_name ]
else:
    file_names = [[ file_name for file_name in os.listdir(perf_dir) if '.txt' in file_name ][int(to_plot)]]

print file_names

perfs = {}
for file_name in file_names:
    perf_text = open(perf_dir + file_name, 'r').readlines()[1:]
    perfs[file_name] = [ float(token) for token in ''.join(perf_text).replace('\n', '').replace('[', '').replace(']', '').split()  ]

print '\n'

plots = []
for perf in sorted(perfs.keys()):
    perf_plot, = plt.plot(perfs[perf][:100], label='performance')
    average_plot, = plt.plot([ np.mean(perfs[perf][:100]) ] * len(perfs[perf][:100]), label='average: ' + str(np.mean(perfs[perf][:100])))
    upper_std_plot, = plt.plot([ np.mean(perfs[perf][:100]) + np.std(perfs[perf][:100]) ] * len(perfs[perf][:100]), label='plus one standard deviation: ' + str(np.mean(perfs[perf][:100]) + np.std(perfs[perf][:100])))
    lower_std_plot, = plt.plot([ np.mean(perfs[perf][:100]) - np.std(perfs[perf][:100]) ] * len(perfs[perf][:100]), label='minus one standard deviation: ' + str(np.mean(perfs[perf][:100]) - np.std(perfs[perf][:100])))
    
    plt.legend(handles=[perf_plot, average_plot, upper_std_plot, lower_std_plot])
    
    fig = plt.gcf()
    fig.set_size_inches(16, 12)

    plt.xlabel('Iteration number (1 through ' + str(len(perfs[perf][:100])) + ')')
    plt.ylabel('Classification accuracy (out of 100%)')

    title_strs = perf[5:perf.index('weight') - 1].split('_')
    print perf

    plt.title('Classification accuracy by iteration number (' + title_strs[0] + 'x' + title_strs[0] + ' convolution, stride ' + title_strs[1] + ', ' + title_strs[2] + ' convolution features, giving ' + title_strs[3] + ' excitatory neurons per convolutional patch')
    plt.savefig(perf_dir + 'performance_plots/Classification accuracy by iteration number (' + title_strs[0] + 'x' + title_strs[0] + ' convolution, stride ' + title_strs[1] + ', ' + title_strs[2] + ' convolution features, gving ' + title_strs[3] + ' excitatory neurons per convolutional patch')
    plt.show()

print '\n'

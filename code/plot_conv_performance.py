from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import os


perf_dir = '../performance/'

perfs = {}
for file_name in os.listdir(perf_dir):
    if '59900' in file_name and 'XeAe100' not in file_name:
        perf_text = open(perf_dir + file_name, 'r').readlines()
        perfs[file_name] = [ float(token) for token in ''.join(perf_text).replace('\n', '').replace('[', '').replace(']', '').split()  ]

plots = []
for perf in sorted(perfs.keys()):
    # plots.append(plt.plot(perfs[perf]), label=perf[5:perf.index('weight') - 1])[0])
    perf_plot, = plt.plot(perfs[perf], label='performance')
    average_plot, = plt.plot([ np.mean(perfs[perf]) ] * len(perfs[perf]), label='average: ' + str(np.mean(perfs[perf])))
    upper_std_plot, = plt.plot([ np.mean(perfs[perf]) + np.std(perfs[perf]) ] * len(perfs[perf]), label='plus one standard deviation: ' + str(np.mean(perfs[perf]) + np.std(perfs[perf])))
    lower_std_plot, = plt.plot([ np.mean(perfs[perf]) - np.std(perfs[perf]) ] * len(perfs[perf]), label='minus one standard deviation: ' + str(np.mean(perfs[perf]) - np.std(perfs[perf])))
    
    plt.legend(handles=[perf_plot, average_plot, upper_std_plot, lower_std_plot])
    
    fig = plt.gcf()
    fig.set_size_inches(16, 12)

    plt.xlabel('Iteration number (1 through 60,000)')
    plt.ylabel('Classification accuracy (out of 100%)')

    title_strs = perf[5:perf.index('weight') - 1].split('_')
    print title_strs
    print perf
    print perf[5:perf.index('weight') - 1].split('_')

    plt.title('Classification accuracy by iteration number (' + title_strs[0] + 'x' + title_strs[0] + ' convolution, stride ' + title_strs[1] + ', ' + title_strs[2] + ' convolution features, gving ' + title_strs[3] + ' excitatory neurons per convolutional patch')
    plt.savefig(perf_dir + 'performance_plots/Classification accuracy by iteration number (' + title_strs[0] + 'x' + title_strs[0] + ' convolution, stride ' + title_strs[1] + ', ' + title_strs[2] + ' convolution features, gving ' + title_strs[3] + ' excitatory neurons per convolutional patch')
    plt.show()

# plt.legend(handles=[ plot for plot in plots ])

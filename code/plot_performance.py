from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import os


perf_dir = '../performance/'

file_name = 'XeAe100_weight_dependence_postpre_iter_59900' # raw_input('Enter in name of performance file to plot: ')

perf_text = open(perf_dir + file_name, 'r').readlines()
perf = [ float(token) for token in ''.join(perf_text).replace('\n', '').replace('[', '').replace(']', '').split()  ]

# plots.append(plt.plot(perfs[perf]), label=perf[5:perf.index('weight') - 1])[0])
perf_plot, = plt.plot(perf, label='performance')
average_plot, = plt.plot([ np.mean(perf) ] * len(perf), label='average: ' + str(np.mean(perf)))
upper_std_plot, = plt.plot([ np.mean(perf) + np.std(perf) ] * len(perf), label='plus one standard deviation: ' + str(np.mean(perf) + np.std(perf)))
lower_std_plot, = plt.plot([ np.mean(perf) - np.std(perf) ] * len(perf), label='minus one standard deviation: ' + str(np.mean(perf) - np.std(perf)))

plt.legend(handles=[perf_plot, average_plot, upper_std_plot, lower_std_plot])

fig = plt.gcf()
fig.set_size_inches(16, 12)

plt.xlabel('Iteration number (1 through 60,000)')
plt.ylabel('Classification accuracy (out of 100%)')

n_e = file_name[file_name.index('Ae') + 2 : file_name.index('_')]

plt.title('Classification accuracy by iteration number (' + n_e + ' excitatory, inhibitory neurons)')
plt.savefig(perf_dir + 'performance_plots/Classification accuracy by iteration number (' + n_e + ' excitatory, inhibitory neurons)')
plt.show()

# plt.legend(handles=[ plot for plot in plots ])
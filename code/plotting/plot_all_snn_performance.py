import os

import matplotlib.pyplot as plt


top_level_path = os.path.join('..', '..')

performances = [ 51.86, 66.18, 78.16, 84.59, 90.54, 91.98, 92.17, 93.01, 95.22 ]
x = [ 25, 50, 100, 200, 400, 800, 1600, 3200, 6400 ]

performance_plot, = plt.semilogx(x, performances, label='Spiking Neural Network (Diehl & Cook 2015)')
conv_plot, = plt.plot(x, [ 99.0 ] * len(x), label='Convolutional neural network baseline')

fig = plt.gcf()
fig.set_size_inches(16, 12)

plt.legend(handles=[performance_plot, conv_plot], prop={ 'size' : 16 })
plt.xlim(25, 6400)
plt.title('Classification performance by number of excitatory, inhibitory neurons', fontsize=20)
plt.xticks([ 25, 50, 100, 200, 400, 800, 1600, 3200, 6400 ], [ 25, 50, 100, 200, 400, 800, 1600, 3200, 6400 ])
plt.xlabel('No. of excitatory neurons', fontsize=16)
plt.ylabel('Test dataset accuracy', fontsize=16)

plt.savefig(os.path.join(top_level_path, 'plots', 'snn_performance.png'))

plt.show()

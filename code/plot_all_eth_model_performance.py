import matplotlib.pyplot as plt

performances = [ 51.86, 66.18, 78.16, 84.59, 90.54, 91.98, 92.17, 93.01, 95.22 ]
x = [ 25, 50, 100, 200, 400, 800, 1600, 3200, 6400 ]

performance_plot, = plt.semilogx(x, performances, label='Spiking Neural Network (Diehl & Cook 2015)')
conv_plot, = plt.plot(x, [ 99.0 ] * len(x), label='Convolutional Neural Network Baseline')

fig = plt.gcf()
fig.set_size_inches(16, 12)

plt.legend(handles=[performance_plot, conv_plot])
plt.xlim(0, 7000)
plt.title('Classification performance by number of excitatory, inhibitory neurons')
plt.xlabel('Number of excitatory, inhibitory neurons')
plt.ylabel('Test dataset classification accuracy')

plt.show()

import matplotlib.pyplot as plt

performances = {}
x_axes = {}

x_axes['Convolutional neural network baseline'] = [ 784 * 100, 784 * 7000 ]
performances['Convolutional neural network baseline'] = [ 99.0 ] * len(x_axes['Convolutional neural network baseline'])

x_axes['Spiking neural network (Diehl & Cook 2015)'] = [ 784 * n_e for n_e in [ 100, 200, 400, 800, 1600, 3200, 6400 ]]
performances['Spiking neural network (Diehl & Cook 2015)'] = [ 78.16, 84.59, 90.54, 91.98, 92.17, 93.01, 95.22 ]

# 51.86, 66.18,

x_axes['Convolution spiking neural network (4 lattice, no connectivity, weight sharing, all voting)'] = [ 50 * n_e for n_e in [ 1024, 1600, 2304, 3600, 6084 ] ]
performances['Convolution spiking neural network (4 lattice, no connectivity, weight sharing, all voting)'] = [ 54.18, 41.5, 55.22, 58.02, 63.72 ]

# x_axes['Convolution spiking neural network (4 lattice, pairs connectivity, weight sharing, all voting)'] = [ 50 * n_e for n_e in [ 1024, 1600, 2304, 3600, 6084 ] ]
# performances['Convolution spiking neural network (4 lattice, pairs connectivity, weight sharing, all voting)'] = [ 54.18, 41.5, 55.22, 58.02, 63.72 ]

# x_axes['Convolution spiking neural network (4 lattice, weight sharing, most-spiked voting)'] = []
# performances['Convolution spiking neural network (4 lattice, weight sharing, most-spiked voting)'] = []

# x_axes['Convolution spiking neural network (4 lattice, no weight sharing, all voting)'] = []
# performances['Convolution spiking neural network (4 lattice, no weight sharing, all voting)'] = []

# x_axes['Convolution spiking neural network (4 lattice, no connectivity, no weight sharing, most-spiked voting)'] = [ 50 * n_e for n_e in [ 2704, 2916, 5184, 7744,  ] ]
# performances['Convolution spiking neural network (4 lattice, no connectivity, no weight sharing, most-spiked voting)'] = [  ]

# x_axes['Convolution spiking neural network (4 lattice, no weight sharing, top percent voting)'] = []
# performances['Convolution spiking neural network (4 lattice, no weight sharing, top percent voting)'] = []

x_axes['Convolution spiking neural network (8 lattice, no connectivity, weight sharing, all voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, no connectivity, weight sharing, all voting)'] = [ 59.06, 65.42, 68.33, 62.74, 60.32, 58.28, 63.14, 61.77, 63.52, 62.32, 64.52 ]

x_axes['Convolution spiking neural network (8 lattice, no connectivity, weight sharing, most-spiked voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, no connectivity, weight sharing, most-spiked voting)'] = [ 60.01, 63.53, 68.55, 66.9, 68.28, 66.35, 66.36, 66.79, 66.72, 68.99, 61.07 ]

x_axes['Convolution spiking neural network (8 lattice, no connectivity, weight sharing, top percent voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, no connectivity, weight sharing, top percent voting)'] = [ 61.7, 64.31, 70.17, 69.22, 70.22, 67.4, 67.76, 69.19, 71.27, 72.5, 67.97 ]

x_axes['Convolution spiking neural network (8 lattice, pairs connectivity, weight sharing, all voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, pairs connectivity, weight sharing, all voting)'] = [ 54.34, 66.98, 64.08, 62.96, 62.14, 61.09, 60.29, 65.51, 65.65, 59.54, 56.14 ]

x_axes['Convolution spiking neural network (8 lattice, pairs connectivity, weight sharing, most-spiked voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, pairs connectivity, weight sharing, most-spiked voting)'] = [ 58.47, 66.73, 67.35, 66.91, 68.87, 66.19, 67.3, 69.74, 67.6, 63.86, 60.21 ]

x_axes['Convolution spiking neural network (8 lattice, pairs connectivity, weight sharing, top percent voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, pairs connectivity, weight sharing, top percent voting)'] = [ 58.38, 67.64, 67.23, 67.32, 68.66, 69.85, 67.32, 72.24, 72.07, 68.33, 65.66 ]

x_axes['Convolution spiking neural network (8 lattice, all connectivity, weight sharing, all voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, all connectivity, weight sharing, all voting)'] = [ 53.06, 65.33, 67.71, 64.52, 62.53, 60.76, 59.92, 62.08, 66.52, 62.56, 61.94 ]

x_axes['Convolution spiking neural network (8 lattice, all connectivity, weight sharing, most-spiked voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, all connectivity, weight sharing, most-spiked voting)'] = [ 56.42, 63.96, 68.65, 69.68, 63.51, 64.53, 66.04, 68.45, 69.51, 66.82, 60.81 ]

x_axes['Convolution spiking neural network (8 lattice, all connectivity, weight sharing, top percent voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, all connectivity, weight sharing, top percent voting)'] = [ 55.02, 66.21, 69.9, 69.11, 66.24, 68.72, 66.08, 72.2, 71.45, 69.32, 64.78 ]

x_axes['Convolution spiking neural network (8 lattice, no connectivity, no weight sharing, all voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, no connectivity, no weight sharing, all voting)'] = [ 64.36, 67.56, 67.53, 72.81, 75.82, 78.93, 80.02, 84.18, 83.56, 82.58, 80.21 ]

x_axes['Convolution spiking neural network (8 lattice, no connectivity, no weight sharing, most-spiked voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, no connectivity, no weight sharing, most-spiked voting)'] = [ 64.71, 67.47, 66.51, 73.25, 76.22, 78.76, 79.39, 82.86, 81.18, 79.68, 76.69 ]

x_axes['Convolution spiking neural network (8 lattice, no connectivity, no weight sharing, top percent voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, no connectivity, no weight sharing, top percent voting)'] = [ 62.23, 67.16, 65.65, 70.64, 74.38, 76.41, 76.82, 82.04, 80.5, 78.16, 73.83 ]

x_axes['Convolution spiking neural network (8 lattice, pairs connectivity, no weight sharing, all voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, pairs connectivity, no weight sharing, all voting)'] = [ 60.99, 67.06, 68.77, 73.29, 75.13, 78.89, 79.48, 84.18, 83.31, 82.95, 80.79 ]

x_axes['Convolution spiking neural network (8 lattice, pairs connectivity, no weight sharing, most-spiked voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, pairs connectivity, no weight sharing, most-spiked voting)'] = [ 61.68, 66.25, 68.73, 73.1, 75.95, 78.77, 78.88, 83.19, 81.31, 79.44, 77.21 ]

x_axes['Convolution spiking neural network (8 lattice, pairs connectivity, no weight sharing, top percent voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, pairs connectivity, no weight sharing, top percent voting)'] = [ 60.41, 65.59, 67.97, 71.9, 73.75, 76.33, 76.37, 82.02, 80.56, 78.39, 75.22 ]

x_axes['Convolution spiking neural network (8 lattice, all connectivity, no weight sharing, all voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, all connectivity, no weight sharing, all voting)'] = [ 59.15, 64.24, 67.65, 68.57, 73.76, 76.25, 76.62, 80.5, 80.85, 79.75, 68.64 ]

x_axes['Convolution spiking neural network (8 lattice, all connectivity, no weight sharing, most-spiked voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, all connectivity, no weight sharing, most-spiked voting)'] = [ 58.83, 62.36, 67.46, 68.58, 73.29, 75.85, 76.12, 80.01, 77.99, 76.42, 70.83 ]

x_axes['Convolution spiking neural network (8 lattice, all connectivity, no weight sharing, top percent voting)'] = [ (27 ** 2) * 4 * 50, (26 ** 2) * 9 * 50 ] + [ 50 * (conv_size ** 2) * (((28 - conv_size) / 2) ** 2) for conv_size in xrange(24, 6, -2) ]
performances['Convolution spiking neural network (8 lattice, all connectivity, no weight sharing, top percent voting)'] = [ 57.98, 60.54, 67.72, 66.6, 70.72, 73.75, 73.64, 78.61, 77.46, 74.11, 69.43 ]


plots = []
for comparator in performances.keys():
	print sorted(x_axes[comparator])
	plots.append(plt.semilogx(sorted(x_axes[comparator]), [ performance for (x_value, performance) in sorted(zip(x_axes[comparator], performances[comparator])) ], label=comparator)[0])

fig = plt.gcf()
fig.set_size_inches(18.5, 12)

plt.legend(handles=plots, fontsize='x-small')
plt.xlim(0, 784 * 7000)
plt.title('Classification performance by number of excitatory, inhibitory neurons')
plt.xlabel('Number of network weights (input -> excitatory layer)')
plt.ylabel('Test dataset classification accuracy')

plt.show()

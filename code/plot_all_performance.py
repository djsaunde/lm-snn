import matplotlib.pyplot as plt

fifty_features_y = [ 66.18, 68.75, 65.23, 64.04, 61.18, 58.43, 56.08, 54.84, 51.47, 43.14, 43.25, 40.90, 39.09, 35.64, 23.08, 31.35, 31.9, 30.52, 21.08 ]
one_hundred_features_y = [ 78.16, 82.53, 75.54, 67.86, 63.53, 58.95, 55.43, 51.72, 46.78, 41.34, 24.75, 35.7, 31.35, 28.29, 23.26, 22.39, 24.07, 26.01, 0 ]
x = range(28, 9, -1)

fifty_plot, = plt.plot(x, fifty_features_y, label='50 convolution patches')
one_hundred_plot, = plt.plot(x, one_hundred_features_y, label='100 convolution patches')

fig = plt.gcf()
fig.set_size_inches(16, 12)

plt.legend(handles=[fifty_plot, one_hundred_plot])
plt.xlim(29, 10)
plt.title('Classification performance by convolution window size, number of convolution windows')
plt.xlabel('Convolution window side length')
plt.ylabel('Test dataset classification accuracy')

plt.show()

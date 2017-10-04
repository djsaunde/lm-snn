import numpy as np
import matplotlib.pyplot as plt


def increasing(x_i, inhib_const=5.0, max_inhib=17.5):
	if np.abs(x_i) <= 1.0:
		return 0
	else:
		return -np.minimum(max_inhib, inhib_const * np.sqrt(np.abs(x_i)))

x = np.linspace(-15, 15, 10000)

plt.rc('text', usetex=True)

for inhib_const in [ 0.0, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5 ]:
	y = [ increasing(x_i, inhib_const=inhib_const) for x_i in x ]
	plt.plot(x, y, 'r');

plt.ylabel('Inhibition strength')
plt.ylim([-20, 1])
plt.title('Increasing inhibition')
plt.xlabel('Euclidean distance from spiking neuron')
plt.show()
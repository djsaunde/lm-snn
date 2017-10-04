import numpy as np
import matplotlib.pyplot as plt


def increasing(x_i, max_inhib=17.5):
	if np.abs(x_i) <= 1.0:
		return 0
	else:
		return -np.minimum(max_inhib, 4.8*np.sqrt(np.abs(x_i)))

x = np.linspace(-15, 15, 10000)
y = [ increasing(x_i) for x_i in x ]

plt.plot(x, y, 'k', label='Inhibition'); plt.plot(x, [ 0.0 ] * len(x), 'k--', label='No inhibition');
plt.plot(x, [ -17.5 ] * len(x), color='k', linestyle='-.', label='Max inhibition')
plt.axvline(-1.0, color='k', linestyle=':'); plt.axvline(1.0, color='k', linestyle=':'); plt.ylabel('Inhibition strength')
plt.legend(); plt.ylim([-20, 1]); plt.title('Increasing inhibition'); plt.xlabel('Euclidean distance from spiking neuron')
plt.show()
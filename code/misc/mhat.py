import argparse
import numpy as np
import matplotlib.pyplot as plt

def mhat(t, sigma=1.0, scale=1.0, shift=0.0, max_excite=np.inf, max_inhib=-np.inf):
	'''
	Truly, this is the Ricker wavelet, which is the negative normalized second derivative
	of a Gaussian function; i.e., up to scale and normalization, the second Hermite function.
	It is frequently employed to model seismic data, or is used as a broad spectrum source
	term in computational electrodynamics.

	It is only referred to as the Mexican hat wavelet in the Americas due to its taking the
	shape of a sombrero when used as a 2D image processing kernel.

	See https://en.wikipedia.org/wiki/Mexican_hat_wavelet for more details and references.
	'''

	if np.abs(t) <= 1.0:
		return 0.0
	else:
		return np.maximum(np.minimum(scale * np.divide(2, np.sqrt(3 * sigma) * (np.pi ** 0.25)) * (1.0 - np.square(np.divide(t, sigma))) * np.exp(-np.divide(np.square(t), 2 * np.square(sigma))) + shift, max_excite), max_inhib)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--sigma', type=float, default=1.0)
	parser.add_argument('--scale', type=float, default=1.0)
	parser.add_argument('--shift', type=float, default=0.0)
	parser.add_argument('--max_excite', type=float, default=np.inf)
	parser.add_argument('--max_inhib', type=float, default=-np.inf)
	args = parser.parse_args()
	
	sigma, scale, shift, max_excite, max_inhib = args.sigma, args.scale, args.shift, args.max_excite, args.max_inhib

	x = np.linspace(-10, 10, 10000)
	y = np.array([ mhat(x_i, sigma=sigma, scale=scale, shift=shift, max_excite=max_excite, max_inhib=max_inhib) for x_i in x ])

	plt.plot(x, y, 'r'); plt.axhline(0, color='k', linestyle='-.'); plt.axvline(-1, color='b', linestyle=':')
	plt.axvline(1, color='b', linestyle=':'); plt.title('Mexican hat wavelet'); plt.show()

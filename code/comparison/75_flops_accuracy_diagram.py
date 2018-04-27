import os
import numpy as np
import matplotlib.pyplot as plt

sizes = [225, 400, 625]
eth = [8000, 11500, 16000]
lmsnn = [6500, 6250, 6000]

plt.plot(sizes, eth, label='ETH')
plt.plot(sizes, lmsnn, label='LM-SNN')

plt.xlabel('Network size (no. neurons)')
plt.ylabel('Training samples required')
plt.title('ETH vs. LM-SNN: 75% accuracy sample requirements')

plt.legend(); plt.grid()
plt.savefig('../../plots/75_percent_samples_needed.png')

plt.show(); plt.clf()

def per_sample_flops(size, input_size=28 ** 2, time=350):
	return time * (3 * size + 2 * input_size * size + (size - 1) ** 2)

eth_flops = [samples * per_sample_flops(size) for (size, samples) in zip(sizes, eth)]
lmsnn_flops = [samples * per_sample_flops(size) for (size, samples) in zip(sizes, lmsnn)]

plt.semilogy(sizes, eth_flops, label='ETH')
plt.semilogy(sizes, lmsnn_flops, label='LM-SNN')

plt.xlabel('Network size (no. neurons)')
plt.ylabel('Floating point operations required')
plt.title('ETH vs. LM-SNN: 75% accuracy FLOPs requirements')

plt.legend(); plt.grid()
plt.savefig('../../plots/75_percent_flops_needed.png')

plt.show(); plt.clf()

nn_curves = {}
kmeans_curves = {}
for fname in os.listdir('.'):
	if '.npy' in fname:
		curve = np.load(fname)
		if 'kmeans' in fname and ('225' in fname or '400' in fname or '625' in fname):
			n_clusters = int(fname.split('.')[0].split('_')[1])
			kmeans_curves[n_hidden] = np.array([0] + list(curve))

	if '.npy' in fname:
		curve = np.load(fname)
		if 'nn' in fname and ('225' in fname or '400' in fname or '625' in fname):
			n_hidden = int(fname.split('.')[0].split('_')[1])
			nn_curves[n_hidden] = np.array([0] + list(curve))

def kmeans_per_sample_flops(size, input_size=28 ** 2):
	return 0

def nn_per_sample_flops(size, input_size=28 ** 2):
	return 2 * ((input_size + 1) * size + size + (size + 1) * 10)
			
sizes = [225, 400, 625]
	
kmeans = [np.argmax(curve > 75) * 250 for curve in kmeans_curves.values()]
nns = [np.argmax(curve > 75) * 250 for curve in nn_curves.values()]

plt.plot(sizes, eth, label='ETH')
plt.plot(sizes, lmsnn, label='LM-SNN')
plt.plot(sizes, nns, label='2-layer NN')

plt.xlabel('Network size (no. neurons)')
plt.ylabel('Training samples required')
plt.title('ETH vs. LM-SNN: 75% accuracy sample requirements')

plt.legend(); plt.grid()
plt.savefig('../../plots/75_percent_samples_needed_nn_kmeans.png')

plt.show(); plt.clf()

kmeans_flops = [samples * kmeans_per_sample_flops(size) for (size, samples) in zip(sizes, kmeans)]
nn_flops = [samples * nn_per_sample_flops(size) for (size, samples) in zip(sizes, nns)]

plt.semilogy(sizes, eth_flops, label='ETH')
plt.semilogy(sizes, lmsnn_flops, label='LM-SNN')
plt.semilogy(sizes, nn_flops, label='2-layer NN')

plt.xlabel('Network size (no. neurons)')
plt.ylabel('Floating point operations required')
plt.title('ETH vs. LM-SNN: 75% accuracy FLOPs requirements')

plt.grid(); plt.legend()
plt.savefig('../../plots/75_percent_flops_needed_nn_kmeans.png')
plt.show()

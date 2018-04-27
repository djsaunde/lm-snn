import os
import numpy as np
import matplotlib.pyplot as plt

sizes = [225, 400, 625]
accuracies = [70, 75, 80, 85]

eth = [[7000, 11000, 15500],
	   [8000, 11500, 16000],
	   [10500, 12500, 18500],
	   [60000, 36500, 35000]]

lmsnn= [[2400, 2600, 3750],
		[6500, 6250, 6000],
		[7000, 7500, 6500],
		[9000, 8750, 8500]]

def per_sample_flops(size, input_size=28 ** 2, time=350):
	return time * (3 * size + 2 * input_size * size + (size - 1) ** 2)

for i in range(len(accuracies)):
	for j, size in enumerate(sizes):
		eth[i][j] = eth[i][j] * per_sample_flops(size)
		lmsnn[i][j] = lmsnn[i][j] * per_sample_flops(size)

ratios = [[eth[i][j] / lmsnn[i][j] for j in range(len(sizes))] for i in range(len(accuracies))]

for line, acc in zip(ratios, accuracies):
	plt.plot(sizes, line, label='%d%% accuracy' % acc)

plt.xlabel('Network size (no. neurons)')
plt.ylabel('(LM-SNN / ETH SNN) FLOPs ratio')
plt.title('ETH SNN vs. LM-SNN - FLOPs requirements ratio')

plt.legend(); plt.grid()
plt.savefig('../../plots/flops_ratio.png')

plt.show()
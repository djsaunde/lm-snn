import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

model_name = 'csnn_pc_inhibit_far'

confusion_histogram_dir = os.path.join('..', '..', 'confusion_histograms', model_name)
plots_path = os.path.join('..', '..', 'plots', model_name)
correct_trials_path = os.path.join('..', '..', 'correct_trials', model_name)
incorrect_trials_path = os.path.join('..', '..', 'incorrect_trials', model_name)

filenames = os.listdir(confusion_histogram_dir)

for d in [ plots_path, correct_trials_path, incorrect_trials_path ]:
	if not os.path.isdir(d):
		os.makedirs(d)

print '\n'
print '\n'.join([ str(idx + 1) + ' - ' + filename for idx, filename in enumerate(filenames) ])
print '\n'

model_idx = int(raw_input('\nEnter the index of the model to confusion matrix and histograms for: ')) - 1
ending = '_'.join(filenames[model_idx].split('_')[1:-1])

n_e_total = int(ending.split('_')[2])

print '\n'

confusion_histogram = np.load(os.path.join(confusion_histogram_dir, filenames[model_idx]))

print 'Model accuray on the test dataset:', np.sum(confusion_histogram[:, 0] == \
				confusion_histogram[:, 2]) / float(np.shape(confusion_histogram)[0]), '\n'

conf_matrix = confusion_matrix(confusion_histogram[:, 0], confusion_histogram[:, 2])
conf_matrix = normalize(conf_matrix, axis=0)

plt.matshow(conf_matrix)
plt.xticks(xrange(10))
plt.yticks(xrange(10))
plt.colorbar()
plt.title('Confusion matrix')
plt.show()

# to_plot = np.bincount(confusion_histogram[confusion_histogram[:, 0] == \
# 				confusion_histogram[:, 2], 1].astype(np.int64)).astype(np.float64) / \
# 				np.bincount(confusion_histogram[:, 1].astype(np.int64)).astype(np.float64)

# to_plot[~np.isfinite(to_plot)] = 1.0

# plt.bar(xrange(n_e_total), to_plot)
# plt.title('Percent correct out of all most-spiked trials (un-fired neurons set to 100%)')
# plt.savefig(os.path.join(plots_path, '_'.join(['percent_correct_most_spiked', ending]) + '.png'))
# plt.show()

# to_plot = np.bincount(confusion_histogram[confusion_histogram[:, 0] != \
# 				confusion_histogram[:, 2], 1].astype(np.int64)).astype(np.float64) / \
# 				np.bincount(confusion_histogram[:, 1].astype(np.int64)).astype(np.float64)

# to_plot[~np.isfinite(to_plot)] = 0.0

# plt.bar(xrange(n_e_total), to_plot)
# plt.title('Percent incorrect out of all most-spiked trials (un-fired neurons set to 0%)')
# plt.savefig(os.path.join(plots_path, '_'.join(['percent_incorrect_most_spiked', ending]) + '.png'))
# plt.show()

# to_plot = np.bincount(confusion_histogram[confusion_histogram[:, 0] == \
# 				confusion_histogram[:, 2], 3].astype(np.int64))

# plt.bar(xrange(len(to_plot)), to_plot)
# plt.title('Max spikes on correctly classified inputs')
# plt.savefig(os.path.join(plots_path, '_'.join(['max_spikes_correct_inputs', ending]) + '.png'))

# plt.show()

# to_plot = np.bincount(confusion_histogram[confusion_histogram[:, 0] != \
# 				confusion_histogram[:, 2], 3].astype(np.int64))

# plt.bar(xrange(len(to_plot)), to_plot)
# plt.title('Max spikes on incorrectly classified inputs')
# plt.savefig(os.path.join(plots_path, '_'.join(['max_spikes_incorrect_inputs', ending]) + '.png'))
# plt.show()

# plt.hist(confusion_histogram[confusion_histogram[:, 0] == confusion_histogram[:, 2], 1], bins=np.shape(confusion_histogram)[0])
# plt.title('Times most-spiked neuron on correct trials')
# plt.savefig(os.path.join(plots_path, '_'.join(['times_most_spiked_correct', ending]) + '.png'))
# plt.show()

array = np.zeros(400)
array[confusion_histogram[confusion_histogram[:, 0] == confusion_histogram[:, 2], 1].astype(np.int64)] = \
			confusion_histogram[confusion_histogram[:, 0] == confusion_histogram[:, 2]]

np.save(os.path.join(correct_trials_path, 'correct_trials.npy'), array)

plt.matshow(array.reshape((20, 20)))
plt.colorbar()
plt.title('Times most-spiked neuron on correct trials')
plt.savefig(os.path.join(plots_path, '_'.join(['times_most_spiked_correct', ending]) + '.png'))

# plt.hist(confusion_histogram[confusion_histogram[:, 0] != confusion_histogram[:, 2], 1], bins=np.shape(confusion_histogram)[0])
# plt.title('Times most-spiked neuron on incorrect trials')
# plt.savefig(os.path.join(plots_path, '_'.join(['times_most_spiked_incorrect', ending]) + '.png'))
# plt.show()

array = np.zeros(400)
array[confusion_histogram[confusion_histogram[:, 0] != confusion_histogram[:, 2], 1].astype(np.int64)] = \
			confusion_histogram[confusion_histogram[:, 0] != confusion_histogram[:, 2]]

np.save(os.path.join(incorrect_trials_path, 'incorrect_trials.npy'), array)

plt.matshow(array.reshape((20, 20)))
plt.colorbar()
plt.title('Times most-spiked neuron on incorrect trials')
plt.savefig(os.path.join(plots_path, '_'.join(['times_most_spiked_incorrect', ending]) + '.png'))
plt.show()
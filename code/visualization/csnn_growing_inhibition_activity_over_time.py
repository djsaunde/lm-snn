import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--n_files', type=int, default=-1)
args = parser.parse_args()
n_files = args.n_files

model_name = 'csnn_growing_inhibition'

top_level_path = os.path.join('..', '..')
spikes_dir = os.path.join(top_level_path, 'spikes', model_name)

files = sorted([ file for file in os.listdir(spikes_dir) if '_0.npy' in file and 'spike_counts' in file ])

print '\n'
print '\n'.join([ str(idx + 1) + ' - ' + file[:file.index('_spike_counts')] for idx, file in enumerate(files) ])
print '\n'

model_idx = int(raw_input('Enter the index of the model to plot activity over time for: ')) - 1
model = files[model_idx][:file.index('_spike_counts')]

n_input_sqrt = 28
conv_features = int(model.split('_')[2])
conv_features_sqrt = int(np.sqrt(conv_features))

activity_files = sorted([ file for file in os.listdir(spikes_dir) \
			if 'spike_counts' in file and model in file ], key=lambda x: int(x.split('_')[-1][:-4]))
rates_files = sorted([ file for file in os.listdir(spikes_dir) \
			if 'rates' in file and model in file ], key=lambda x: int(x.split('_')[-1][:-4]))

FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=5)

fig, (ax1, ax2) = plt.subplots(1, 2)
image1 = ax1.imshow(np.zeros([ conv_features_sqrt, conv_features_sqrt ]), cmap='binary', vmin=0, vmax=1, interpolation='nearest')
image2 = ax2.imshow(np.zeros([ n_input_sqrt, n_input_sqrt ]), cmap='binary', vmin=0, vmax=64, interpolation='nearest')

fig.colorbar(image1, ax=ax1)
fig.colorbar(image2, ax=ax2)

with writer.saving(fig, os.path.join('movies', '_'.join([ model, str(n_files) ]) + '.mp4'), len(activity_files)):
	if n_files == -1:
		for activity_file, rates_file in tqdm(zip(activity_files, rates_files)):
			activity = np.load(os.path.join(spikes_dir, activity_file))
			activity = activity / float(np.sum(activity))

			rates = np.load(os.path.join(spikes_dir, rates_file))

			image1.set_data(activity.reshape([conv_features_sqrt, conv_features_sqrt]))
			image2.set_data(rates)
			writer.grab_frame()
	else:
		for activity_file, rates_file in tqdm(zip(activity_files[:n_files], rates_files[:n_files])):
			activity = np.load(os.path.join(spikes_dir, activity_file))
			activity = activity / float(np.sum(activity))

			rates = np.load(os.path.join(spikes_dir, rates_file))

			image1.set_data(activity.reshape([conv_features_sqrt, conv_features_sqrt]))
			image2.set_data(rates)
			writer.grab_frame()
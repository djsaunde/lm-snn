import os
import numpy as np
import pandas as pd

top_level_path = '../../'
model_name = 'csnn_pc_inhibit_far'
results_path = os.path.join(top_level_path, 'results', model_name, 'results.csv')

results = pd.read_csv(results_path)
results['random seed'] = pd.Series([entry.split('_')[22] for entry in results['Model']])
results['inhibition'] = pd.Series([entry.split('_')[19] for entry in results['Model']])
results['strengthen'] = pd.Series([entry.split('_')[20] for entry in results['Model']])

print results.columns.values

del results['most_spiked_location']
del results['most_spiked_patch']
del results['top_percent']
del results['most-spiked (overall)']
del results['most-spiked (per patch)']

del results['Model']

averaged_results = results.groupby(['inhibition', 'strengthen']).mean()

print averaged_results.to_latex(float_format='%.2f')

import os
import numpy as np
import pandas as pd

top_level_path = '../../'
model_name = 'csnn_growing_inhibition'
results_path = os.path.join(top_level_path, 'results', model_name, 'results.csv')

results = pd.read_csv(results_path)

results['random seed'] = pd.Series([entry.split('_')[5] for entry in results['Model']])

del results['most_spiked_patch']

results = results[results['Model'].str.contains('400')]
results = results[results['Model'].str.contains('False_True')]

averaged_results = results.groupby(['random seed']).mean()

print averaged_results.to_latex(float_format='%.2f')

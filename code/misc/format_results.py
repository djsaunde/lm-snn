import os
import numpy as np
import pandas as pd

top_level_path = '../../'
results_path = os.path.join(top_level_path, 'docs', 'results.csv')

results = pd.read_csv(results_path)
results['random seed'] = pd.Series([entry.split('_')[22] for entry in results['Model']])
results['inhibition'] = pd.Series([entry.split('_')[19] for entry in results['Model']])
results['strengthen'] = pd.Series([entry.split('_')[20] for entry in results['Model']])

del results['Model']

averaged_results = results.groupby(['inhibition', 'strengthen']).mean()

print averaged_results.to_latex(float_format='%.2f')
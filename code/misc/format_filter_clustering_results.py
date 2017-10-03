import os
import numpy as np
import pandas as pd

top_level_path = '../../'
model_name = 'csnn_pc_inhibit_far_cluster_filters'
results_path = os.path.join(top_level_path, 'results', model_name, 'results.csv')

results = pd.read_csv(results_path)

results['random seed'] = pd.Series([entry.split('_')[22] for entry in results['Model']], dtype=int)
results['inhibition'] = pd.Series([entry.split('_')[19] for entry in results['Model']], dtype=float)
results['strengthen'] = pd.Series([entry.split('_')[20] for entry in results['Model']], dtype=float)
results['clusters'] = pd.Series([entry.split('_')[25] for entry in results['Model']], dtype=int)

print results.columns.values

results = results.drop_duplicates(['Model'])

del results['Model']

results = results[results['strengthen'] == 0.0]

averaged_results = results.groupby(['inhibition', 'clusters']).mean()

del averaged_results['random seed']
del averaged_results['strengthen']

print averaged_results.to_latex(float_format='%.2f')

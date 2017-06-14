'''
Script which parses .csv file of all results into desired LaTeX table. Abandoned (for now) in favor of Jupyter notebook
entitled "Format Results.ipynb".
'''

import pandas as pd

top_level_path = '../../'

results = pd.read_csv(top_level_path + 'data/all_accuracy_results.csv', index_col=0, delimiter=',', encoding="utf-8-sig", header=0)

print results

# results = results.set_index('Network')

print results.columns
print results.index

# for ending in results.index.values:
# 	print results.get(ending)

ending_mapping = pd.DataFrame.from_dict({ results[idx] : accuracy for idx, accuracy in enumerate(results['All']) })
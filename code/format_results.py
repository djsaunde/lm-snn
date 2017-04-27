'''
Script which parses .csv file of all results into desired LaTeX table.

@author: Dan Saunders (djsaunde.github.io)
'''

import pandas as pd


results = pd.read_csv('../data/all_accuracy_results.csv', index_col=0, delimiter=',', encoding="utf-8-sig", usecols=[0, 1, 2, 3, 4])

print results.columns
print results.index

for ending in results.index.values:
	print results.get(ending)

ending_mapping = pd.DataFrame.from_dict({ results[idx] : accuracy for idx, accuracy in enumerate(results['All']) })
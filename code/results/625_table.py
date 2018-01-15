import os
import pandas as pd

df = pd.read_csv('625_results.csv')

column_names = ['Kernel Size', 'Stride', 'Neurons', 'Filter Size', '# Train',
			'Random Seed', 'Proportion Low', 'Low Inhibition', 'High Inhibition']
df[column_names] = df['Model'].apply(lambda x: pd.Series([item for item in x.split('_')]))

df = df.drop(['Kernel Size', 'Stride', 'Neurons', 'Filter Size', 'all_active', 'most_spiked_patch',
				'most_spiked_location', 'most_spiked_neighborhood', 'activity_neighborhood'], axis=1)

df = df[df['# Train'] == '60000']
df = df[df['Proportion Low'] != '1.0']

temp = df.groupby(['Proportion Low', 'Low Inhibition', 'High Inhibition']).mean()
temp[['std1', 'std2', 'std3']] = df.groupby(['Proportion Low', 'Low Inhibition', 'High Inhibition']).std()

df = temp

df = df.round(decimals=2)

df['all'] = df['all'].map(str) + r' $\pm$ ' + df['std1'].map(str)
df['confidence_weighting'] = df['confidence_weighting'].map(str) + r' $\pm$ ' + df['std3'].map(str)
df['distance'] = df['distance'].map(str) + r' $\pm$ ' + df['std3'].map(str)

df = df.drop(['std1', 'std2', 'std3'], axis=1)

index = [r'$p_{\textrm{low}}$', r'$c_{\textrm{min}}$', r'$c_{\textrm{max}}$']
df.index = df.index.set_names(index)

columns = {'all' : r'\textit{all}', 'confidence_weighting' : r'\textit{confidence}', 'distance' : r'\textit{distance}'}
df = df.rename(index=str, columns=columns)

df = pd.DataFrame(df.to_records())

print df.to_latex(escape=False, column_format='||c|c|c|c|c|c||', multicolumn=False, index=False)
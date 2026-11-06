import pandas as pd

df = pd.DataFrame({ 'DS' : [40,43,38], 'TOC': [45, 48, 42], 'CD': [50, 35, 48] })

corr_matrix = df.corr(method='pearson')
print("Pearson Correlation Matrix:\n", corr_matrix)


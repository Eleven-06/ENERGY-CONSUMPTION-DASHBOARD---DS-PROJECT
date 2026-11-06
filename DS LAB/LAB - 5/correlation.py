import pandas as pd

df = pd.DataFrame({ 'X' : [10, 20, 30, 40, 50], 'Y' : [12, 24, 33, 45, 60] })

corr_matrix = df.corr(method='pearson')
print("Pearson Correlation Matrix:\n", corr_matrix)


import pandas as pd

df = pd.read_csv('Lab 5/Iris.csv')

corr_matrix = df.corr(method='pearson', numeric_only=True)

print("Pearson Correlation Matrix:\n", corr_matrix)
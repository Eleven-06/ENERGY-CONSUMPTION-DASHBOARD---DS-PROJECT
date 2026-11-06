import pandas as pd
import os

file_path = os.path.join(os.path.dirname(__file__), 'Iris.csv')

df = pd.read_csv(file_path)

print("Spearman Correlation Matrix:\n")
print(df.corr(method='spearman', numeric_only=True))
import pandas as pd
df = pd.read_csv('Lab 3/Iris.csv')
print(df.describe())
df.to_csv('Iris2.csv', index=False)
print(df.describe(include='all'))
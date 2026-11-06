import pandas as pd
data = { 'apples': [3, 2, 0, 1], 'oranges': [0, 3, 7, 2] }
df = pd.DataFrame(data)
df = pd.DataFrame(data, index=['June', 'Robert', 'Lily', 'David'])
print(df)
print(df.loc['June'])

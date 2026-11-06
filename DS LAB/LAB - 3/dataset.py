import pandas as pd
data = { 'col_1' : [3, 2, 0, 1], 'col_2' : [0, 3, 7, 2] }
df = pd.DataFrame.from_dict(data)
print(df)

data = {
    'row_1': [3, 2, 0, 1],
    'row_2': ['a', 'b', 'c', 'd']
}

df2 = pd.DataFrame.from_dict(data, orient='index')

print(df2)
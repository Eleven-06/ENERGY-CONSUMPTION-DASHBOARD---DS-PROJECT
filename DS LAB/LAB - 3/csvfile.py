import pandas as pd
df = pd.read_csv('Lab 3/Iris.csv')
print(df.head(5))
print(df.tail(5))
print(df.info())
print(df.shape)
dup_df=pd.concat([df,df])
dup_df.shape
print(dup_df)
dup_df.drop_duplicates(inplace=True)
print(dup_df)
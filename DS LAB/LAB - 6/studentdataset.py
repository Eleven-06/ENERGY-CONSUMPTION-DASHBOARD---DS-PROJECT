import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")

df['age'] = df['age'].fillna(df['age'].mean())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

df.drop_duplicates(inplace=True)

df = pd.get_dummies(df, columns=['sex', 'class', 'embarked'], drop_first=True)

df['family_size'] = df['sibsp'] + df['parch'] + 1

print(df.head())
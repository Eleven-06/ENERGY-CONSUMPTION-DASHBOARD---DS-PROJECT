import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("titanic")

sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

df = sns.load_dataset("tips")

sns.histplot(df['total_bill'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

import pandas as pd
from scipy.stats import spearmanr

df = pd.DataFrame({ 'X' : [10, 20, 30, 40, 50], 'Y' : [12, 18, 33, 47, 55] })

corr_value, p_value = spearmanr(df['X'], df['Y'])
print(f"Spearman Correlation Coefficient: {corr_value}")
print(f"P-Value: {p_value}")
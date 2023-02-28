import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Churn_Modelling.csv', index_col = 'CustomerId')
"""# **Model Evaluation and Visualization**"""

df['HasCrCard'] = df['HasCrCard'].astype('category')
df['IsActiveMember'] = df['IsActiveMember'].astype('category')
df['Exited'] = df['Exited'].astype('category')
print(df)


df = df.drop('Balance', axis=1)



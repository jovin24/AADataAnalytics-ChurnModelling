import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Churn_Modelling.csv', index_col = 'CustomerId')
"""# **Churn Modelling Dataset**"""

df['HasCrCard'] = df['HasCrCard'].astype('category')
df['IsActiveMember'] = df['IsActiveMember'].astype('category')
df['Exited'] = df['Exited'].astype('category')

df = df.drop('Balance', axis=1)

import scipy.stats as stats

Z_score_CreditScore = stats.zscore(df['CreditScore'])

len(df[(Z_score_CreditScore<-3) | (Z_score_CreditScore>3)])

df1= df[(Z_score_CreditScore>-3) & (Z_score_CreditScore<3)].reset_index()


# Standardization for CreditScore
avg_CreditScore = df1['CreditScore'].mean()
std_CreditScore = df1['CreditScore'].std()

df1['Z_Score_CreditScore'] = (df1['CreditScore'] - avg_CreditScore)/std_CreditScore
zscore_rate=stats.zscore(df1['CreditScore'])

df1


from sklearn.preprocessing import StandardScaler
"The skewness for the original data is {}.".format(df1.CreditScore.skew())

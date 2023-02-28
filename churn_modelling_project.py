import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


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


from sklearn.model_selection import train_test_split
Y = df1['Exited']
X = df1.drop(['Exited', 'CustomerId' , 'EstimatedSalary', 'CreditScore'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.8, random_state =0)

st.write("The shape of X_train is:", X_train.shape)
st.write("The shape of X_test is:", X_test.shape)

st.write('')
st.write("The shape of Y_train is:", y_train.shape)
st.write("The shape of Y_test is:", y_test.shape)

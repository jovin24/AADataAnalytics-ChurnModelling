import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Churn_Modelling.csv', index_col = 'CustomerId')

df['HasCrCard'] = df['HasCrCard'].astype('category')
df['IsActiveMember'] = df['IsActiveMember'].astype('category')
df['Exited'] = df['Exited'].astype('category')


df = df.drop('Balance', axis=1)


#Calculate Z Score to treat Outliers
import scipy.stats as stats

Z_score_CreditScore = stats.zscore(df['CreditScore'])

len(df[(Z_score_CreditScore<-3) | (Z_score_CreditScore>3)])

df1= df[(Z_score_CreditScore>-3) & (Z_score_CreditScore<3)].reset_index()
print(df1)

# Standardization for CreditScore
avg_CreditScore = df1['CreditScore'].mean()
std_CreditScore = df1['CreditScore'].std()

df1['Z_Score_CreditScore'] = (df1['CreditScore'] - avg_CreditScore)/std_CreditScore

import scipy.stats as s
zscore_rate=s.zscore(df1['CreditScore'])


min_EstimatedSalary = df1.EstimatedSalary.min()
max_EstimatedSalary = df1.EstimatedSalary.max()


"""# **Data Splitting into train and test**"""

from sklearn.model_selection import train_test_split

Y = df1['Exited']
X = df1.drop(['Exited', 'CustomerId' , 'EstimatedSalary', 'CreditScore'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.8, random_state =0)

print("The shape of X_train is:", X_train.shape)
print("The shape of X_test is:", X_test.shape)

print('')
print("The shape of Y_train is:", y_train.shape)
print("The shape of Y_test is:", y_test.shape)

#Logistic Regression Method
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train, y_train)
y_pred_LR = lm.predict(X_test)


"""# **Model Evaluation and Visualization**"""

#Logistic Regression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred_LR)

print('Logistic Regression')
print('Confusion Matrix:\n' , confusion_matrix)

print('\nAccuracy:', accuracy_score(y_test, y_pred_LR))
print('Precision:', precision_score(y_test, y_pred_LR))
print('Recall:', recall_score(y_test, y_pred_LR))
print('F1-score:', f1_score(y_test, y_pred_LR))

"""#**Step 10: Creating WebApp using Streamlit**"""

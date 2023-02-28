import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Churn_Modelling.csv', index_col = 'CustomerId')

df['HasCrCard'] = df['HasCrCard'].astype('category')
df['IsActiveMember'] = df['IsActiveMember'].astype('category')
df['Exited'] = df['Exited'].astype('category')


df = df.drop('Balance', axis=1)

print(df)


rom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred_LR)

print('Logistic Regression')
print('Confusion Matrix:\n' , confusion_matrix)

print('\nAccuracy:', accuracy_score(y_test, y_pred_LR))
print('Precision:', precision_score(y_test, y_pred_LR))
print('Recall:', recall_score(y_test, y_pred_LR))
print('F1-score:', f1_score(y_test, y_pred_LR))
max_EstimatedSalary = df1.EstimatedSalary.max()

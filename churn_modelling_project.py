import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Churn_Modelling.csv')


df['HasCrCard'] = df['HasCrCard'].astype('category')
df['IsActiveMember'] = df['IsActiveMember'].astype('category')
df['Exited'] = df['Exited'].astype('category')

# Checking is there any null values mean for each columns
print("Missing values distribution: ")
print(df.isnull().mean())
print("")

# Checking is there any null values sum for each columns
df.isnull().sum()

#Check what are the rows that have '0' values
zero_rows = df[df['Balance'] == 0]
print(zero_rows)

#check the total rows for 'Balance' that has '0' value
zero_rows.shape

#Drop the column 'Balance' due to too many '0' values
df = df.drop('Balance', axis=1)

#Outliers Treatment
#CreditScore
df['CreditScore'].plot(kind='box')
plt.show()

#Age
df['Age'].plot(kind='box')
plt.show()

#Tenure
df['Tenure'].plot(kind='box')
plt.show()

#NumOfProducts
df['NumOfProducts'].plot(kind='box')
plt.show()

#EstimatedSalary
df['EstimatedSalary'].plot(kind='box')
plt.show()

#Calculate Z Score to treat Outliers
import scipy.stats as stats

Z_score_CreditScore = stats.zscore(df['CreditScore'])

len(df[(Z_score_CreditScore<-3) | (Z_score_CreditScore>3)])

#Cleaned Data: without outliers so z>-3 and z< +3

df1= df[(Z_score_CreditScore>-3) & (Z_score_CreditScore<3)].reset_index()
print(df1)

# Standardization for CreditScore
avg_CreditScore = df1['CreditScore'].mean()
avg_CreditScore

std_CreditScore = df1['CreditScore'].std()
std_CreditScore

# Step 1 : transform using Z-score
df1['Z_Score_CreditScore'] = (df1['CreditScore'] - avg_CreditScore)/std_CreditScore

df1.head()

import scipy.stats as s
zscore_rate=s.zscore(df1['CreditScore'])
zscore_rate


#Normalization: Min Max Scalar
# For EstimatedSalary:
min_EstimatedSalary = df1.EstimatedSalary.min()
max_EstimatedSalary = df1.EstimatedSalary.max()

print('Min:' , min_EstimatedSalary)
print('Max:' , max_EstimatedSalary)

df1['Min_Max_EstimatedSalary'] = (df1['EstimatedSalary'] - min_EstimatedSalary)/ (max_EstimatedSalary - min_EstimatedSalary)

# checking if the skewness and kurtosis post scaling or not:


df1

#Factorize the data
for i in df1.columns:
    if df1[i].dtypes=='object':
        df1[i] = pd.Categorical(pd.factorize(df1[i])[0])

df1

df1.info()

"""# **Step 6: Data Splitting into train and test**"""

from sklearn.model_selection import train_test_split

## Splitting for X and Y variables:

Y = df1['Exited']
X = df1.drop(['Exited', 'CustomerId' , 'EstimatedSalary', 'CreditScore'], axis=1)

# Independent Variable

X.head()

# Dependent or Target Variable

Y.head()

## Splitting dataset into 80% Training and 20% Testing Data:

X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.8, random_state =0)

# random_state ---> is seed -- fixing the sample selection for Training & Testing dataset

# check the dimensions of the train & test subset for 

print("The shape of X_train is:", X_train.shape)
print("The shape of X_test is:", X_test.shape)

print('')
print("The shape of Y_train is:", y_train.shape)
print("The shape of Y_test is:", y_test.shape)

"""# **Step 7: Model Building**


"""

#Logistic Regression Method
from sklearn.linear_model import LogisticRegression

lm = LogisticRegression()
lm.fit(X_train, y_train)

"""# **Step 8: Model Validation**"""

y_pred_LR = lm.predict(X_test)
y_pred_LR

"""# **Step 9:Model Evaluation and Visualization**"""

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


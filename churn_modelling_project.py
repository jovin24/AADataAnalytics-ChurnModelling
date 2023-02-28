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

from sklearn.preprocessing import StandardScaler



print("The skewness for the original data is {}.".format(df1.CreditScore.skew()))
print("The kurtosis for the original data is {}.".format(df1.CreditScore.kurt()))

print('')

print("The skewness for the Zscore Scaled column is {}.".format(df1.Z_Score_CreditScore.skew()))
print("The kurtosis for the Zscore Scaled columns is {}.".format(df1.Z_Score_CreditScore.kurt()))


fig, axes = plt.subplots(2, figsize=(15,8))

sns.distplot(df1['CreditScore'], ax=axes[0])
sns.distplot(df1['Z_Score_CreditScore'], ax=axes[1])


plt.show()


min_EstimatedSalary = df1.EstimatedSalary.min()
max_EstimatedSalary = df1.EstimatedSalary.max()




print("The skewness for the original data is {}.".format(df1.EstimatedSalary.skew()))
print("The skewness for the Zscore Scaled column is {}.".format(df1.Z_Score_CreditScore.skew()))
print("The skewness for the Min Max Scaled Data is {}.".format(df1.Min_Max_EstimatedSalary.skew()))


print('')

print("The kurtosis for the original data is {}.".format(df1.EstimatedSalary.kurt()))
print("The kurtosis for the Zscore Scaled columns is {}.".format(df1.Z_Score_CreditScore.kurt()))
print("The kurtosis for the Min Max Scaled Data is {}.".format(df1.Min_Max_EstimatedSalary.kurt()))

# Distribution of the columns

# For CreditScore
# EstimatedSalary
fig, axes = plt.subplots(1,3, figsize=(15,5))

sns.distplot(df1['EstimatedSalary'], ax=axes[0])
sns.distplot(df1['Z_Score_CreditScore'], ax=axes[1])
sns.distplot(df1['Min_Max_EstimatedSalary'], ax=axes[2])

plt.tight_layout()
plt.show()

#Factorize the data
for i in df1.columns:
    if df1[i].dtypes=='object':
        df1[i] = pd.Categorical(pd.factorize(df1[i])[0])

data1 = df1['Age']
data2 = df1['Geography']
data3 = df1['Gender']
data4 = df1['CreditScore']
data5 = df1['Exited']


a=data1.value_counts()
plt.bar(data1.unique(),a)

plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Customer Age Bar Chart")
plt.show()

df1.groupby('Geography').size().plot(kind = 'pie', autopct = '%.0f%%', label = '');
plt.legend(title='Geography', loc='best', labels=['France', 'Spain', 'Germany'] )
age_groups = pd.cut(df1['Age'], bins=[18, 25, 35, 45, 55, 65, 75, 85])
exit_counts = df1.groupby([age_groups, 'Exited'])['Exited'].count().unstack()
exit_counts.plot(kind='bar', stacked=True)

plt.xlabel('Age Group')
plt.ylabel('Number of Customers')
plt.title('Customer Churn by Age Group')
plt.legend(title='Exited', loc='upper right', labels=['No', 'Yes'])
age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84']
plt.xticks(range(len(age_labels)), age_labels)
plt.show()

sns.violinplot(data = df1, x = 'Gender', y = 'CreditScore');
sns.histplot(data = df1, x = 'CreditScore', hue = 'Exited', bins = 15)
plt.legend(['Exited', 'Not Exited'])

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

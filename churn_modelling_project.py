"""Churn Modelling Project.ipynb

# **Churn Modelling**

Churn modeling is a predictive modeling technique used to identify and analyze customers who are likely to leave a company's products or services. 

Credit Score, Geography, Gender, Age, Tenure, Balance, Number Of Products, Has Credit Card, Is Active Member, Estimated Salary, and Exited are commonly used as input features in this dataset.

The target for this data set will be 'Exited".

# **About Dataset:**

# **Step 1:Import necessary libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Churn_Modelling.csv', index_col = 'CustomerId')

"""# **Step 3:Data Exploration**

Data exploration is the process of discovering and understanding the underlying patterns, trends, and relationships in a dataset using Python programming language. It involves examining the data visually and statistically to gain insights and to identify any issues or anomalies in the data.
"""

df.head()

df.shape

df.info()

df.dtypes

df.describe

"""# **Step 4:Data Cleaning**

Data cleaning in Python is the process of identifying and correcting or removing errors, inconsistencies, and inaccuracies in a dataset using Python programming language. It is an essential step in the data analysis process, as raw data often contains missing values, invalid values, outliers, or other issues that can affect the accuracy of the results.

Check for missing values,duplicate values,categorical values and outliers and handle them accordingly.

"""

# Converting data types of columns

df['HasCrCard'] = df['HasCrCard'].astype('category')
df['IsActiveMember'] = df['IsActiveMember'].astype('category')
df['Exited'] = df['Exited'].astype('category')

#Check Data Types again
df.dtypes

# Summary statistics
df.describe()

df.describe(exclude=[np.number])

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

#Check for column dropped
listView = list(df.columns)
listView

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

from sklearn.preprocessing import StandardScaler

# checking if the skewness and kurtosis post scaling or not:

# For CreditScore:

print("The skewness for the original data is {}.".format(df1.CreditScore.skew()))
print("The kurtosis for the original data is {}.".format(df1.CreditScore.kurt()))

print('')

print("The skewness for the Zscore Scaled column is {}.".format(df1.Z_Score_CreditScore.skew()))
print("The kurtosis for the Zscore Scaled columns is {}.".format(df1.Z_Score_CreditScore.kurt()))

# Distribution of the columns

fig, axes = plt.subplots(2, figsize=(15,8))

sns.distplot(df1['CreditScore'], ax=axes[0])
sns.distplot(df1['Z_Score_CreditScore'], ax=axes[1])


plt.show()

#Normalization: Min Max Scalar
# For EstimatedSalary:
min_EstimatedSalary = df1.EstimatedSalary.min()
max_EstimatedSalary = df1.EstimatedSalary.max()

print('Min:' , min_EstimatedSalary)
print('Max:' , max_EstimatedSalary)

df1['Min_Max_EstimatedSalary'] = (df1['EstimatedSalary'] - min_EstimatedSalary)/ (max_EstimatedSalary - min_EstimatedSalary)

# checking if the skewness and kurtosis post scaling or not:

# For CreditScore:

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

df1.dtypes

df1

#Factorize the data
for i in df1.columns:
    if df1[i].dtypes=='object':
        df1[i] = pd.Categorical(pd.factorize(df1[i])[0])

df1

df1.info()

"""# **Step 5:Data Visualization**
Explain the findings on visualizing the data
"""

#Sweet Viz 

!pip install sweetviz

# importing sweetviz
import sweetviz as sv

#analyzing the dataset
advert_report = sv.analyze(df1)

#display the report
advert_report.show_html('Churn_Modelling.html')

#analyzing the dataset
myreport = sv.analyze(df1)
#display the report
myreport.show_html('Churn_Modelling.html')
myreport.show_notebook()

myreport = sv.compare(df1[100:200], df1[0:100])
myreport.show_notebook()

advert_report.show_notebook()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

data1 = df1['Age']
data2 = df1['Geography']
data3 = df1['Gender']
data4 = df1['CreditScore']
data5 = df1['Exited']

data1.head

data1.unique()

a=data1.value_counts()
a

plt.bar(data1.unique(),a)

# Set labels and title
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Customer Age Bar Chart")

# Show the plot
plt.show()

data2

df1.groupby('Geography').size()

df1.groupby('Geography').size().plot(kind = 'pie', autopct = '%.0f%%', label = '');
plt.legend(title='Geography', loc='best', labels=['France', 'Spain', 'Germany'] )

# Create a new DataFrame that groups customers by age group and churn status
age_groups = pd.cut(df1['Age'], bins=[18, 25, 35, 45, 55, 65, 75, 85])
exit_counts = df1.groupby([age_groups, 'Exited'])['Exited'].count().unstack()

# Plot the stacked bar chart
exit_counts.plot(kind='bar', stacked=True)

# Add chart labels and titles
plt.xlabel('Age Group')
plt.ylabel('Number of Customers')
plt.title('Customer Churn by Age Group')

# Add a legend to show the exit status
plt.legend(title='Exited', loc='upper right', labels=['No', 'Yes'])

# Add the age group labels to the x-axis
age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84']
plt.xticks(range(len(age_labels)), age_labels)

# Display the chart
plt.show()

data3.head()

sns.violinplot(data = df1, x = 'Gender', y = 'CreditScore');

sns.histplot(data = df1, x = 'CreditScore', hue = 'Exited', bins = 15)
plt.legend(['Exited', 'Not Exited'])

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

# Kaggle https://www.kaggle.com/c/titanic/

# Cristian Díaz, 2020

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Imports
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix

# Load Data
data_test = pd.read_csv('/kaggle/input/titanic/test.csv')
data_train = pd.read_csv('/kaggle/input/titanic/train.csv')

data_test.head()
data_train.head()

data_test.describe()
data_train.describe()

# Correlation Matrix
corr = data_train.corr()
corr.style.background_gradient(cmap='plasma').set_precision(2)

# Constants
CLASS_NAMES = ["Survived", "Dead"]
COLUMNS_TO_ENCODER = ['Sex', 'Embarked']

# Functions
def create_confusion_matrix(model, data_X, data_y):
  disp = plot_confusion_matrix(model, data_X, data_y,
                            display_labels=CLASS_NAMES,
                            cmap=plt.cm.Blues,
                            normalize=None)
  plt.show()

# Drop Columns in both data sets
data_train.drop(['PassengerId'],axis = 1, inplace = True)
data_train.drop(['Ticket'],axis = 1, inplace = True)
data_train.drop(['Name'],axis = 1, inplace = True)
data_train.drop(['Cabin'],axis = 1, inplace = True)

data_test.drop(['Ticket'],axis = 1, inplace = True)
data_test.drop(['Name'],axis = 1, inplace = True)
data_test.drop(['Cabin'],axis = 1, inplace = True)

# Check null/empty Values in Data Train
sns.heatmap(data_train.isnull(), cbar=False)

sns.heatmap(data_test.isnull(), cbar=False)

# Fill null values with the mean
data_test.fillna(data_test.mean(), inplace=True)
data_train.fillna(data_train.mean(), inplace=True)
data_train.Embarked = data_train.Embarked.fillna(data_train.Embarked.mode()[0], inplace=True)

# Check values in datasets 
sns.heatmap(data_test.isnull(), cbar=False)
sns.heatmap(data_train.isnull(), cbar=False)

data_train

# Convert to Dummies some Variables
# dummies_sex = pd.get_dummies(data_train.Sex, prefix="category")
# data_train = data_train.drop('Sex',axis = 1)
# data_train = data_train.join(dummies_sex)

# dummies_embarked = pd.get_dummies(data_train.Embarked, prefix="category")
# data_train = data_train.drop('Embarked',axis = 1)
# data_train = data_train.join(dummies_embarked)

for column in COLUMNS_TO_ENCODER:
    encoder = OrdinalEncoder()
    
    encoder.fit(data_train[[column]])
    data_train[[column]] = encoder.transform(data_train[[column]])
    
    encoder.fit(data_test[[column]])
    data_test[[column]] = encoder.transform(data_test[[column]])

data_train.head()

## Feature Engineering

# Create Variables

# Gonna work with data_train

# Set Variables
X = data_train.iloc[:, 1:]
y = data_train.Survived

# Create Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Logistic Regression Model

# Training Model and display cross val mean score

LR_model = LogisticRegression(multi_class='multinomial', max_iter=1000).fit(X_train, y_train)
y_prima_LR = LR_model.predict(X_train)
print('Cross_val LogisticRegression =', cross_val_score(LR_model, X_train, y_train, cv=5).mean())

print('Classification Report LogisticRegression')
print(classification_report(y_train, y_prima_LR, target_names=CLASS_NAMES))

# Display Confusion Matrix
create_confusion_matrix(LR_model, X_train, y_train)

## Random Forest
# Training Model and display cross val mean score

RF_model = RandomForestClassifier().fit(X_train, y_train)
y_prima_RF = RF_model.predict(X_train)
print('Cross_val Score RandomForestClassifier = ', cross_val_score(RF_model, X_train, y_train, cv=5).mean())
print('Accuracy Score RandomForestClassifier =', accuracy_score(y_train, y_prima_RF))

# Display Confusion Matrix
create_confusion_matrix(RF_model, X_train, y_train)

# Gets the feature_importances
importances = RF_model.feature_importances_
print('Feature Importances = ', importances)

# Sort Indices
indices_sorted = np.argsort(importances)[::-1]
print('Indices Sorted = ', indices_sorted)
Indices Sorted =  [5 1 2 0 3 4 6]

# Sort Dataset
columns_sorted = data_train.columns[indices_sorted]
print('Columns Sorted = ', columns_sorted)
Columns Sorted =  Index(['Parch', 'Pclass', 'Sex', 'Survived', 'Age', 'SibSp', 'Fare'], dtype='object')

## Predictions and Submission
data_test

X_TEST_SUBMIT = data_test.iloc[:, 1:]
y_prima_LR = LR_model.predict(X_TEST_SUBMIT)
print('Data Test Predictions = \n', y_prima_LR)


# Adding prediction in Test dataset
data_test['Survived'] = y_prima_LR
data_test


In [33]:
# Remove unnecessary columns to final dataset
data_test.drop(data_test.columns.difference(['PassengerId', 'Survived']), axis=1, inplace=True)
new_dataset = data_test
new_dataset


new_dataset.to_csv('my_dataset_submission.csv', index=False) 
print("Your submission was successfully saved!")

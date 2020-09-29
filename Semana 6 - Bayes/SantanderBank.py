# Kaggle https://www.kaggle.com/c/santander-customer-transaction-prediction/data

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix

# load data
# data = pd.read_csv('kaggle_data/sample_submission.csv')
# data_test = pd.read_csv('kaggle_data/test.csv')
data_train = pd.read_csv('kaggle_data/train.csv')
data_train

# Set Constants
CLASS_NAMES = ["0", "1"]

# Set Variables
X = data_train.iloc[:, 2:]
y = data_train.target

# Functions
def create_confusion_matrix(model, data_X, data_y):
  disp = plot_confusion_matrix(model, data_X, data_y,
                            display_labels=CLASS_NAMES,
                            cmap=plt.cm.Blues,
                            normalize=None)
  plt.show()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Naive Bayes
# First attempt without Feature Engineering
gauss_nb = GaussianNB()
y_prima_gauss_nb = gauss_nb.fit(X_train, y_train)

y_predict_train = gauss_nb.predict(X_train)
y_predict_test = gauss_nb.predict(X_test)

# Cross Validation Score
print(cross_val_score(gauss_nb, X_train, y_train, cv=5).mean())

# Classification Report (Test Set)
print(classification_report(y_test, y_predict_test, target_names=CLASS_NAMES))

# Accuracy Score (Train Set)
accuracy_score(y_train, y_predict_train)

# Display Plot non-normalized confusion matrix
create_confusion_matrix(gauss_nb, X_train, y_train)

## KN Neighbors Classifier

# First attempt without Feature Engineering
# knn_model = KNeighborsClassifier(n_neighbors=3)
# y_prima_knn = knn_model.fit(X_train, y_train)

# y_predict_train = knn_model.predict(X_train)
# y_predict_test = knn_model.predict(X_test)

# Classification Report
# print(classification_report(y_test, y_predict_test, target_names=CLASS_NAMES))

# Accuracy Score (Train Set)
# accuracy_score(y_train, y_predict_train)

# Display Plot non-normalized confusion matrix
# create_confusion_matrix(knn_model, X_train, y_train)

## Feature Engineering

# Split target class
data_target_zero = data_train.loc[data_train.target == 0]
data_target_zero = data_target_zero.sample(20098)
data_target_zero.shape

data_target_one = data_train.loc[data_train.target == 1]
data_target_one.shape

new_data = pd.concat([data_target_zero, data_target_one])
new_data.shape

## Naive Bayes with Random Sample Data

# Set Variables
X = new_data.iloc[:, 2:]
y = new_data.target

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

gauss_nb = GaussianNB()
y_prima_gauss_nb = gauss_nb.fit(X_train, y_train)

y_predict_train = gauss_nb.predict(X_train)
y_predict_test = gauss_nb.predict(X_test)

# Cross Validation Score
print(cross_val_score(gauss_nb, X_train, y_train, cv=5).mean())

# Classification Report (Test Set)
print(classification_report(y_test, y_predict_test, target_names=CLASS_NAMES))

# Accuracy Score (Train Set)
accuracy_score(y_train, y_predict_train)

# Display Plot non-normalized confusion matrix
create_confusion_matrix(gauss_nb, X_train, y_train)

## Submission
# Loading data_test
data_test

# Set Variables
X = data_test.iloc[:, 1:]
y = data_test.ID_code 

## Naive Bayes
gauss_nb = GaussianNB()
y_prima_gauss_nb = gauss_nb.fit(X, y)
y_predict = gauss_nb.predict(X)

# Save information in Kaggle

kaggle_data = pd.DataFrame({ 'ID_CODE': y, 'Target': y_predict })
kaggle_data.to_csv('nb_subset.csv', index=False)
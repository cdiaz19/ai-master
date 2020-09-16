# ANTORCHA!

## Cargamos bibliotecas importantes
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime 
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

## Cargamos el dataset de proyectos
projects_data = pd.read_csv('data/proyectos.csv')
    
## feature engineering (transformacion, norm, eliminar var, etc...): 5 pts

# correlation matrix
corr = projects_data.corr()
corr.style.background_gradient(cmap='plasma').set_precision(2)

# Revisar si existen datos nulos

# Drop columns
projects_data = projects_data.drop('name',axis = 1)
projects_data = projects_data.drop('usd_pledged',axis = 1)
projects_data = projects_data.drop('country',axis = 1)
projects_data = projects_data.drop('currency',axis = 1)
projects_data = projects_data.drop('category',axis = 1)
projects_data = projects_data.drop('main_category',axis = 1)
projects_data = projects_data.drop('deadline',axis = 1)
projects_data = projects_data.drop('launched',axis = 1)
projects_data = projects_data.drop(projects_data.index[90135])

# Remove Live State rows
projects_without_live_state = projects_data.loc[projects_data['state'] != 'live']

# create columns

# categorize some columns

# dummies_category = pd.get_dummies(projects_without_live_state.category, prefix="category")
# projects_without_live_state = projects_without_live_state.drop('category',axis = 1)
# projects_without_live_state = projects_without_live_state.join(dummies_category)

# dummies_main_category = pd.get_dummies(projects_without_live_state.main_category, prefix="main_category")
# projects_without_live_state = projects_without_live_state.drop('main_category',axis = 1)
# projects_without_live_state = projects_without_live_state.join(dummies_main_category)

# dummies_state = pd.get_dummies(projects_without_live_state.state, prefix="state")
# projects_without_live_state = projects_without_live_state.drop('state',axis = 1)
# projects_without_live_state = projects_without_live_state.join(dummies_state)

# transformar en DATETIME
# projects_without_live_state.deadline = [datetime.strptime(dl, '%Y-%m-%d') for dl in projects_without_live_state.deadline]
# projects_without_live_state.launched= [datetime.strptime(dl, '%Y-%m-%d %H:%M:%S') for dl in projects_without_live_state.launched]

# Eliminar Nulos
projects_without_live_state.goal = projects_without_live_state.goal.fillna(projects_without_live_state.goal.median())
projects_without_live_state.pledged = projects_without_live_state.pledged.fillna(projects_without_live_state.pledged.median())
projects_without_live_state.backers = projects_without_live_state.backers.fillna(projects_without_live_state.backers.median())
projects_without_live_state.usd_pledged_real = projects_without_live_state.usd_pledged_real.fillna(projects_without_live_state.usd_pledged_real.median())
projects_without_live_state.usd_goal_real = projects_without_live_state.usd_goal_real.fillna(projects_without_live_state.usd_goal_real.median())
projects_without_live_state.isnull().any()


projects_without_live_state.state = projects_without_live_state.state.astype('category')
projects_without_live_state.state.cat.categories.unique()


## X, Y
X = np.array((projects_without_live_state.loc[:, ~projects_without_live_state.columns.isin(['state'])]))
y = np.array((projects_without_live_state.state))

## entrenamiento de modelos: 5pts

# KNN
                                  
# knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
# y_prima = knn.predict(X.T)

# Naibe Bayes Gaussiano
# NB Gaussiano
model = GaussianNB()

# 10-k-fold CV
scores = cross_val_score(model, X, y, cv=10)
print("CV:",scores)
print("Accuracy:",scores.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

model = GaussianNB()
y_prima = model.fit(X_train, y_train).predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_prima))

## model selection: uso de metricas para comprar modelos y matrices de confusion: 5 pts

class_names = ['canceled', 'failed', 'successful', 'suspended', 'undefined']
classifier = LogisticRegressionCV(max_iter=1000, cv=10, random_state=0).fit(X, y)
np.set_printoptions(precision=2)

disp = plot_confusion_matrix(classifier,X, y,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
plt.show()

# Plot non-normalized confusion matrix
disp = plot_confusion_matrix(classifier, X_test, y_test,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
plt.show()

## considere crear un resumen al final donde se puedan comparar las metricas "taco-a-taco", para que quede bien explicado cual modelo es el mejor.
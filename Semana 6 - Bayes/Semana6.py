# ANTORCHA!

## Cargamos bibliotecas importantes
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
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
projects_data = projects_data.drop('currency',axis = 1)
projects_data = projects_data.drop('deadline',axis = 1)
projects_data = projects_data.drop('launched',axis = 1)
projects_data = projects_data.drop('goal',axis = 1)
projects_data = projects_data.drop(projects_data.index[90135])

# Categorizar columns
dummies_country = pd.get_dummies(projects_data.country, prefix="category")
projects_data = projects_data.drop('country',axis = 1)
projects_data = projects_data.join(dummies_country)

dummies_category = pd.get_dummies(projects_data.category, prefix="category")
projects_data = projects_data.drop('category',axis = 1)
projects_data = projects_data.join(dummies_category)

dummies_main_category = pd.get_dummies(projects_data.main_category, prefix="main_category")
projects_data = projects_data.drop('main_category',axis = 1)
projects_data = projects_data.join(dummies_main_category)

# Remove Live State rows
projects_without_live_state = projects_data.loc[projects_data.state != 'live']

# Eliminar Nulos
projects_without_live_state.pledged = projects_without_live_state.pledged.fillna(projects_without_live_state.pledged.median())
projects_without_live_state.backers = projects_without_live_state.backers.fillna(projects_without_live_state.backers.median())
projects_without_live_state.usd_pledged_real = projects_without_live_state.usd_pledged_real.fillna(projects_without_live_state.usd_pledged_real.median())
projects_without_live_state.usd_goal_real = projects_without_live_state.usd_goal_real.fillna(projects_without_live_state.usd_goal_real.median())

# ## X, Y
X = np.array((projects_without_live_state.loc[:, ~projects_without_live_state.columns.isin(['state'])]))
y = np.array((projects_without_live_state.state))

# # 70 - 30 train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# ## entrenamiento de modelos and model selection

# CONSTANTS
class_names = ['canceled', 'failed', 'successful', 'suspended', 'undefined']
np.set_printoptions(precision=2)

# KNN
                                  
knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_prima_knn = knn.predict(X_train)

knn_cros_val_score = cross_val_score(knn, X, y, cv=5).mean()
accurancy_score_knn = accuracy_score(y_train, y_prima_knn)

disp = plot_confusion_matrix(knn, X_train, y_train,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
plt.show()

knn_cros_val_score = cross_val_score(knn, X, y, cv=5).mean()

knn_results = [knn_cros_val_score, accurancy_score_knn, disp.confusion_matrix, 6]

# NB Gaussiano
gauss = GaussianNB()
y_prima_gauss = gauss.fit(X_train, y_train).predict(X_test)

gauss_cros_val_score = cross_val_score(gauss, X, y, cv=5).mean()
accurancy_score_gaussian = accuracy_score(y_test, y_prima_gauss)

disp = plot_confusion_matrix(knn, X_train, y_train,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
plt.show()


gaussian_results = [gauss_cros_val_score, accurancy_score_gaussian, disp.confusion_matrix, 4]


## considere crear un resumen al final donde se puedan comparar las metricas "taco-a-taco", para que quede bienexplicado cual modelo es el mejor.

#!pip3 install plotly==4.10.0
#!pip3 install jupyterlab "ipywidgets>=7.5"

headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

fig = go.Figure(data=[go.Table(
  header=dict(
    values=['<b></b>','<b>KNeighbors</b>','<b>GaussianNB</b>'],
    line_color='darkslategray',
    fill_color=headerColor,
    align=['left','center'],
    font=dict(color='white', size=12)
  ),
  cells=dict(
    values=[
      ['Accurancy Cross Validation', 'Train/Test Sets Accurancy', 'Confusion Matrix', 'Classification Report'],
      knn_results,
      gaussian_results],
    line_color='darkslategray',
    fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*5],
    align = ['left', 'center'],
    font = dict(color = 'darkslategray', size = 11)
    ))
])

fig.show()


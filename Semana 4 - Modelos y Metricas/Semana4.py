# La variable a predecir es la variable "target" la cual tiene valores de 1 y 0. Donde 1 es que el paciente tiene un padecimiento cardiaco.

# Los atributos X son los siguientes:
# age
# sex
# chest pain type (4 values)
# resting blood pressure
# serum cholestoral in mg/dl
# fasting blood sugar > 120 mg/dl
# resting electrocardiographic results (values 0,1,2)
# maximum heart rate achieved
# exercise induced angina
# oldpeak = ST depression induced by exercise relative to rest
# the slope of the peak exercise ST segment
# number of major vessels (0-3) colored by flourosopy
# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# El dataset contiene 303 registros de un hospital de Cleveland. Es recomendable que lea el siguiente paper para que obtenga un mejor conocimiento sobre el trasfondo del problema.

# Cargamos bibliotecas importantes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/heart.csv")
data.head()

# EJERCICIO - Feature Engineering
# En esta seccion incluya transformaciones, scaling, eliminar o crear variables segun considere apropiado.
X = data.loc[:, ~data.columns.isin(['target'])]
y = data.target


# EJERCICIO - Entrenar Modelo & Resampling
# Utilize SKlearn para entrenar el modelo de logistic regression. 
# Como ud solo tiene este dataset, recuerde utilizar la mejor tecnica de resampling.

cv_y = np.array((y)).reshape(-1,1)
# Cross Validation
model = LogisticRegression(max_iter=1000, random_state=0)
scores = cross_val_score(model, X, cv_y.reshape(-1), cv=10)

print("Exactitud de cada particion:", scores)
print("Exactitud Promedio:", scores.mean())

# EJERCICIO - Metricas y Evaluacion
# Implemente la matrix de confusion y calcule todas las metricas del notebook #4 utilizando classification_report 
# y accuracy_score. Considere el resultado obtenido e itere sobre todo el notebook hasta sentirse satisfecho con
# los resultados. 

X = np.array((data.loc[:, ~data.columns.isin(['target'])]))
y = np.array((data.target))
class_names = ["Sin padecimiento", "Tiene padecimiento"]

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)
classifier = LogisticRegression(multi_class='multinomial', max_iter=1000).fit(X_train, y_train)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
disp = plot_confusion_matrix(classifier, X_test, y_test,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
plt.show()

y_prima = classifier.predict(X_test)
print(classification_report(y_test, y_prima, target_names=class_names))
print("Accurancy_score:", accuracy_score(y_test, y_prima))

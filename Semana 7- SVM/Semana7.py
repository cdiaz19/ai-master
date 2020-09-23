# Utilize SVM para predecir Ham vs Spam.
# Use las tecnicas de NLP vistas en la clase y utilice SVM para realizar la estimacion!
# use cross validation y todo lo que ha aprendido en el curso!

# imports
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix

# Constants
class_names = ['ham', 'spam']

# Functions
def confusion_matrix(model, data_X, data_y):
    print("Model: ", model)
    disp = plot_confusion_matrix(model, data_X, data_y,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
    plt.show()

# CountVectorizer
cv = CountVectorizer(binary=True, stop_words= 'english').fit(X)
X = cv.transform(X)

# Models
print("== KERNEL LINEAR ==")
linear_model = svm.SVC(kernel='linear').fit(X, y)
y_prima_linear = linear_model.predict(X)
print("Classification Report Kernel Linear")
print(classification_report(y, y_prima_linear, target_names=class_names))
print("Cross-Validation Score:",cross_val_score(linear_model, X, y, cv=5).mean())
confusion_matrix(linear_model, X, y)

print("== KERNEL POLY ==")
poly_model = svm.SVC(kernel='poly', degree=3).fit(X, y)
y_prima_poly = poly_model.predict(X)
print("Classification Report Kernel Poly")
print(classification_report(y, y_prima_poly, target_names=class_names))
print("Cross-Validation Score:",cross_val_score(poly_model, X, y, cv=5).mean())
confusion_matrix(poly_model, X, y)

print("== KERNEL GAMMA ==")
gamma_model = svm.SVC(kernel='rbf', gamma=2).fit(X, y)
y_prima_gamma = gamma_model.predict(X)
print("Classification Report Kernel Gamma")
print(classification_report(y, y_prima_gamma, target_names=class_names))
print("Cross-Validation Score:",cross_val_score(gamma_model, X, y, cv=5).mean())
confusion_matrix(gamma_model, X, y)

# ANTORCHA!
# ejercicio extra: 
# https://www.kaggle.com/c/nlp-getting-started ingrese en la competencia y envie sus resultados!
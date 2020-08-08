from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X, y = datasets.load_diabetes(return_X_y=True)

# variables X & y
bmi = X[:, np.newaxis, 2]
progreso = y

# separacion de test de entrenamiento y test de prueba.
bmi_train, bmi_test, progreso_train, progreso_test = train_test_split(bmi, progreso, test_size=0.30, random_state=42)

# <=== Ejercicio en Clase ===>
# a) Aplique el modelo generado sobre el set de pruebas.

from sklearn.linear_model import LinearRegression

x = bmi_test
y = progreso_test.reshape(-1,1)

model = LinearRegression().fit(x, y)

b_1 = model.coef_[0]
b_0 = model.intercept_
rss = np.sum((y - b_0 - b_1 * x)**2)

print("b0:",b_0, "b1:", b_1, "rss:", rss)

# b) Despliegue el set de pruebas y el modelo (la función de regresión) con Matplotlib

plt.scatter(bmi_test,progreso_test, marker = ".", s = 60, c = "blue")
plt.xlabel("Bmi")
plt.ylabel("Progreso de la Enfermedad")

y_prima = model.predict(bmi_test)

plt.plot(bmi_test, y_prima, 'r--', c = "magenta")
plt.show()

# c) Obtenga el MSE del set de pruebas y comparelo con el MSE del set de entrenamiento

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(progreso_test.reshape(-1,1), y_prima)
print("MSE:",mse)

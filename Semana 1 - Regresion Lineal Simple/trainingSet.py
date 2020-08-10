# Parte 1: Training Set

# 1 imports:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("data/precios-casas-1.csv")
data.head()

# 2 separe el dataset (data) en train (con 80%) y test set (con 20%) de los datos
# y = precio, X = sqft_living
X, y = data['sqft_living'].to_numpy(), data['price'].to_numpy()

sqft = X[:, np.newaxis]
price = y

sqft_train, sqft_set, price_train, price_set = train_test_split(sqft, price, test_size=0.30, random_state=42)

# 3 despliegue el train set y el test set con matplotlib. Despliegue los charts usando
# scatter plots con puntos verdes para el training set y azules para el test set.
# ambos charts deben desplegarse de forma horizonal (uno al lado del otro)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
ax1.scatter(price_train, sqft_train, marker = ".", s = 60, c = "green")
ax2.scatter(price_set, sqft_set, marker = ".", s = 60, c = "blue")

fig.text(0.5, 0.04, 'SQFT Living', ha='center')
fig.text(0.09, 0.5, 'Price', va='center', rotation='vertical')

plt.show()

# 4 utilize Sklearn para generar el modelo de regresion lineal sobre el training set.
# imprima el valor de los coeficientes
x = sqft_set
y = price_set.reshape(-1,1)

model = LinearRegression().fit(x, y)

b_1 = model.coef_[0]
b_0 = model.intercept_
rss = np.sum((y - b_0 - b_1 * x)**2)

print("b0:",b_0, "b1:", b_1, "rss:", rss)

# 5 despliegue con Matplotlib el trainig set y el modelo (funcion de regresion) sobre
# los datos. Aplique el color magenta a la linea de regresion.
plt.scatter(sqft_train, price_train, marker = ".", s = 60, c = "blue")
plt.xlabel("SQFT Living")
plt.ylabel("Price")

y_prima = model.predict(sqft_train)
plt.plot(sqft_train, y_prima, 'r--', c = "magenta")

plt.show()

# 6 calcule el MSE del training set
mse = mean_squared_error(price_train.reshape(-1,1), y_prima)
MSE_train = mse
print("MSE:",mse)

# Parte 2: Test Set

# 7 Aplique el modelo generado (en el paso #4) sobre el test set. 
# Aqui debe generar la prediccion sobre el test set usando el modelo ya generado.
y_prima_test = model.predict(sqft_set)

# 8 despliegue con Matplotlib el test set y el modelo (funcion de regresion) sobre
# los datos. Aplique el color naranja a la linea de regresion.
plt.scatter(sqft_set, price_set, marker = ".", s = 60, c = "blue")
plt.xlabel("SQFT Living")
plt.ylabel("Price")

plt.plot(sqft_set, y_prima_test, 'r--', c = "orange")

plt.show()

# 9 calcule el MSE del test set
mse = mean_squared_error(price_set.reshape(-1,1), y_prima_test)
MSE_test = mse
print("MSE:",mse)


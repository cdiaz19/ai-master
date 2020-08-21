# Parte 1: Training Set

# 1 imports:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/precios-casas-1.csv")
data.head()

# AJUSTE - Las indicaciones estan en la Parte 3. (no resuelva esto sin haber completado 
# la Parte 1 y 2)
# el ajuste debe realizarse sobre la variable data.

sqft_max = data.sqft_living.argmax()
price_max = data.price.argmax()

data.drop(sqft_max, inplace=True)
data.drop(price_max, inplace=True)
data.head()

# 2 separe el dataset (data) en train (con 80%) y test set (con 20%) de los datos
# y = precio, X = sqft_living
X, y = data.sqft_living.to_numpy(), data.price.to_numpy()

sqft = X[:, np.newaxis]
price = y

# separacion de test de entrenamiento y test de prueba.
sqft_train, sqft_test, price_train, price_test = train_test_split(sqft, price, test_size=0.20, random_state=42)

# 3 despliegue el train set y el test set con matplotlib. Despliegue los charts usando
# scatter plots con puntos verdes para el training set y azules para el test set.
# ambos charts deben desplegarse de forma horizonal (uno al lado del otro)
# vamos a crear una figura con 2 slots con matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

ax1.scatter(sqft_train, price_train, marker = ".", s = 60, c = "green")
ax2.scatter(sqft_test, price_test, marker = ".", s = 60, c = "blue")

fig.text(0.5, 0.04, 'SQFT Living', ha='center')
fig.text(0.09, 0.5, 'Price', va='center', rotation='vertical')

plt.show()
ax.plot(price, sqft)
ax.yaxis.set_major_formatter(y_format)  # set formatter to needed axis

plt.show()

# 4 utilize Sklearn para generar el modelo de regresion lineal sobre el training set.
# imprima el valor de los coeficientes
x = sqft_train
y = price_train.reshape(-1,1)

model = LinearRegression().fit(x, y)

b_1 = model.coef_[0]
b_0 = model.intercept_
rss = np.sum((y - b_0 - b_1 * x)**2)

print("b0:",b_0, "b1:", b_1, "rss:", rss)

# 5 despliegue con Matplotlib el trainig set y el modelo (funcion de regresion) sobre
# los datos. Aplique el color magenta a la linea de regresion.
plt.scatter(sqft_train, price_train, marker = ".", s = 60, c = "green")
plt.xlabel("SQFT Living")
plt.ylabel("Price")

y_prima = model.predict(sqft_train)
plt.plot(sqft_train, y_prima, 'r--', c = "magenta")

plt.show()

# 6 calcule el MSE del training set
mse = mean_squared_error(price_train.reshape(-1,1), y_prima)
# MSE_train = mse 
MSE_train_adj = mse
print("MSE:",mse)

# Parte 2: Test Set

# 7 Aplique el modelo generado (en el paso #4) sobre el test set. 
# Aqui debe generar la prediccion sobre el test set usando el modelo ya generado.
y_prima_test = model.predict(sqft_set)

# 8 despliegue con Matplotlib el test set y el modelo (funcion de regresion) sobre
# los datos. Aplique el color naranja a la linea de regresion.
plt.scatter(sqft_set, price_set, marker = ".", s = 60, c = "blue")
plt.xlabel("Price")
plt.ylabel("SQFT Living")

plt.plot(sqft_set, y_prima_test, 'r--', c = "orange")

plt.show()

# 9 calcule el MSE del test set
mse = mean_squared_error(price_set.reshape(-1,1), y_prima_test)
# MSE_test = mse
MSE_test_adj = mse
print("MSE:",mse)

# 10 Existe un problema con los datos que afecta el modelo de regresion lineal. 
# Considere eliminar los datos que ud considere que estan dando problemas y
# vuelva a ejecutar los pasos del 1 al 9.
# PISTA: son 2 datos que afectan el modelo...

# es importante que este ajuste se realize en la casilla de AJUSTE, que esta antes
# de la Parte 1.

# antes de realizar el ajuste, salve el MSE_train y el MSE_test en variables para 
# que pueda comparar estos datos despues del ajuste

# Obtenga: 
# MSE_train
# MSE_train_adj
# MSE_test
# MSE_test_adj

print("MSE_train: ", MSE_train)
print("MSE_train_adj: ", MSE_train_adj)
print("MSE_test: ", MSE_test)
print("MSE_test_adj: ", MSE_test_adj)


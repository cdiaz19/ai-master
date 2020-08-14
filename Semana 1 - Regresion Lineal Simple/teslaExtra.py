# imports
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Cargar los datos de la bateria de Tesla
data = pd.read_csv('data/tesla-battery.csv')
print(data.shape)
data.head()

# variables
# Kilometraje Actual de la Bateria (formato 32,080 km) / # Porcentaje del rango disponible (formato 95.00%)
X = data["kmstand huidige batterij"]
y = data["percentage"]

#  ====  Otros datos para formato ====
# hay dos tipos de baterias 85kWh y 60kWh
potencia = data["batterijvermogen"]

# limpieza de datos
X = [int(re.sub("[^0-9]", "", a)) for a in X]
y = [int(re.sub("[^0-9]", "", b.split(".")[0])) for b in y]

# Muestra de Datos
plt.figure(figsize=(20,10)) # tamano del cuadro

plt.title("")

potencia_colores = {'85 kWh':'#26648E', '60 kWh':'#53D2DC'}
colors = potencia.apply(lambda x: potencia_colores[x])

plt.scatter(X,y, c = colors, s = 80, alpha = 0.8)

plt.xlabel("Kilometraje de la Bateria")
plt.xticks(rotation=45) 
plt.xticks(np.arange(0, 95000, 5000.0))

plt.yticks(np.arange(76, 105, 1.0))
plt.ylabel("Porcentaje Del Rango Disponible")

plt.show()

# Ejercicio Opcional:

# Utilice Sklearn para estimar un modelo que se ajuste a los datos recolectados.
# Calcule los coeficientes
# Estime el MSE
# Realice la prediccion del rango disponible (y) para una bateria con (x = 80,000 km y x = 140,000) y despliegue su prediccion en el grafico como un punto de color rojo.
# Para desplegar el grafico y el modelo Utilize Plotly no utilize matplotlib.

# 1. Estimacion de Coeficientes 
XKm = np.array(X)
peY = np.array(y)
km = XKm[:, np.newaxis]
percentage = peY

# # separacion de test de entrenamiento y test de prueba.
km_train, km_test, percentage_train, percentage_test = train_test_split(km, percentage, test_size=0.20, random_state=42)

# 2. Estimacion del MSE
x = km_train
y = percentage_train.reshape(-1,1)

model = LinearRegression().fit(x, y)

b_1 = model.coef_[0]
b_0 = model.intercept_
rss = np.sum((y - b_0 - b_1 * x)**2)

print("b0:",b_0, "b1:", b_1, "rss:", rss)

y_prima = model.predict(km_train)

mse = mean_squared_error(km_train.reshape(-1,1), y_prima)
print("MSE:",mse)

# 3. Prediccion de f(80,000)

## <SU CODIGO AQUI>


# 4. Desplegar todos los datos + la prediccion como un punto rojo (2)
plt.scatter(km_train, percentage_train, marker = ".", s = 60, c = "green")
plt.xlabel("Kilometraje de la Bateria")
plt.ylabel("Porcentaje Del Rango Disponible")

plt.plot(km_train, y_prima, 'r--', c = "magenta")

plt.show()
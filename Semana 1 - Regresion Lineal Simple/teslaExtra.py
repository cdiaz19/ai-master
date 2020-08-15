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
x_ = np.array(X).reshape(-1,1)
y_ = np.array(y).reshape(-1,1)

model = LinearRegression().fit(x_, y_)

b_1 = model.coef_[0]
b_0 = model.intercept_

print("b0:",b_0, "b1:", b_1)

# 2. Estimacion del MSE
y_prima = model.predict(x_)

mse = mean_squared_error(y, y_prima)
print("MSE:", mse)

# 3. Prediccion de f(80000, 140000)
x1 = 80000
y1 = model.predict([[x1]])

x2 = 140000
y2 = model.predict([[x2]])

# 4. Desplegar todos los datos + la prediccion como un punto rojo (2)
!pip3 install plotly

import numpy as np
import plotly.graph_objects as go


t = np.linspace(0, 10, 100)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=X, y=y,
    name='Datos',
    mode='markers',
    marker_color=colors
))

fig.add_trace(go.Scatter(x=X, y=y_prima.transpose()[0],
                    mode='lines',
                    name='Tendencia'))

fig.add_trace(go.Scatter(
    x=[x1], y=y1[0],
    name='80k',
    mode='markers',
    marker=dict(size=[15],color="green", symbol="star")
))

fig.add_trace(go.Scatter(
    x=[x2], y=y2[0],
    name='140k',
    mode='markers',
    marker=dict(size=[15],color="red", symbol="star")
))

fig.update_layout(yaxis=dict(range=[50,100]), title="Porcentaje Disponible de la Bateria del Tesla Modelo S")

fig.show()
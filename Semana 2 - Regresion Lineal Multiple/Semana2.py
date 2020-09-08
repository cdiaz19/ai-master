# imports
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/casas-california.csv")
data.head()

# Visualizar casas en California
data.plot(kind ="scatter", x ="longitude", y ="latitude", alpha = 0.4,
             s = data.population / 100, label ="population", figsize =(11,8),
             c ="median_house_value", cmap = plt.get_cmap("jet"), colorbar = True, ) 
plt.legend()

# Correlaciones
# EJERCICIO 1
# utilize matplotlib para desplegar la matriz de correlaciones
corr = data.corr()
corr.style.background_gradient(cmap='plasma').set_precision(2)

# EJERCICIO 2
# despliegue las correlaciones de la variable median_house_value
corr = data.corr()
corr.median_house_value.sort_values(ascending=False)

### EJERCICIO 3
# utilize seaborn para revisar de que no hay valores en blanco. despliegue el grafico de seaborn.
# !pip3 install seaborn
sns.heatmap(data.isnull(), cbar=False)

data.total_bedrooms = data.total_bedrooms.fillna(data.total_bedrooms.median())
sns.heatmap(data.isnull(), cbar=False)

### EJERCICIO 4
# Revise los tipos de las columnas y verifique que cada uno esta definido correctamente 
# (ejemplo, la variable OCEAN_PROXIMITY debe ser categorica)
data.ocean_proximity = data.ocean_proximity.astype('category')
data.ocean_proximity.cat.categories.unique()

# test & training set
# EJERCICIO 5 
# separe el dataset train:80%, test:20%, random_state=42 
# donde y = median_house_value
dummies = pd.get_dummies(data.ocean_proximity, prefix="ocean_proximity")
data = data.drop('ocean_proximity',axis = 1)
data = data.join(dummies)

X = data.loc[:, ~data.columns.isin(['median_house_value'])]
y = data.median_house_value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# regresion lineal multiple
# EJERCICIO 6 
# estime el modelo de regresion lineal utilizando Sklearn.
# imprima los coeficientes
model = LinearRegression().fit(X.values, y.values)

print("b",0,":","%.2f"%model.intercept_)
for i,b in zip(np.arange(1,len(model.coef_)+1), model.coef_):
    print("b",i,":", "%.2f"%b)

# EJERCICIO 7
# utilize statsmodels para estimar las estadisticas del modelo actual.
# estime el MSE del train/test
y_prima_train = model.predict(X_train)
y_prima_test = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_prima_train)
mse_train = np.sqrt(mse_train)

mse_test = mean_squared_error(y_test, y_prima_test)
mse_test = np.sqrt(mse_test)

print("MSE train:", mse_train)
print("MSE test:", mse_test)

X2 = sm.add_constant(X_test)
model_ols = sm.OLS(y_test, X2)
pred = model_ols.fit()
print(pred.summary())

### Estimacion Avanzada - Feature Engineering

### EJERCICIO: Aplique Escalamiento y Transformaciones
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

### EJERCICIO: Cree o Elimine variables 

data["households_over_population"] = data.households / data.population
data["bedrooms_per_room"] = data.total_bedrooms / data.total_rooms


### EJERCICIO: Aplique Escalamiento y Transformaciones
RX = data.loc[:, ~data.columns.isin(['median_house_value'])]
scaler = StandardScaler().fit(RX)
RX2 = scaler.transform(RX)

### EJERCICIO: Estime nuevamente el modelo de regresion lineal
model = LinearRegression().fit(RX2, y.values)

print("b",0,":","%.2f"%model.intercept_)
for i,b in zip(np.arange(1,len(model.coef_)+1), model.coef_):
    print("b",i,":", "%.2f"%b)

### EJERCICIO: Calcule nuevamente las estadisticas para determinar si hay mejoras sobre el modelo.
# estime el MSE del train/test

RX_train, RX_test, ry_train, ry_test = train_test_split(RX2, y, test_size=0.20, random_state=42)

ry_prima_train = model.predict(RX_train)
ry_prima_test = model.predict(RX_test)

rmse_train = mean_squared_error(ry_train, ry_prima_train)
rmse_train = np.sqrt(rmse_train)

rmse_test = mean_squared_error(ry_test, ry_prima_test)
rmse_test = np.sqrt(rmse_test)

print("RMSE train:", rmse_train)
print("RMSE test:", rmse_test)

RX2 = sm.add_constant(RX_test)
model_ols = sm.OLS(ry_test, RX2)
pred = model_ols.fit()
print(pred.summary())
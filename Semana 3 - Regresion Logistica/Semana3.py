# Data importante, analizadfa desde el dataset
# meses_ultima: número de meses desde la última donación de sangre
# frecuencia: número de veces que la persona ha donado sangre
# volumen: cantidad total de sangre (en ml) donada por la persona
# meses_primera: número de meses desde la primera donación de sangre
# donante: si la persona ha donado sangre durante el último mes

# Cargamos bibliotecas importantes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

# Cargamos el dataset de transfusion
transfusion_data = pd.read_csv('data/transfusion.csv')
transfusion_data.head()

# EJERCICIO: Colocar los datos en formato numpy
transfusion_x = transfusion_data.iloc[:, ~transfusion_data.columns.isin(['donante'])] # Independientes: contiene las columnas meses_ultima, frecuencia, volumen y meses_primera
transfusion_y = transfusion_data.donante # Dependiente: contiene la columna donante

# EJERCICIO: Entrenar un modelo de regresión logística con scikit-learn
Mtransfusion_x = np.array(transfusion_x)
Mtransfusion_y = np.array(transfusion_y)

transfusion_modelo = LogisticRegression()
transfusion_modelo.fit(Mtransfusion_x, Mtransfusion_y)

# EJERCICIO: Crear una función de predicción
def es_donante(meses_ultima, frecuencia, volumen, meses_primera):
    return True if (transfusion_modelo.predict([[meses_ultima, frecuencia, volumen, meses_primera]])) else False

# --- CELDA PARA PROBAR SU MODELO UNA VEZ ESTÉ ENTRENADO --- #
## Precisión del modelo debe ser aproximadamente 0.77
print('Precisión: ', (transfusion_modelo.predict(transfusion_x) == transfusion_y).mean())

# Debe dar False
print('es_donante(3, 20, 1000, 8): ', es_donante(3, 20, 1000, 8))

# Debe dar True
print('es_donante(6, 12, 5000, 20): ', es_donante(6, 12, 5000, 20))

## Clasificación multiclase
# Cargamos el dataset de provincias
provincias_data = pd.read_csv('data/provincias_scaled.csv')

# Y mostramos los puntos en el mapa
# Primero, codifiquemos las provincias en 0-6
le = preprocessing.LabelEncoder()
le.fit(provincias_data['provincia'])
c = le.transform(provincias_data['provincia'])

# Leemos el archivo
im = plt.imread('data/mapaCR.png')
implot = plt.imshow(im)

# Escalamos los arrays en 0-1
p_long = provincias_data['long']
p_lat = provincias_data['lat']

# Dibujamos (con un pequeño offset empírico para que caiga en la imágen)
plt.scatter((im.shape[0]*p_long)*0.95 + 25, (im.shape[1]*(1-p_lat))*0.8 + 20, c=c)

# EJERCICIO: Entrenar un modelo de regresión softmax para distinguir la probabilidad de que un punto 
# esté en cada provincia
p_x = np.array((p_long, p_lat)).T
p_y = np.array((provincias_data.provincias))

provincias_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
provincias_model.fit(p_x, p_y)

# EJERCICIO (2pt): Función que devuelve un diccionario con la probabilidad de que un punto (longitud, latitud) 
# se encuentre en una provincia dada, 
def prob_provincias(long, lat): 
    alajuela, cartago, puntarenas, limon, guanacaste, heredia, sj = provincias_model.predict_proba([[long, lat]])[0]

    dic_provicias = {
        'ALAJUELA': round(alajuela, 3), 
        'CARTAGO': round(cartago, 3), 
        'PUNTARENAS': round(puntarenas, 3), 
        'LIMON': round(limon, 3),
        'GUANACASTE': round(guanacaste, 3), 
        'HEREDIA': round(heredia, 3), 
        'SAN JOSE': round(sj, 3),
    }

    return dic_provicias

def prob_provincias(long, lat):
    inpx = np.array([[long, lat]])
    [out] = provincias_model.predict_proba(inpx)
    le_out = dict(zip(provincias_model.classes_, out))
    return le_out

# EJERCICIO: Función que devuelve la provincia de mayor probabilidad a la que un punto (longitud, latitud)
# pertenece.
def cual_provincia(long, lat):
    vec_provincias = prob_provincias(long, lat)
    return max(vec_provincias, key=lambda key: vec_provincias[key])  

# --- CELDA PARA PROBAR SU MODELO UNA VEZ ESTÉ ENTRENADO --- #
from scipy.spatial import Delaunay
points = np.array((p_long, p_lat)).T
hull = Delaunay(points)

samples = np.random.rand(20000,2)
samples = np.array([x for x in samples if hull.find_simplex(x) >= 0]).T

im = plt.imread('data/mapaCR.png')
implot = plt.imshow(im)

plt.scatter((samples[0]*im.shape[0])*0.95 + 25, ((1.0-samples[1])*im.shape[1])*0.8 + 20, c=le.transform(provincias_model.predict(samples.T)), alpha=0.05)
# Cargamos bibliotecas importantes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

## Primer ejercicio: *k*-NN
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
plt.show()

# Obtenemos la envolvente convexa del territorio nacional
from scipy.spatial import Delaunay
points = np.array((p_long, p_lat)).T
hull = Delaunay(points)

# Generaremos un grid de prueba filtrado por la envolvente convexa
XX, YY = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
pos = np.vstack([XX.ravel(), YY.ravel()]).T
samples = np.array([x for x in pos if hull.find_simplex(x) >= 0]).T

### EJERCICIO 1 
## Entrenaremos los modelos usando los datos 'points' contra la variable provincias_data['provincia'] como target
# osea X = points, y = provincias_data['provincia']
# debe generar un modelo para cada K,
# debe obtener el y_prima para cada modelo

X = points
y = provincias_data.provincia

### Entrenar modelos de clasificación KNN para k={1, 3, 5, 10, 30, 100, 200}
k_a_probar = [1, 3, 5, 10, 30, 100, 200]
y_primas = []

# Probamos
for val_k in k_a_probar:
    knn = KNeighborsClassifier(n_neighbors=val_k).fit(X, y)
    y_prima = knn.predict(samples.T)
    y_primas.append(y_prima) 

### Utilice esta celda para probar los resultados de sus modelos ###
# imprima el mapa de costa rica con cada y_prima obtenida en el ejercicio 1.
for val_prima in y_primas:
    print(val_prima)
    im = plt.imread('data/mapaCR.png')
    implot = plt.imshow(im)
    plt.scatter((samples[0]*im.shape[0])*0.95 + 25, ((1.0-samples[1])*im.shape[1])*0.8 + 20,c=le.transform(val_y_prima), alpha=0.1)
    plt.title("k=#" +  str(k_a_probar.pop(0)))
    plt.show()

## Segundo ejercicio: reducción de la dimensionalidad
# Leemos el dataset
waveform = pd.read_csv('data/waveform.csv')

# Tomamos las columnas independientes
wave_x_all = np.array(waveform.loc[:, waveform.columns != 'class'])

# Y la dependiente
wave_y_all = np.array(waveform['class'])

# Los separamos en dos conjuntos entrenamiento/pruebas
wave_x_train = wave_x_all[:4000]
wave_y_train = wave_y_all[:4000]
wave_x_test = wave_x_all[4000:]
wave_y_test = wave_y_all[4000:]

## EJERCICIO 1: Entrenar un modelo k-NN con k=10 y reportar la precisión (accuracy)
## Debe dar 0.826
knn = KNeighborsClassifier(n_neighbors=10).fit(wave_x_all, wave_y_all)
y_prima = knn.predict(wave_x_all)

print("Accurancy_score:", accuracy_score(wave_y_all, y_prima))

## EJERCICIO 2 : Reducir la dimensionalidad utilizando análisis de discriminante lineal a 2 dimensiones y dibujarlo
## IMPORTANTE: utilizar el mismo modelo para reducir los datos de prueba
lda_model = LinearDiscriminantAnalysis(n_components=2).fit(wave_x_train, wave_y_train)
wave_x_reduced_train = lda_model.transform(wave_x_train).T
wave_x_reduced_test = lda_model.transform(wave_x_test).T

# Imprimit el training set REDUCIDO a 2 componentes
plt.scatter(wave_x_reduced_train[0], wave_x_reduced_train[1], c=['red' if x==0 else 'blue' if x==1 else 'black' for x in wave_y_train])

## EJERCICIO 3: Entrenar un clasificador k-NN con los datos reducidos por discriminante lineal. 
## Con n_neighbors=10 
## debe dar 0.848
X = wave_x_reduced_train.T
y = wave_y_train

knn = KNeighborsClassifier(n_neighbors=10).fit(X, y)

# imprimir el accuracy
print("Accurancy_score:", accuracy_score(wave_y_all, y_prima))
#nota  No such file or directory: 'movies2.csv'

#importar librerias 
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
#leer el dataset
peliculas = pd.read_csv("movies2.csv")
#seleccionar solo los valores númericos 
datos_numericos = peliculas.select_dtypes(np.number).fillna(0)
#configurar las variables objetivo e independientes 
objetivo = "ventas"
#las variables independientes serían todas las demas venos ventas 
independientes = datos_numericos.drop(columns = objetivo).columns
#creamos el modelo y lo ajustamos
modelo = LinearRegression()
modelo.fit(X = datos_numericos[independientes], y = datos_numericos[objetivo])
#agregamos las predicciones como una nueva columna del dataset original
peliculas["ventas_prediccion"] = modelo.predict(datos_numericos[independientes])
#mostrar solo los campos ventas y ventas predicción
print(peliculas[["ventas", "ventas_prediccion"]].head())

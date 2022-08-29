import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
#leer el archivo y establecer las variables objetivo e independientes 
ventas = pd.read_csv("ventas2.csv")
objetivo = "monto"
independientes = ventas.drop(columns = ['monto']).columns
#crear el modelo
modelo = LinearRegression()
modelo.fit(X = ventas[independientes], y= ventas[objetivo])
#crear el conjunto
ventas["ventas_prediccion"] = modelo.predict(ventas[independientes])
preds = ventas[["monto", "ventas_prediccion"]].head(50)
#realizar prediccion
talvez = modelo.predict([[41,1,1,1]])
print("tal vez compre: ")
print (talvez)
#graficar
import matplotlib.pyplot as plt 
preds.plot(kind = 'bar', figsize = (18,8))
plt.grid(linewidth ='2')
plt.grid(linewidth ='2')
plt.grid(None)
plt.show()
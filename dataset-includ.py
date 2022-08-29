#librerías
from sklearn import datasets
from sklearn.linear_model import LinearRegression 
import pandas as pd
import numpy as np

#cargar el dataset 
dataset = datasets.load_boston()
#configurar las variables objetivo e independientes 
objetivo = dataset['target']
independientes = dataset['data']
#crear modelo
modelo = LinearRegression()
#método fit
modelo.fit(X = independientes, y = objetivo)
#método predict
predicciones = modelo.predict(independientes)
for y, y_pred in list (zip(objetivo, predicciones))[:5]:
    print("valor real: {:.3f} valor estimado: {:.5f}".format(y,y_pred))


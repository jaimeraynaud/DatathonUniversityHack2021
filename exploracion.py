'''En este script Exploracion realizamos una exploracion basica pero efectiva de los datos 
que nos permitira tomar decisiones sobre el tratamiento de los mismos para el reto Altamira Stock Prediction  
del Datathon Cajamar UniversityHack 2021,
realizado por sus miembros: Manuel Bueno Gómez, Pablo Santos Ortiz y Jaime Raynaud Sánchez'''

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


df = pd.read_csv('Modelar_UH2021.txt', sep='|', header=0, names=['fecha', 'id', 'visitas', 'categoria_uno',
                                                                                            'categoria_dos', 'estado', 'precio', 'dia_atipico', 'campaña', 'antiguedad', 'unidades_vendidas'])

'''Para comenzar a analizar los datos hacemos print de las variables y las primeras 5 filas de datos'''
print(df.head(5))

print('Cantidad de Filas y columnas:',df.shape)
print('Nombre de las features:\n',df.columns)

#Tipo de los distintos datos que aparecen:
print(df.info())

#Pandas filtra las features numericas y calcula datos estadísticos que pueden ser útiles: 
#cantidad, media, desvío estandar, valores máximo y mínimo
print(df.describe())
plt.figure(figsize=(9, 8))
sns.histplot(df['unidades_vendidas'], color='g', bins=100)
#Podemos observar que una gran mayoria de los datos para unidades_vendidas se encuentran por debajo del valor 200
#Es por ello que en versiones futuras del codigo seria interesante probar los resultados con una eliminacion de outliers

#Veamos si hay correlacion entre las variables:
corr = df.set_index('unidades_vendidas').corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()

#Missing data
#Veamos que porcentaje de valores nulos hay en cada feature:
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
print(missing_data.head(20))

#precio tiene un 65% de nulos, pero podemos arreglarlo con las indicaciones del enunciado.
#antiguedad tiene un 21% de nulos, por ello decidimos eliminar esta feature al no tener indicaciones de como solventarlo
#categoria_dos tiene 5844 nulos, por tanto trataremos estos datos usando ffill y bfill al tener 
#un único valor de categoria_dos por cada id'''





''' Este es el script Prediccion presentado por el equipo ETSII-Corp para el reto Atmira Stock Prediction
 del Datathon Cajamar UniversityHack 2021, 
 realizado por sus miembros: Manuel Bueno Gómez, Pablo Santos Ortiz y Jaime Raynaud Sánchez'''
 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt


def read_file():
    df = pd.read_csv('Modelar_UH2021.txt', sep='|', header=0, names=['fecha', 'id', 'visitas', 'categoria_uno',
     
                                                                                            'categoria_dos', 'estado', 'precio', 'dia_atipico', 'campaña', 'antiguedad', 'unidades_vendidas'])  
    return df

def read_estimar():
    df_estimar = pd.read_csv('Estimar2.txt', sep='|', header=0, names=['fecha', 'id', 'visitas', 'categoria_uno',
     
                                                                                            'categoria_dos', 'estado', 'precio', 'dia_atipico', 'campaña', 'antiguedad']) 
    return df_estimar

def reload_prices(df):
    
    ''' Teniendo en cuenta la descripcion que se nos da en el enunciado sobre la feature 'precios' 
    llevaremos a cabo la transformacion que se nos indica en el mismo, con el metodo fillna de pandas
    haciendo un forward fill y despues un backward fill de los datos nulos,
     agrupando primero los datos por producto'''
    
    frames = []
    for i in list(set(df['id'])):
        df_id = df[df['id'] == i]
        df_id.loc[:, 'precio'].fillna(method = 'ffill', inplace = True)
        df_id.loc[:, 'precio'].fillna(method = 'bfill', inplace = True)
        
        frames.append(df_id)
    
    final_df = pd.concat(frames)
    
    '''Eliminamos la columna antiguedad puesto que, a diferencia de precio, 
    no se nos da ninguna indicacion de como rellenar dichos datos, 
    y tiene un 21% de datos nulos, como vimos en el script Exploracion '''
    final_df = final_df.drop(['antiguedad'], axis=1)
    
    return final_df

def reload_categoria_dos(df):
    
    '''La misma logica es aplicada con categoria_dos, 
     puesto que todos los datos de un mismo producto tienen el mismo valor de categoria_dos,
     como pudimos ver durante la exploracion del dataset'''
    
    frames = []
    for i in list(set(df['id'])):
        df_id = df[df['id'] == i]
        df_id.loc[:, 'categoria_dos'].fillna(method = 'ffill', inplace = True)
        df_id.loc[:, 'categoria_dos'].fillna(method = 'bfill', inplace = True)
        
        frames.append(df_id)
    
    final_df = pd.concat(frames)
    
    return final_df



def refactor_data(df):
    
    '''Convertimos los datos de la columna fecha del dataset Modelar 
    al tipo datetime para poder trabajar con ellos'''
    
    df['fecha'] =  pd.to_datetime(df.loc[:,'fecha'], format='%d/%m/%Y %H:%M:%S')
 
    '''Ahora usaremos el metodo LabelEncoder que nos proporciona sklearn 
    para asi poder trabajar con las features categoria_uno y estado, 
    ya que ambas no son de tipo int o float, como pudimos ver en el script Exploracion,
    y necesitamos todos los datos de esos tipos 
    para poder aplicar el xgboost'''
    
    enc1 = LabelEncoder()
    enc1.fit(df['categoria_uno'])
    df['categoria_uno'] = enc1.transform(df['categoria_uno'])
    
    enc2 = LabelEncoder()
    enc2.fit(df['estado'])
    df['estado'] = enc2.transform(df['estado'])
    return df

def refactor_estimar(df):
    '''Aplicamos la misma logica que en la funcion refactor_data 
    solo que esta vez para el formato dado en el Dataframe Estimar'''
    
    df['fecha'] =  pd.to_datetime(df.loc[:,'fecha'], format='%Y-%m-%d')
    
    enc1 = LabelEncoder()
    enc1.fit(df['categoria_uno'])
    df['categoria_uno'] = enc1.transform(df['categoria_uno'])
    
    enc2 = LabelEncoder()
    enc2.fit(df['estado'])
    df['estado'] = enc2.transform(df['estado'])
    
    df['categoria_dos'] = df['categoria_dos'].astype(int)
    
    return df
    

def fecha_to_dia_mes_año(df):
    
    '''Para poder aplicar nuestro modelo predictivo dividimos la feature fecha en tres: dia, mes y año'''
    df['dia'] = df.loc[:,'fecha'].dt.day
    df['mes'] = df.loc[:,'fecha'].dt.month
    df['año'] = df.loc[:,'fecha'].dt.year
    
    '''Pasamos la feature precio al tipo float para poder trabajar con ella'''
    df['precio'] = df['precio'].apply(lambda x: x.replace(',','.') if type(x) == str else x)
    df['precio'] = df['precio'].astype(float)
    
    '''Eliminamos la feature fecha, que ya no es necesaria'''
    final_df = df.drop(['fecha'], axis=1)
    
    return final_df

def xgboost_f(df):
    
    '''La funcion xgbboost_f nos permitira validar nuestro modelo usando Extreme Gradient Boost Regression'''
    
    X = df.loc[:,['dia','mes','año','id', 'visitas', 'categoria_uno','categoria_dos', 'estado', 'precio', 'dia_atipico', 'campaña']]
    y = df.loc[:, 'unidades_vendidas']
    
    '''Dividimos el dataframe en train y test set para poder entrenar nuestro modelo 
    con train y validar los resultados en test, para ello usamos el metodo train_test_split de sklearn
    con un tamaño sobre el original del 25% y 75% para el test y train set respectivamente,
    con una semilla aleatoria fija de 42 para permitir la reproducibilidad de nuestros resultados'''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, train_size = 0.75, random_state=42)
    
    '''El siguiente paso es iniciar nuestro XGBoost regressor object llamando a 
    XGBRegressor con sus respectivos hyper-parameters como argumentos'''
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
    
    '''Ahora ajustamos el regressor a nuestro training set y predecimos los valores de y_test con X_test'''
    xg_reg.fit(X_train,y_train)
    preds = xg_reg.predict(X_test)
    
    '''Redondeamos con floor al tratarse de unidades_vendidas, que debe ser del tipo int y usamos floor ya que consideramos que, 
    por ejemplo, 1.6 unidades_vendidas deberia traducirse como 1 unidadades_vendidas'''
    preds = np.floor(preds)
    
    '''Calculamos el RMSE con el metodo mean_squared_error de sklearn'''
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    '''Obtenemos por pantalla un RMSE: 16.310883, teniendo en cuenta que 
    los valores de unidades_vendidas varian de 0 a 4881, nos parece un resultado satisfactorio'''
    print("RMSE: %f" % (rmse))
    
    '''Como parte de la solucion mostramos una grafica con aquellas variables 
    que han resultado mas relevantes para el modelo usando el metodo plot_importance de xgb'''
    xgb.plot_importance(xg_reg)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()

def prediccion(df, df_apredecir):
    
    '''Finalmente, tras haber validado nuestro modelo con los train y test sets 
    obteniendo un resultado satisfactorio, usaremos este mismo modelo pero ajustandolo 
    a todo el dataframe Modelar para obtener una mayor precision en las medidas sobre el dataframe Estimar'''
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
    X = df.loc[:,['dia','mes','año','id', 'visitas', 'categoria_uno','categoria_dos', 'estado', 'precio', 'dia_atipico', 'campaña']]
    y = df.loc[:, 'unidades_vendidas']
    xg_reg.fit(X,y)
    
    '''Obtenemos los valores predichos de unidades_vendidas para el dataframe Estimar'''
    preds = xg_reg.predict(df_apredecir)
    
    return preds

if __name__ == "__main__":
    
    '''Cargamos y leemos el archivo Modelar en la variable df'''
    df = read_file()
    
    '''Le aplicamos las modificaciones necesarias para trabajar con el obteniendo final_df'''
    final_df = fecha_to_dia_mes_año(refactor_data(reload_categoria_dos(reload_prices(df))))
    
    '''Cargamos y leemos el archivo Estimar en la variable df_estimar'''
    df_estimar = read_estimar()
    
    '''Le aplicamos los cambios pertinentes, en el caso de categoria_dos, no encontramos Nan's 
    sino '-' por lo que, como estan ordenados por id, aplicamos un ffill para sustituirlos, 
    ya que dentro de un mismo producto se tienen los mismos valores de categoria_dos'''
    df_estimar['categoria_dos'].replace('-', method='ffill', inplace = True)
    df_estimar = reload_prices(df_estimar)
    df_estimar_sin_fecha = fecha_to_dia_mes_año(refactor_estimar(df_estimar))
    
    df_estimar_sin_fecha = df_estimar_sin_fecha[['dia','mes','año','id', 'visitas', 'categoria_uno','categoria_dos', 'estado', 'precio', 'dia_atipico', 'campaña']]
    
    #Obtenemos los valores de las unidades_vendidas para Estimar llamando a la funcion prediccion
    preds = prediccion(final_df, df_estimar_sin_fecha)

    df_estimar['unidades_vendidas'] = preds.tolist()
    df_estimar['unidades_vendidas'] = df_estimar['unidades_vendidas'].apply(np.floor).astype(int)
    
    '''Finalmente habremos obtenido el Dataframe df_a_presentar en el que se encuentran los datos 
    en el formato indicado por Cajamar para su correcta presentacion, ademas generamos un fichero txt con los resultados'''
    df_a_presentar = df_estimar[['fecha','id','unidades_vendidas']]
    df_a_presentar['fecha'] = df_a_presentar['fecha'].dt.strftime('%d/%m/%Y')
    #Para generar el txt:
    #df_a_presentar.to_csv(r'C:\Users\Jaime\Desktop\Reto Cajamar\ETSII-Corp.txt', sep = '|', index = False, header = ['FECHA','ID','UNIDADES'])
    
    # Si queremos ver el valor de RMSE para el train y test set debemos descomentar la siguiente linea:
    xgboost_f(final_df)
    
    print(df_a_presentar.head(10))
    
    '''ACLARACION: Se deben ignorar los errores del tipo: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    See the caveats in the documentation: 
    https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    puesto que no estan afectando a la resolucion del problema'''
    
    
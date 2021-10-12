#!/usr/bin/env python
# coding: utf-8

# # Clasificador de tejido sano y tejido cancerígeno en imagenes microscópicas del esófago

# ![image.png](attachment:image.png)

# #### Paso 1 - Cargar librerías, módulos y funciones que utilizaremos.
# 
# #### Paso 2 - Lectura de datos y adecuación de formato.
# 
# #### Paso 3 - Entrenamiento utilizando perceptrón simple con conjunto de entrenamiento y prueba
# 
# #### Paso 4. Entrenamiento de perceptron utilizando el método de validación K-fold Cross Validation

# ## Paso 1. Cargar librerías y módulos requerdidos

# In[1]:


import os            #módulo para manejar carpetas y archivos en nuestro ordenador
import random        #módulo para aleatorizar

import numpy as np   #librería para manejar matrices y operaciones de matrices 
import pandas as pd  #librería para manejar tablas de datos


                                       #Skimage (Scikit-image): librería para procesamiento de imagenes
from skimage import io                 #Modulo para leer una imagen (librería para procesamiento de imagenes)
from skimage.transform import rescale  #Función dentro del modulo transform, para cambiar el tamaño de una imagen 


                                                       #Sklearn (Scikit-learn): librería para machine learning 
from sklearn.model_selection import train_test_split   
from sklearn.linear_model import Perceptron 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt        #Para graficar y visualizar


# para especificar que queremos las graficas en linea sin necesidad de usar plt.show()
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Paso 2. Lectura de datos. 

# X=Imagenes
# Y=Etiqueta con la clase a la que pertenece
# 
# Para X, Scikilearn admite arreglos de dos dimensiones ( ya sea un np.arrays de dos dimensiones o un data frame de pandas)
# 
# Para Y, puede ser un vector, o matriz de una sola dimensión, de np.arrays o una serie de pandas)
# 

# ### 2.1 Lectura de etiquetas (Y)

# Para leer un archivo en Python, debemos indicar la carpeta donde se encuentra.
# 
# Para obtener la ruta podemos copiarla desde nuestro explorador de archivos.
# En Windows deberemos remplazar el símbolo la diagonal invertida \ por / o bien \\  
# 
# Ejemplo:
# 
# Remplazar  'C:\mydir'
# 
# Por        'C:/mydir'
# o bien     'C:\\mydir'
# 
# Así la ruta          'C:\Users\anais\Documents\AnIs Analytics\Perceptrón\Implementación clasificación cáncer\DatosCancer/'
# se declarará como    'C:/Users/anais/Documents/AnIs Analytics/Perceptrón/Implementación clasificación cáncer/DatosCancer/'  

# In[2]:


#declaramos la ruta donde se encuentran nuestros datos

path = 'C:/Users/rafae/Documents/'


#Leemos los datos

datos = pd.read_csv(path+"ClasesImagenes.csv")


# In[3]:


datos.head()  #muestra los primeros cinco registros


# In[4]:


#¿Con cuántas imagenes vamos a trabajar?

datos.shape[0]


# In[5]:


#¿cuántas imagenes tenemos de cada clase?

datos['class_number'].value_counts(sort=False)  


# Clase 0 corresponde a tejido sano y Clase 1 a tejido Cancerígeno

# In[6]:


Y=datos.class_number   #Guardamos las etiquetas de las imagenes como serie de pandas


# ### 2.2 Lectura de imágenes (X)

# In[7]:


#Al nombre de cada imagen necesitamos añadirle la ruta del directorio en el que se encuentra 

image_path = "C:\\Users\\rafae\\Documents\\CarpetaImagenesCorregida.zip\\"
datos['image_filename'] = datos['image_filename'].apply(lambda x: [os.path.join(image_path,x)])
datos['image_filename'] = datos['image_filename'].str.get(0)

datos.head(3)


# In[8]:


#Cargamos las imagenes del conjunto d entrenamiento como lista de matrices (np.arrays)
# (4 min)

get_ipython().run_line_magic('time', "img= [io.imread(i, as_gray=True) for i in datos['image_filename'][:]]")


# In[9]:


img[0].shape


# - Dimensión 0: 260 renglones de pixeles de arriba a abajo
# - Dimension 1: 260 columnas de pixeles de izquierda a derecha
# 

# ####  Transformar imágenes en vectores
# Lo primero es transformar la lista img en una matriz
# 

# In[10]:


IMG =np.stack(img, axis=0)  # Toma una secuencia de matrices y las apila a lo largo 
                                # de un tercer eje para hacer una sola matriz              
IMG.shape


# Despues aplanar la matriz de 2D (260x260 pixeles) de cada imagen, para obtener un vector por imagen. 

# In[11]:


n_imagenes=len(IMG)
X = IMG.reshape((n_imagenes, -1))
X.shape


# ##### Visualizamos ejemplos de ambas clases de imagenes

# In[12]:


#El método GroupBy de Pandas separa un data frame en varios data frames
porClase=datos.groupby('class_number')

#elije al azar n muestras de cada subconjunto y guarda la posición de las figuras elegidas en una lista
n=20
c=random.sample(porClase.get_group(1).index.tolist(),n)
s=random.sample(porClase.get_group(0).index.tolist(),n)

# Grafica 20 imágenes aleatorias de tejido con cáncer y 20 de tejido sano

fig=plt.figure(figsize=(16, 8))
columns = 10
rows = 4
for i in range(0, columns*rows):
    fig.add_subplot( rows, columns, i+1)
    if i < 20:
        plt.imshow(img[c[i]], cmap='Greys_r')
        plt.title('cancer')
        plt.xticks([])
        plt.yticks([])
    else:
        plt.imshow(img[s[i-20]], cmap='Greys_r')
        plt.title('tejido sano')
        plt.xticks([])
        plt.yticks([])
plt.show()


# ## Paso 3. Entrenamiento utilizando perceptrón simple con conjunto de entrenamiento y prueba

# ### 3.1  Crear subconjuntos de entrenamiento y prueba
# 
# 

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=0)


# In[14]:


# ¿Cómo son los conjuntos de entrenamiento y prueba?
print("Training set")
print("X: ", X_train.shape)
print("Y: ", y_train.shape)
unique, counts= np.unique(y_train, return_counts=True)
print('Tejido Sano:        ', counts[0],'\nDisplasia o Cáncer: ', counts[1],'\n')


print("Test set")
print("X: ", X_test.shape)
print("Y: ", y_test.shape)
unique, counts= np.unique(y_test, return_counts=True)
print('Tejido Sano:         ', counts[0],'\nDisplasia o Cáncer: ', counts[1],'\n')


# ### 3.2  Entrenamiento
# 
# 

# In[15]:


get_ipython().run_cell_magic('time', '', 'model = Perceptron(alpha=0.0001, fit_intercept=True, max_iter=1000, tol=0.001, eta0=1.0, random_state=0, early_stopping=False) \nmodel.fit(X_train,y_train)                                   \n\nprint("Accuracy: %.2f%%" % (model.score(X_test,y_test)*100.0)) ')


# In[16]:


y_predict=model.predict(X_test)


# In[19]:


accuracy_score(y_test, y_predict)


# ## Paso 4. Entrenamiento de perceptron utilizando el método de validación K-fold Cross Validation 

# (4 min aprox)

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nkfold=KFold(n_splits=5,random_state=0,  shuffle=True)\nmodel_kfold=Perceptron(alpha=0.0001, fit_intercept=True, max_iter=1000, tol=0.001, eta0=1.0, random_state=0, early_stopping=False)\nresult_kfold=cross_val_score(model_kfold, X, Y, cv=kfold )\nprint("Accuracy: %.2f (+/- %02f)" % (result_kfold.mean(),result_kfold.std()))')


# In[30]:


result_kfold


# In[ ]:





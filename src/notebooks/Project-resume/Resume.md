- Objetivo:  

El objetivo principal del proyecto es construir un modelo de clasificación que pueda predecir la calidad de una cerveza basándose en diversas características asociadas.  

- Importación de Datos:  

Desde la ruta src/data/raw se puede importar el arcvivo CSV denominado "beer_ML". Este se almacena en un DataFrame de pandas con el nombre de "beer" donde se realiza un análisis exploratorio y manipulaciones iniciales de los datos en el notebook "1-Análisis_y_Limpieza_de_Datos.ipynb". 

Luego de sufrir modificaciones en dicho notebook, se guarda otro archivo CSV con el nombre de "cerveza", el cual tiene la exploración y limpieza de los datos ya realizada. A éste último se lo puede encontrar en la ruta src/data/processed. Este es almacenado en un DataFrame de pandas con el nombre de "beer_ML" donde se llevan a cabo transformaciones, entrenamiento, predicción y evaluación de diferentes modelos de Machine Learning en el notebook "2-Modelos_Machine_Learning.ipynb". Y finalmente el modelo elegido que mejor explica el problema de negocio se guarda en el archivo llamado "MLP.model.pkl".  

- Preprocesamiento de Datos:  

En el notebook "1-Análisis_y_Limpieza_de_Datos.ipynb":  

Lectura preliminar de la información que proporciona el DataSet original (tipos de datos, variables categóricas, valores únicos, entre otros.) y la información estadística de los parámetros (media, mediana, cuartiles, desviación estandar, mínimo y máximo). 
Detección de observaciones duplicados, nulos y extremos.
Visualización de gráficos como histograma para analizar la distribución normal de los datos, diagrama de caja para evaluar la presencia de outliers, heatmap para analizar la significancia y posible correlación de los parámetros.
Eliminación de variables de entrada irrelevantes.
Renombrado de parámetros y modificación de decimales.

En el notebook "2-Modelos_Machine_Learning.ipynb":

Análisis de componentes principales (PCA) para reducir la dimensionalidad de las características.  
Se estandarizan y escalan las características para asegurar que todas tengan la misma escala.  
Discretización de la variable objetivo (opinión general) utilizando diferentes números de bins y se aplica el algoritmo SMOTE para balancear las clases.  
División de los datos en conjuntos de entrenamiento y prueba.  
Ajuste de varios modelos de clasificación, incluyendo Regresión Logística, Random Forest, AdaBoost, Árbol de Decisión, SVM, KNN y MLP, utilizando GridSearchCV para optimizar los hiperparámetros.  
Evaluación de los modelos en el conjunto de prueba utilizando métricas como precisión, recall, y F1-score.
Selección del mejor modelo de MLP basado en la puntuación de precisión obtenida durante la validación cruzada.  
Guardado del mejor modelo MLP en un archivo utilizando pickle.  

- Resultados y Conclusiones:  

El modelo MLP entrenado muestra un rendimiento aceptable en la clasificación de opiniones sobre cervezas.
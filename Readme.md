# README #

Este repositorio contiene un análisis y desarrollo de modelos de machine learning para predecir la opinión general sobre diferentes tipos de cerveza. A continuación, se detalla el contenido y los pasos realizados:

***Contenido del Repositorio:*** 

    beer_ML.csv: archivo CSV que contiene el DataSet original del proyecto.  

    cerveza.csv: archivo CSV que contiene los datos limpios.



**Análisis_y_Limpieza_de_Datos.ipynb**  
Notebook 1 de Jupyter que contiene el código Python para el análisis exploratorio y la limpieza de los datos.  

**Modelos_Machine_Learning.ipynb**  
Notebook 2 de Jupyter que contiene el código Python para el preprocesamiento, la construcción y evaluación de modelos de machine learning.  

**model**: modelo final entrenado y guardado utilizando el mejor clasificador según los resultados obtenidos.  

**app.py**: aplicación de Streamlit para predicción de datos.

***Pasos Realizados en los Notebooks:***  

- Lectura de los datos desde el archivo beer_ML.csv
- Observación de la distribución de la variable objetivo o target (opinion_general).
- Análisis de la correlación entre las variables.  
- Aplicación de la transformación de Box-Cox para normalizar la distribución de los datos.
- Estandarización y escalado de las variables numéricas.
- Reducción de dimensionalidad mediante Análisis de Componentes Principales (PCA).
- Discretización de la variable objetivo para convertirla en un modelo de clasificación.
- Evaluación de varios modelos de clasificación, incluyendo Regresión Logística, Random Forest, AdaBoost, Árbol de Decisión, Support Vector Machine (SVM,) K-Nearest Neighbors (KNN), Redes Neuronales, entre otros.
- Optimización de hiperparámetros mediante búsqueda de cuadrícula (GridSearchCV) para mejorar el rendimiento del modelo seleccionado.
- Selección del mejor modelo basado en métricas de rendimiento como precisión, recall y F1-score.

***Para Utilizar el Modelo Entrenado:*** 

Cargar el archivo finished_model.model y utilizarlo para realizar predicciones sobre nuevos datos.
- Objetivo:  

El objetivo principal del proyecto es construir un modelo de clasificación que pueda predecir la calidad de una cerveza basándose en diversas características asociadas.  

- Importación de Datos:  

Desde la ruta src/data/raw se puede importar el arcvivo CSV denominado "beer_ML". Este se almacena en un DataFrame de pandas con el nombre de "beer" donde se realiza un análisis exploratorio y manipulaciones iniciales de los datos en el notebook "1-Análisis_y_Limpieza_de_Datos.ipynb".  

Luego de sufrir modificaciones en dicho notebook, se guarda otro archivo CSV con el nombre de "cerveza", el cual tiene la exploración y limpieza de los datos ya realizada. A éste último se lo puede encontrar en la ruta src/data/processed. Este es almacenado en un DataFrame de pandas con el nombre de "beer_ML" donde se llevan a cabo transformaciones, entrenamiento, predicción y evaluación de diferentes modelos de Machine Learning en el notebook "2-Modelos_Machine_Learning.ipynb". Y finalmente el modelo elegido que mejor explica el problema de negocio se guarda en el archivo llamado "MLP.model.pkl".  

|columna|cardinalidad|% cardinalidad|tipo de dato|valores unicos(si son menos de 15)|tipo de variable|
|---|---|---|---|---|---|
|id|1460|100|int64|NaN|id|
|mssubclass||||||
|MSZoning||||||
|LotFrontage||||||
---|---|---|


- Preprocesamiento de Datos:  

En el notebook "1-Análisis_y_Limpieza_de_Datos.ipynb":  


En el notebook "2-Modelos_Machine_Learning.ipynb":

Se realiza un análisis de componentes principales (PCA) para reducir la dimensionalidad de las características.
Se estandarizan y escalan las características para asegurar que todas tengan la misma escala.
Se discretiza la variable objetivo (opinión general) utilizando diferentes números de bins y se aplica el algoritmo SMOTE para balancear las clases.  

5. Modelado:  

Se dividen los datos en conjuntos de entrenamiento y prueba.
Se ajustan varios modelos de clasificación, incluyendo Regresión Logística, Random Forest, AdaBoost, Árbol de Decisión, SVM, KNN y MLP, utilizando GridSearchCV para optimizar los hiperparámetros.
Se selecciona el mejor modelo de MLP basado en la puntuación de precisión obtenida durante la validación cruzada.
Se guarda el mejor modelo MLP en un archivo utilizando pickle.  

6. Evaluación del Modelo:  

Se evalúa el modelo MLP en el conjunto de prueba utilizando métricas como precisión, recall, y F1-score.
Se muestran las predicciones realizadas por el modelo en los datos de prueba.  

7. Resultados y Conclusiones:  

El modelo MLP entrenado muestra un rendimiento aceptable en la clasificación de opiniones sobre cervezas. Las futuras mejoras pueden incluir la exploración de diferentes algoritmos de clasificación y técnicas de ingeniería de características para mejorar aún más la precisión del modelo.
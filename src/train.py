#!/usr/bin/env python
# coding: utf-8

# LIBRERIAS

# In[1]:


import numpy as np
import pandas as pd
 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split,cross_val_score, cross_validate, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, silhouette_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter


# IMPORTACIÓN DE DATASET

# In[2]:


# beer_ML = pd.read_csv("./cerveza.csv")
beer_ML = pd.read_csv("../data/processed/cerveza.csv")
beer_ML = beer_ML.copy()
beer_ML.head()


# In[3]:


# Observo la brecha de valores de mi Target
beer_ML['opinion_general'].describe()[["min","max"]]


# PCA - Análisis de los Componentes Principales

# In[4]:


# DataFrame sin strings
beer_ML_numerico = beer_ML.drop(["nombre",	"estilo"], axis=1)
beer_ML_numerico


# Estandarizo las variables que se encuentran mayormente correlacionadas

# In[5]:


variables_stand = beer_ML_numerico.drop(["alcohol_%",	"min_IBU",	"max_IBU",	"astringencia",	"cuerpo",	"alcohol_gr/lt",	"amargor",	"dulzor",	"acidez",	"salado",	"frutado",	"lupulo",	"especias",	"malteado"], axis=1)

stand = StandardScaler()
beer_ML_numerico[variables_stand.columns] = stand.fit_transform(beer_ML_numerico[variables_stand.columns])


# Escalo todas las columnas y convierto el resultado en un DataFrame de pandas

# In[6]:


min_max_scaler = MinMaxScaler()
beer_ML_numerico_scaled = min_max_scaler.fit_transform(beer_ML_numerico)

beer_ML_numerico_scaled = pd.DataFrame(beer_ML_numerico_scaled, columns=beer_ML_numerico.columns)


# In[7]:


# Observo la media y la mediana luego de estandarizar
beer_ML_numerico_scaled.describe()


# Las medias cercanas a cero y las desviaciones estándar cercanas a uno indican que los datos están centrados y escalados adecuadamente.

# In[8]:


beer_ML_PCA = beer_ML_numerico_scaled.drop(["opinion_general"], axis=1)


# In[9]:


pca = PCA()
beer_pos_PCA = pca.fit_transform(beer_ML_PCA)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Número de Componentes Principales")
plt.ylabel("Varianza Explicada Acumulada")
plt.title("Análisis de Varianza Explicada")
plt.grid(True)
plt.show()


# In[10]:


# Averiguo el número de componentes principales para el 95% de varianza

varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
num_componentes_95 = np.argmax(varianza_acumulada >= 0.95) + 1

print(f"Número de componentes principales para el 95% de varianza: {num_componentes_95}")


# In[11]:


beer_pos_PCA_ = pd.DataFrame(beer_pos_PCA)

# Corroboro la nueva matriz de correlación
correlation_matrix = beer_pos_PCA_.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Heatmap de Correlación después de PCA')
plt.show()


# MODELO: CLASIFICACION (REGRESIÓN LOGÍSTICA)  
# 
# TARGET = "opinión_general"  

# División en datos de entrenamiento y datos de prueba:

# In[12]:


# x_clasif = beer_ML.drop(columns=["nombre", "estilo", "opinion_general"])
# y_clasif = beer_ML["opinion_general"]

x_clasif = beer_pos_PCA_
y_clasif = beer_ML_numerico_scaled["opinion_general"]

x_train, x_test, y_train, y_test = train_test_split(x_clasif,y_clasif,test_size=0.25, random_state=40)


# In[13]:


possible_bins = range(2, 4) # Valores de bins a probar

mean_accuracy_scores = []

# Itero sobre cada valor de bins y evalúo el rendimiento del modelo
for num_bins in possible_bins:
    # Discretizo la variable
    y_discretized = pd.cut(y_clasif, bins=num_bins, labels=False)
    
    # Calculo la precisión media utilizando CV
    clf = DecisionTreeClassifier()
    accuracy_scores = cross_val_score(clf, x_clasif, y_discretized, cv=5)  # Utiliza 5-fold cross-validation
    mean_accuracy = np.mean(accuracy_scores)
    mean_accuracy_scores.append(mean_accuracy)

# Averiguo el número óptimo de bins con la precisión media más alta
optimal_bins = possible_bins[np.argmax(mean_accuracy_scores)]
print("Número óptimo de bins:", optimal_bins)


# In[14]:


inertia_values = []         # inercia
silhouette_scores = []      # silueta

# Valores de "k" a probar
k_values = range(2, 6)

# Calculo la inercia y el coeficiente de silueta para diferentes valores de "k"
for k in k_values:

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(y_clasif.values.reshape(-1, 1))
    inertia_values.append(kmeans.inertia_)
    
    y_pred = kmeans.predict(y_clasif.values.reshape(-1, 1))
    silhouette_avg = silhouette_score(y_clasif.values.reshape(-1, 1), y_pred)
    silhouette_scores.append(silhouette_avg)


# Valor de k que maximiza el coeficiente de silueta
optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
print(f"El número óptimo de clusters según el Método de la Silueta es: {optimal_k_silhouette}")

def find_elbow(inertia_values):
    deltas = [inertia_values[i] - inertia_values[i+1] for i in range(len(inertia_values)-1)]
    max_delta_idx = deltas.index(max(deltas))
    return max_delta_idx + 1

# Punto donde se encuentra el codo
elbow_idx = find_elbow(inertia_values)

# Número óptimo de clusters según el método del codo
optimal_k_elbow = k_values[elbow_idx]
print(f"El número óptimo de clusters según el Método del Codo es: {optimal_k_elbow}")

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('Inercia', color=color)
ax1.plot(k_values, inertia_values, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Coeficiente de Silueta', color=color)  
ax2.plot(k_values, silhouette_scores, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.title('Método del Codo y Método de la Silueta')
plt.show()


# In[15]:


# Agrego la columna de etiquetas discretizadas al nuevo DataFrame
beer_ML_plus = beer_ML.copy()
beer_ML_plus['etiqueta'] = y_discretized  


# In[16]:


etiquetas_unicas = beer_ML_plus['etiqueta'].nunique()
etiquetas_ = beer_ML_plus['etiqueta'].unique()
type_etiquetas = beer_ML_plus['etiqueta'].dtype

print(f"La TARGET queda conformada por {etiquetas_unicas} etiquetas, que son {etiquetas_} y el tipo de dato es {type_etiquetas}.")


# In[17]:


k_values = range(2, 5)
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_clasif)
    silhouette_scores.append(kmeans.inertia_)

optimal_k = k_values[np.argmin(silhouette_scores)]
print("Número óptimo de clusters (k):", optimal_k)

# Ajusto K-Means con el número óptimo de clusters
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_optimal.fit(x_clasif)

# Asigno las etiquetas de los clusters a los datos
cluster_labels = kmeans_optimal.labels_

plt.scatter(x_clasif.iloc[:, 0], x_clasif.iloc[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering with Cluster Centers')
plt.show()


# In[18]:


# Calculo la frecuencia de cada clase
frecuencia_clases = beer_ML_plus['etiqueta'].value_counts()

porcentaje_clases = (frecuencia_clases / len(beer_ML_plus['etiqueta'])) * 100

print("Frecuencia de cada clase:")
print(frecuencia_clases)
print("\nPorcentaje de datos en cada clase:")
print(porcentaje_clases)


# NUEVA DIVISIÓN DE DATOS LUEGO DE PCA Y DISCRETIZACIÓN DE LA TARGET

# In[19]:


X = beer_pos_PCA_
Y = beer_ML_plus['etiqueta']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# BALANCEADO DE DATASET

# In[20]:


# Conteo de etiquetas antes de aplicar SMOTE
conteo_etiquetas_original = Counter(Y)
print("Conteo de Etiquetas (antes de SMOTE):", conteo_etiquetas_original)

smote = SMOTE(random_state=42)

# Aplico SMOTE para generar ejemplos sintéticos
X_resampled, Y_resampled = smote.fit_resample(X, Y)

# Conteo de etiquetas después de aplicar SMOTE
nuevo_conteo_etiquetas = Counter(Y_resampled)
print("Nuevo conteo de Etiquetas (después de SMOTE):", nuevo_conteo_etiquetas)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(conteo_etiquetas_original.keys(), conteo_etiquetas_original.values())
plt.title("Distribución de Etiquetas (Antes de SMOTE)")
plt.xlabel("Etiqueta")
plt.ylabel("Frecuencia")

plt.subplot(1, 2, 2)
plt.bar(nuevo_conteo_etiquetas.keys(), nuevo_conteo_etiquetas.values())
plt.title("Distribución de Etiquetas (Después de SMOTE)")
plt.xlabel("Etiqueta")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()


# Cross-Validation:

# In[25]:


clf = RandomForestClassifier(random_state=40)

cross_validation_scores = cross_val_score(clf, X_train, Y_train, cv=5, scoring='balanced_accuracy')

print("Precisión promedio en validación cruzada: {:.2f}".format(cross_validation_scores.mean()))


# ENTRENAMIENTO, PREDICCIÓN Y EVALUACIÓN DE MODELOS

# - REGRESIÓN LOGÍSTICA

# In[26]:


beer_ML_rlog = LogisticRegression()
beer_ML_rlog.fit(X, Y)

prediccion_RL = beer_ML_rlog.predict(X_test)

accuracy = accuracy_score(Y_test, prediccion_RL)
print("Precisión del modelo de Regresión Logística:", accuracy)


# - SVC

# In[27]:


svm_model = SVC()

svm_model.fit(X, Y)

prediccion_SVC = svm_model.predict(X_test)

accuracy = accuracy_score(Y_test, prediccion_SVC)
print("Precisión del modelo de SVM:", accuracy)


# - KNN

# In[28]:


knn_model = KNeighborsClassifier()

knn_model.fit(X, Y)

prediccion_KNN = knn_model.predict(X_test)

accuracy = accuracy_score(Y_test, prediccion_KNN)
print("Precisión del modelo de KNN:", accuracy)


# - ÁRBOLES DE DECISIÓN

# In[29]:


arbol_model = DecisionTreeClassifier()

arbol_model.fit(X, Y)

prediccion_AD = arbol_model.predict(X_test)

accuracy = accuracy_score(Y_test, prediccion_AD)
print("Precisión del modelo de Árbol de Decisión:", accuracy)


# - RANDOM FOREST

# In[30]:


forest_model = RandomForestClassifier()

forest_model.fit(X, Y)

prediccion_RF = forest_model.predict(X_test)

accuracy = accuracy_score(Y_test, prediccion_RF)
print("Precisión del modelo de Random Forest:", accuracy)


# - ADABOOST

# In[31]:


adaboost_model = AdaBoostClassifier()

adaboost_model.fit(X, Y)
prediccion_ADABOOST = adaboost_model.predict(X_test)

accuracy = accuracy_score(Y_test, prediccion_ADABOOST)
print("Precisión del modelo de AdaBoost:", accuracy)


# - GRADIENT BOOSTING - XGBOOST

# In[32]:


xgboost_model = XGBClassifier()

xgboost_model.fit(X, Y)
prediccion_XGBoost = xgboost_model.predict(X_test)

accuracy = accuracy_score(Y_test, prediccion_XGBoost)
print("Precisión del modelo de XGBoost:", accuracy)


# - BAGGING

# In[33]:


bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=300,
                            max_samples=10, n_jobs=-1, random_state=42)
bag_clf.fit(X, Y)

prediccion_Bagging = bag_clf.predict(X_test)

accuracy = accuracy_score(Y_test, prediccion_Bagging)
print("Precisión del modelo de Bagging:", accuracy)


# - REDES NEURONALES

# In[34]:


red_model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', alpha=0.001)

red_model.fit(X, Y)
prediccion_RED_NEU = red_model.predict(X_test)

accuracy = accuracy_score(Y_test, prediccion_RED_NEU)
print("Precisión del modelo de Redes Neuronales:", accuracy)


# ANÁLISIS DE LAS MÉTRICAS DE LOS MODELOS - REPORTES DE CLASIFICACIÓN

# In[35]:


modelos = {
    "Regresión Logística": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Arbol de Decisión": DecisionTreeClassifier(),
    "SVC": SVC(),
    "KNN": KNeighborsClassifier(),
}

hiperparametros = {
    "Regresión Logística": {'modelo__C': [1, 5, 10], 'modelo__penalty': ['l1','l2']},
    "Random Forest": {'modelo__n_estimators': [75, 100, 125], 'modelo__max_depth': [None, 5, 10, 15, 20], 'modelo__min_samples_split': [1, 5, 10], 'modelo__min_samples_leaf': [1, 5, 10]},
    "AdaBoost": {'modelo__n_estimators': [50, 100, 150], 'modelo__learning_rate': [0.1, 1, 10]},
    "Arbol de Decisión": {'modelo__criterion': ['gini','entropy'], 'modelo__splitter': ['best','random'], 'modelo__max_features': [None,'sqrt']},
    "SVC": {'modelo__C': [0.01, 0.1, 1, 10], 'modelo__kernel': ['rbf', 'sigmoid'], 'modelo__gamma': ['scale','auto']},
    "KNN": {'modelo__n_neighbors': [1, 5, 10], 'modelo__weights': ['uniform','distance'], 'modelo__algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto']},
}

pipeline = []

for nombre_modelo, modelo in modelos.items():
    print(f"Entrenando y evaluando el modelo {nombre_modelo}")

    # Obtengo hiperparámetros específicos para el modelo
    hiperparametros_modelo = hiperparametros.get(nombre_modelo, {})

    pipeline = Pipeline([('modelo', modelo)])

    # Verifico si el modelo admite búsqueda de cuadrícula
    if hiperparametros_modelo:
        grid_search = GridSearchCV(estimator=pipeline, param_grid=hiperparametros_modelo, scoring='accuracy', cv=3)
        grid_search.fit(X, Y)
        mejor_modelo = grid_search.best_estimator_
    else:
        mejor_modelo = pipeline.fit(X, Y)

    prediccion_entrenamiento = mejor_modelo.predict(X_train)
    prediccion_prueba = mejor_modelo.predict(X_test)

    reporte_entrenamiento = classification_report(Y_train, prediccion_entrenamiento)
    reporte_prueba = classification_report(Y_test, prediccion_prueba)

    print(f"Informe de Clasificación para {nombre_modelo} en datos de entrenamiento:")
    print(reporte_entrenamiento)
    print(f"Informe de Clasificación para {nombre_modelo} en datos de prueba:")
    print(reporte_prueba)


# In[36]:


report = classification_report(Y_test, prediccion_RED_NEU)
print("\nInforme de clasificación:\n", report)


# In[37]:


print(classification_report(Y_test, bag_clf.predict(X_test)))


# In[38]:


modelos = {
    "Regresión Logística": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Arbol de Decisión": DecisionTreeClassifier(),
    "SVC": SVC(),
    "KNN": KNeighborsClassifier(),
    "Redes Neuronales": MLPClassifier()
    }

metricas = ["accuracy", "f1_macro", "recall_macro", "precision_macro", "roc_auc_ovr"]

resultados_dict = {}

for nombre_modelo, modelo in modelos.items():
    cv_resultados = cross_validate(modelo, X, Y, cv=5, scoring=metricas)
    
    for metrica in metricas:
        
        clave = f"{nombre_modelo}_{metrica}"
        resultados_dict[clave] = cv_resultados[f"test_{metrica}"].mean()
    
resultados = pd.DataFrame([resultados_dict])


# In[39]:


resultados.T.sort_values(by=0, ascending=False)


# In[40]:


models_gridsearch = {}

models = [
    ("Regresión Logística", LogisticRegression()),
    ("Random Forest", RandomForestClassifier()),
    ("AdaBoost", AdaBoostClassifier()),
    ("Arbol de Decisión", DecisionTreeClassifier()),
    ("SVC", SVC()),
    ("KNN", KNeighborsClassifier()),
    ("Redes Neuronales", MLPClassifier())
]

for model_name, model_instance in models:
    models_gridsearch[model_name] = GridSearchCV(model_instance,
                                                 param_grid={},
                                                 cv=10,
                                                 scoring="accuracy",
                                                 verbose=1,
                                                 n_jobs=-1)
    
    models_gridsearch[model_name].fit(X, Y)


# In[41]:


best_grids = [(i, j.best_score_) for i, j in models_gridsearch.items()]

best_grids = pd.DataFrame(best_grids, columns=["Grid", "Best score"]).sort_values(by="Best score", ascending=False)
best_grids


# In[42]:


models_gridsearch['Redes Neuronales'].best_estimator_


# MODELO ELEGIDO

# In[43]:


models_gridsearch['Redes Neuronales'].best_estimator_.score(X_test, Y_test)


# GUARDADO DE MODELO

# In[44]:


import pickle

with open('finished_model.model', "wb") as archivo_salida:
    pickle.dump(models_gridsearch['Redes Neuronales'].best_estimator_, archivo_salida)


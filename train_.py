# LIBRERIAS

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

import pickle

# IMPORTACIÓN DE DATASET

# beer_ML = pd.read_csv("./cerveza.csv")
beer_ML = pd.read_csv("../src/data/processed/cerveza.csv")
beer_ML = beer_ML.copy()
beer_ML.head()

# TRANSFORMACIONES

# PCA - Análisis de los Componentes Principales
beer_ML_numerico = beer_ML.drop(["nombre",	"estilo"], axis=1)
beer_ML_numerico

# Estandarizo las variables que se encuentran mayormente correlacionadas
variables_stand = beer_ML_numerico.drop(["alcohol_%",	"min_IBU",	"max_IBU",	"astringencia",	"cuerpo",	"alcohol_gr/lt",	"amargor",	"dulzor",	"acidez",	"salado",	"frutado",	"lupulo",	"especias",	"malteado"], axis=1)

stand = StandardScaler()
beer_ML_numerico[variables_stand.columns] = stand.fit_transform(beer_ML_numerico[variables_stand.columns])

# Escalo todas las columnas y convierto el resultado en un DataFrame de pandas
min_max_scaler = MinMaxScaler()
beer_ML_numerico_scaled = min_max_scaler.fit_transform(beer_ML_numerico)

beer_ML_numerico_scaled = pd.DataFrame(beer_ML_numerico_scaled, columns=beer_ML_numerico.columns)

beer_ML_PCA = beer_ML_numerico_scaled.drop(["opinion_general"], axis=1)

pca = PCA()
beer_pos_PCA = pca.fit_transform(beer_ML_PCA)
beer_pos_PCA_ = pd.DataFrame(beer_pos_PCA)

x_clasif = beer_pos_PCA_
y_clasif = beer_ML_numerico_scaled["opinion_general"]

x_train, x_test, y_train, y_test = train_test_split(x_clasif,y_clasif,test_size=0.25, random_state=40)

possible_bins = range(2, 4) # Valores de bins a probar

mean_accuracy_scores = []

# Itero sobre cada valor de bins y evalúo el rendimiento del modelo
for num_bins in possible_bins:
    # Discretizo la variable
    y_discretized = pd.cut(y_clasif, bins=num_bins, labels=False)
    

# Agrego la columna de etiquetas discretizadas al nuevo DataFrame
beer_ML_plus = beer_ML.copy()
beer_ML_plus['etiqueta'] = y_discretized  

# MODELO: CLASIFICACION (REGRESIÓN LOGÍSTICA)  
# TARGET = "opinión_general"  

# DIVISIÓN DE DATOS LUEGO DE PCA Y DISCRETIZACIÓN DE LA TARGET
X = beer_pos_PCA_
Y = beer_ML_plus['etiqueta']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# BALANCEO DE LAS ETIQUETAS

smote = SMOTE(random_state=42)

# Aplico SMOTE para generar ejemplos sintéticos
X_resampled, Y_resampled = smote.fit_resample(X, Y)

# ENTRENAMIENTO Y PREDICCIÓN DEL MODELO: MULTI LAYER PERCEPTRON (MLP)

mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', alpha=0.001)

mlp_model.fit(X, Y)
prediccion_MLP = mlp_model.predict(X_test)

accuracy = accuracy_score(Y_test, prediccion_MLP)
print("Precisión del modelo de MLP:", accuracy)

models_gridsearch = {}

models = [
    ("Regresión Logística", LogisticRegression()),
    ("Random Forest", RandomForestClassifier()),
    ("AdaBoost", AdaBoostClassifier()),
    ("Arbol de Decisión", DecisionTreeClassifier()),
    ("SVC", SVC()),
    ("KNN", KNeighborsClassifier()),
    ("MLP", MLPClassifier())
]

for model_name, model_instance in models:
    models_gridsearch[model_name] = GridSearchCV(model_instance,
                                                 param_grid={},
                                                 cv=10,
                                                 scoring="accuracy",
                                                 verbose=1,
                                                 n_jobs=-1)
    
    models_gridsearch[model_name].fit(X, Y)
best_grids = [(i, j.best_score_) for i, j in models_gridsearch.items()]

best_grids = pd.DataFrame(best_grids, columns=["Grid", "Best score"]).sort_values(by="Best score", ascending=False)
best_grids
models_gridsearch['MLP'].best_estimator_
models_gridsearch['MLP'].best_estimator_.score(X_test, Y_test)
with open('MLP.model.pkl', "wb") as archivo_salida:
    pickle.dump(models_gridsearch['MLP'].best_estimator_, archivo_salida)

# Usar el modelo MLP para hacer predicciones en los datos de prueba
predicciones_clasificacion = models_gridsearch['MLP'].best_estimator_.predict(X_test)

predicciones_df = pd.DataFrame({'Predicciones': predicciones_clasificacion}, index=X_test.index)

# Unir los datos de prueba con las predicciones
datos_prueba_con_predicciones = pd.concat([X_test, predicciones_df], axis=1)

# Mostrar los datos con las predicciones
print(datos_prueba_con_predicciones)
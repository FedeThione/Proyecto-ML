import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
import requests

# Función para cargar el modelo desde GitHub
@st.cache(allow_output_mutation=True)
def cargar_modelo_desde_github(url_modelo):
    response = requests.get(url_modelo)
    model = load_model(BytesIO(response.content))
    return model

# URL del modelo en GitHub
url_modelo_github = 'URL_DEL_MODELO_EN_GITHUB'

# Cargar el modelo
modelo = cargar_modelo_desde_github(url_modelo_github)

# Interfaz de usuario
st.title('Predicción con Modelo de Regresión de Redes Neuronales')

# Obtener entradas del usuario
input_variable_1 = st.slider('Variable 1', min_value=0.0, max_value=10.0, value=5.0)
input_variable_2 = st.slider('Variable 2', min_value=0.0, max_value=10.0, value=5.0)

# Predecir con el modelo
prediccion = modelo.predict([[input_variable_1, input_variable_2]])

# Mostrar la predicción al usuario
st.write(f'La predicción es: {prediccion[0][0]}')

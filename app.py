import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Cargar el modelo entrenado
def load_mlp_model(MLP_model):
    with open(MLP_model, 'rb') as f:
        model = pickle.load(f)
    return model

# Función para predecir
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    # Agregar imagen de fondo
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url('beer.fondo.jpg');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Predicción con Modelo de Regresión Logística de Multi Layer Perceptron')
    st.sidebar.header('Parámetros de Entrada')

    # Definir límites personalizados para cada característica
    feature_limits =  {
        'opinion_aroma': (2.59, 4.75),
        'opinion_apariencia': (3.01, 4.59),
        'opinion_paladar': (2.73, 4.67),
        'opinion_sabor': (2.71, 4.75)
    }

    # Crear campos de entrada para cada característica con límites personalizados
    input_data = []
    for feature_name, (min_value, max_value) in feature_limits.items():
        input_value = st.sidebar.number_input(feature_name, min_value=min_value, max_value=max_value, value=min_value)
        input_data.append(input_value)

    # Convertir los datos de entrada en un arreglo de numpy
    input_data = np.array([input_data])

    mlp_model_ = Path().cwd() / "model" / "MLP.model.pkl"
    
    model = load_mlp_model(mlp_model_)

    # Botón de predicción
    if st.button('Predecir'):
        prediction = predict(model, input_data)
        st.write('La predicción es:', prediction)

if __name__ == '__main__':
    main()

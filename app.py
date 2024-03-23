import streamlit as st
import numpy as np
import pickle
from pathlib import Path

# Cargo el modelo entrenado
def load_mlp_model(MLP_model):
    with open(MLP_model, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    st.markdown(
        """
        <div style="background-color:#f4f4f4;padding:10px;border-radius:10px;">
        <h1 style="color:black;text-align:center;">Predicción con Modelo de Regresión Logística de Multi Layer Perceptron</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Cargo imagen para el fondo
    st.image("beer.fondo.jpg", use_column_width=True)

    st.sidebar.header('Parámetros de Entrada')

    # Defino límites personalizados para cada parámetro
    feature_limits =  {
        'opinion_aroma': (2.59, 4.75),
        'opinion_apariencia': (3.01, 4.59),
        'opinion_paladar': (2.73, 4.67),
        'opinion_sabor': (2.71, 4.75),
        'alcohol_%': (1.2, 11.5),
        'min_IBU': (0.0, 40.0),
        'max_IBU 3': (0.0, 70.0),
        'astringencia': (0.0, 39.0),
        'cuerpo': (0.0, 101.0),
        'alcohol_gr/lt': (0.0, 46.0),
        'amargor': (0.0, 104.0),
        'dulzor': (0.0, 143.0),
        'acidez': (0.0, 88.0),
        'salado': (0.0, 2.0),
        'frutado': (0.0, 132.0),
        'lupulo': (0.0, 113.0),
        'especias': (0.0, 51.0),
        'malteado': (0.0, 190.0),
    }

    input_data = []
    for feature_name, (min_value, max_value) in feature_limits.items():
        st.sidebar.markdown(f"**{feature_name}:** Min: {min_value}, Max: {max_value}")
        input_value = st.sidebar.number_input(feature_name, min_value=min_value, max_value=max_value, value=min_value)
        input_data.append(input_value)

    input_data = np.array([input_data])

    mlp_model_ = Path().cwd() / "model" / "MLP.model.pkl"
    model = load_mlp_model(mlp_model_)

    # Realizar la predicción
    if st.button('Predecir'):
        prediction = predict(model, input_data)
        # Mapear los valores predichos a las etiquetas deseadas
        label_mapping = {
            0: 'Calidad baja',
            1: 'Calidad media',
            2: 'Calidad alta'
        }
        # Obtener la etiqueta correspondiente a la predicción
        predicted_label = label_mapping[prediction[0]]
        st.write('La calidad de la cerveza es:', predicted_label)

if __name__ == '__main__':
    main()

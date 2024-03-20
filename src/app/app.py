from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open('C:/Users/feder/OneDrive/Escritorio/Proyecto_Machine_Learning/src/notebooks/finished_model.model', 'rb') as archivo_entrada:
    modelo_entrenado = pickle.load(archivo_entrada)

@app.route('/')
def index():
    return render_template('index.html')

# Ruta para la predicción de la opinión general sobre la cerveza
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos de entrada del formulario
    aroma = float(request.form['aroma'])
    apariencia = float(request.form['apariencia'])
    paladar = float(request.form['paladar'])
    sabor = float(request.form['sabor'])

    # Realizar la predicción utilizando el modelo cargado
    prediction = model.predict(np.array([[aroma, apariencia, paladar, sabor]]))[0]

    # Devolver la predicción como respuesta
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, render_template
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Función para procesar texto
def procesar_texto(texto):
    stop_words = set(stopwords.words('spanish'))
    lemmatizer = WordNetLemmatizer()
    texto = unidecode(texto)
    r = re.sub('[^a-zA-Z]', ' ', texto)
    r = r.lower()
    r = r.split()
    r = [word for word in r if word not in stop_words]
    r = [lemmatizer.lemmatize(word) for word in r]
    r = ' '.join(r)
    return r

# Función para leer el contenido de los archivos de texto
def leer_archivo(ruta):
    with open(ruta, 'r', encoding='utf-8') as archivo:
        contenido = archivo.read()
    return contenido

# Leer y procesar los archivos de culpables e inocentes
culpable_vector = procesar_texto(leer_archivo('culpables_txt.txt'))
inocente_vector = procesar_texto(leer_archivo('inocentes_txt.txt'))

# Crear un DataFrame con los nuevos textos
df_nuevos = pd.DataFrame({'texto': [culpable_vector, inocente_vector], 'etiqueta': ['culpable', 'inocente']})

# Vectorización y cálculo de similitud de coseno
vector = TfidfVectorizer()
matriz_TF_IDF = vector.fit_transform(df_nuevos['texto'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clasificar', methods=['POST'])
def clasificar():
    sentencia = request.form['sentencia']
    texto_procesado = procesar_texto(sentencia)

    # Crear un DataFrame temporal para la nueva sentencia
    df_nuevo = pd.DataFrame({'texto': [texto_procesado]})
    
    # Concatenar con los vectores de culpables e inocentes
    df_completo = pd.concat([df_nuevos, df_nuevo], ignore_index=True)
    
    # Recalcular la vectorización y la similitud
    matriz_TF_IDF_completa = vector.fit_transform(df_completo['texto'])
    matriz_coseno_completa = cosine_similarity(matriz_TF_IDF_completa, matriz_TF_IDF_completa)
    
    # Comparar la nueva sentencia con los vectores de culpabilidad e inocencia
    similitud_culpable = matriz_coseno_completa[-1, 0]  # Similaridad con el vector de culpabilidad
    similitud_inocente = matriz_coseno_completa[-1, 1]  # Similaridad con el vector de inocencia
    
    # Clasificar como culpable o inocente
    clasificacion = "culpable" if similitud_culpable > similitud_inocente else "inocente"
    
    return render_template('resultado.html', sentencia=sentencia, clasificacion=clasificacion,
                        similitud_culpable=similitud_culpable, similitud_inocente=similitud_inocente)

if __name__ == '__main__':
    app.run(debug=True)

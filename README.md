Este proyecto es una aplicación web construida con Flask que permite clasificar textos legales o sentencias en dos categorías: culpable o inocente.
El sistema funciona de la siguiente manera:
Preprocesa el texto ingresado eliminando caracteres especiales, stopwords y aplicando lematización.
Compara la sentencia con textos de referencia ("culpables_txt.txt" e "inocentes_txt.txt").
Utiliza TF-IDF para vectorizar los textos y la similitud de coseno para medir qué tan parecido es el texto nuevo a cada clase.
Devuelve el resultado en una interfaz web (index.html → resultado.html), indicando si el texto ingresado es más similar a "culpable" o a "inocente".

Tecnologías utilizadas:
1. Python
2. Flask
3. NLTK (stopwords, lematización)
4. scikit-learn (TF-IDF, similitud de coseno)
5. NumPy y Pandas
6. HTML (templates index.html y resultado.html)

# Sistemas Inteligentes para la Gestión de la Empresa

## Máster en Ingeniería Informática 19-20.

### Ejercicios 

1. Descripción de los principios básicos de funcionamiento del algoritmo *AlphaGo* (máximo 5 páginas). 

2. Uso de redes neuronales con Keras para resolver el problema de clasificación de la supervivencia en el Titanic a partir de un primer fichero con un modelo sencillo. El objetivo reside en mejorar este modelo predictivo realizando diversas modifiaciones en la arquitectura de la red, en los algoritmos de optimización, hiperparámetros, como la tasa de aprendizaje, o en otros aspectos como el número de épocas, el tamaño del *batch*, etc.

### Trabajo

Trabajo teórico con dos casos prácticos y presentación sobre las Redes Generativas Adversarias (*GANs*).

### Prácticas

1. Práctica 1: Pre-procesamiento de datos y clasificación binaria.

Para ello hemos hecho uso del *dataset* Kaggle IEEE-CIS Fraud Detection

El problema consiste en predecir si una transacción online es fraudulenta o no (isFraud) a partir del resto de variables usando como *dataset* el denominado **Kaggle IEEE-CIS Fraud Detection** que se puede encontrar en Kaggle . Los datos están separados en dos ficheros:
* *transaction*, con los datos de la propia transacción (393 variables + 1 identificador de transacción)
* *identity*, con los datos de identidad de la persona que realiza la transacción (40 variables + 1 identificador de transacción asociada)

Ambos ficheros pueden combinarse a través del atributo TransactionID. No todas las transacciones tienen asociada información de identidad. A su vez, los ficheros aparecen ya separados en conjuntos de entrenamiento y validación.

El ejercicio se abordará como un problema de clasificación binaria, con dos posibles salidas: {Yes, No}. La selección de datos y el procesamiento queda a criterio del estudiante.

2. Práctica 2: Deep Learning para clasificación de imágenes.

En esta segunda práctica se pretende desarrollar un modelo de clasificación de imágenes basado en redes neuronales profundas. Para ello, se trabajará sobre el conjunto de imágenes **FASHION-MNIST** que se puede cargar directamente desde Keras.

El problema consiste en predecir la prenda a la que corresponde cada imagen, codificada con un valor numérico que representa un tipo de prenda:
* 0 - T-shirt/top
* 1 - Trouser
* 2 - Pullover
* 3 - Dress
* 4 - Coat
* 5 - Sandal
* 6 - Shirt
* 7 - Sneaker
* 8 - Bag
* 9 - Ankle boot

El repositorio para esta práctica se encuentra [aquí](https://github.com/ProyectosComunes-MII/SIGE-P2)

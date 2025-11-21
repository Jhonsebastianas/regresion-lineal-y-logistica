# USO DE REGRESIÓN LOGÍSTICA Y REDES NEURONALES PARA LA CLASIFICACIÓN DE TIPO DE VIVIENDA Y PRECIO EN EL DATASET DE “ARRIENDOS - FINCA RAÍZ”

**Autores:**
Breyner Andres Taborda Agudelo, Jhon Sebastian Agudelo Sierra
Politécnico Colombiano Jaime Isaza Cadavid
breyner_taborda23191@elpoli.edu.co, jhon_agudelo23191@elpoli.edu.co

## Resumen
En este proyecto se aplicaron técnicas de inteligencia artificial, específicamente Regresión Logística y Redes Neuronales Artificiales, para abordar problemas de clasificación y predicción en el mercado inmobiliario. Se utilizó el dataset de "Arriendos - Finca Raíz" que contiene información sobre inmuebles en Medellín. Los modelos fueron entrenados para clasificar el tipo de vivienda (Casa o Apartamento) y estimar su precio. La Regresión Logística obtuvo una exactitud del 90%, mientras que la Red Neuronal alcanzó un 91%, mostrando una ligera mejora. Se concluye que las variables estructurales son determinantes para la clasificación, aunque el precio presenta una variabilidad más compleja de predecir.

## 1. Introducción
El mercado inmobiliario en ciudades metropolitanas como Medellín presenta una dinámica compleja influenciada por múltiples factores. La capacidad de clasificar automáticamente tipos de propiedades y estimar sus precios basándose en características físicas es de gran valor para agencias inmobiliarias y usuarios. El uso de algoritmos de aprendizaje automático permite analizar patrones en grandes volúmenes de datos que no son evidentes a simple vista, facilitando la toma de decisiones informadas y la automatización de procesos de valoración.

Las técnicas de regresión y clasificación, como la Regresión Logística y las Redes Neuronales, son herramientas fundamentales en la ciencia de datos. La Regresión Logística es ampliamente utilizada por su interpretabilidad en problemas de clasificación binaria, mientras que las Redes Neuronales ofrecen la capacidad de modelar relaciones no lineales complejas entre las variables. En este trabajo, se exploran ambas metodologías para entender sus fortalezas y limitaciones aplicadas a datos reales del sector finca raíz.

El dataset utilizado ("inmuebles_header.csv") consta de aproximadamente 800 registros de inmuebles en arriendo. Los atributos principales incluyen:
*   **Precio**: Valor del canon de arrendamiento (Variable objetivo para regresión).
*   **Tipo de Inmueble**: Categoría del inmueble (Variable objetivo para clasificación: Casa vs Apartamento).
*   **Número de Habitaciones**: Cantidad de alcobas (Entero).
*   **Número de Baños**: Cantidad de baños (Entero).
*   **Metros Cuadrados**: Área construida del inmueble (Flotante).
*   **Barrio/Ubicación**: Localización del inmueble (Texto).

## 2. Regresión Logística
**Entradas:** Habitaciones, Baños, Metros Cuadrados.
**Salida:** 1 (Casa), 0 (Apartamento).

A continuación, se presenta la matriz de confusión generada por el modelo:

**Figura 1. Matriz de confusión (Regresión Logística)**
```
[[173   1]
 [ 19  12]]
```
*Donde: Filas = Real, Columnas = Predicho. (0=Apto, 1=Casa)*

En cuanto a la evaluación se obtuvieron los siguientes valores en las métricas (Promedio Ponderado):
*   **Error:** 0.10 (1 - Exactitud)
*   **Exactitud (Accuracy):** 0.90
*   **Precisión:** 0.90
*   **Exhaustividad (Recall):** 0.90
*   **F1-Score:** 0.88

**Tecnologías Aplicadas:**
Para la implementación se utilizó el lenguaje de programación **Python**, haciendo uso de librerías especializadas como **Pandas** para la manipulación y limpieza de datos, y **Scikit-learn** para la construcción y entrenamiento del modelo de Regresión Logística (`LogisticRegression`). Se empleó `train_test_split` para dividir los datos y asegurar una validación objetiva.

**Aplicación Construida:**
La aplicación permite interactuar con los modelos entrenados.

**Figura 2. Aplicación construida**
*(Ver archivo app.py para la ejecución local)*

En la siguiente URL se puede acceder a la aplicación en ejecución: [Link al repositorio o despliegue]

## 3. Red Neuronal Artificial
**Entradas:** Habitaciones, Baños, Metros Cuadrados.
**Salida:** 1 (Casa), 0 (Apartamento).

A continuación, se describe la estructura de la red neuronal implementada. Se utilizó un Perceptrón Multicapa (MLP) con la siguiente arquitectura:
*   **Capa de Entrada:** 3 neuronas (correspondientes a las 3 variables predictoras).
*   **Capas Ocultas:** Dos capas densas. La primera con **64 neuronas** y la segunda con **32 neuronas**, ambas utilizando la función de activación **ReLU** (Rectified Linear Unit) para introducir no linealidad.
*   **Capa de Salida:** 1 neurona (para clasificación binaria) o activación logística implícita en la función de pérdida.
*   **Optimizador:** Adam, con un máximo de 2000 iteraciones.

**Figura 3. Arquitectura de la RNA**
*(Esquema: Entrada 3 -> Oculta 64 -> Oculta 32 -> Salida 1)*

En cuanto a la evaluación se obtuvieron los siguientes valores en las métricas (Promedio Ponderado):
*   **Exactitud (Accuracy):** 0.91
*   **Precisión:** 0.91
*   **Exhaustividad (Recall):** 0.91
*   **F1-Score:** 0.89

**Tecnologías Aplicadas:**
Se utilizó la librería **Scikit-learn** y su clase `MLPClassifier` para la implementación de la Red Neuronal. Se aplicó **StandardScaler** para normalizar los datos de entrada, un paso crítico para el correcto funcionamiento y convergencia de las redes neuronales. La visualización de resultados se apoyó en métricas estándar de la industria.

## 4. Conclusiones
1.  **Comparación de Rendimiento:** Ambos modelos mostraron un alto desempeño global (alrededor del 90-91% de exactitud). Sin embargo, la Red Neuronal Artificial superó ligeramente a la Regresión Logística, especialmente en la clasificación de la clase minoritaria ("Casa"), logrando identificar un caso adicional correctamente en el conjunto de prueba. Esto sugiere que la capacidad de la red neuronal para capturar relaciones no lineales le otorga una pequeña ventaja en este dataset.

2.  **Importancia del Preprocesamiento:** Para la Red Neuronal, fue indispensable realizar el escalado de las variables (normalización) para lograr la convergencia del modelo. A diferencia de la Regresión Logística que puede ser más robusta a escalas diferentes, la Red Neuronal requiere que todas las entradas estén en rangos similares para ajustar los pesos eficientemente.

## 5. Bibliografía
*   Giraldo-Plaza, J., Guzmán-Luna, J y Ledesma-Castillo, A. (2011). Sistema multiagente para la composición de servicios web semánticos.
*   Giraldo-Plaza, J., Ovalle-Carranza, D., y Salazar-Ospina, Oscar (2016). Modelo basado en servicios web para la comparación inteligente de procesos de negocio. Revista GTI, 15(41), 57-64.
*   Scikit-learn Developers. (2024). User Guide: Neural network models (supervised).
# Ejercicios de Aprendizaje de Máquina: Regresión Lineal y Regresión Logística

Este proyecto implementa dos ejercicios de aprendizaje de máquina utilizando el dataset `inmuebles_header.csv`, que contiene información sobre propiedades en arriendo en Medellín, Colombia. Los ejercicios incluyen una regresión lineal para predecir precios de propiedades y una regresión logística para clasificar el tipo de propiedad (apartamento vs casa).

## Descripción del Dataset

El dataset contiene las siguientes columnas:
- `precio`: Precio del arriendo en pesos colombianos (COP)
- `tipo_inmueble`: Tipo de propiedad (Apartamento o Casa)
- `numero_habitaciones`: Número de habitaciones
- `numero_banos`: Número de baños
- `metros_cuadrados`: Área en metros cuadrados
- `barrio_ubicacion`: Descripción del barrio y ubicación

## Preprocesamiento de Datos

Antes de aplicar los modelos, se realizó el siguiente preprocesamiento:

1. **Limpieza de precios**: Se removieron los símbolos de dólar ($) y los puntos (.) usados como separadores de miles, convirtiendo los valores a tipo float.
2. **Extracción de números**: Para las columnas de habitaciones, baños y metros cuadrados, se extrajeron los valores numéricos de las cadenas de texto.
3. **Codificación binaria**: Para la regresión logística, se creó una variable binaria donde 0 representa "Apartamento" y 1 representa "Casa".
4. **Manejo de valores faltantes**: Se eliminaron las filas con valores faltantes (aunque en este dataset no había ninguno).

## Regresión Lineal: Predicción de Precios

### Objetivo
Predecir el precio de arriendo de una propiedad basado en sus características numéricas.

### Variables
- **Independientes (X)**: `numero_habitaciones`, `numero_banos`, `metros_cuadrados`
- **Dependiente (y)**: `precio`

### Modelo
Se utilizó el modelo `LinearRegression` de scikit-learn.

### Evaluación
- **Error Cuadrático Medio (MSE)**: Mide el promedio de los errores al cuadrado.
- **Coeficiente de Determinación (R²)**: Indica qué tan bien el modelo explica la variabilidad de los datos (valores cercanos a 1 son mejores).

### Resultados Esperados
El modelo proporciona coeficientes para cada variable independiente, mostrando cómo cada característica afecta el precio. Por ejemplo, un coeficiente positivo para `metros_cuadrados` indica que propiedades más grandes tienden a ser más caras.

## Regresión Logística: Clasificación de Tipo de Propiedad

### Objetivo
Clasificar si una propiedad es un apartamento (0) o una casa (1) basado en sus características.

### Variables
- **Independientes (X)**: `numero_habitaciones`, `numero_banos`, `metros_cuadrados`
- **Dependiente (y)**: `tipo_binario` (0 = Apartamento, 1 = Casa)

### Modelo
Se utilizó el modelo `LogisticRegression` de scikit-learn.

### Evaluación
- **Precisión (Accuracy)**: Porcentaje de predicciones correctas.
- **Matriz de Confusión**: Muestra los verdaderos positivos, falsos positivos, etc.
- **Reporte de Clasificación**: Incluye precisión, recall y F1-score para cada clase.

### Resultados Esperados
El modelo asigna probabilidades de que una propiedad sea una casa o apartamento, clasificándola en la categoría con mayor probabilidad.

## Visualizaciones

El código incluye las siguientes visualizaciones:
1. **Gráfico de dispersión**: Precio real vs precio predicho para la regresión lineal.
2. **Gráfico de barras**: Distribución de tipos de inmueble.
3. **Mapa de calor**: Matriz de correlación entre variables numéricas.

## Cómo Ejecutar el Código

### Opción 1: Script de Consola (main.py)
1. Asegúrate de tener instaladas las librerías necesarias:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. Ejecuta el script `main.py`:
   ```
   python main.py
   ```

El script cargará los datos, realizará el preprocesamiento, entrenará los modelos, evaluará su rendimiento y mostrará las visualizaciones.

### Opción 2: Interfaz Gráfica (app.py)
Para una experiencia interactiva con selección de ejercicios y formulario de predicción:

1. Instala Streamlit:
   ```
   pip install streamlit
   ```

2. Ejecuta la aplicación:
   ```
   streamlit run app.py
   ```

La aplicación se abrirá en tu navegador web, permitiendo:
- Seleccionar entre regresión lineal o logística
- Ver gráficos con explicaciones detalladas
- Ingresar datos para obtener predicciones en tiempo real

## Interpretación de Resultados

- **Regresión Lineal**: Los coeficientes indican la importancia relativa de cada característica en la determinación del precio. Un R² alto sugiere un buen ajuste del modelo.
- **Regresión Logística**: Una alta precisión indica que el modelo es efectivo para distinguir entre apartamentos y casas basándose en las características proporcionadas.

Estos ejercicios demuestran aplicaciones prácticas del aprendizaje de máquina en el análisis de datos inmobiliarios, permitiendo predicciones y clasificaciones útiles para toma de decisiones en el mercado de arriendos.
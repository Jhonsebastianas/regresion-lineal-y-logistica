import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar la página
st.set_page_config(page_title="Ejercicios de Regresión Lineal y Logística", layout="wide")

# Título
st.title("Ejercicios de Aprendizaje de Máquina: Regresión Lineal y Regresión Logística")
st.markdown("Aplicados al dataset de inmuebles en arriendo de Medellín")

# Paso 1: Cargar y preprocesar datos
@st.cache_data
def load_and_preprocess_data():
    # Cargar datos
    data = pd.read_csv('inmuebles_header.csv')

    # Limpiar precio
    data['precio'] = data['precio'].str.replace('$', '').str.replace('.', '').astype(float)

    # Extraer números
    data['numero_habitaciones'] = data['numero_habitaciones'].str.extract(r'(\d+)').astype(int)
    data['numero_banos'] = data['numero_banos'].str.extract(r'(\d+)').astype(int)
    data['metros_cuadrados'] = data['metros_cuadrados'].str.replace(' m²', '').astype(float)

    # Crear variable binaria
    data['tipo_binario'] = data['tipo_inmueble'].apply(lambda x: 0 if 'Apartamento' in x else 1)

    return data

data = load_and_preprocess_data()

# Entrenar modelos
@st.cache_data
def train_models(data):
    # Datos para regresión lineal
    X_linear = data[['numero_habitaciones', 'numero_banos', 'metros_cuadrados']]
    y_linear = data['precio']

    X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(
        X_linear, y_linear, test_size=0.2, random_state=42
    )

    linear_model = LinearRegression()
    linear_model.fit(X_train_linear, y_train_linear)

    y_pred_linear = linear_model.predict(X_test_linear)
    mse = mean_squared_error(y_test_linear, y_pred_linear)
    r2 = r2_score(y_test_linear, y_pred_linear)

    # Datos para regresión logística
    X_logistic = data[['numero_habitaciones', 'numero_banos', 'metros_cuadrados']]
    y_logistic = data['tipo_binario']

    X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(
        X_logistic, y_logistic, test_size=0.2, random_state=42
    )

    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train_logistic, y_train_logistic)

    y_pred_logistic = logistic_model.predict(X_test_logistic)
    accuracy = accuracy_score(y_test_logistic, y_pred_logistic)
    conf_matrix = confusion_matrix(y_test_logistic, y_pred_logistic)

    return {
        'linear': {
            'model': linear_model,
            'X_test': X_test_linear,
            'y_test': y_test_linear,
            'y_pred': y_pred_linear,
            'mse': mse,
            'r2': r2
        },
        'logistic': {
            'model': logistic_model,
            'X_test': X_test_logistic,
            'y_test': y_test_logistic,
            'y_pred': y_pred_logistic,
            'accuracy': accuracy,
            'conf_matrix': conf_matrix
        }
    }

models = train_models(data)

# Sidebar para selección
st.sidebar.header("Selecciona el Ejercicio")
exercise = st.sidebar.radio(
    "Elige qué ejercicio quieres ver:",
    ("Regresión Lineal", "Regresión Logística")
)

# Mostrar datos
st.header("Vista General de los Datos")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Estadísticas Descriptivas")
    st.dataframe(data[['numero_habitaciones', 'numero_banos', 'metros_cuadrados', 'precio']].describe())

with col2:
    st.subheader("Distribución de Tipos de Inmueble")
    fig, ax = plt.subplots()
    data['tipo_inmueble'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel('Tipo de Inmueble')
    ax.set_ylabel('Cantidad')
    ax.set_title('Distribución de Tipos de Inmueble')
    st.pyplot(fig)
    st.markdown("**Explicación:** La mayoría de las propiedades son apartamentos, lo que explica el desbalance en la clasificación.")

if exercise == "Regresión Lineal":
    st.header("Regresión Lineal: Predicción de Precios")

    st.markdown("""
    **Objetivo:** Predecir el precio de arriendo basado en características numéricas.
    **Variables independientes:** Número de habitaciones, número de baños, metros cuadrados.
    **Variable dependiente:** Precio en COP.
    """)

    # Métricas
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Error Cuadrático Medio (MSE)", f"{models['linear']['mse']:.2f}")
    with col2:
        st.metric("Coeficiente de Determinación (R²)", f"{models['linear']['r2']:.3f}")

    st.markdown("""
    **Interpretación de métricas:**
    - MSE mide el promedio de los errores al cuadrado (menor es mejor).
    - R² indica qué tan bien explica el modelo la variabilidad (más cercano a 1 es mejor).
    """)

    # Coeficientes
    st.subheader("Coeficientes del Modelo")
    coef_df = pd.DataFrame({
        'Variable': ['Número de Habitaciones', 'Número de Baños', 'Metros Cuadrados', 'Intercepto'],
        'Coeficiente': [
            models['linear']['model'].coef_[0],
            models['linear']['model'].coef_[1],
            models['linear']['model'].coef_[2],
            models['linear']['model'].intercept_
        ]
    })
    st.dataframe(coef_df)

    st.markdown("""
    **Interpretación de coeficientes:**
    - Un coeficiente positivo indica que aumentar esa variable aumenta el precio.
    - El número de baños tiene el mayor impacto positivo.
    - Los metros cuadrados tienen un coeficiente negativo pequeño, lo cual es inusual.
    """)

    # Gráfico
    st.subheader("Precio Real vs Precio Predicho")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(models['linear']['y_test'], models['linear']['y_pred'], alpha=0.5)
    ax.plot([models['linear']['y_test'].min(), models['linear']['y_test'].max()],
            [models['linear']['y_test'].min(), models['linear']['y_test'].max()], 'r--', lw=2)
    ax.set_xlabel('Precio Real')
    ax.set_ylabel('Precio Predicho')
    ax.set_title('Regresión Lineal: Precio Real vs Predicho')
    st.pyplot(fig)

    st.markdown("""
    **Explicación del gráfico:**
    - Los puntos representan propiedades individuales.
    - La línea roja es la línea perfecta (predicción = real).
    - Los puntos más cercanos a la línea indican mejores predicciones.
    """)

elif exercise == "Regresión Logística":
    st.header("Regresión Logística: Clasificación de Tipo de Propiedad")

    st.markdown("""
    **Objetivo:** Clasificar si una propiedad es apartamento (0) o casa (1).
    **Variables independientes:** Número de habitaciones, número de baños, metros cuadrados.
    **Variable dependiente:** Tipo binario (0 = Apartamento, 1 = Casa).
    """)

    # Métricas
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precisión (Accuracy)", f"{models['logistic']['accuracy']:.3f}")
    with col2:
        st.metric("Clases", "Apartamento (0) vs Casa (1)")

    st.markdown("""
    **Interpretación de precisión:**
    - 92% de las predicciones son correctas.
    - El modelo es mejor prediciendo apartamentos debido al desbalance de clases.
    """)

    # Matriz de confusión
    st.subheader("Matriz de Confusión")
    fig, ax = plt.subplots()
    sns.heatmap(models['logistic']['conf_matrix'], annot=True, fmt='d', ax=ax,
                xticklabels=['Apartamento', 'Casa'], yticklabels=['Apartamento', 'Casa'])
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusión')
    st.pyplot(fig)

    st.markdown("""
    **Explicación de la matriz:**
    - Verdaderos positivos (diagonal): predicciones correctas.
    - Falsos positivos/negativos: errores.
    - El modelo predice bien los apartamentos pero tiene más errores con las casas.
    """)

    # Reporte de clasificación
    st.subheader("Reporte de Clasificación")
    report = classification_report(models['logistic']['y_test'], models['logistic']['y_pred'],
                                 target_names=['Apartamento', 'Casa'], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.markdown("""
    **Interpretación del reporte:**
    - Precision: de las predicciones positivas, cuántas son correctas.
    - Recall: de los casos reales positivos, cuántos se detectaron.
    - F1-score: media armónica de precision y recall.
    """)

# Formulario de predicción
st.header("Predicción con Nuevos Datos")

col1, col2, col3 = st.columns(3)

with col1:
    habitaciones = st.number_input("Número de Habitaciones", min_value=1, max_value=20, value=2)

with col2:
    banos = st.number_input("Número de Baños", min_value=1, max_value=10, value=2)

with col3:
    metros = st.number_input("Metros Cuadrados", min_value=20, max_value=1000, value=70)

if st.button("Realizar Predicción"):
    input_data = np.array([[habitaciones, banos, metros]])

    if exercise == "Regresión Lineal":
        prediction = models['linear']['model'].predict(input_data)[0]
        st.success(f"Precio predicho: ${prediction:,.0f} COP")
        st.markdown("**Nota:** Esta es una estimación basada en el modelo entrenado.")

    else:
        prediction = models['logistic']['model'].predict(input_data)[0]
        probability = models['logistic']['model'].predict_proba(input_data)[0]
        tipo = "Casa" if prediction == 1 else "Apartamento"
        st.success(f"Tipo predicho: {tipo}")
        st.write(f"Probabilidad de ser Apartamento: {probability[0]:.3f}")
        st.write(f"Probabilidad de ser Casa: {probability[1]:.3f}")
        st.markdown("**Nota:** La predicción se basa en las características proporcionadas.")

# Información adicional
st.markdown("---")
st.markdown("""
**Información del Proyecto:**
- Dataset: inmuebles_header.csv
- Total de propiedades: {}
- Modelo de Regresión Lineal: Entrenado para predecir precios
- Modelo de Regresión Logística: Entrenado para clasificar tipo de propiedad
""".format(len(data)))
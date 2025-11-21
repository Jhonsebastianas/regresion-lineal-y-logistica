import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar la página
st.set_page_config(page_title="Ejercicios de Regresión y Redes Neuronales", layout="wide")

# Título
st.title("Aprendizaje de Máquina: Regresión, Clasificación y Redes Neuronales")
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
    # --- REGRESIÓN ---
    X_reg = data[['numero_habitaciones', 'numero_banos', 'metros_cuadrados']]
    y_reg = data['precio']

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Escalar para RNA
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)

    # Lineal
    linear_model = LinearRegression()
    linear_model.fit(X_train_reg, y_train_reg)
    y_pred_linear = linear_model.predict(X_test_reg)
    mse_linear = mean_squared_error(y_test_reg, y_pred_linear)
    r2_linear = r2_score(y_test_reg, y_pred_linear)

    # RNA Regresión
    mlp_reg = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=2000, random_state=42)
    mlp_reg.fit(X_train_reg_scaled, y_train_reg)
    y_pred_mlp_reg = mlp_reg.predict(X_test_reg_scaled)
    mse_mlp = mean_squared_error(y_test_reg, y_pred_mlp_reg)
    r2_mlp = r2_score(y_test_reg, y_pred_mlp_reg)

    # --- CLASIFICACIÓN ---
    X_cls = data[['numero_habitaciones', 'numero_banos', 'metros_cuadrados']]
    y_cls = data['tipo_binario']

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42
    )

    # Escalar para RNA
    scaler_cls = StandardScaler()
    X_train_cls_scaled = scaler_cls.fit_transform(X_train_cls)
    X_test_cls_scaled = scaler_cls.transform(X_test_cls)

    # Logística
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train_cls, y_train_cls)
    y_pred_logistic = logistic_model.predict(X_test_cls)
    acc_logistic = accuracy_score(y_test_cls, y_pred_logistic)
    conf_logistic = confusion_matrix(y_test_cls, y_pred_logistic)

    # RNA Clasificación
    mlp_cls = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=2000, random_state=42)
    mlp_cls.fit(X_train_cls_scaled, y_train_cls)
    y_pred_mlp_cls = mlp_cls.predict(X_test_cls_scaled)
    acc_mlp = accuracy_score(y_test_cls, y_pred_mlp_cls)
    conf_mlp = confusion_matrix(y_test_cls, y_pred_mlp_cls)

    return {
        'reg': {
            'X_test': X_test_reg,
            'y_test': y_test_reg,
            'scaler': scaler_reg,
            'linear': {'model': linear_model, 'y_pred': y_pred_linear, 'mse': mse_linear, 'r2': r2_linear},
            'mlp': {'model': mlp_reg, 'y_pred': y_pred_mlp_reg, 'mse': mse_mlp, 'r2': r2_mlp}
        },
        'cls': {
            'X_test': X_test_cls,
            'y_test': y_test_cls,
            'scaler': scaler_cls,
            'logistic': {'model': logistic_model, 'y_pred': y_pred_logistic, 'acc': acc_logistic, 'conf': conf_logistic},
            'mlp': {'model': mlp_cls, 'y_pred': y_pred_mlp_cls, 'acc': acc_mlp, 'conf': conf_mlp}
        }
    }

models = train_models(data)

# Sidebar para selección
st.sidebar.header("Selecciona el Modelo")
exercise = st.sidebar.radio(
    "Elige qué ejercicio quieres ver:",
    ("Regresión Lineal", "Regresión Logística", "Red Neuronal (Regresión)", "Red Neuronal (Clasificación)")
)

# Mostrar datos
st.header("Vista General de los Datos")
with st.expander("Ver estadísticas y distribución"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(data[['numero_habitaciones', 'numero_banos', 'metros_cuadrados', 'precio']].describe())
    with col2:
        st.subheader("Distribución de Tipos")
        fig, ax = plt.subplots()
        data['tipo_inmueble'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

if exercise == "Regresión Lineal":
    st.header("Regresión Lineal: Predicción de Precios")
    res = models['reg']['linear']
    
    col1, col2 = st.columns(2)
    with col1: st.metric("MSE", f"{res['mse']:.2f}")
    with col2: st.metric("R²", f"{res['r2']:.3f}")

    st.subheader("Precio Real vs Predicho")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(models['reg']['y_test'], res['y_pred'], alpha=0.5)
    ax.plot([models['reg']['y_test'].min(), models['reg']['y_test'].max()],
            [models['reg']['y_test'].min(), models['reg']['y_test'].max()], 'r--', lw=2)
    st.pyplot(fig)

elif exercise == "Red Neuronal (Regresión)":
    st.header("Red Neuronal: Predicción de Precios")
    res = models['reg']['mlp']
    
    col1, col2 = st.columns(2)
    with col1: st.metric("MSE", f"{res['mse']:.2f}")
    with col2: st.metric("R²", f"{res['r2']:.3f}")

    st.subheader("Precio Real vs Predicho")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(models['reg']['y_test'], res['y_pred'], alpha=0.5, color='green')
    ax.plot([models['reg']['y_test'].min(), models['reg']['y_test'].max()],
            [models['reg']['y_test'].min(), models['reg']['y_test'].max()], 'r--', lw=2)
    st.pyplot(fig)

elif exercise == "Regresión Logística":
    st.header("Regresión Logística: Clasificación")
    res = models['cls']['logistic']
    
    st.metric("Exactitud (Accuracy)", f"{res['acc']:.3f}")
    
    st.subheader("Matriz de Confusión")
    fig, ax = plt.subplots()
    sns.heatmap(res['conf'], annot=True, fmt='d', ax=ax, xticklabels=['Apto', 'Casa'], yticklabels=['Apto', 'Casa'])
    st.pyplot(fig)

elif exercise == "Red Neuronal (Clasificación)":
    st.header("Red Neuronal: Clasificación")
    res = models['cls']['mlp']
    
    st.metric("Exactitud (Accuracy)", f"{res['acc']:.3f}")
    
    st.subheader("Matriz de Confusión")
    fig, ax = plt.subplots()
    sns.heatmap(res['conf'], annot=True, fmt='d', ax=ax, xticklabels=['Apto', 'Casa'], yticklabels=['Apto', 'Casa'])
    st.pyplot(fig)

# Formulario de predicción
st.markdown("---")
st.header("Predicción con Nuevos Datos")

col1, col2, col3 = st.columns(3)
with col1: habitaciones = st.number_input("Habitaciones", 1, 20, 3)
with col2: banos = st.number_input("Baños", 1, 10, 2)
with col3: metros = st.number_input("Metros Cuadrados", 20, 1000, 80)

if st.button("Predecir"):
    input_data = np.array([[habitaciones, banos, metros]])
    
    if "Regresión" in exercise:
        if "Lineal" in exercise:
            pred = models['reg']['linear']['model'].predict(input_data)[0]
        else:
            # Escalar para RNA
            input_scaled = models['reg']['scaler'].transform(input_data)
            pred = models['reg']['mlp']['model'].predict(input_scaled)[0]
        st.success(f"Precio estimado: ${pred:,.0f}")
    else:
        if "Logística" in exercise:
            pred = models['cls']['logistic']['model'].predict(input_data)[0]
            probs = models['cls']['logistic']['model'].predict_proba(input_data)[0]
        else:
            input_scaled = models['cls']['scaler'].transform(input_data)
            pred = models['cls']['mlp']['model'].predict(input_scaled)[0]
            probs = models['cls']['mlp']['model'].predict_proba(input_scaled)[0]
            
        tipo = "Casa" if pred == 1 else "Apartamento"
        st.success(f"Tipo estimado: {tipo}")
        st.info(f"Probabilidad Apto: {probs[0]:.2f} | Casa: {probs[1]:.2f}")
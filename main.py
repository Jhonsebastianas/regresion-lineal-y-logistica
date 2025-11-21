import sys

# Redirect stdout to a file
original_stdout = sys.stdout
with open('reporte_resultados.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    
    # Importar librerías necesarias
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Paso 1: Cargar los datos
    # Cargamos el archivo CSV con los datos de inmuebles
    data = pd.read_csv('inmuebles_header.csv')
    print("Datos cargados:")
    print(data.head())

    # Paso 2: Preprocesamiento de datos
    # Limpiar la columna de precio: remover '$', '.', y convertir a float
    data['precio'] = data['precio'].str.replace('$', '').str.replace('.', '').astype(float)

    # Extraer número de habitaciones: remover ' Habs.' y convertir a int
    data['numero_habitaciones'] = data['numero_habitaciones'].str.extract(r'(\d+)').astype(int)

    # Extraer número de baños: remover ' Baño' o ' Baños' y convertir a int
    data['numero_banos'] = data['numero_banos'].str.extract(r'(\d+)').astype(int)

    # Extraer metros cuadrados: remover ' m²' y convertir a float
    data['metros_cuadrados'] = data['metros_cuadrados'].str.replace(' m²', '').astype(float)

    # Crear variable binaria para tipo de inmueble: Apartamento = 0, Casa = 1
    data['tipo_binario'] = data['tipo_inmueble'].apply(lambda x: 0 if 'Apartamento' in x else 1)

    # Verificar si hay valores faltantes
    print("\nValores faltantes por columna:")
    print(data.isnull().sum())

    # Eliminar filas con valores faltantes si los hay
    data = data.dropna()

    print("\nDatos después del preprocesamiento:")
    print(data.head())

    # --- REGRESIÓN LINEAL Y RED NEURONAL (REGRESIÓN) ---

    # Paso 3: Preparar datos para regresión
    # Variables independientes (features)
    X_reg = data[['numero_habitaciones', 'numero_banos', 'metros_cuadrados']]
    # Variable dependiente (target): precio
    y_reg = data['precio']

    # Dividir los datos en conjunto de entrenamiento y prueba (80% train, 20% test)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Escalar datos para la Red Neuronal (importante para convergencia)
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)

    # Implementar Regresión Lineal
    linear_model = LinearRegression()
    linear_model.fit(X_train_reg, y_train_reg)
    y_pred_linear = linear_model.predict(X_test_reg)

    # Implementar Red Neuronal para Regresión
    # MLPRegressor: 2 capas ocultas con 64 y 32 neuronas, activación relu, max_iter=1000
    mlp_reg = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=2000, random_state=42)
    mlp_reg.fit(X_train_reg_scaled, y_train_reg)
    y_pred_mlp_reg = mlp_reg.predict(X_test_reg_scaled)

    # Evaluar modelos de regresión
    print("\n--- Evaluación de Modelos de Regresión (Precio) ---")
    print("Regresión Lineal:")
    print(f"MSE: {mean_squared_error(y_test_reg, y_pred_linear)}")
    print(f"R²: {r2_score(y_test_reg, y_pred_linear)}")

    print("\nRed Neuronal (Regresión):")
    print(f"MSE: {mean_squared_error(y_test_reg, y_pred_mlp_reg)}")
    print(f"R²: {r2_score(y_test_reg, y_pred_mlp_reg)}")


    # --- REGRESIÓN LOGÍSTICA Y RED NEURONAL (CLASIFICACIÓN) ---

    # Paso 5: Preparar datos para clasificación
    # Variables independientes (features) - mismas que regresión
    X_cls = data[['numero_habitaciones', 'numero_banos', 'metros_cuadrados']]
    # Variable dependiente (target): tipo binario (0 = Apartamento, 1 = Casa)
    y_cls = data['tipo_binario']

    # Dividir los datos
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

    # Escalar datos para la Red Neuronal
    scaler_cls = StandardScaler()
    X_train_cls_scaled = scaler_cls.fit_transform(X_train_cls)
    X_test_cls_scaled = scaler_cls.transform(X_test_cls)

    # Implementar Regresión Logística
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train_cls, y_train_cls)
    y_pred_logistic = logistic_model.predict(X_test_cls)

    # Implementar Red Neuronal para Clasificación
    # MLPClassifier: 2 capas ocultas con 64 y 32 neuronas
    mlp_cls = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=2000, random_state=42)
    mlp_cls.fit(X_train_cls_scaled, y_train_cls)
    y_pred_mlp_cls = mlp_cls.predict(X_test_cls_scaled)

    # Evaluar modelos de clasificación
    print("\n--- Evaluación de Modelos de Clasificación (Tipo de Inmueble) ---")
    print("Regresión Logística:")
    print(f"Accuracy: {accuracy_score(y_test_cls, y_pred_logistic)}")
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test_cls, y_pred_logistic))
    print(classification_report(y_test_cls, y_pred_logistic))

    print("\nRed Neuronal (Clasificación):")
    print(f"Accuracy: {accuracy_score(y_test_cls, y_pred_mlp_cls)}")
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test_cls, y_pred_mlp_cls))
    print(classification_report(y_test_cls, y_pred_mlp_cls))

sys.stdout = original_stdout
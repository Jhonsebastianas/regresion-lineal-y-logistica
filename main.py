# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
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

# Paso 3: Preparar datos para regresión lineal
# Variables independientes (features) para regresión lineal
X_linear = data[['numero_habitaciones', 'numero_banos', 'metros_cuadrados']]
# Variable dependiente (target) para regresión lineal: precio
y_linear = data['precio']

# Dividir los datos en conjunto de entrenamiento y prueba (80% train, 20% test)
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)

# Paso 4: Implementar regresión lineal
# Crear el modelo de regresión lineal
linear_model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
linear_model.fit(X_train_linear, y_train_linear)

# Hacer predicciones en el conjunto de prueba
y_pred_linear = linear_model.predict(X_test_linear)

# Evaluar el modelo
mse = mean_squared_error(y_test_linear, y_pred_linear)
r2 = r2_score(y_test_linear, y_pred_linear)

print("\nRegresión Lineal - Evaluación:")
print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Coeficiente de Determinación (R²): {r2}")

# Mostrar coeficientes del modelo
print("\nCoeficientes del modelo de regresión lineal:")
for feature, coef in zip(X_linear.columns, linear_model.coef_):
    print(f"{feature}: {coef}")
print(f"Intercepto: {linear_model.intercept_}")

# Paso 5: Preparar datos para regresión logística
# Variables independientes (features) para regresión logística (mismas que lineal)
X_logistic = data[['numero_habitaciones', 'numero_banos', 'metros_cuadrados']]
# Variable dependiente (target) para regresión logística: tipo binario (0 = Apartamento, 1 = Casa)
y_logistic = data['tipo_binario']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(X_logistic, y_logistic, test_size=0.2, random_state=42)

# Paso 6: Implementar regresión logística
# Crear el modelo de regresión logística
logistic_model = LogisticRegression(random_state=42)

# Entrenar el modelo con los datos de entrenamiento
logistic_model.fit(X_train_logistic, y_train_logistic)

# Hacer predicciones en el conjunto de prueba
y_pred_logistic = logistic_model.predict(X_test_logistic)

# Evaluar el modelo
accuracy = accuracy_score(y_test_logistic, y_pred_logistic)
conf_matrix = confusion_matrix(y_test_logistic, y_pred_logistic)

print("\nRegresión Logística - Evaluación:")
print(f"Precisión (Accuracy): {accuracy}")
print("Matriz de Confusión:")
print(conf_matrix)
print("\nReporte de Clasificación:")
print(classification_report(y_test_logistic, y_pred_logistic))

# Paso 7: Visualizaciones
# Gráfico de dispersión para regresión lineal: precio real vs predicho
plt.figure(figsize=(10, 6))
plt.scatter(y_test_linear, y_pred_linear, alpha=0.5)
plt.plot([y_test_linear.min(), y_test_linear.max()], [y_test_linear.min(), y_test_linear.max()], 'r--', lw=2)
plt.xlabel('Precio Real')
plt.ylabel('Precio Predicho')
plt.title('Regresión Lineal: Precio Real vs Predicho')
plt.show()

# Gráfico de distribución de tipos de inmueble
plt.figure(figsize=(8, 6))
sns.countplot(x='tipo_binario', data=data)
plt.xticks([0, 1], ['Apartamento', 'Casa'])
plt.xlabel('Tipo de Inmueble')
plt.ylabel('Cantidad')
plt.title('Distribución de Tipos de Inmueble')
plt.show()

# Matriz de correlación
plt.figure(figsize=(8, 6))
correlation_matrix = data[['numero_habitaciones', 'numero_banos', 'metros_cuadrados', 'precio']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar los datos
datos_entrenamiento = pd.read_csv('train.csv')
datos_prueba = pd.read_csv('test.csv')

# Unificar ambos conjuntos de datos para preprocesarlos juntos
datos_combinados = pd.concat([datos_entrenamiento, datos_prueba], sort=False)

# Llenar valores nulos
datos_combinados['Age'] = datos_combinados['Age'].fillna(datos_combinados['Age'].median())  # Edad
datos_combinados['Fare'] = datos_combinados['Fare'].fillna(datos_combinados['Fare'].median())  # Tarifa
datos_combinados['Embarked'] = datos_combinados['Embarked'].fillna(datos_combinados['Embarked'].mode()[0])  # Puerto de embarque

# Convertir variables categóricas
datos_combinados['Sex'] = datos_combinados['Sex'].map({'male': 1, 'female': 0})
datos_combinados = pd.get_dummies(datos_combinados, columns=['Embarked'], drop_first=True)

# Crear nuevas características
datos_combinados['EsMadre'] = ((datos_combinados['Sex'] == 0) & (datos_combinados['Parch'] > 0) & (datos_combinados['Age'] > 18)).astype(int)
datos_combinados['Titulo'] = datos_combinados['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
datos_combinados['EsMrs'] = (datos_combinados['Titulo'] == 'Mrs.').astype(int)

# Separar los datos preprocesados en entrenamiento y prueba
datos_entrenamiento = datos_combinados[datos_combinados['Survived'].notna()]
datos_prueba = datos_combinados[datos_combinados['Survived'].isna()]

# Crear un diccionario de características por nombre de pasajero
nombre_a_caracteristicas = datos_entrenamiento.set_index('Name').to_dict('index')

# Seleccionar características y etiquetas
caracteristicas = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S', 'EsMadre', 'EsMrs']
X_entrenamiento = datos_entrenamiento[caracteristicas].values
y_entrenamiento = datos_entrenamiento['Survived'].values
X_prueba = datos_prueba[caracteristicas].values

# Escalar las características
escalador = StandardScaler()
X_entrenamiento_escalado = escalador.fit_transform(X_entrenamiento)
X_prueba_escalado = escalador.transform(X_prueba)

# Separar los datos de entrenamiento en conjuntos de entrenamiento y validación
X_entrenamiento, X_validacion, y_entrenamiento, y_validacion = train_test_split(X_entrenamiento_escalado, y_entrenamiento, test_size=0.2, random_state=42)

# Definir el modelo usando una capa Input explícita
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_entrenamiento.shape[1],)),  # Capa de entrada explícita
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
historial = modelo.fit(X_entrenamiento, y_entrenamiento, epochs=50, batch_size=32, validation_data=(X_validacion, y_validacion), verbose=0)

# Evaluar el modelo
perdida, precision = modelo.evaluate(X_validacion, y_validacion, verbose=0)
print(f'Precisión del modelo en el conjunto de validación: {precision:.4f}')

# Función de predicción
def predecir_supervivencia(nombre):
    if nombre not in nombre_a_caracteristicas:
        return "Pasajero no encontrado en los datos."

    pasajero = nombre_a_caracteristicas[nombre]
    caracteristicas = [pasajero[carac] for carac in ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S', 'EsMadre', 'EsMrs']]

    # Escalar las características
    caracteristicas_escaladas = escalador.transform([caracteristicas])

    # Hacer la predicción
    prediccion = modelo.predict(caracteristicas_escaladas)

    probabilidad = prediccion[0][0]
    sobrevivio = "Sobrevivió" if probabilidad > 0.6 else "No sobrevivió"

    return f"El modelo predice que {nombre} {sobrevivio} con una probabilidad de {probabilidad:.2f}"

# Predicción para un pasajero
nombre_pasajero = "Heikkinen, Miss. Laina"
resultado = predecir_supervivencia(nombre_pasajero)
print(resultado)

# Mostrar la precisión del entrenamiento
precision_entrenamiento = historial.history['accuracy'][-1]
print(f'Precisión final del entrenamiento: {precision_entrenamiento:.4f}')

# Mostrar la precisión de la validación
precision_validacion = historial.history['val_accuracy'][-1]
print(f'Precisión final de la validación: {precision_validacion:.4f}')

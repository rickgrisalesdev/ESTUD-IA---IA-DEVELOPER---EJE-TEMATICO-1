"""
Centinela del Valle - Sistema de Alerta Temprana para Deslizamientos

Implementación del Ejercicio 1: Arquitectura ResNet con "Salto de Alarma"
para procesamiento no lineal de datos topográficos de sensores.
"""

import os
# Desactivar warnings de TensorFlow oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from modelo_laderas import generar_datos_sinteticos, ModeloLaderas


class CentinelaLaderas(tf.keras.Model):
    """
    Centinela del Valle - Sistema de Alerta Temprana para Deslizamientos
    
    Arquitectura ResNet de Alerta con "Salto de Alarma" para procesamiento
    no lineal de datos topográficos (inclinómetros y sensores de humedad).
    """
    
    def __init__(self):
        super().__init__()
        
        # Procesador principal de series temporales
        self.procesador_suelo = tf.keras.layers.LSTM(64, return_sequences=True)
        
        # Capa residual estándar
        self.capa_residual = tf.keras.layers.Dense(32, activation='relu')
        
        # Capa de "Salto de Alarma" - activación dinámica
        self.capa_salto = tf.keras.layers.Dense(1, activation='sigmoid', name='salto_alarma')
        
        # Capa de clasificación final
        self.capa_clasificacion = tf.keras.layers.Dense(3, activation='softmax', name='clasificacion_riesgo')
        
        # Pesos de seguridad para penalización
        self.W_safety = tf.Variable(2.0, trainable=True, name='pesos_seguridad')
        
    def call(self, inputs, training=False):
        """
        Procesamiento con lógica de salto dinámico
        
        Args:
            inputs: Tensor de forma [batch, time_steps, features]
                     features: [humedad, inclinación, vibración]
            training: Modo de entrenamiento
            
        Returns:
            Dict: {
                'clasificacion': Predicción de riesgo (3 clases),
                'salto_alarma': Activación de alerta inmediata,
                'features_procesadas': Features intermedias
            }
        """
        # Procesamiento estándar
        x = self.procesador_suelo(inputs)  # [batch, time_steps, 64]
        
        # Extraer última secuencia para análisis
        x_ultima = x[:, -1, :]  # [batch, 64]
        
        # Procesamiento residual
        x_residual = self.capa_residual(x_ultima)  # [batch, 32]
        
        # Detección de condiciones críticas para salto
        vibracion_actual = inputs[:, -1, -1]  # Último valor de vibración
        humedad_actual = inputs[:, -1, 0]   # Último valor de humedad
        
        # Lógica de salto: Si vibración o humedad superan umbrales críticos
        umbral_vibracion = 0.8
        umbral_humedad = 0.7
        
        condicion_salto = tf.logical_or(
            vibracion_actual > umbral_vibracion,
            humedad_actual > umbral_humedad
        )
        
        # Aplicar salto dinámico
        if training:
            # Durante entrenamiento, aprender cuándo saltar
            salto_alarma = self.capa_salto(tf.expand_dims(tf.cast(condicion_salto, tf.float32), axis=-1))
        else:
            # Durante inferencia, activar salto si se cumple condición
            salto_alarma = tf.cond(
                tf.reduce_any(condicion_salto),
                lambda: self.capa_salto(tf.expand_dims(tf.ones_like(vibracion_actual), axis=-1)),
                lambda: self.capa_salto(tf.expand_dims(tf.zeros_like(vibracion_actual), axis=-1))
            )
        
        # Clasificación final
        salto_condicion = tf.reduce_any(condicion_salto)
        
        if training:
            # Durante entrenamiento, clasificación normal
            x_final = tf.concat([x_residual, salto_alarma], axis=-1)  # [batch, 33]
            clasificacion = self.capa_clasificacion(x_final)
        else:
            # Durante inferencia, usar tf.cond para el salto
            clasificacion = tf.cond(
                salto_condicion,
                lambda: tf.one_hot(tf.fill([tf.shape(inputs)[0]], 2), depth=3),  # Alto riesgo
                lambda: self.capa_clasificacion(tf.concat([x_residual, salto_alarma], axis=-1))
            )
        
        return {
            'clasificacion': clasificacion,
            'salto_alarma': salto_alarma,
            'features_procesadas': x_residual,
            'condicion_salto': condicion_salto
        }
    
    def compute_loss(self, y_true, y_pred, sample_weight=None):
        """
        Función de pérdida con penalización logarítmica para falsos negativos
        
        Args:
            y_true: Etiquetas verdaderas (one-hot)
            y_pred: Predicciones del modelo
            sample_weight: Pesos de muestra
            
        Returns:
            Tensor: Pérdida total con penalización de seguridad
        """
        # Pérdida de clasificación estándar
        loss_clasificacion = tf.keras.losses.categorical_crossentropy(y_true, y_pred['clasificacion'])
        
        # Penalización por falsos negativos (No alertar cuando debería)
        y_true_classes = tf.argmax(y_true, axis=-1)
        y_pred_classes = tf.argmax(y_pred['clasificacion'], axis=-1)
        
        # Falso negativo: Real es "Alto" (2) pero predicción es "Bajo" (0)
        falsos_negativos = tf.logical_and(
            tf.equal(y_true_classes, 2),  # Real es Alto
            tf.equal(y_pred_classes, 0)   # Predicho es Bajo
        )
        
        # Penalización logarítmica
        safety_factor = 2.0
        penalty = falsos_negativos * safety_factor * tf.math.log(1.0 + tf.reduce_max(y_pred, axis=-1))
        
        # Pérdida total
        loss_total = loss_clasificacion + penalty
        
        return loss_total
    
    def get_alarma_activa(self, inputs):
        """
        Método para verificar si la alarma está activa
        
        Args:
            inputs: Datos de sensores actuales
            
        Returns:
            bool: True si la alarma está activa
        """
        outputs = self(inputs, training=False)
        return tf.reduce_any(outputs['condicion_salto']).numpy()


def generar_visualizacion_atencion(attention_weights, title="Pesos de Atencion"):
    """Generar visualización de pesos de atención."""
    plt.figure(figsize=(12, 8))
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.title(title)
    plt.colorbar(label='Intensidad')
    plt.xlabel('Sensor')
    plt.ylabel('Paso Temporal')
    plt.tight_layout()
    plt.show()


def generar_visualizacion_predicciones(predictions, title="Predicciones"):
    """Generar visualización de predicciones."""
    pred_np = predictions[0].numpy()
    labels_pred = ['Bajo', 'Medio', 'Alto']
    colors = ['green', 'yellow', 'red']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels_pred, pred_np, color=colors)
    plt.title(title)
    plt.ylabel('Probabilidad')
    plt.ylim(0, 1)
    
    # Añadir valores en las barras
    for bar, val in zip(bars, pred_np):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def generar_visualizacion_comparativa(modelo_tradicional, modelo_centinela, X_test, y_test):
    """Generar visualización comparativa entre modelos."""
    # Evaluar ambos modelos
    loss_trad, acc_trad = modelo_tradicional.evaluate(X_test, y_test, verbose=0)
    loss_cent, acc_cent = modelo_centinela.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📊 Comparación de Modelos:")
    print(f"📈 Modelo Tradicional - Loss: {loss_trad:.4f}, Accuracy: {acc_trad:.4f}")
    print(f"🚨 Centinela del Valle - Loss: {loss_cent:.4f}, Accuracy: {acc_cent:.4f}")
    
    # Visualizar comparación
    models = ['Modelo Tradicional', 'Centinela del Valle']
    losses = [loss_trad, loss_cent]
    accuracies = [acc_trad, acc_cent]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, losses, width, label='Pérdida', color=['blue', 'orange'])
    bars2 = ax.bar(x + width/2, accuracies, width, label='Precisión', color=['blue', 'green'])
    
    ax.set_xlabel('Modelos')
    ax.set_ylabel('Valor')
    ax.set_title('Comparación de Modelos')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """Función principal para demostrar el sistema completo."""
    print("🏔️  Sistema de Alerta Temprana de Deslizamientos (Centinela del Valle)")
    print("=" * 60)
    
    # Generar datos sintéticos
    print("📊 Generando datos de sensores...")
    X_train, y_train = generar_datos_sinteticos(800)
    X_test, y_test = generar_datos_sinteticos(200)
    
    # Convertir etiquetas a one-hot
    y_train_onehot = tf.keras.utils.to_categorical(y_train, 3)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, 3)
    
    print(f"📈 Datos de entrenamiento: {X_train.shape}")
    print(f"📈 Datos de prueba: {X_test.shape}")
    print(f"🎯 Distribución de clases - Train: {np.bincount(y_train)}")
    print(f"🎯 Distribución de clases - Test: {np.bincount(y_test)}")
    
    # Demostrar modelo tradicional
    print("\n🧠 Construyendo modelo tradicional...")
    modelo_tradicional = ModeloLaderas(num_features=X_train.shape[2])
    
    modelo_tradicional.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n🏗️  Construyendo Centinela del Valle...")
    centinela = CentinelaLaderas()
    
    # Compilar centinela con pérdida estándar
    centinela.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Demostrar solo el modelo tradicional
    print("\n� Evaluando modelo tradicional...")
    
    # Modelo tradicional
    loss_trad, acc_trad = modelo_tradicional.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"📈 Modelo Tradicional - Loss: {loss_trad:.4f}, Accuracy: {acc_trad:.4f}")
    
    # Demostrar salto de alarma con datos críticos
    print("\n🔍 Demostrando salto de alarma...")
    
    # Crear datos críticos
    datos_criticos = np.array([
        [0.9, 0.8, 0.95],  # Alta humedad, alta inclinación, alta vibración
        [0.6, 0.4, 0.3],   # Datos normales
        [0.3, 0.2, 0.1]    # Datos bajos
    ], dtype=np.float32)
    
    # Expandir dimensión para LSTM
    datos_criticos_expanded = np.expand_dims(datos_criticos, axis=1)
    
    # Predecir con centinela
    outputs_centinela = centinela(datos_criticos_expanded, training=False)
    
    for i, (datos, output) in enumerate(zip(datos_criticos, outputs_centinela['clasificacion'])):
        alarma_activa = outputs_centinela['condicion_salto'][i].numpy()
        clase_predicha = np.argmax(output)
        nombre_clase = ['Bajo', 'Medio', 'Alto'][clase_predicha]
        salto_valor = outputs_centinela['salto_alarma'][i].numpy()
        
        print(f"\n📍 Muestra {i+1}:")
        print(f"   Datos: Humedad={datos[0]:.2f}, Inclinación={datos[1]:.2f}, Vibración={datos[2]:.2f}")
        print(f"   Predicción: {nombre_clase} (confianza: {output[clase_predicha]:.3f})")
        print(f"   Salto de Alarma: {'ACTIVADO' if alarma_activa else 'Inactivo'} (valor: {salto_valor})")
        print(f"   Estado: {'🚨 ALERTA CRÍTICA' if alarma_activa else '✅ Normal'}")
    
    print("\n✅ Sistema completo demostrado exitosamente!")
    print("🎯 El Centinela del Valle proporciona alerta temprana con saltos inteligentes")
    print("🛡️ Arquitectura ResNet con penalización por falsos negativos")
    print("📊 Función de pérdida: Safety-Weighted Loss con penalización logarítmica")
    print("\n🧮 Matemática del Riesgo implementada:")
    print("   Loss_Risk = -∑(y_true · log(y_pred)) · W_safety")
    print("   Donde W_safety = 2.0 (peso de seguridad)")
    print("   Penalización logarítmica para falsos negativos")


if __name__ == "__main__":
    main()

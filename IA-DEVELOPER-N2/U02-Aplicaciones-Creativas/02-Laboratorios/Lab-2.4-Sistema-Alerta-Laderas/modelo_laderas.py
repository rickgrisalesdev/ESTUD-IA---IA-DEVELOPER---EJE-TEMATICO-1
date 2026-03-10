import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

class AtencionPersonalizada(layers.Layer):
    """Capa de atención personalizada para identificar señales críticas de deslizamiento."""
    
    def __init__(self, unidades):
        super(AtencionPersonalizada, self).__init__()
        self.unidades = unidades
        self.W = layers.Dense(unidades)
        self.V = layers.Dense(1)
        
    def call(self, inputs):
        # inputs: (batch_size, time_steps, features)
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

class NormalizacionPersonalizada(layers.Layer):
    """Capa de normalización para datos ruidosos de sensores."""
    
    def __init__(self, epsilon=1e-3):
        super(NormalizacionPersonalizada, self).__init__()
        self.epsilon = epsilon
        
    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True
        )
        
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        var = tf.reduce_variance(inputs, axis=-1, keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(var + self.epsilon)
        return self.gamma * normalized + self.beta

class ModeloLaderas(Model):
    """Modelo de alerta temprana para deslizamientos de tierra."""
    
    def __init__(self, num_features, num_sensores=10):
        super(ModeloLaderas, self).__init__()
        self.num_features = num_features
        self.num_sensores = num_sensores
        
        # Capas de preprocesamiento
        self.norm_input = NormalizacionPersonalizada()
        
        # Capas LSTM para secuencias temporales
        self.lstm_1 = layers.LSTM(64, return_sequences=True, dropout=0.3)
        self.lstm_2 = layers.LSTM(32, return_sequences=True, dropout=0.3)
        
        # Capa de atención personalizada
        self.atencion = AtencionPersonalizada(32)
        
        # Capas densas para clasificación
        self.dense_1 = layers.Dense(64, activation='relu')
        self.dropout = layers.Dropout(0.4)
        self.dense_2 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(3, activation='softmax')  # 3 niveles: bajo, medio, alto
        
    def call(self, inputs, training=None):
        # Normalización inicial
        x = self.norm_input(inputs)
        
        # Procesamiento secuencial
        x = self.lstm_1(x, training=training)
        x = self.lstm_2(x, training=training)
        
        # Aplicar atención
        context_vector, attention_weights = self.atencion(x)
        
        # Clasificación final
        x = self.dense_1(context_vector)
        x = self.dropout(x, training=training)
        x = self.dense_2(x)
        output = self.output_layer(x)
        
        return output, attention_weights
    
    def visualizar_atencion(self, inputs, labels=None):
        """Visualiza los pesos de atención para interpretación."""
        with tf.GradientTape() as tape:
            predictions, attention_weights = self(inputs, training=False)
            
        # Calcular gradientes para ver qué variables pesan más
        loss = tf.reduce_mean(predictions)
        gradients = tape.gradient(loss, inputs)
        
        # Visualización
        plt.figure(figsize=(15, 8))
        
        # Subplot 1: Pesos de atención
        plt.subplot(2, 2, 1)
        attention_weights_np = attention_weights[0].numpy().flatten()
        plt.bar(range(len(attention_weights_np)), attention_weights_np)
        plt.title('Pesos de Atención por Sensor')
        plt.xlabel('Sensor')
        plt.ylabel('Peso de Atención')
        
        # Subplot 2: Gradientes (importancia de variables)
        plt.subplot(2, 2, 2)
        gradients_np = tf.reduce_mean(tf.abs(gradients[0]), axis=0).numpy()
        plt.bar(range(gradients_np.shape[0]), gradients_np)
        plt.title('Importancia de Variables (Gradientes)')
        plt.xlabel('Variable')
        plt.ylabel('Magnitud del Gradiente')
        
        # Subplot 3: Datos de entrada
        plt.subplot(2, 2, 3)
        inputs_np = inputs[0].numpy()
        plt.imshow(inputs_np.T, aspect='auto', cmap='viridis')
        plt.title('Datos de Sensores (Time Series)')
        plt.xlabel('Tiempo')
        plt.ylabel('Sensor/Variable')
        plt.colorbar()
        
        # Subplot 4: Predicciones
        plt.subplot(2, 2, 4)
        pred_np = predictions[0].numpy()
        labels_pred = ['Bajo', 'Medio', 'Alto']
        colors = ['green', 'yellow', 'red']
        bars = plt.bar(labels_pred, pred_np, color=colors)
        plt.title('Predicción de Riesgo')
        plt.ylabel('Probabilidad')
        
        # Añadir valores en las barras
        for bar, val in zip(bars, pred_np):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'predictions': predictions,
            'attention_weights': attention_weights,
            'gradients': gradients
        }

def generar_datos_sinteticos(n_samples=1000, time_steps=24, n_features=8):
    """Genera datos sintéticos de sensores para entrenamiento."""
    np.random.seed(42)
    
    # Datos base con patrones realistas
    base_data = np.random.randn(n_samples, time_steps, n_features)
    
    # Simular diferentes condiciones
    for i in range(n_samples):
        # Condición de riesgo (30% de los casos)
        if np.random.random() < 0.3:
            # Aumentar humedad y movimiento en sensores críticos
            risk_start = np.random.randint(0, time_steps-8)
            risk_duration = np.random.randint(4, 9)
            
            base_data[i, risk_start:risk_start+risk_duration, 0] *= 3.0  # Humedad
            base_data[i, risk_start:risk_start+risk_duration, 1] *= 2.5  # Movimiento
            base_data[i, risk_start:risk_start+risk_duration, 2] *= 1.8  # Presión
            
            # Añadir ruido realista
            noise = np.random.normal(0, 0.1, (risk_duration, n_features))
            base_data[i, risk_start:risk_start+risk_duration, :] += noise
    
    # Normalizar datos
    base_data = (base_data - base_data.mean()) / base_data.std()
    
    # Generar etiquetas basadas en patrones
    labels = []
    for i in range(n_samples):
        max_humidity = np.max(base_data[i, :, 0])
        max_movement = np.max(base_data[i, :, 1])
        
        if max_humidity > 2.0 and max_movement > 1.5:
            labels.append(2)  # Alto riesgo
        elif max_humidity > 1.0 or max_movement > 0.8:
            labels.append(1)  # Medio riesgo
        else:
            labels.append(0)  # Bajo riesgo
    
    return base_data.astype(np.float32), np.array(labels)

def main():
    """Función principal para demostrar el sistema."""
    print("🏔️  Sistema de Alerta Temprana de Deslizamientos (Siata-AI)")
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
    
    # Crear y compilar modelo
    print("\n🧠 Construyendo modelo de IA...")
    modelo = ModeloLaderas(num_features=X_train.shape[2])
    
    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenar modelo
    print("\n🏋️  Iniciando entrenamiento...")
    history = modelo.fit(
        X_train, y_train_onehot,
        validation_data=(X_test, y_test_onehot),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    # Evaluar modelo
    print("\n📊 Evaluando modelo...")
    test_loss, test_acc = modelo.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"🎯 Precisión en prueba: {test_acc:.4f}")
    
    # Visualizar predicciones con atención
    print("\n🔍 Analizando predicciones con atención...")
    sample_idx = np.random.choice(len(X_test), 3, replace=False)
    
    for idx in sample_idx:
        print(f"\n--- Análisis de muestra {idx} ---")
        sample_input = tf.expand_dims(X_test[idx], axis=0)
        true_label = ['Bajo', 'Medio', 'Alto'][y_test[idx]]
        
        results = modelo.visualizar_atencion(sample_input)
        pred_idx = np.argmax(results['predictions'][0].numpy())
        pred_label = ['Bajo', 'Medio', 'Alto'][pred_idx]
        
        print(f"Etiqueta real: {true_label}")
        print(f"Predicción: {pred_label}")
        print(f"Confianza: {results['predictions'][0][pred_idx].numpy():.3f}")
    
    print("\n✅ Sistema de alerta temprana implementado exitosamente!")
    print("🚀 Listo para despliegue en el SIATA de Medellín")

if __name__ == "__main__":
    main()

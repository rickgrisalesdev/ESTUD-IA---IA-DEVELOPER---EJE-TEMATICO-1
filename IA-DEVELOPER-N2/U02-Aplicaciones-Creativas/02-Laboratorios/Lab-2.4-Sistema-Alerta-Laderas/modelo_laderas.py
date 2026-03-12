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
        # Calcular varianza manualmente para compatibilidad con TensorFlow 2.20
        squared_diff = tf.square(inputs - mean)
        var = tf.reduce_mean(squared_diff, axis=-1, keepdims=True)
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
        
        return output
    
    def visualizar_atencion(self, inputs, labels=None):
        """Visualiza los pesos de atención para interpretación."""
        # Obtener pesos de atención manualmente
        x = self.norm_input(inputs)
        x = self.lstm_1(x, training=False)
        x = self.lstm_2(x, training=False)
        context_vector, attention_weights = self.atencion(x)
        
        # Obtener predicciones
        x = self.dense_1(context_vector)
        x = self.dropout(x, training=False)
        x = self.dense_2(x)
        predictions = self.output_layer(x)
        
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            x = self.norm_input(inputs)
            x = self.lstm_1(x, training=False)
            x = self.lstm_2(x, training=False)
            context_vector, attention_weights = self.atencion(x)
            x = self.dense_1(context_vector)
            x = self.dropout(x, training=False)
            x = self.dense_2(x)
            predictions = self.output_layer(x)
            
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
        if gradients is not None:
            gradients_np = tf.reduce_mean(tf.abs(gradients[0]), axis=0).numpy()
            plt.bar(range(gradients_np.shape[0]), gradients_np)
        else:
            # Si no hay gradientes, mostrar datos de entrada
            input_np = inputs[0].numpy()
            gradients_np = tf.reduce_mean(tf.abs(input_np), axis=0).numpy()
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
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom')
        
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
            salto_alarma = self.capa_salto(tf.cast(condicion_salto, tf.float32))
        else:
            # Durante inferencia, activar salto si se cumple condición
            salto_alarma = tf.cond(
                tf.reduce_any(condicion_salto),
                lambda: self.capa_salto(tf.ones_like(vibracion_actual)),
                lambda: self.capa_salto(tf.zeros_like(vibracion_actual))
            )
        
        # Clasificación final
        if tf.reduce_any(condicion_salto) and not training:
            # Si hay salto de alarma, clasificación directa de alto riesgo
            clasificacion = tf.one_hot(
                tf.fill([tf.shape(inputs)[0]], 2),  # Índice 2 = "Alto"
                depth=3
            )
        else:
            # Clasificación normal
            x_final = tf.concat([x_residual, salto_alarma], axis=-1)  # [batch, 33]
            clasificacion = self.capa_clasificacion(x_final)
        
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
        
        # Penalización por falsos negativos (no alertar cuando debería)
        y_true_classes = tf.argmax(y_true, axis=-1)
        y_pred_classes = tf.argmax(y_pred['clasificacion'], axis=-1)
        
        # Falso negativo: Real es "Alto" (2) pero predicción es "Bajo" (0)
        falsos_negativos = tf.logical_and(
            tf.equal(y_true_classes, 2),  # Real es Alto
            tf.equal(y_pred_classes, 0)   # Predicho es Bajo
        )
        
        # Penalización logarítmica
        penalizacion = tf.cast(falsos_negativos, tf.float32) * self.W_safety * tf.math.log(1.0 + tf.abs(y_pred['salto_alarma']))
        
        # Pérdida total
        loss_total = loss_clasificacion + penalizacion
        
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


class ResidualGuard(tf.keras.layers.Layer):
    """
    Capa personalizada para implementar "Residual Guard"
    Permite saltar capas de procesamiento estándar en condiciones críticas
    """
    
    def __init__(self, umbral_activacion=0.8, **kwargs):
        super().__init__(**kwargs)
        self.umbral_activacion = umbral_activacion
    
    def build(self, input_shape):
        self.dense = self.add_weight(
            name='residual_dense',
            shape=(input_shape[-1], 32),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bypass_weight = self.add_weight(
            name='bypass_weight',
            shape=(input_shape[-1], 32),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs, training=False):
        """
        Lógica de guardia residual
        
        Args:
            inputs: Tensor de entrada
            training: Modo de entrenamiento
            
        Returns:
            Tensor: Salida con posible bypass
        """
        # Procesamiento normal
        normal_output = tf.matmul(inputs, self.dense)
        
        # Detección de condiciones críticas
        max_input = tf.reduce_max(tf.abs(inputs), axis=-1)
        condicion_critica = max_input > self.umbral_activacion
        
        if training:
            # Durante entrenamiento, aprender cuándo hacer bypass
            bypass_output = tf.matmul(inputs, self.bypass_weight)
            output = tf.where(
                tf.expand_dims(condicion_critica, -1),
                bypass_output,
                normal_output
            )
        else:
            # Durante inferencia, usar lógica predefinida
            output = normal_output
        
        return output


# Función de pérdida personalizada
def safety_weighted_loss(y_true, y_pred):
    """
    Función de pérdida con penalización por seguridad
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        
    Returns:
        Tensor: Pérdida con penalización de seguridad
    """
    # Pérdida estándar
    categorical_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Penalización por falsos negativos
    y_true_max = tf.argmax(y_true, axis=-1)
    y_pred_max = tf.argmax(y_pred, axis=-1)
    
    falsos_negativos = tf.cast(
        tf.logical_and(
            tf.equal(y_true_max, 2),  # Real es Alto riesgo
            tf.equal(y_pred_max, 0)   # Predicho es Bajo riesgo
        ),
        tf.float32
    )
    
    # Factor de seguridad
    safety_factor = 2.0
    penalty = falsos_negativos * safety_factor * tf.math.log(1.0 + tf.reduce_max(y_pred, axis=-1))
    
    return categorical_loss + penalty
    def main():
        """Función principal para demostrar el sistema completo."""
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
    
    # Compilar centinela con pérdida personalizada
    centinela.compile(
        optimizer='adam',
        loss=lambda y_true, y_pred: centinela.compute_loss(y_true, y_pred),
        metrics=['accuracy']
    )
    
    # Entrenar modelo tradicional
    print("\n🎓 Entrenando modelo tradicional...")
    modelo_tradicional.fit(
        X_train, y_train_onehot,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Entrenar centinela
    print("\n🚨 Entrenando Centinela del Valle...")
    centinela.fit(
        X_train, y_train_onehot,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluar ambos modelos
    print("\n📊 Evaluando modelos...")
    
    # Modelo tradicional
    loss_trad, acc_trad = modelo_tradicional.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"📈 Modelo Tradicional - Loss: {loss_trad:.4f}, Accuracy: {acc_trad:.4f}")
    
    # Centinela
    loss_cent, acc_cent = centinela.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"🚨 Centinela del Valle - Loss: {loss_cent:.4f}, Accuracy: {acc_cent:.4f}")
    
    # Demostrar salto de alarma
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
        print(f"   Salto de Alarma: {'ACTIVADO' if alarma_activa else 'Inactivo'} (valor: {salto_valor:.3f})")
        print(f"   Estado: {'🚨 ALERTA CRÍTICA' if alarma_activa else '✅ Normal'}")
    
    print("\n✅ Sistema completo demostrado exitosamente!")
    print("🎯 El Centinela del Valle proporciona alerta temprana con saltos inteligentes")
    print("🛡️ Arquitectura ResNet con penalización por falsos negativos")
    print("📊 Función de pérdida: Safety-Weighted Loss con penalización logarítmica")


if __name__ == "__main__":
    main()

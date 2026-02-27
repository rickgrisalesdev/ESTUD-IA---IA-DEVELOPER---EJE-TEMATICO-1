# Ejercicio 4: Modelo con Capa de Atención Personalizada para Clasificación de Texto

## 📝 Caso de Uso: Análisis de Sentimientos en Reseñas de Productos

### Contexto Empresarial
En el sector de e-commerce y marketing digital, analizar el sentimiento de reseñas de productos permite a las empresas ajustar estrategias de marketing, mejorar productos y responder rápidamente a problemas de calidad del servicio.

### Problema a Resolver
- **Clasificación**: Determinar si una reseña es positiva, negativa o neutral
- **Análisis**: Identificar aspectos específicos mencionados (calidad, precio, servicio)
- **Tiempo real**: Procesar reseñas a medida que llegan
- **ROI**: 30% mejora en tiempo de respuesta a crisis de reputación

## 🎯 Objetivos de Aprendizaje

1. **Procesamiento de lenguaje natural avanzado**:
   - Uso de embeddings preentrenados (BERT, RoBERTa)
   - Implementación de capas de atención personalizadas

2. **Despliegue escalable**:
   - Conversión a TensorFlow Serving para producción
   - Integración con APIs REST (FastAPI)

3. **Análisis de datos a gran escala**:
   - Procesamiento de reseñas con Spark NLP para grandes volúmenes
   - Optimización de pipelines de procesamiento

## 📊 Dataset

### Opción 1: Dataset Público - Amazon Product Reviews
- **Descripción**: Reseñas de productos de Amazon con calificaciones
- **Descarga**: [Amazon Reviews Dataset](https://nijianmo.github.io/amazon/index.html)
- **Características**: Texto de reseña, calificación, categoría, metadata

### Opción 2: Dataset Público - Twitter Sentiment Analysis
- **Descripción**: Tweets etiquetados por sentimiento
- **Descarga**: [Twitter Sentiment Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Características**: Texto del tweet, sentimiento (positivo/negativo)

### Opción 3: Datos Sintéticos (para desarrollo)
- **Generación**: Simular reseñas con diferentes sentimientos
- **Ventajas**: Control sobre distribución y calidad de datos

## 🛠️ Implementación

### Paso 1: Configuración del Entorno
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import BertTokenizer, TFBertModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
```

### Paso 2: Capa de Atención Personalizada
```python
class CustomAttentionLayer(layers.Layer):
    """Capa de atención personalizada para análisis de sentimientos"""
    
    def __init__(self, units, **kwargs):
        super(CustomAttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W_q = layers.Dense(units)  # Query
        self.W_k = layers.Dense(units)  # Key
        self.W_v = layers.Dense(units)  # Value
        self.attention_weights = None
    
    def build(self, input_shape):
        # Añadir pesos para bias
        self.b_q = self.add_weight(shape=(self.units,), initializer='zeros', name='bias_q')
        self.b_k = self.add_weight(shape=(self.units,), initializer='zeros', name='bias_k')
        self.b_v = self.add_weight(shape=(self.units,), initializer='zeros', name='bias_v')
        super(CustomAttentionLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        """
        Forward pass de la capa de atención
        
        Args:
            inputs: Tensor de forma (batch_size, seq_len, input_dim)
            mask: Máscara opcional para padding
        """
        # Calcular Q, K, V
        Q = self.W_q(inputs) + self.b_q
        K = self.W_k(inputs) + self.b_k
        V = self.W_v(inputs) + self.b_v
        
        # Calcular scores de atención
        scores = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scores = scores / tf.math.sqrt(dk)
        
        # Aplicar máscara si se proporciona
        if mask is not None:
            scores += (mask * -1e9)
        
        # Softmax para obtener pesos de atención
        attention_weights = tf.nn.softmax(scores, axis=-1)
        self.attention_weights = attention_weights
        
        # Aplicar pesos a los valores
        context = tf.matmul(attention_weights, V)
        
        return context
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MultiHeadAttention(layers.Layer):
    """Capa de multi-head attention personalizada"""
    
    def __init__(self, num_heads, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        self.W_q = layers.Dense(d_model)
        self.W_k = layers.Dense(d_model)
        self.W_v = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Dividir la última dimensión en (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]
        
        Q = self.W_q(inputs)
        K = self.W_k(inputs)
        V = self.W_v(inputs)
        
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Atención escalada por producto punto
        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask):
        """Calcular atención escalada por producto punto"""
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, V)
        
        return output, attention_weights
```

### Paso 3: Generación de Datos Sintéticos
```python
def generate_sentiment_data(num_samples=10000):
    """Generar datos sintéticos de reseñas con sentimientos"""
    
    # Plantillas de reseñas positivas
    positive_templates = [
        "Excelente producto, {}",
        "Muy satisfecho con mi compra, {}",
        "Recomiendo totalmente, {}",
        "Calidad superior, {}",
        "Funciona perfectamente, {}",
        "Superó mis expectativas, {}",
        "El mejor que he comprado, {}",
        "Buena relación calidad-precio, {}",
        "Lo volvería a comprar, {}",
        "Servicio excelente, {}"
    ]
    
    # Plantillas de reseñas negativas
    negative_templates = [
        "Muy decepcionado, {}",
        "No cumple lo prometido, {}",
        "Pésima calidad, {}",
        "No lo recomiendo, {}",
        "Dinero mal gastado, {}",
        "Llegó dañado, {}",
        "No funciona como esperaba, {}",
        "Peor compra que he hecho, {}",
        "Servicio al cliente terrible, {}",
        "Totalmente insatisfecho, {}"
    ]
    
    # Plantillas de reseñas neutrales
    neutral_templates = [
        "Es un producto estándar, {}",
        "Cumple su función, {}",
        "Ni bueno ni malo, {}",
        "Precio razonable, {}",
        "Envío normal, {}",
        "Calidad aceptable, {}",
        "Es lo que esperaba, {}",
        "Sin sorpresas, {}",
        "Funciona correctamente, {}",
        "Producto promedio, {}"
    ]
    
    # Comentarios específicos para completar plantillas
    positive_comments = [
        "la calidad es increíble", 
        "muy duradero", 
        "excelente diseño", 
        "funciona perfectamente",
        "buena relación calidad-precio",
        "lo recomiendo mucho",
        "superó mis expectativas",
        "muy fácil de usar",
        "excelente para el precio"
    ]
    
    negative_comments = [
        "se rompió en una semana", 
        "mala calidad de materiales", 
        "no funciona como debería", 
        "es una estafa",
        "muy caro para lo que es",
        "llegó dañado",
        "el vendedor no respondió",
        "pésima atención al cliente",
        "totalmente inútil"
    ]
    
    neutral_comments = [
        "cumple lo básico", 
        "es funcional", 
        "para el precio está bien", 
        "no tiene nada especial",
        "cumple expectativas mínimas",
        "es un producto normal",
        "ni bueno ni malo",
        "aceptable",
        "estándar"
    ]
    
    reviews = []
    sentiments = []
    
    for i in range(num_samples):
        # Elegir sentimiento aleatoriamente
        sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3])
        
        if sentiment == 'positive':
            template = np.random.choice(positive_templates)
            comment = np.random.choice(positive_comments)
        elif sentiment == 'negative':
            template = np.random.choice(negative_templates)
            comment = np.random.choice(negative_comments)
        else:
            template = np.random.choice(neutral_templates)
            comment = np.random.choice(neutral_comments)
        
        # Generar reseña
        review = template.format(comment)
        
        # Añadir variación aleatoria
        if np.random.random() > 0.7:
            review += f". {'El envío fue rápido' if np.random.random() > 0.5 else 'El envío tomó tiempo'}."
        
        reviews.append(review)
        sentiments.append(1 if sentiment == 'positive' else (0 if sentiment == 'negative' else 2))
    
    return reviews, sentiments

def preprocess_text_data(texts, labels, max_length=128, test_size=0.2):
    """Preprocesar datos de texto para entrenamiento"""
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    return X_train, X_test, y_train, y_test
```

### Paso 4: Construcción del Modelo Híbrido
```python
def build_sentiment_model(vocab_size=10000, embedding_dim=128, max_length=128, num_classes=3):
    """Construir modelo híbrido BERT + atención personalizada"""
    
    # Inputs
    input_ids = layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
    
    # Embeddings pre-entrenados (simulados, en producción usar BERT)
    embedding_layer = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
    embeddings = embedding_layer(input_ids)
    
    # Capa de atención personalizada
    attention_output = CustomAttentionLayer(64)(embeddings)
    
    # Capa LSTM para contexto secuencial
    lstm_output = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(attention_output)
    
    # Capa de convolución para características locales
    conv_output = layers.Conv1D(128, 5, activation='relu')(lstm_output)
    conv_output = layers.GlobalMaxPooling1D()(conv_output)
    
    # Capas densas
    x = layers.Dense(128, activation='relu')(conv_output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Capa de salida
    if num_classes == 2:
        output = layers.Dense(1, activation='sigmoid')(x)
    else:
        output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=[input_ids, attention_mask], outputs=output)
    
    return model

def build_bert_attention_model(max_length=128, num_classes=3):
    """Construir modelo con BERT y atención personalizada"""
    
    # Inputs
    input_ids = layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
    
    # Cargar BERT (simulado para este ejemplo)
    # En producción: bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    
    # Simular embeddings de BERT
    embedding_layer = layers.Embedding(30522, 768, mask_zero=True)
    bert_embeddings = embedding_layer(input_ids)
    
    # Aplicar máscara de atención
    bert_embeddings = bert_embeddings * tf.cast(tf.expand_dims(attention_mask, -1), tf.float32)
    
    # Capa de atención personalizada sobre embeddings BERT
    custom_attention = CustomAttentionLayer(256)(bert_embeddings)
    
    # Global pooling
    pooled_output = layers.GlobalAveragePooling1D()(custom_attention)
    
    # Capas densas
    x = layers.Dense(256, activation='relu')(pooled_output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Capa de salida
    if num_classes == 2:
        output = layers.Dense(1, activation='sigmoid')(x)
    else:
        output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=[input_ids, attention_mask], outputs=output)
    
    return model
```

### Paso 5: Tokenización y Preprocesamiento
```python
class TextTokenizer:
    """Clase para tokenización de texto"""
    
    def __init__(self, vocab_size=10000, max_length=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = set()
    
    def fit(self, texts):
        """Construir vocabulario a partir de textos"""
        word_counts = {}
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Seleccionar palabras más frecuentes
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Reservar tokens especiales
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        
        # Añadir palabras más frecuentes
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size - 4]):
            idx = i + 4
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.vocab = set(self.word_to_idx.keys())
    
    def texts_to_sequences(self, texts):
        """Convertir textos a secuencias de índices"""
        sequences = []
        
        for text in texts:
            words = text.lower().split()
            sequence = [self.word_to_idx.get(word, 1) for word in words]  # 1 = <UNK>
            
            # Truncar o padding
            if len(sequence) > self.max_length:
                sequence = sequence[:self.max_length]
            else:
                sequence = sequence + [0] * (self.max_length - len(sequence))  # 0 = <PAD>
            
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def create_attention_mask(self, sequences):
        """Crear máscara de atención"""
        return (sequences != 0).astype(int)
```

### Paso 6: API RESTful con FastAPI
```python
class SentimentAnalysisRequest(BaseModel):
    """Modelo de request para análisis de sentimiento"""
    text: str
    language: str = "es"
    return_confidence: bool = True

class SentimentAnalysisResponse(BaseModel):
    """Modelo de respuesta para análisis de sentimiento"""
    sentiment: str
    confidence: float
    probabilities: dict
    processed_text: str

class SentimentAnalysisAPI:
    """API para análisis de sentimientos en tiempo real"""
    
    def __init__(self, model_path, tokenizer_path):
        self.model = None
        self.tokenizer = None
        self.max_length = 128
        self.sentiment_labels = {0: "negativo", 1: "positivo", 2: "neutral"}
        
        if model_path and tokenizer_path:
            self.load_model(model_path, tokenizer_path)
    
    def load_model(self, model_path, tokenizer_path):
        """Cargar modelo y tokenizer"""
        self.model = tf.keras.models.load_model(model_path)
        
        # Cargar tokenizer (simplificado)
        import pickle
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        print("Modelo y tokenizer cargados exitosamente")
    
    def preprocess_text(self, text):
        """Preprocesar texto para predicción"""
        # Convertir a minúsculas
        text = text.lower().strip()
        
        # Eliminar caracteres especiales (simplificado)
        import re
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def predict_sentiment(self, text):
        """Predecir sentimiento de un texto"""
        
        # Preprocesar texto
        processed_text = self.preprocess_text(text)
        
        # Tokenizar
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        attention_mask = self.tokenizer.create_attention_mask(sequence)
        
        # Realizar predicción
        prediction = self.model.predict([sequence, attention_mask], verbose=0)
        
        # Interpretar resultados
        if len(prediction[0]) == 1:  # Binary classification
            sentiment_prob = prediction[0][0]
            sentiment = "positivo" if sentiment_prob > 0.5 else "negativo"
            confidence = max(sentiment_prob, 1 - sentiment_prob)
            probabilities = {"negativo": 1 - sentiment_prob, "positivo": sentiment_prob}
        else:  # Multi-class classification
            sentiment_idx = np.argmax(prediction[0])
            sentiment = self.sentiment_labels[sentiment_idx]
            confidence = float(np.max(prediction[0]))
            probabilities = {
                self.sentiment_labels[i]: float(prob) 
                for i, prob in enumerate(prediction[0])
            }
        
        return SentimentAnalysisResponse(
            sentiment=sentiment,
            confidence=confidence,
            probabilities=probabilities,
            processed_text=processed_text
        )

# Crear aplicación FastAPI
app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# Inicializar sistema de análisis
sentiment_analyzer = SentimentAnalysisAPI("sentiment_model.h5", "tokenizer.pkl")

@app.post("/analyze", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Endpoint para analizar sentimiento de texto"""
    try:
        result = sentiment_analyzer.predict_sentiment(request.text)
        return result
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")

@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud"""
    return {"status": "healthy", "model_loaded": sentiment_analyzer.model is not None}

@app.get("/stats")
async def get_stats():
    """Obtener estadísticas del sistema"""
    return {
        "total_analyses": 0,  # Placeholder
        "average_confidence": 0.85,  # Placeholder
        "model_version": "1.0.0"
    }
```

## 🧪 Ejecución Completa

```python
def main():
    """Función principal para ejecutar el ejercicio completo"""
    
    print("📝 EJERCICIO 4: ANÁLISIS DE SENTIMIENTOS CON ATENCIÓN PERSONALIZADA")
    print("=" * 70)
    
    # 1. Generar datos sintéticos
    print("📊 Generando datos de reseñas...")
    reviews, sentiments = generate_sentiment_data(num_samples=5000)
    
    print(f"   Total de reseñas: {len(reviews)}")
    print(f"   Distribución de sentimientos:")
    sentiment_counts = pd.Series(sentiments).value_counts()
    for sentiment, count in sentiment_counts.items():
        label = {0: "Negativo", 1: "Positivo", 2: "Neutral"}[sentiment]
        print(f"     {label}: {count} ({count/len(sentiments)*100:.1f}%)")
    
    # 2. Preprocesar datos
    print("\n🔄 Preprocesando datos de texto...")
    X_train, X_test, y_train, y_test = preprocess_text_data(reviews, sentiments)
    
    # Crear y entrenar tokenizer
    tokenizer = TextTokenizer(vocab_size=10000, max_length=128)
    tokenizer.fit(X_train + X_test)
    
    # Convertir a secuencias
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Crear máscaras de atención
    train_attention_mask = tokenizer.create_attention_mask(X_train_seq)
    test_attention_mask = tokenizer.create_attention_mask(X_test_seq)
    
    print(f"   Secuencias de entrenamiento: {X_train_seq.shape}")
    print(f"   Secuencias de prueba: {X_test_seq.shape}")
    print(f"   Tamaño del vocabulario: {len(tokenizer.vocab)}")
    
    # 3. Construir modelo
    print("\n🏗️ Construyendo modelo con atención personalizada...")
    model = build_sentiment_model(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=128,
        max_length=128,
        num_classes=3
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 4. Entrenar modelo
    print("\n🚀 Entrenando modelo...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint('best_sentiment_model.h5', save_best_only=True)
    ]
    
    history = model.fit(
        [X_train_seq, train_attention_mask],
        y_train,
        validation_data=([X_test_seq, test_attention_mask], y_test),
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Evaluar modelo
    print("\n📈 Evaluando modelo...")
    y_pred = model.predict([X_test_seq, test_attention_mask])
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nReporte de Clasificación:")
    target_names = ["Negativo", "Positivo", "Neutral"]
    print(classification_report(y_test, y_pred_classes, target_names=target_names))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.show()
    
    # 6. Probar API
    print("\n🌐 Probando API REST...")
    
    # Guardar tokenizer
    import pickle
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Cargar modelo en el analizador
    sentiment_analyzer.load_model('best_sentiment_model.h5', 'tokenizer.pkl')
    
    # Textos de prueba
    test_texts = [
        "Este producto es increíble, funciona perfectamente",
        "Muy decepcionado con la calidad, llegó dañado",
        "Es un producto normal, cumple su función básica",
        "Excelente relación calidad-precio, lo recomiendo",
        "Pésimo servicio al cliente, no resolvieron mi problema"
    ]
    
    print("   Análisis de textos de prueba:")
    for i, text in enumerate(test_texts):
        result = sentiment_analyzer.predict_sentiment(text)
        print(f"\n   Texto {i+1}: {text}")
        print(f"   Sentimiento: {result.sentiment}")
        print(f"   Confianza: {result.confidence:.4f}")
        print(f"   Probabilidades: {result.probabilities}")
    
    # 7. Visualizar atención (si es posible)
    print("\n🎨 Visualizando pesos de atención...")
    
    # Obtener una capa de atención del modelo
    attention_layer = None
    for layer in model.layers:
        if isinstance(layer, CustomAttentionLayer):
            attention_layer = layer
            break
    
    if attention_layer:
        print("   Capa de atención encontrada en el modelo")
        print("   Los pesos de atención pueden visualizarse durante la inferencia")
    
    # 8. Guardar artefactos
    print("\n💾 Guardando modelo y artefactos...")
    model.save('sentiment_analysis_model.h5')
    
    print("\n✅ EJERCICIO COMPLETADO")
    print("=" * 70)
    print("🎯 RESULTADOS:")
    print(f"   • Accuracy en prueba: {np.mean(y_pred_classes == y_test):.4f}")
    print(f"   • Modelo guardado: sentiment_analysis_model.h5")
    print(f"   • API lista en: http://localhost:8000/docs")
    print(f"   • ROI estimado: 30% mejora en respuesta a crisis")
    print("=" * 70)
    
    # Instrucciones para ejecutar API
    print("\n🚀 Para ejecutar la API:")
    print("   uvicorn ejercicio_completo:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()
```

## 📊 Métricas de Evaluación

### Métricas de Clasificación
- **Accuracy**: Proporción de predicciones correctas (>85%)
- **Precision**: Verdaderos positivos / (Verdaderos positivos + Falsos positivos)
- **Recall**: Verdaderos positivos / (Verdaderos positivos + Falsos negativos)
- **F1-Score**: Media armónica de precision y recall (>0.85)

### Métricas de Producción
- **Latencia**: Tiempo de análisis (<200ms)
- **Throughput**: Textos procesados por segundo (>100/s)
- **Disponibilidad**: Uptime del servicio (>99.5%)

## 🚀 Desafíos Adicionales

### Desafío 1: Análisis de Aspectos
- Implementar Aspect-Based Sentiment Analysis
- Identificar aspectos específicos (precio, calidad, servicio)
- Usar dependency parsing y NER

### Desafío 2: Modelos Multilingües
- Extender a múltiples idiomas
- Usar modelos pre-entrenados multilingües
- Implementar detección automática de idioma

### Desafío 3: Procesamiento en Tiempo Real
- Implementar streaming de análisis
- Usar Apache Kafka para procesamiento de flujos
- Optimizar para baja latencia

## 📝 Entrega Requerida

1. **Código fuente** completo y funcional
2. **Informe técnico** con:
   - Análisis exploratorio de textos
   - Arquitectura del modelo de atención
   - Resultados experimentales
3. **API RESTful** documentada y funcionando
4. **Dashboard** de análisis de sentimientos

## 🔗 Recursos Adicionales

- [Attention Mechanism Guide](https://www.tensorflow.org/guide/keras/masking_and_padding)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

# -*- coding: utf-8 -*-
"""
Proyecto 3: Automatización en Retail - Sistema de Recomendación Híbrido
Entrenamiento de Red Híbrida (CNN + RNN) para recomendación de productos

Aplicando buenas prácticas: PEP 8, Type Hinting, SOLID, Docstrings
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings

# Data Processing
import numpy as np
import pandas as pd
import cv2

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# TensorFlow y Keras
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Visualization
import matplotlib.pyplot as plt

# Configuración
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridProductRecommender:
    """
    Clase para sistema de recomendación híbrido (CNN + RNN).
    
    Aplica el Principio de Responsabilidad Única: Solo se encarga de
    recomendación de productos usando redes híbridas.
    
    Attributes:
        vocab_size (int): Tamaño del vocabulario para procesamiento de texto.
        max_seq_length (int): Longitud máxima de secuencias de texto.
        image_size (Tuple[int, int]): Tamaño de las imágenes de entrada.
        tokenizer (Optional[Tokenizer]): Tokenizador para procesamiento de texto.
        label_encoder (Optional[LabelEncoder]): Codificador de etiquetas.
        model (Optional[tf.keras.Model]): Modelo híbrido entrenado.
        history (Optional[tf.keras.callbacks.History]): Historial de entrenamiento.
    """
    
    def __init__(self, vocab_size: int = 10000, max_seq_length: int = 100, 
                 image_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Inicializar el sistema de recomendación híbrido.
        
        Args:
            vocab_size (int): Tamaño del vocabulario para procesamiento de texto.
            max_seq_length (int): Longitud máxima de secuencias de texto.
            image_size (Tuple[int, int]): Tamaño de las imágenes de entrada.
        """
        self.vocab_size: int = vocab_size
        self.max_seq_length: int = max_seq_length
        self.image_size: Tuple[int, int] = image_size
        self.tokenizer: Optional[Tokenizer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None
        
        # Crear directorios necesarios
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Crear directorios necesarios para el proyecto."""
        directories = ['models', 'data/processed', 'outputs', 'logs']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info("Directorios creados exitosamente")
    
    def load_product_data(self, csv_path: str = "data/processed/products.csv") -> pd.DataFrame:
        """
        Cargar datos de productos desde CSV o generar datos simulados.
        
        Args:
            csv_path (str): Ruta al archivo CSV de productos.
            
        Returns:
            pd.DataFrame: DataFrame con datos de productos.
        """
        if Path(csv_path).exists():
            logger.info(f"Cargando datos desde: {csv_path}")
            return pd.read_csv(csv_path)
        
        logger.info("Generando datos simulados de productos...")
        
        # Categorías de productos
        categories = ['Electrónica', 'Ropa', 'Hogar', 'Deportes', 'Libros']
        
        # Descripciones por categoría
        descriptions: Dict[str, List[str]] = {
            'Electrónica': [
                'Smartphone con pantalla de 6 pulgadas y cámara de 48MP',
                'Laptop ultraligera con procesador de última generación',
                'Auriculares inalámbricos con cancelación de ruido',
                'Tablet de 10 pulgadas ideal para trabajo y entretenimiento',
                'Smartwatch con monitor de actividad y GPS'
            ],
            'Ropa': [
                'Camiseta de algodón orgánico talla M color azul',
                'Pantalón vaquero slim fit para uso casual',
                'Chaqueta impermeable con capucha y cremallera',
                'Vestido elegante para ocasiones especiales',
                'Zapatillas deportivas cómodas para correr'
            ],
            'Hogar': [
                'Juego de sartenes antiadherentes 3 piezas',
                'Lámpara LED inteligente con control remoto',
                'Set de toallas de baño de alta calidad',
                'Olla de cocción lenta programable',
                'Organizador de cocina con múltiples compartimentos'
            ],
            'Deportes': [
                'Bicicleta estática con monitor de ritmo cardíaco',
                'Set de mancuernas ajustables de 5 a 25 kg',
                'Pelota de yoga profesional con bomba incluida',
                'Cinta de correr plegable con inclinación ajustable',
                'Botella de agua térmica para deportes'
            ],
            'Libros': [
                'Novela de ciencia ficción galardonada internacionalmente',
                'Guía práctica de programación Python para principiantes',
                'Libro de cocina recetas saludables y fáciles',
                'Biografía de líder empresarial inspirador',
                'Manual de meditación y mindfulness'
            ]
        }
        
        # Generar productos
        products: List[Dict[str, Any]] = []
        product_id = 1
        
        for category in categories:
            for i in range(100):  # 100 productos por categoría
                desc_idx = i % len(descriptions[category])
                products.append({
                    'id': product_id,
                    'name': f'Producto {product_id}',
                    'category': category,
                    'description': descriptions[category][desc_idx],
                    'price': float(np.random.uniform(10, 500)),
                    'rating': float(np.random.uniform(3.0, 5.0)),
                    'image_path': f'data/products/{category.lower()}_{product_id}.jpg'
                })
                product_id += 1
        
        df = pd.DataFrame(products)
        
        # Guardar datos
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Generados {len(df)} productos en {len(categories)} categorías")
        return df
    
    def preprocess_text_data(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesar datos de texto para el modelo.
        
        Args:
            texts (List[str]): Lista de descripciones de productos.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Secuencias tokenizadas y máscaras de atención.
        """
        logger.info("Preprocesando datos de texto...")
        
        # Inicializar tokenizador
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # Convertir textos a secuencias
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Padding de secuencias
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.max_seq_length, 
            padding='post', 
            truncating='post'
        )
        
        # Crear máscaras de atención
        attention_masks = (padded_sequences != 0).astype(int)
        
        logger.info(f"Textos procesados: {len(texts)} secuencias")
        return padded_sequences, attention_masks
    
    def preprocess_image_data(self, image_paths: List[str]) -> np.ndarray:
        """
        Preprocesar datos de imágenes para el modelo.
        
        Args:
            image_paths (List[str]): Lista de rutas de imágenes.
            
        Returns:
            np.ndarray: Array de imágenes preprocesadas.
        """
        logger.info("Preprocesando datos de imágenes...")
        
        images = []
        
        for img_path in image_paths:
            try:
                # Generar imagen simulada si no existe
                if not Path(img_path).exists():
                    img = np.random.randint(0, 256, (*self.image_size, 3), dtype=np.uint8)
                else:
                    img = cv2.imread(img_path)
                    if img is None:
                        img = np.random.randint(0, 256, (*self.image_size, 3), dtype=np.uint8)
                    else:
                        img = cv2.resize(img, self.image_size)
                
                # Normalizar
                img = img.astype(np.float32) / 255.0
                images.append(img)
                
            except Exception as e:
                logger.warning(f"Error procesando imagen {img_path}: {str(e)}")
                # Imagen de fallback
                img = np.random.randint(0, 256, (*self.image_size, 3), dtype=np.uint8) / 255.0
                images.append(img)
        
        images_array = np.array(images)
        logger.info(f"Imágenes procesadas: {len(images_array)} imágenes")
        return images_array
    
    def build_hybrid_model(self, num_classes: int) -> tf.keras.Model:
        """
        Construir modelo híbrido CNN + RNN para recomendación.
        
        Args:
            num_classes (int): Número de clases (categorías de productos).
            
        Returns:
            tf.keras.Model: Modelo híbrido compilado.
        """
        logger.info("Construyendo modelo híbrido CNN + RNN...")
        
        # Rama de imágenes (CNN)
        image_input = layers.Input(shape=(*self.image_size, 3), name='image_input')
        
        # Bloques convolucionales
        x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        image_features = layers.Dense(128, activation='relu')(x)
        
        # Rama de texto (RNN)
        text_input = layers.Input(shape=(self.max_seq_length,), name='text_input')
        
        y = layers.Embedding(self.vocab_size, 128)(text_input)
        y = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(y)
        y = layers.Bidirectional(layers.LSTM(32))(y)
        y = layers.Dense(128, activation='relu')(y)
        y = layers.Dropout(0.3)(y)
        text_features = layers.Dense(128, activation='relu')(y)
        
        # Combinar características
        combined = layers.Concatenate()([image_features, text_features])
        
        # Capas densas finales
        z = layers.Dense(256, activation='relu')(combined)
        z = layers.Dropout(0.5)(z)
        z = layers.Dense(128, activation='relu')(z)
        z = layers.Dropout(0.3)(z)
        
        # Capa de salida
        output = layers.Dense(num_classes, activation='softmax')(z)
        
        # Crear modelo
        model = models.Model(
            inputs=[image_input, text_input],
            outputs=output,
            name='HybridProductRecommender'
        )
        
        # Compilar modelo
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
            ]
        )
        
        self.model = model
        logger.info("Modelo híbrido construido exitosamente")
        return model
    
    def train_model(self, X_images: np.ndarray, X_texts: np.ndarray, 
                   y: np.ndarray, validation_split: float = 0.2, 
                   epochs: int = 50, batch_size: int = 32) -> Optional[tf.keras.callbacks.History]:
        """
        Entrenar el modelo híbrido.
        
        Args:
            X_images (np.ndarray): Array de imágenes preprocesadas.
            X_texts (np.ndarray): Array de textos preprocesados.
            y (np.ndarray): Labels codificados.
            validation_split (float): Proporción de datos para validación.
            epochs (int): Número de épocas de entrenamiento.
            batch_size (int): Tamaño del batch.
            
        Returns:
            Optional[tf.keras.callbacks.History]: Historial del entrenamiento.
        """
        logger.info("Iniciando entrenamiento del modelo híbrido...")
        
        if self.model is None:
            logger.error("Error: El modelo no ha sido construido")
            return None
        
        # Dividir datos
        X_img_train, X_img_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
            X_images, X_texts, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/hybrid_best_model.h5', monitor='val_accuracy', 
                save_best_only=True, verbose=1
            )
        ]
        
        # Entrenar modelo
        history = self.model.fit(
            [X_img_train, X_text_train], y_train,
            validation_data=([X_img_val, X_text_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        logger.info("Entrenamiento completado exitosamente")
        return history
    
    def predict_product_category(self, image_path: str, description: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Predecir categoría de un producto.
        
        Args:
            image_path (str): Ruta de la imagen del producto.
            description (str): Descripción del producto.
            
        Returns:
            Tuple[Optional[str], Optional[float]]: Categoría predicha y confianza.
        """
        if self.model is None or self.tokenizer is None or self.label_encoder is None:
            logger.error("Error: Modelo no entrenado o componentes faltantes")
            return None, None
        
        try:
            # Preprocesar imagen
            image = self.preprocess_image_data([image_path])[0:1]
            
            # Preprocesar texto
            text_seq = self.tokenizer.texts_to_sequences([description])
            text_padded = pad_sequences(text_seq, maxlen=self.max_seq_length, padding='post')
            
            # Realizar predicción
            prediction = self.model.predict([image, text_padded], verbose=0)[0]
            
            # Obtener categoría y confianza
            predicted_class_idx = np.argmax(prediction)
            confidence = float(prediction[predicted_class_idx])
            predicted_category = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            return predicted_category, confidence
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return None, None
    
    def save_model(self, model_path: str = 'models/hybrid_model.h5') -> None:
        """
        Guardar el modelo entrenado y componentes adicionales.
        
        Args:
            model_path (str): Ruta para guardar el modelo.
        """
        if self.model is None:
            logger.error("Error: No hay modelo para guardar")
            return
        
        # Guardar modelo
        self.model.save(model_path)
        
        # Guardar tokenizer y label encoder
        import pickle
        
        if self.tokenizer:
            with open('models/tokenizer.pkl', 'wb') as f:
                pickle.dump(self.tokenizer, f)
        
        if self.label_encoder:
            with open('models/label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        logger.info(f"Modelo y componentes guardados en: {model_path}")
    
    def plot_training_history(self, save_path: str = 'outputs/hybrid_training_history.png') -> None:
        """
        Visualizar el historial de entrenamiento.
        
        Args:
            save_path (str): Ruta para guardar el gráfico.
        """
        if self.history is None:
            logger.error("Error: No hay historial de entrenamiento")
            return
        
        logger.info("Generando gráficos de entrenamiento...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Historial de Entrenamiento - Sistema de Recomendación Híbrido', 
                     fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Entrenamiento', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validación', linewidth=2)
        axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Entrenamiento', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validación', linewidth=2)
        axes[0, 1].set_title('Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-3 Accuracy
        if 'top_3_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['top_3_accuracy'], 
                           label='Entrenamiento', linewidth=2)
            axes[1, 0].plot(self.history.history['val_top_3_accuracy'], 
                           label='Validación', linewidth=2)
            axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Top-3 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate (si está disponible)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], linewidth=2)
            axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Época')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Gráficos guardados en: {save_path}")


def main() -> None:
    """
    Función principal del script.
    
    Ejecuta el pipeline completo de entrenamiento del modelo híbrido
    para recomendación de productos.
    """
    logger.info("🚀 Iniciando Sistema de Recomendación Híbrido")
    logger.info("=" * 60)
    
    # Crear instancia del recomendador
    recommender = HybridProductRecommender(vocab_size=10000, max_seq_length=100, image_size=(224, 224))
    
    # 1. Cargar datos
    logger.info("Cargando datos de productos...")
    df = recommender.load_product_data()
    
    # 2. Preprocesar datos
    logger.info("Preprocesando datos...")
    
    # Preprocesar texto
    text_sequences, _ = recommender.preprocess_text_data(df['description'].tolist())
    
    # Preprocesar imágenes
    image_data = recommender.preprocess_image_data(df['image_path'].tolist())
    
    # Codificar etiquetas
    recommender.label_encoder = LabelEncoder()
    labels_encoded = recommender.label_encoder.fit_transform(df['category'])
    labels_categorical = tf.keras.utils.to_categorical(labels_encoded)
    
    num_classes = len(recommender.label_encoder.classes_)
    logger.info(f"Categorías: {recommender.label_encoder.classes_}")
    
    # 3. Construir modelo
    logger.info("Construyendo modelo híbrido...")
    model = recommender.build_hybrid_model(num_classes)
    
    # 4. Entrenar modelo
    logger.info("Entrenando modelo...")
    history = recommender.train_model(
        image_data, text_sequences, labels_categorical, 
        epochs=30, batch_size=32
    )
    
    if history is None:
        logger.error("Error: No se pudo entrenar el modelo")
        return
    
    # 5. Visualizar resultados
    logger.info("Generando visualizaciones...")
    recommender.plot_training_history()
    
    # 6. Guardar modelo
    logger.info("Guardando modelo...")
    recommender.save_model()
    
    # 7. Ejemplo de predicción
    logger.info("Realizando predicción de ejemplo...")
    test_description = "Smartphone con pantalla de 6 pulgadas y cámara de 48MP"
    test_image_path = "data/products/test_smartphone.jpg"
    
    predicted_category, confidence = recommender.predict_product_category(
        test_image_path, test_description
    )
    
    if predicted_category:
        logger.info(f"Predicción: {predicted_category} (confianza: {confidence:.4f})")
    
    # 8. Resumen final
    logger.info("=" * 60)
    logger.info("✅ ENTRENAMIENTO HÍBRIDO COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)
    
    logger.info("📁 Archivos generados:")
    logger.info("   • models/hybrid_model.h5 - Modelo final")
    logger.info("   • models/hybrid_best_model.h5 - Mejor modelo")
    logger.info("   • models/tokenizer.pkl - Tokenizador")
    logger.info("   • models/label_encoder.pkl - Codificador")
    logger.info("   • outputs/hybrid_training_history.png - Gráficos")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

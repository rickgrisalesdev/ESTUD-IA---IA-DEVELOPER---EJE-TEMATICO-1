# -*- coding: utf-8 -*-
"""
Proyecto 1: Automatización en Logística - Script Principal de Entrenamiento CNN
Detección automática de daños en paquetes usando Redes Neuronales Convolucionales

Aplicando buenas prácticas: PEP 8, Type Hinting, SOLID, Docstrings
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
import warnings

# Data Processing
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow y Keras
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Configuración de logging en lugar de prints
import logging

# Configuración
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PackageDamageDetector:
    """
    Clase principal para el detector de daños en paquetes.
    
    Aplica el Principio de Responsabilidad Única: Solo se encarga de
    detección de daños en paquetes usando CNN.
    
    Attributes:
        img_size (Tuple[int, int]): Tamaño de las imágenes de entrada.
        batch_size (int): Tamaño del batch para entrenamiento.
        model (Optional[tf.keras.Model]): Modelo CNN entrenado.
        history (Optional[tf.keras.callbacks.History]): Historial de entrenamiento.
        class_indices (Optional[Dict[str, int]]): Índices de clases.
    """
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224), batch_size: int = 32) -> None:
        """
        Inicializar el detector de daños.
        
        Args:
            img_size (Tuple[int, int]): Tamaño de las imágenes de entrada.
            batch_size (int): Tamaño del batch para entrenamiento.
        """
        self.img_size: Tuple[int, int] = img_size
        self.batch_size: int = batch_size
        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None
        self.class_indices: Optional[Dict[str, int]] = None
        
        # Crear directorios necesarios
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Crear directorios necesarios para el proyecto."""
        directories = ['models', 'notebooks', 'logs', 'data/processed']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info("Directorios creados exitosamente")
    
    def create_data_generators(self, data_dir: str = "data/processed", validation_split: float = 0.2) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Crear generadores de datos para entrenamiento y validación.
        
        Args:
            data_dir (str): Directorio con las imágenes organizadas por clase.
            validation_split (float): Proporción de datos para validación.
            
        Returns:
            Tuple[Optional[Any], Optional[Any]]: Generadores de datos (train, val).
        """
        logger.info("Creando generadores de datos...")
        
        # Data augmentation para entrenamiento
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Solo rescalado para validación
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        try:
            # Generador de entrenamiento
            train_generator = train_datagen.flow_from_directory(
                data_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                subset='training',
                shuffle=True
            )
            
            # Generador de validación
            val_generator = val_datagen.flow_from_directory(
                data_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                subset='validation',
                shuffle=False
            )
            
            self.class_indices = train_generator.class_indices
            
            logger.info(f"Clases encontradas: {self.class_indices}")
            logger.info(f"Muestras de entrenamiento: {len(train_generator)} batches")
            logger.info(f"Muestras de validación: {len(val_generator)} batches")
            
            return train_generator, val_generator
            
        except Exception as e:
            logger.error(f"Error al crear generadores: {str(e)}")
            return None, None
    
    def build_cnn_model(self, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
        """
        Construir el modelo CNN para detección de daños.
        
        Args:
            input_shape (Tuple[int, int, int]): Forma de entrada de las imágenes.
            
        Returns:
            tf.keras.Model: Modelo CNN compilado.
        """
        logger.info("Construyendo modelo CNN...")
        
        model = models.Sequential([
            # Primera capa convolucional
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Segunda capa convolucional
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Tercera capa convolucional
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Cuarta capa convolucional
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Capas completamente conectadas
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Capa de salida
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        return model
    
    def create_callbacks(self, model_path: str = 'models/cnn_best_model.h5') -> list:
        """
        Crear callbacks para el entrenamiento.
        
        Args:
            model_path (str): Ruta para guardar el mejor modelo.
            
        Returns:
            list: Lista de callbacks configurados.
        """
        callbacks = [
            # Early stopping para evitar overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reducir learning rate cuando no hay mejora
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Guardar el mejor modelo
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, train_generator: Any, val_generator: Any, epochs: int = 50, model_path: str = 'models/cnn_best_model.h5') -> Optional[tf.keras.callbacks.History]:
        """
        Entrenar el modelo CNN.
        
        Args:
            train_generator (Any): Generador de datos de entrenamiento.
            val_generator (Any): Generador de datos de validación.
            epochs (int): Número de épocas de entrenamiento.
            model_path (str): Ruta para guardar el modelo.
            
        Returns:
            Optional[tf.keras.callbacks.History]: Historial del entrenamiento.
        """
        logger.info("Iniciando entrenamiento del modelo...")
        
        if self.model is None:
            logger.error("Error: El modelo no ha sido construido")
            return None
        
        # Crear callbacks
        callbacks = self.create_callbacks(model_path)
        
        # Entrenar modelo
        start_time = datetime.now()
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        logger.info(f"Tiempo de entrenamiento: {training_time}")
        
        self.history = history
        return history
    
    def evaluate_model(self, val_generator: Any) -> Optional[Dict[str, float]]:
        """
        Evaluar el modelo en datos de validación.
        
        Args:
            val_generator (Any): Generador de datos de validación.
            
        Returns:
            Optional[Dict[str, float]]: Diccionario con métricas de evaluación.
        """
        logger.info("Evaluando modelo...")
        
        if self.model is None:
            logger.error("Error: No hay modelo para evaluar")
            return None
        
        # Evaluar modelo
        results = self.model.evaluate(val_generator, verbose=0)
        metrics_names = self.model.metrics_names
        
        # Crear diccionario de resultados
        results_dict = dict(zip(metrics_names, results))
        
        logger.info("\nResultados de Evaluación:")
        for name, value in results_dict.items():
            logger.info(f"   {name.capitalize()}: {value:.4f}")
        
        return results_dict
    
    def plot_training_history(self, save_path: str = 'notebooks/cnn_training_history.png') -> None:
        """
        Visualizar el historial de entrenamiento.
        
        Args:
            save_path (str): Ruta para guardar el gráfico.
        """
        if self.history is None:
            logger.error("Error: No hay historial de entrenamiento")
            return
        
        logger.info("Generando gráficos de entrenamiento...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Historial de Entrenamiento - Detección de Daños en Paquetes', fontsize=16, fontweight='bold')
        
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
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Entrenamiento', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Validación', linewidth=2)
        axes[1, 0].set_title('Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Entrenamiento', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Validación', linewidth=2)
        axes[1, 1].set_title('Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Gráficos guardados en: {save_path}")
    
    def save_model(self, model_path: str = 'models/cnn_model.h5') -> None:
        """
        Guardar el modelo entrenado.
        
        Args:
            model_path (str): Ruta para guardar el modelo.
        """
        if self.model is None:
            logger.error("Error: No hay modelo para guardar")
            return
        
        self.model.save(model_path)
        logger.info(f"Modelo guardado en: {model_path}")
    
    def load_model(self, model_path: str = 'models/cnn_model.h5') -> bool:
        """
        Cargar un modelo entrenado.
        
        Args:
            model_path (str): Ruta del modelo a cargar.
            
        Returns:
            bool: True si el modelo se cargó exitosamente.
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Modelo cargado desde: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            return False
    
    def predict_image(self, image_path: str, img_size: Tuple[int, int] = (224, 224)) -> Tuple[Optional[str], Optional[float]]:
        """
        Realizar predicción sobre una imagen.
        
        Args:
            image_path (str): Ruta de la imagen.
            img_size (Tuple[int, int]): Tamaño para redimensionar la imagen.
            
        Returns:
            Tuple[Optional[str], Optional[float]]: Predicción y confianza.
        """
        if self.model is None:
            logger.error("Error: No hay modelo cargado")
            return None, None
        
        try:
            from tensorflow.keras.preprocessing import image
            
            # Cargar y preprocesar imagen
            img = image.load_img(image_path, target_size=img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # Realizar predicción
            prediction = self.model.predict(img_array)[0][0]
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            # Determinar clase
            class_name = "dañado" if prediction > 0.5 else "sano"
            
            return class_name, confidence
            
        except Exception as e:
            logger.error(f"Error al predecir imagen: {str(e)}")
            return None, None
    
    def print_model_summary(self) -> None:
        """Imprimir resumen del modelo."""
        if self.model is None:
            logger.error("Error: No hay modelo construido")
            return
        
        logger.info("\n🏗️ Resumen del Modelo:")
        logger.info("=" * 60)
        self.model.summary()
        logger.info("=" * 60)
        
        # Información adicional
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        logger.info("\n📊 Información del Modelo:")
        logger.info(f"   • Parámetros totales: {total_params:,}")
        logger.info(f"   • Parámetros entrenables: {trainable_params:,}")
        logger.info(f"   • Parámetros no entrenables: {total_params - trainable_params:,}")
        logger.info(f"   • Número de capas: {len(self.model.layers)}")


def main() -> None:
    """
    Función principal del script.
    
    Ejecuta el pipeline completo de entrenamiento del modelo CNN
    para detección de daños en paquetes.
    """
    logger.info("🚀 Iniciando Sistema de Detección de Daños en Paquetes")
    logger.info("=" * 60)
    
    # Crear instancia del detector
    detector = PackageDamageDetector(img_size=(224, 224), batch_size=32)
    
    # 1. Preparar datos
    logger.info("Preparando datos...")
    train_gen, val_gen = detector.create_data_generators()
    
    if train_gen is None or val_gen is None:
        logger.error("Error: No se pudieron crear los generadores de datos")
        logger.info("💡 Asegúrate de tener las imágenes en 'data/processed/' con las carpetas 'danado/' y 'sano/'")
        return
    
    # 2. Construir modelo
    logger.info("Construyendo modelo CNN...")
    model = detector.build_cnn_model()
    detector.print_model_summary()
    
    # 3. Entrenar modelo
    logger.info("Entrenando modelo...")
    history = detector.train_model(train_gen, val_gen, epochs=50)
    
    if history is None:
        logger.error("Error: No se pudo entrenar el modelo")
        return
    
    # 4. Evaluar modelo
    logger.info("Evaluando modelo...")
    results = detector.evaluate_model(val_gen)
    
    # 5. Visualizar resultados
    logger.info("Generando visualizaciones...")
    detector.plot_training_history()
    
    # 6. Guardar modelo final
    logger.info("Guardando modelo final...")
    detector.save_model()
    
    # 7. Resumen final
    logger.info("=" * 60)
    logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)
    
    if results:
        logger.info("📊 Métricas Finales:")
        logger.info(f"   • Accuracy: {results['accuracy']:.4f}")
        logger.info(f"   • Precision: {results['precision']:.4f}")
        logger.info(f"   • Recall: {results['recall']:.4f}")
        logger.info(f"   • Loss: {results['loss']:.4f}")
    
    logger.info("📁 Archivos generados:")
    logger.info("   • models/cnn_model.h5 - Modelo final")
    logger.info("   • models/cnn_best_model.h5 - Mejor modelo durante entrenamiento")
    logger.info("   • notebooks/cnn_training_history.png - Gráficos de entrenamiento")
    
    logger.info(f"🎯 Clases detectadas: {detector.class_indices}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

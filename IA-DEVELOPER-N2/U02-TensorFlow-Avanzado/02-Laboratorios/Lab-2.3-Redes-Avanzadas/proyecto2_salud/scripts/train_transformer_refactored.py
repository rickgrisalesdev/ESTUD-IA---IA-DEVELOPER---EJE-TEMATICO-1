# -*- coding: utf-8 -*-
"""
Proyecto 2: Automatización en Salud - Análisis de Informes Médicos
Entrenamiento de Transformers para análisis de informes médicos

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

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Transformers y TensorFlow
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalReportAnalyzer:
    """
    Clase para analizar informes médicos usando Transformers.
    
    Aplica el Principio de Responsabilidad Única: Solo se encarga de
    análisis de informes médicos usando modelos Transformer.
    
    Attributes:
        model_name (str): Nombre del modelo Transformer pre-entrenado.
        tokenizer (DistilBertTokenizer): Tokenizador para procesamiento de texto.
        model (Optional[TFDistilBertForSequenceClassification]): Modelo Transformer entrenado.
        history (Optional[tf.keras.callbacks.History]): Historial de entrenamiento.
        label_mapping (Dict[int, str]): Mapeo de etiquetas numéricas a texto.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        """
        Inicializar el analizador de informes médicos.
        
        Args:
            model_name (str): Nombre del modelo Transformer pre-entrenado.
        """
        self.model_name: str = model_name
        self.tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model: Optional[TFDistilBertForSequenceClassification] = None
        self.history: Optional[tf.keras.callbacks.History] = None
        self.label_mapping: Dict[int, str] = {0: "Normal", 1: "Neumonía"}
        
        # Crear directorios necesarios
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Crear directorios necesarios para el proyecto."""
        directories = ['models', 'data/processed', 'outputs', 'logs']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info("Directorios creados exitosamente")
    
    def load_medical_data(self, csv_path: str = "data/processed/reports.csv") -> pd.DataFrame:
        """
        Cargar datos de informes médicos desde CSV o generar datos simulados.
        
        Args:
            csv_path (str): Ruta al archivo CSV de informes médicos.
            
        Returns:
            pd.DataFrame: DataFrame con datos de informes médicos.
        """
        if Path(csv_path).exists():
            logger.info(f"Cargando datos desde: {csv_path}")
            return pd.read_csv(csv_path)
        
        logger.info("Generando datos simulados de informes médicos...")
        
        # Informes médicos simulados
        reports_data: List[Dict[str, Any]] = [
            {
                'text': "El paciente presenta opacidades en lóbulo inferior derecho con infiltrados.",
                'label': 1
            },
            {
                'text': "Radiografía de tórax sin hallazgos patológicos significativos.",
                'label': 0
            },
            {
                'text': "Infiltrados bilaterales sugestivos de neumonía en base pulmonar.",
                'label': 1
            },
            {
                'text': "Neumotórax derecho sin signos evidentes de neumonía.",
                'label': 0
            },
            {
                'text': "Cardiomegalia moderada con congestión pulmonar leve.",
                'label': 1
            },
            {
                'text': "Campos pulmonares claros, sin evidencia de proceso infeccioso.",
                'label': 0
            },
            {
                'text': "Consolidación en lóbulo superior izquierdo compatible con neumonía.",
                'label': 1
            },
            {
                'text': "Patrón intersticial bilateral, posible fibrosis pulmonar.",
                'label': 1
            },
            {
                'text': "Radiografía normal para la edad del paciente.",
                'label': 0
            },
            {
                'text': "Atelectasia segmentaria en base derecha, resto normal.",
                'label': 0
            }
        ]
        
        # Expandir datos para tener más muestras
        expanded_data = []
        for i in range(50):  # 50 repeticiones
            for report in reports_data:
                expanded_data.append({
                    'text': report['text'],
                    'label': report['label']
                })
        
        df = pd.DataFrame(expanded_data)
        
        # Guardar datos
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Generados {len(df)} informes médicos")
        return df
    
    def preprocess_texts(self, texts: List[str], max_length: int = 128) -> Dict[str, np.ndarray]:
        """
        Preprocesar textos para el modelo Transformer.
        
        Args:
            texts (List[str]): Lista de textos de informes médicos.
            max_length (int): Longitud máxima de las secuencias.
            
        Returns:
            Dict[str, np.ndarray]: Diccionario con input_ids y attention_mask.
        """
        logger.info("Preprocesando textos para Transformer...")
        
        # Tokenizar textos
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        )
        
        logger.info(f"Textos procesados: {len(texts)} secuencias")
        return {
            'input_ids': encoded_inputs['input_ids'].numpy(),
            'attention_mask': encoded_inputs['attention_mask'].numpy()
        }
    
    def build_transformer_model(self, num_labels: int = 2, learning_rate: float = 2e-5) -> TFDistilBertForSequenceClassification:
        """
        Construir y configurar el modelo Transformer.
        
        Args:
            num_labels (int): Número de etiquetas de clasificación.
            learning_rate (float): Tasa de aprendizaje.
            
        Returns:
            TFDistilBertForSequenceClassification: Modelo Transformer configurado.
        """
        logger.info("Construyendo modelo Transformer...")
        
        # Cargar modelo pre-entrenado
        model = TFDistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        
        # Compilar modelo
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model
        logger.info("Modelo Transformer construido exitosamente")
        return model
    
    def train_model(self, X_train: Dict[str, np.ndarray], y_train: np.ndarray,
                   X_val: Dict[str, np.ndarray], y_val: np.ndarray,
                   epochs: int = 3, batch_size: int = 16) -> Optional[tf.keras.callbacks.History]:
        """
        Entrenar el modelo Transformer.
        
        Args:
            X_train (Dict[str, np.ndarray]): Datos de entrenamiento.
            y_train (np.ndarray): Labels de entrenamiento.
            X_val (Dict[str, np.ndarray]): Datos de validación.
            y_val (np.ndarray): Labels de validación.
            epochs (int): Número de épocas de entrenamiento.
            batch_size (int): Tamaño del batch.
            
        Returns:
            Optional[tf.keras.callbacks.History]: Historial del entrenamiento.
        """
        logger.info("Iniciando entrenamiento del modelo Transformer...")
        
        if self.model is None:
            logger.error("Error: El modelo no ha sido construido")
            return None
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=2, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=1, min_lr=1e-7, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/transformer_best_model.h5', monitor='val_accuracy',
                save_best_only=True, save_weights_only=True, verbose=1
            )
        ]
        
        # Entrenar modelo
        history = self.model.fit(
            {'input_ids': X_train['input_ids'], 'attention_mask': X_train['attention_mask']},
            y_train,
            validation_data=(
                {'input_ids': X_val['input_ids'], 'attention_mask': X_val['attention_mask']},
                y_val
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        logger.info("Entrenamiento completado exitosamente")
        return history
    
    def predict_report(self, text: str, max_length: int = 128) -> Tuple[Optional[str], Optional[float]]:
        """
        Predecir diagnóstico para un informe médico.
        
        Args:
            text (str): Texto del informe médico.
            max_length (int): Longitud máxima de la secuencia.
            
        Returns:
            Tuple[Optional[str], Optional[float]]: Diagnóstico predicho y confianza.
        """
        if self.model is None:
            logger.error("Error: Modelo no entrenado")
            return None, None
        
        try:
            # Preprocesar texto
            inputs = self.preprocess_texts([text], max_length)
            
            # Realizar predicción
            predictions = self.model.predict(
                {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']},
                verbose=0
            )
            
            # Obtener predicción
            logits = predictions.logits[0]
            probabilities = tf.nn.softmax(logits).numpy()
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            # Mapear a etiqueta
            diagnosis = self.label_mapping.get(predicted_class, "Desconocido")
            
            return diagnosis, confidence
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return None, None
    
    def evaluate_model(self, X_test: Dict[str, np.ndarray], y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluar el modelo en datos de prueba.
        
        Args:
            X_test (Dict[str, np.ndarray]): Datos de prueba.
            y_test (np.ndarray): Labels de prueba.
            
        Returns:
            Dict[str, Any]: Métricas de evaluación.
        """
        logger.info("Evaluando modelo...")
        
        if self.model is None:
            logger.error("Error: No hay modelo para evaluar")
            return {}
        
        # Realizar predicciones
        predictions = self.model.predict(
            {'input_ids': X_test['input_ids'], 'attention_mask': X_test['attention_mask']},
            verbose=0
        )
        
        # Obtener clases predichas
        predicted_classes = np.argmax(predictions.logits, axis=1)
        
        # Calcular métricas
        report = classification_report(
            y_test, predicted_classes, 
            target_names=list(self.label_mapping.values()),
            output_dict=True
        )
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, predicted_classes)
        
        logger.info("Evaluación completada")
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy']
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = 'outputs/confusion_matrix.png') -> None:
        """
        Visualizar matriz de confusión.
        
        Args:
            cm (np.ndarray): Matriz de confusión.
            save_path (str): Ruta para guardar el gráfico.
        """
        logger.info("Generando matriz de confusión...")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=list(self.label_mapping.values()),
            yticklabels=list(self.label_mapping.values())
        )
        plt.title('Matriz de Confusión - Análisis de Informes Médicos', fontsize=16, fontweight='bold')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Matriz de confusión guardada en: {save_path}")
    
    def plot_training_history(self, save_path: str = 'outputs/transformer_training_history.png') -> None:
        """
        Visualizar el historial de entrenamiento.
        
        Args:
            save_path (str): Ruta para guardar el gráfico.
        """
        if self.history is None:
            logger.error("Error: No hay historial de entrenamiento")
            return
        
        logger.info("Generando gráficos de entrenamiento...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Historial de Entrenamiento - Modelo Transformer', fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Entrenamiento', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validación', linewidth=2)
        axes[0].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Entrenamiento', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validación', linewidth=2)
        axes[1].set_title('Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Gráficos guardados en: {save_path}")
    
    def save_model(self, model_path: str = 'models/transformer_model') -> None:
        """
        Guardar el modelo entrenado y componentes adicionales.
        
        Args:
            model_path (str): Ruta para guardar el modelo.
        """
        if self.model is None:
            logger.error("Error: No hay modelo para guardar")
            return
        
        # Guardar modelo
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        logger.info(f"Modelo y tokenizer guardados en: {model_path}")


def main() -> None:
    """
    Función principal del script.
    
    Ejecuta el pipeline completo de entrenamiento del modelo Transformer
    para análisis de informes médicos.
    """
    logger.info("🚀 Iniciando Sistema de Análisis de Informes Médicos")
    logger.info("=" * 60)
    
    # Crear instancia del analizador
    analyzer = MedicalReportAnalyzer(model_name="distilbert-base-uncased")
    
    # 1. Cargar datos
    logger.info("Cargando datos de informes médicos...")
    df = analyzer.load_medical_data()
    
    # 2. Dividir datos
    logger.info("Dividiendo datos...")
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    # 3. Preprocesar datos
    logger.info("Preprocesando datos...")
    
    # Procesar textos
    X_train = analyzer.preprocess_texts(train_df['text'].tolist())
    X_val = analyzer.preprocess_texts(val_df['text'].tolist())
    X_test = analyzer.preprocess_texts(test_df['text'].tolist())
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # 4. Construir modelo
    logger.info("Construyendo modelo Transformer...")
    model = analyzer.build_transformer_model(num_labels=2)
    
    # 5. Entrenar modelo
    logger.info("Entrenando modelo...")
    history = analyzer.train_model(
        X_train, y_train, X_val, y_val,
        epochs=3, batch_size=16
    )
    
    if history is None:
        logger.error("Error: No se pudo entrenar el modelo")
        return
    
    # 6. Evaluar modelo
    logger.info("Evaluando modelo...")
    results = analyzer.evaluate_model(X_test, y_test)
    
    # 7. Visualizar resultados
    logger.info("Generando visualizaciones...")
    analyzer.plot_training_history()
    analyzer.plot_confusion_matrix(results['confusion_matrix'])
    
    # 8. Guardar modelo
    logger.info("Guardando modelo...")
    analyzer.save_model()
    
    # 9. Ejemplos de predicción
    logger.info("Realizando predicciones de ejemplo...")
    
    test_reports = [
        "El paciente presenta infiltrados bilaterales compatibles con neumonía.",
        "Radiografía de tórax normal sin hallazgos patológicos.",
        "Opacidades en lóbulo inferior derecho con signos de infección."
    ]
    
    for report in test_reports:
        diagnosis, confidence = analyzer.predict_report(report)
        if diagnosis:
            logger.info(f"Reporte: {report[:50]}...")
            logger.info(f"Diagnóstico: {diagnosis} (confianza: {confidence:.4f})")
        logger.info("-" * 40)
    
    # 10. Resumen final
    logger.info("=" * 60)
    logger.info("✅ ENTRENAMIENTO TRANSFORMER COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)
    
    if results:
        logger.info("📊 Métricas Finales:")
        logger.info(f"   • Accuracy: {results['accuracy']:.4f}")
        
        # Mostrar métricas por clase
        for label in analyzer.label_mapping.values():
            if label in results['classification_report']:
                metrics = results['classification_report'][label]
                logger.info(f"   • {label}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    logger.info("📁 Archivos generados:")
    logger.info("   • models/transformer_model/ - Modelo y tokenizer")
    logger.info("   • models/transformer_best_model.h5 - Mejor pesos")
    logger.info("   • outputs/transformer_training_history.png - Gráficos")
    logger.info("   • outputs/confusion_matrix.png - Matriz de confusión")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

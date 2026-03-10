#!/usr/bin/env python3
"""
API Server para Modelos de IA del Bootcamp
Servidor FastAPI para servir predicciones de todos los modelos

Aplicando buenas prácticas: PEP 8, Type Hinting, SOLID, Docstrings
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import warnings

# Web Framework
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Data Processing
import numpy as np
import pandas as pd
import cv2

# Machine Learning
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Configuración
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la app
app = FastAPI(
    title="Bootcamp IA Developer API",
    description="API para servir modelos de IA entrenados en el bootcamp",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modelos Pydantic para requests
class ImagePredictionRequest(BaseModel):
    """Modelo para requests de predicción de imágenes."""
    image_path: str = Field(..., description="Ruta de la imagen a analizar")
    model_type: str = Field("cnn", description="Tipo de modelo: 'cnn' o 'hybrid'")


class TextPredictionRequest(BaseModel):
    """Modelo para requests de predicción de texto."""
    text: str = Field(..., description="Texto para analizar")
    model_type: str = Field("transformer", description="Tipo de modelo: 'transformer'")


class HybridPredictionRequest(BaseModel):
    """Modelo para requests de predicción híbrida."""
    image_path: str = Field(..., description="Ruta de la imagen")
    description: str = Field(..., description="Descripción del producto")


class PredictionResponse(BaseModel):
    """Modelo para responses de predicción."""
    success: bool
    prediction: Optional[str]
    confidence: Optional[float]
    model_type: str
    processing_time: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Modelo para response de health check."""
    status: str
    models_loaded: Dict[str, bool]
    api_version: str


# Clases de modelos
class ModelManager:
    """
    Gestor de modelos de IA para la API.
    
    Aplica el Principio de Responsabilidad Única: Solo se encarga de
    cargar y gestionar modelos de IA.
    
    Attributes:
        models (Dict[str, Any]): Diccionario de modelos cargados.
        model_configs (Dict[str, Dict]): Configuración de modelos.
    """
    
    def __init__(self) -> None:
        """Inicializar el gestor de modelos."""
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict] = self._load_model_configs()
        
        # Crear directorios necesarios
        Path("models").mkdir(exist_ok=True)
    
    def _load_model_configs(self) -> Dict[str, Dict]:
        """
        Cargar configuración de modelos.
        
        Returns:
            Dict[str, Dict]: Configuración de los modelos.
        """
        return {
            "cnn": {
                "path": "proyecto1_logistica/scripts/models/cnn_model.h5",
                "class_mapping": {0: "sano", 1: "dañado"},
                "input_shape": (224, 224, 3),
                "description": "CNN para detección de daños en paquetes"
            },
            "transformer": {
                "path": "proyecto2_salud/scripts/models/transformer_model",
                "class_mapping": {0: "Normal", 1: "Neumonía"},
                "description": "Transformer para análisis de informes médicos"
            },
            "hybrid": {
                "path": "proyecto3_retail/scripts/models/hybrid_model.h5",
                "class_mapping": None,  # Se carga dinámicamente
                "description": "Red híbrida CNN+RNN para recomendación"
            }
        }
    
    def load_cnn_model(self) -> bool:
        """
        Cargar modelo CNN para detección de daños.
        
        Returns:
            bool: True si el modelo se cargó exitosamente.
        """
        try:
            model_path = self.model_configs["cnn"]["path"]
            if not Path(model_path).exists():
                logger.warning(f"Modelo CNN no encontrado: {model_path}")
                return False
            
            model = tf.keras.models.load_model(model_path)
            self.models["cnn"] = model
            logger.info("Modelo CNN cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo CNN: {str(e)}")
            return False
    
    def load_transformer_model(self) -> bool:
        """
        Cargar modelo Transformer para análisis médico.
        
        Returns:
            bool: True si el modelo se cargó exitosamente.
        """
        try:
            model_path = self.model_configs["transformer"]["path"]
            if not Path(model_path).exists():
                logger.warning(f"Modelo Transformer no encontrado: {model_path}")
                return False
            
            # Cargar modelo y tokenizer
            model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
            tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            
            self.models["transformer"] = {
                "model": model,
                "tokenizer": tokenizer
            }
            logger.info("Modelo Transformer cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo Transformer: {str(e)}")
            return False
    
    def load_hybrid_model(self) -> bool:
        """
        Cargar modelo híbrido para recomendación.
        
        Returns:
            bool: True si el modelo se cargó exitosamente.
        """
        try:
            model_path = self.model_configs["hybrid"]["path"]
            if not Path(model_path).exists():
                logger.warning(f"Modelo híbrido no encontrado: {model_path}")
                return False
            
            # Cargar modelo principal
            model = tf.keras.models.load_model(model_path)
            
            # Cargar componentes adicionales
            tokenizer_path = "proyecto3_retail/scripts/models/tokenizer.pkl"
            encoder_path = "proyecto3_retail/scripts/models/label_encoder.pkl"
            
            tokenizer = None
            label_encoder = None
            
            if Path(tokenizer_path).exists():
                import pickle
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
            
            if Path(encoder_path).exists():
                import pickle
                with open(encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
            
            self.models["hybrid"] = {
                "model": model,
                "tokenizer": tokenizer,
                "label_encoder": label_encoder
            }
            logger.info("Modelo híbrido cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo híbrido: {str(e)}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Cargar todos los modelos disponibles.
        
        Returns:
            Dict[str, bool]: Estado de carga de cada modelo.
        """
        results = {}
        
        results["cnn"] = self.load_cnn_model()
        results["transformer"] = self.load_transformer_model()
        results["hybrid"] = self.load_hybrid_model()
        
        return results
    
    def preprocess_image(self, image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
        """
        Preprocesar imagen para predicción.
        
        Args:
            image_path (str): Ruta de la imagen.
            target_size (tuple): Tamaño objetivo.
            
        Returns:
            np.ndarray: Imagen preprocesada.
        """
        try:
            # Cargar imagen
            if not Path(image_path).exists():
                # Generar imagen simulada si no existe
                img = np.random.randint(0, 256, (*target_size, 3), dtype=np.uint8)
            else:
                img = cv2.imread(image_path)
                if img is None:
                    img = np.random.randint(0, 256, (*target_size, 3), dtype=np.uint8)
                else:
                    img = cv2.resize(img, target_size)
            
            # Normalizar
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            return img
            
        except Exception as e:
            logger.error(f"Error preprocesando imagen: {str(e)}")
            # Imagen de fallback
            img = np.random.randint(0, 256, (*target_size, 3), dtype=np.uint8) / 255.0
            return np.expand_dims(img, axis=0)


# Instancia global del gestor de modelos
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la API."""
    logger.info("Iniciando API Server para Bootcamp IA Developer")
    
    # Cargar modelos
    model_status = model_manager.load_all_models()
    
    loaded_models = sum(model_status.values())
    total_models = len(model_status)
    
    logger.info(f"Modelos cargados: {loaded_models}/{total_models}")
    for model_name, status in model_status.items():
        status_icon = "✅" if status else "❌"
        logger.info(f"  {status_icon} {model_name}: {'Cargado' if status else 'No disponible'}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz."""
    return {
        "message": "Bootcamp IA Developer API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de health check."""
    model_status = {}
    for model_name in model_manager.model_configs.keys():
        model_status[model_name] = model_name in model_manager.models
    
    return HealthResponse(
        status="healthy",
        models_loaded=model_status,
        api_version="1.0.0"
    )


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(request: ImagePredictionRequest):
    """
    Predicción usando modelo CNN o híbrido.
    
    Args:
        request (ImagePredictionRequest): Request con ruta de imagen y tipo de modelo.
        
    Returns:
        PredictionResponse: Resultado de la predicción.
    """
    import time
    start_time = time.time()
    
    try:
        if request.model_type not in ["cnn", "hybrid"]:
            raise HTTPException(status_code=400, detail="model_type debe ser 'cnn' o 'hybrid'")
        
        if request.model_type not in model_manager.models:
            raise HTTPException(status_code=404, detail=f"Modelo {request.model_type} no disponible")
        
        # Preprocesar imagen
        image = model_manager.preprocess_image(request.image_path)
        
        if request.model_type == "cnn":
            # Predicción con CNN
            model = model_manager.models["cnn"]
            prediction = model.predict(image, verbose=0)[0][0]
            confidence = float(prediction if prediction > 0.5 else 1 - prediction)
            class_name = "dañado" if prediction > 0.5 else "sano"
        
        else:  # hybrid
            # Para modelo híbrido, necesitaríamos también texto
            # Por ahora, solo usamos la parte de imagen
            hybrid_model = model_manager.models["hybrid"]["model"]
            # Simular entrada de texto dummy
            dummy_text = np.zeros((1, 100))  # Dummy sequence
            prediction = hybrid_model.predict([image, dummy_text], verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = float(prediction[predicted_class])
            
            # Mapear clase si tenemos encoder
            if model_manager.models["hybrid"]["label_encoder"]:
                class_name = model_manager.models["hybrid"]["label_encoder"].inverse_transform([predicted_class])[0]
            else:
                class_name = f"Clase_{predicted_class}"
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            success=True,
            prediction=class_name,
            confidence=confidence,
            model_type=request.model_type,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error en predicción de imagen: {str(e)}")
        
        return PredictionResponse(
            success=False,
            prediction=None,
            confidence=None,
            model_type=request.model_type,
            processing_time=processing_time,
            error=str(e)
        )


@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text(request: TextPredictionRequest):
    """
    Predicción usando modelo Transformer.
    
    Args:
        request (TextPredictionRequest): Request con texto y tipo de modelo.
        
    Returns:
        PredictionResponse: Resultado de la predicción.
    """
    import time
    start_time = time.time()
    
    try:
        if request.model_type != "transformer":
            raise HTTPException(status_code=400, detail="model_type debe ser 'transformer'")
        
        if "transformer" not in model_manager.models:
            raise HTTPException(status_code=404, detail="Modelo transformer no disponible")
        
        # Obtener modelo y tokenizer
        transformer_data = model_manager.models["transformer"]
        model = transformer_data["model"]
        tokenizer = transformer_data["tokenizer"]
        
        # Tokenizar texto
        inputs = tokenizer(
            request.text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='tf'
        )
        
        # Realizar predicción
        predictions = model.predict(
            {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']},
            verbose=0
        )
        
        # Obtener resultado
        logits = predictions.logits[0]
        probabilities = tf.nn.softmax(logits).numpy()
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])
        
        # Mapear a etiqueta
        class_mapping = model_manager.model_configs["transformer"]["class_mapping"]
        class_name = class_mapping.get(predicted_class, "Desconocido")
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            success=True,
            prediction=class_name,
            confidence=confidence,
            model_type=request.model_type,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error en predicción de texto: {str(e)}")
        
        return PredictionResponse(
            success=False,
            prediction=None,
            confidence=None,
            model_type=request.model_type,
            processing_time=processing_time,
            error=str(e)
        )


@app.post("/predict/hybrid", response_model=PredictionResponse)
async def predict_hybrid(request: HybridPredictionRequest):
    """
    Predicción usando modelo híbrido CNN+RNN.
    
    Args:
        request (HybridPredictionRequest): Request con imagen y descripción.
        
    Returns:
        PredictionResponse: Resultado de la predicción.
    """
    import time
    start_time = time.time()
    
    try:
        if "hybrid" not in model_manager.models:
            raise HTTPException(status_code=404, detail="Modelo híbrido no disponible")
        
        hybrid_data = model_manager.models["hybrid"]
        model = hybrid_data["model"]
        tokenizer = hybrid_data["tokenizer"]
        
        # Preprocesar imagen
        image = model_manager.preprocess_image(request.image_path)
        
        # Preprocesar texto
        if tokenizer:
            text_seq = tokenizer.texts_to_sequences([request.description])
            text_padded = tf.keras.preprocessing.sequence.pad_sequences(
                text_seq, maxlen=100, padding='post'
            )
        else:
            # Texto dummy si no hay tokenizer
            text_padded = np.zeros((1, 100))
        
        # Realizar predicción
        prediction = model.predict([image, text_padded], verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class])
        
        # Mapear clase
        if hybrid_data["label_encoder"]:
            class_name = hybrid_data["label_encoder"].inverse_transform([predicted_class])[0]
        else:
            class_name = f"Clase_{predicted_class}"
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            success=True,
            prediction=class_name,
            confidence=confidence,
            model_type="hybrid",
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error en predicción híbrida: {str(e)}")
        
        return PredictionResponse(
            success=False,
            prediction=None,
            confidence=None,
            model_type="hybrid",
            processing_time=processing_time,
            error=str(e)
        )


@app.get("/models/info")
async def get_models_info():
    """Obtener información sobre los modelos disponibles."""
    info = {}
    
    for model_name, config in model_manager.model_configs.items():
        info[model_name] = {
            "description": config["description"],
            "loaded": model_name in model_manager.models,
            "path": config["path"]
        }
    
    return {"models": info}


def main() -> None:
    """
    Función principal para iniciar el servidor API.
    """
    logger.info("🚀 Iniciando API Server para Bootcamp IA Developer")
    
    # Iniciar servidor
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Lab 1.1: Sistema de Reconocimiento de Emociones
Clase principal para detección de emociones faciales usando CNN

Aplicando buenas prácticas: PEP 8, Type Hinting, SOLID, Docstrings
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any, List
import logging
from pathlib import Path
import json
import time

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmotionDetector:
    """
    Clase principal para detección de emociones faciales.
    
    Aplica el Principio de Responsabilidad Única: Solo se encarga de
    detección de emociones usando modelos CNN pre-entrenados.
    
    Attributes:
        model_path (str): Ruta al modelo CNN entrenado.
        cascade_path (str): Ruta al clasificador Haar Cascade para detección facial.
        emotion_labels (List[str]): Lista de etiquetas de emociones.
        model (Optional[tf.keras.Model]): Modelo CNN cargado.
        face_cascade (Optional[cv2.CascadeClassifier]): Clasificador facial.
        input_size (Tuple[int, int]): Tamaño de entrada para el modelo.
    """
    
    def __init__(self, model_path: str = "models/emotion_cnn_model.h5", 
                 cascade_path: str = "haarcascade_frontalface_default.xml") -> None:
        """
        Inicializar el detector de emociones.
        
        Args:
            model_path (str): Ruta al modelo CNN entrenado.
            cascade_path (str): Ruta al clasificador Haar Cascade.
        """
        self.model_path: str = model_path
        self.cascade_path: str = cascade_path
        self.emotion_labels: List[str] = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        self.model: Optional[tf.keras.Model] = None
        self.face_cascade: Optional[cv2.CascadeClassifier] = None
        self.input_size: Tuple[int, int] = (48, 48)
        
        # Cargar componentes
        self._load_model()
        self._load_face_cascade()
    
    def _load_model(self) -> None:
        """Cargar el modelo CNN entrenado."""
        try:
            if Path(self.model_path).exists():
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Modelo cargado exitosamente desde: {self.model_path}")
            else:
                logger.warning(f"Modelo no encontrado en: {self.model_path}")
                logger.info("Creando modelo base para demostración...")
                self._create_demo_model()
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            self._create_demo_model()
    
    def _create_demo_model(self) -> None:
        """Crear un modelo de demostración básico."""
        try:
            # Arquitectura CNN básica para demostración
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(self.emotion_labels), activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info("Modelo de demostración creado exitosamente")
            
        except Exception as e:
            logger.error(f"Error creando modelo de demostración: {str(e)}")
    
    def _load_face_cascade(self) -> None:
        """Cargar el clasificador Haar Cascade para detección facial."""
        try:
            # Intentar cargar desde ruta especificada
            if Path(self.cascade_path).exists():
                self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            else:
                # Usar Haar Cascade por defecto de OpenCV
                haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(haar_path)
                logger.info(f"Usando Haar Cascade por defecto: {haar_path}")
            
            if self.face_cascade.empty():
                raise Exception("No se pudo cargar el clasificador facial")
                
            logger.info("Clasificador facial cargado exitosamente")
            
        except Exception as e:
            logger.error(f"Error cargando clasificador facial: {str(e)}")
            logger.info("Creando detector de rostros básico...")
    
    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocesar una imagen facial para el modelo.
        
        Args:
            face_img (np.ndarray): Imagen facial recortada.
            
        Returns:
            np.ndarray: Imagen preprocesada lista para predicción.
        """
        # Convertir a escala de grises
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar
        face_img = cv2.resize(face_img, self.input_size)
        
        # Normalizar
        face_img = face_img.astype('float32') / 255.0
        
        # Expandir dimensiones para el modelo
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        
        return face_img
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detectar rostros en una imagen.
        
        Args:
            image (np.ndarray): Imagen de entrada.
            
        Returns:
            List[Tuple[int, int, int, int]]: Lista de bounding boxes (x, y, w, h).
        """
        if self.face_cascade is None:
            return []
        
        # Convertir a escala de grises para detección
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def predict_emotion(self, face_img: np.ndarray) -> Dict[str, Any]:
        """
        Predecir la emoción de una imagen facial.
        
        Args:
            face_img (np.ndarray): Imagen facial recortada.
            
        Returns:
            Dict[str, Any]: Resultado de la predicción.
        """
        if self.model is None:
            return {'emotion': 'unknown', 'confidence': 0.0, 'probabilities': {}}
        
        try:
            # Preprocesar imagen
            processed_face = self._preprocess_face(face_img)
            
            # Realizar predicción
            predictions = self.model.predict(processed_face, verbose=0)[0]
            
            # Obtener emoción con mayor probabilidad
            predicted_index = np.argmax(predictions)
            emotion = self.emotion_labels[predicted_index]
            confidence = float(predictions[predicted_index])
            
            # Crear diccionario de probabilidades
            probabilities = {
                label: float(prob) 
                for label, prob in zip(self.emotion_labels, predictions)
            }
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probabilities
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return {'emotion': 'error', 'confidence': 0.0, 'probabilities': {}}
    
    def detect_emotions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detectar emociones en todos los rostros de una imagen.
        
        Args:
            image (np.ndarray): Imagen de entrada.
            
        Returns:
            List[Dict[str, Any]]: Lista de detecciones con bounding boxes y emociones.
        """
        # Detectar rostros
        faces = self.detect_faces(image)
        
        results = []
        for (x, y, w, h) in faces:
            # Extraer región facial
            face_img = image[y:y+h, x:x+w]
            
            # Predecir emoción
            emotion_result = self.predict_emotion(face_img)
            
            # Agregar información de bounding box
            result = {
                'bbox': (x, y, w, h),
                'emotion': emotion_result['emotion'],
                'confidence': emotion_result['confidence'],
                'probabilities': emotion_result['probabilities']
            }
            
            results.append(result)
        
        return results
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Dibujar detecciones en la imagen.
        
        Args:
            image (np.ndarray): Imagen original.
            detections (List[Dict[str, Any]]): Lista de detecciones.
            
        Returns:
            np.ndarray: Imagen con detecciones dibujadas.
        """
        result_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            emotion = detection['emotion']
            confidence = detection['confidence']
            
            # Dibujar bounding box
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Preparar texto
            label = f"{emotion}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Dibujar fondo para texto
            cv2.rectangle(
                result_image,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                (0, 255, 0),
                -1
            )
            
            # Dibujar texto
            cv2.putText(
                result_image,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return result_image
    
    def detect_real_time(self, camera_index: int = 0, save_results: bool = False) -> None:
        """
        Iniciar detección de emociones en tiempo real con webcam.
        
        Args:
            camera_index (int): Índice de la cámara web.
            save_results (bool): Guardar resultados en archivo.
        """
        logger.info("Iniciando detección de emociones en tiempo real...")
        
        # Inicializar cámara
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error(f"No se pudo abrir la cámara {camera_index}")
            return
        
        logger.info("Presiona 'q' para salir, 's' para guardar captura")
        
        # Variables para guardar resultados
        all_detections = []
        
        try:
            while True:
                # Capturar frame
                ret, frame = cap.read()
                if not ret:
                    logger.error("No se pudo capturar frame")
                    break
                
                # Detectar emociones
                detections = self.detect_emotions(frame)
                
                # Dibujar detecciones
                result_frame = self.draw_detections(frame, detections)
                
                # Guardar detecciones si se solicita
                if save_results and detections:
                    timestamp = time.time()
                    for detection in detections:
                        detection['timestamp'] = timestamp
                    all_detections.extend(detections)
                
                # Mostrar frame
                cv2.imshow('Detección de Emociones', result_frame)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Guardar captura
                    timestamp = int(time.time())
                    cv2.imwrite(f'outputs/capture_{timestamp}.jpg', result_frame)
                    logger.info(f"Captura guardada: capture_{timestamp}.jpg")
                
        except KeyboardInterrupt:
            logger.info("Detección interrumpida por el usuario")
        
        finally:
            # Liberar recursos
            cap.release()
            cv2.destroyAllWindows()
            
            # Guardar resultados si se solicitó
            if save_results and all_detections:
                self._save_detections(all_detections)
    
    def _save_detections(self, detections: List[Dict[str, Any]]) -> None:
        """
        Guardar detecciones en archivo JSON.
        
        Args:
            detections (List[Dict[str, Any]]): Lista de detecciones.
        """
        try:
            Path('outputs').mkdir(exist_ok=True)
            
            with open('outputs/emotion_detections.json', 'w') as f:
                json.dump(detections, f, indent=2, default=str)
            
            logger.info(f"Guardadas {len(detections)} detecciones en outputs/emotion_detections.json")
            
        except Exception as e:
            logger.error(f"Error guardando detecciones: {str(e)}")
    
    def process_video(self, input_path: str, output_path: str) -> None:
        """
        Procesar un video completo para detección de emociones.
        
        Args:
            input_path (str): Ruta del video de entrada.
            output_path (str): Ruta del video de salida.
        """
        logger.info(f"Procesando video: {input_path}")
        
        # Abrir video
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el video: {input_path}")
            return
        
        # Obtener propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Configurar writer de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detectar emociones
                detections = self.detect_emotions(frame)
                
                # Dibujar detecciones
                result_frame = self.draw_detections(frame, detections)
                
                # Escribir frame
                out.write(result_frame)
                
                # Guardar detecciones
                if detections:
                    for detection in detections:
                        detection['frame_number'] = frame_count
                    total_detections.extend(detections)
                
                frame_count += 1
                
                # Mostrar progreso
                if frame_count % 100 == 0:
                    logger.info(f"Procesados {frame_count} frames")
        
        finally:
            # Liberar recursos
            cap.release()
            out.release()
            
            # Guardar detecciones
            if total_detections:
                self._save_detections(total_detections)
            
            logger.info(f"Video procesado: {output_path}")
            logger.info(f"Total frames procesados: {frame_count}")
            logger.info(f"Total detecciones: {len(total_detections)}")


def main() -> None:
    """
    Función principal para demostración del detector de emociones.
    """
    logger.info("🧠 Iniciando Sistema de Reconocimiento de Emociones")
    
    # Crear directorios necesarios
    Path('models').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    # Inicializar detector
    detector = EmotionDetector()
    
    # Menú de opciones
    print("\n🎯 Sistema de Reconocimiento de Emociones")
    print("=" * 50)
    print("1. Detección en tiempo real (webcam)")
    print("2. Procesar video")
    print("3. Salir")
    
    choice = input("\nSelecciona una opción (1-3): ").strip()
    
    if choice == '1':
        # Detección en tiempo real
        detector.detect_real_time(save_results=True)
    elif choice == '2':
        # Procesar video
        input_video = input("Ruta del video de entrada: ").strip()
        output_video = input("Ruta del video de salida: ").strip()
        detector.process_video(input_video, output_video)
    elif choice == '3':
        logger.info("Saliendo del sistema...")
    else:
        logger.error("Opción no válida")


if __name__ == "__main__":
    main()

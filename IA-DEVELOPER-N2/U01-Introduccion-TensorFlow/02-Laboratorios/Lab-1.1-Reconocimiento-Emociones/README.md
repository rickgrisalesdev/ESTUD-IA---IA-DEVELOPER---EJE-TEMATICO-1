# 🧠 Lab 1.1: Sistema de Reconocimiento de Emociones

## 🎯 **Objetivo del Laboratorio**

Desarrollar un **sistema de reconocimiento de emociones faciales** en tiempo real utilizando **Redes Neuronales Convolucionales (CNN)** con TensorFlow y OpenCV.

---

## 📋 **Contenido del Laboratorio**

### **🎯 Objetivos de Aprendizaje:**
- **Procesamiento de Imágenes**: Manipulación y preprocesamiento facial
- **Redes CNN**: Arquitectura, entrenamiento y optimización
- **OpenCV**: Detección facial y procesamiento en tiempo real
- **TensorFlow**: Construcción y entrenamiento de modelos
- **Evaluación**: Métricas de rendimiento y validación

### **🔧 Tecnologías Utilizadas:**
- **TensorFlow/Keras**: Para el modelo CNN
- **OpenCV**: Para detección facial y procesamiento
- **NumPy**: Para operaciones numéricas
- **Matplotlib**: Para visualización
- **Scikit-learn**: Para métricas de evaluación

---

## 🏗️ **Estructura del Laboratorio**

```
Lab-1.1-Reconocimiento-Emociones/
├── 📁 src/                           # Código fuente
│   ├── 📄 emotion_detector.py          # Clase principal del detector
│   ├── 📄 data_preprocessor.py        # Preprocesamiento de datos
│   ├── 📄 model_trainer.py           # Entrenamiento del modelo
│   ├── 📄 real_time_detector.py       # Detección en tiempo real
│   └── 📄 utils.py                   # Funciones auxiliares
│
├── 📁 models/                        # Modelos entrenados
│   ├── 📄 emotion_cnn_model.h5       # Modelo CNN entrenado
│   └── 📄 model_architecture.json     # Arquitectura del modelo
│
├── 📁 data/                          # Dataset y datos
│   ├── 📁 raw/                       # Imágenes crudas
│   │   ├── 😊 happy/                 # Feliz
│   │   ├── 😢 sad/                   # Triste
│   │   ├── 😠 angry/                  # Enojado
│   │   ├── 😮 surprised/              # Sorprendido
│   │   ├── 😨 fear/                   # Miedo
│   │   └── 😐 neutral/                # Neutro
│   ├── 📁 processed/                 # Datos preprocesados
│   └── 📄 dataset_info.csv           # Información del dataset
│
├── 📁 notebooks/                     # Jupyter notebooks
│   ├── 📄 01_data_exploration.ipynb  # Exploración de datos
│   ├── 📄 02_model_training.ipynb    # Entrenamiento del modelo
│   └── 📄 03_evaluation.ipynb        # Evaluación de resultados
│
├── 📁 outputs/                       # Resultados y visualizaciones
│   ├── 📁 training_plots/            # Gráficos de entrenamiento
│   ├── 📁 confusion_matrices/        # Matrices de confusión
│   └── 📁 sample_predictions/       # Predicciones de ejemplo
│
└── 📄 requirements.txt              # Dependencias específicas
```

---

## 🎯 **Emociones a Reconocer**

| Emoción | Etiqueta | Descripción | Ejemplos |
|----------|----------|-------------|-----------|
| 😊 Feliz | `happy` | Sonrisa, ojos brillantes | Personas riendo, celebrando |
| 😢 Triste | `sad` | Ceños fruncidos, boca hacia abajo | Personas llorando, melancolía |
| 😠 Enojado | `angry` | Ceño elevado, mandíbula tensa | Personas furiosas, frustradas |
| 😮 Sorprendido | `surprised` | Ojos abiertos, boca abierta | Reacciones inesperadas |
| 😨 Miedo | `fear` | Ojos muy abiertos, ceños levantados | Expresiones de terror |
| 😐 Neutro | `neutral` | Expresión relajada, sin emoción marcada | Rostro en reposo |

---

## 🚀 **Flujo de Trabajo**

### **📋 Paso 1: Preparación del Entorno**
```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### **📋 Paso 2: Exploración de Datos**
```bash
# Ejecutar notebook de exploración
jupyter notebook notebooks/01_data_exploration.ipynb
```

### **📋 Paso 3: Entrenamiento del Modelo**
```bash
# Entrenar modelo CNN
python src/model_trainer.py

# O usar notebook interactivo
jupyter notebook notebooks/02_model_training.ipynb
```

### **📋 Paso 4: Evaluación del Modelo**
```bash
# Evaluar rendimiento
python src/emotion_detector.py --mode eval

# Visualizar resultados
jupyter notebook notebooks/03_evaluation.ipynb
```

### **📋 Paso 5: Detección en Tiempo Real**
```bash
# Iniciar detección con cámara
python src/real_time_detector.py

# O con video
python src/real_time_detector.py --source video.mp4
```

---

## 🧠 **Arquitectura del Modelo CNN**

### **🏗️ Estructura de Red:**
```
Input (48x48x1) 
    ↓
Conv2D(32, 3x3) + ReLU + BatchNorm + MaxPool + Dropout
    ↓
Conv2D(64, 3x3) + ReLU + BatchNorm + MaxPool + Dropout
    ↓
Conv2D(128, 3x3) + ReLU + BatchNorm + MaxPool + Dropout
    ↓
Conv2D(256, 3x3) + ReLU + BatchNorm + MaxPool + Dropout
    ↓
Flatten()
    ↓
Dense(512) + ReLU + BatchNorm + Dropout
    ↓
Dense(256) + ReLU + BatchNorm + Dropout
    ↓
Dense(6) + Softmax
```

### **📊 Parámetros del Modelo:**
- **Input**: 48x48 píxeles, escala de grises
- **Capas convolucionales**: 4 bloques con aumento de filtros
- **Capas densas**: 2 capas fully connected
- **Activación**: ReLU (excepto salida con Softmax)
- **Regularización**: BatchNorm + Dropout (0.25-0.5)
- **Output**: 6 neuronas (una por emoción)

---

## 📊 **Métricas de Evaluación**

### **🎯 Métricas Principales:**
- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión por clase
- **Recall**: Sensibilidad por clase
- **F1-Score**: Balance entre precision y recall
- **Confusion Matrix**: Visualización de errores

### **📈 Objetivos de Rendimiento:**
- **Accuracy**: > 85% en conjunto de prueba
- **F1-Score**: > 0.80 promedio
- **Inference Time**: < 50ms por imagen
- **Model Size**: < 50MB para deploy

---

## 🎮 **Uso del Sistema**

### **📸 Con Webcam:**
```python
from src.emotion_detector import EmotionDetector

# Inicializar detector
detector = EmotionDetector(model_path='models/emotion_cnn_model.h5')

# Iniciar detección en tiempo real
detector.detect_real_time()
```

### **📁 Con Imágenes:**
```python
# Detectar emoción en imagen
result = detector.detect_emotion('path/to/image.jpg')
print(f"Emoción: {result['emotion']}")
print(f"Confianza: {result['confidence']:.2f}")
```

### **📹 Con Video:**
```python
# Procesar video completo
detector.process_video('input_video.mp4', 'output_video.mp4')
```

---

## 🔧 **Configuración y Parámetros**

### **📄 Configuración del Entrenamiento:**
```python
TRAINING_CONFIG = {
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.001,
    'image_size': (48, 48),
    'validation_split': 0.2,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 8
}
```

### **📄 Configuración del Modelo:**
```python
MODEL_CONFIG = {
    'input_shape': (48, 48, 1),
    'num_classes': 6,
    'dropout_rate': 0.5,
    'batch_norm_momentum': 0.99,
    'l2_regularization': 0.001
}
```

---

## 🚨 **Desafíos Comunes y Soluciones**

### **❌ Problema: Overfitting**
- **Síntomas**: Accuracy alta en entrenamiento, baja en validación
- **Solución**: Aumentar dropout, data augmentation, early stopping

### **❌ Problema: Detección Facial Fallida**
- **Síntomas**: No detecta rostros o detecta falsos positivos
- **Solución**: Ajustar parámetros de Haar Cascade, mejorar iluminación

### **❌ Problema: Baja Precisión**
- **Síntomas**: Modelo confunde emociones similares
- **Solución**: Aumentar dataset, usar transfer learning, fine-tuning

---

## 📈 **Extensiones y Mejoras**

### **🚀 Mejoras Futuras:**
1. **Transfer Learning**: Usar modelos pre-entrenados (VGG, ResNet)
2. **Data Augmentation**: Rotaciones, zoom, ruido, brillo
3. **Ensemble Methods**: Combinar múltiples modelos
4. **Optimización**: Cuantización, pruning para mobile
5. **Multi-modal**: Combinar con análisis de voz

### **🌐 Aplicaciones del Mundo Real:**
- **Sistemas de salud**: Monitoreo de bienestar emocional
- **Educación**: Adaptación de contenido según estado emocional
- **Marketing**: Análisis de reacciones a productos
- **Seguridad**: Detección de comportamientos anómalos

---

## 📚 **Recursos Adicionales**

### **📖 Documentación:**
- [OpenCV Face Detection](https://docs.opencv.org/4.x/d7/d8b/tutorial_py_face_detection.html)
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [FER Dataset Documentation](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

### **📊 Datasets Recomendados:**
- **FER-2013**: Facial Expression Recognition 2013
- **CK+**: Extended Cohn-Kanade Dataset
- **JAFFE**: Japanese Female Facial Expression
- **AFEW**: Affective Behavior in-the-wild

---

## 🎯 **Evaluación del Laboratorio**

### **📋 Criterios de Éxito:**
- [ ] **Modelo Entrenado**: Accuracy > 85% en test set
- [ ] **Detección en Tiempo Real**: Funciona con webcam
- [ ] **Código Limpio**: Aplicación de principios de Unidad 0
- [ ] **Documentación**: READMEs y comentarios completos
- [ ] **Visualizaciones**: Gráficos de entrenamiento y resultados

### **🏆 Niveles de Logro:**
- **🥉 Básico**: Modelo funcional con detección básica
- **🥈 Intermedio**: Optimización de parámetros y métricas
- **🥇 Avanzado**: Extensiones y mejoras personalizadas

---

**🚀 ¡Comienza a construir tu sistema de reconocimiento de emociones!**

*Este laboratorio es tu primer paso hacia la visión por computadora y el análisis emocional con IA.*

# 📦 **Proyecto 1: Detección de Daños en Paquetes Logísticos**

## 🎯 **Objetivo del Proyecto**
Desarrollar un sistema de inteligencia artificial basado en Redes Neuronales Convolucionales (CNN) para la detección automática de daños en paquetes logísticos, optimizando los procesos de control de calidad en la cadena de suministro.

## 🏗️ **Arquitectura del Sistema**

### **🧠 Modelo CNN Principal**
- **Arquitectura**: 4 capas convolucionales + 2 capas densas
- **Entrada**: Imágenes de 224x224 píxeles (RGB)
- **Salida**: Clasificación binaria (sano/dañado)
- **Activación**: ReLU (capas ocultas), Sigmoid (salida)

### **🔄 Data Augmentation**
- **Rotación**: ±20 grados
- **Traslación**: Horizontal y vertical (20%)
- **Zoom**: 20% variación
- **Shear**: 20% distorsión
- **Flip**: Horizontal activado

## 📊 **Métricas de Evaluación**
- **Accuracy**: Precisión general del modelo
- **Precision**: Calidad de predicciones positivas
- **Recall**: Capacidad de detección de daños
- **Loss**: Función de pérdida (Binary Crossentropy)

## 🚀 **Implementaciones Desarrolladas**

### **1. 📋 Script de Entrenamiento Principal**
- **Archivo**: `scripts/train_cnn.py`
- **Funcionalidad**: Entrenamiento completo del modelo CNN
- **Características**: Callbacks, visualización, guardado automático

### **2. 🖥️ Interfaces Web (Streamlit)**
- **Prototipo Básico**: `app.py`
- **Prototipo Mejorado**: `app_mejorada.py` (drag-and-drop, inventario)
- **Dashboard Completo**: `app/dashboard_completo.py`
- **Dashboard Modular**: `app/dashboard_modular.py`

### **3. 📚 Documentación y Tutoriales**
- **README Completo**: Guía técnica detallada
- **Tutorial Interactivo**: Jupyter notebook completo
- **Glosario de Términos**: Conceptos y definiciones

## 🛠️ **Tecnologías Utilizadas**

### **Core ML/DL**
- **TensorFlow 2.20+**: Framework principal
- **Keras**: API de alto nivel
- **OpenCV 4.13+**: Procesamiento de imágenes
- **Albumentations 2.0.8**: Data augmentation avanzada

### **Visualización**
- **Matplotlib 3.10.8**: Gráficos estáticos
- **Plotly**: Visualizaciones interactivas
- **Seaborn**: Gráficos estadísticos

### **Web Interface**
- **Streamlit 1.55.0**: Dashboards interactivos
- **HTML/CSS**: Personalización de UI

### **Data Processing**
- **NumPy**: Operaciones numéricas
- **Pandas**: Manipulación de datos
- **Scikit-learn**: Métricas y utilidades

## 📁 **Estructura del Proyecto**
```
proyecto1_logistica/
├── scripts/
│   └── train_cnn.py                 # Script principal de entrenamiento
├── app/
│   ├── app.py                       # Prototipo básico
│   ├── app_mejorada.py              # Prototipo mejorado
│   ├── dashboard_completo.py         # Dashboard completo
│   └── dashboard_modular.py         # Dashboard modular
├── components/
│   ├── ui_components.py             # Componentes reutilizables
│   └── training_panel.py            # Panel de entrenamiento
├── docs/
│   ├── README_completo.md            # Documentación completa
│   └── glosario_terminos.md         # Glosario técnico
├── notebooks/
│   └── tutorial_completo.ipynb      # Tutorial interactivo
├── data/
│   └── processed/                   # Dataset organizado
├── models/                          # Modelos entrenados
├── requirements.txt                 # Dependencias
└── README.md                       # Este archivo
```

## 🎯 **Características Principales**

### **🔧 Configuración Flexible**
- Hiperparámetros ajustables en tiempo real
- Arquitectura CNN configurable
- Data augmentation personalizable

### **📊 Visualización Completa**
- Gráficos de entrenamiento en vivo
- Métricas del sistema en tiempo real
- Análisis de dataset interactivo

### **🎨 Interfaz Profesional**
- Diseño responsive y moderno
- Navegación intuitiva
- Feedback visual inmediato

### **📈 Monitoreo Avanzado**
- Callbacks inteligentes
- Early stopping automático
- Model checkpointing

## 🚀 **Cómo Usar el Sistema**

### **1. Preparación del Entorno**
```bash
# Instalar dependencias
pip install -r requirements.txt

# Preparar dataset
mkdir -p data/processed/{danado,sano}
# Colocar imágenes en las carpetas correspondientes
```

### **2. Entrenamiento del Modelo**
```bash
# Entrenar modelo
python scripts/train_cnn.py

# O usar dashboard interactivo
streamlit run app/dashboard_modular.py
```

### **3. Despliegue Web**
```bash
# Dashboard principal
streamlit run app/dashboard_modular.py --server.port 8506

# Prototipo mejorado
streamlit run app/app_mejorada.py --server.port 8502
```

## 📊 **Resultados Esperados**

### **Métricas de Rendimiento**
- **Accuracy**: >90% (con dataset balanceado)
- **Precision**: >85% (minimizar falsos positivos)
- **Recall**: >85% (detectar la mayoría de daños)
- **Training Time**: 10-20 minutos (GPU recomendada)

### **Casos de Uso**
- **Control de Calidad**: Inspección automática en almacenes
- **Logística**: Verificación de estado de paquetes
- **E-commerce**: Control antes del envío
- **Industria**: Monitoreo de productos

## 🔮 **Mejoras Futuras**

### **Técnicas Avanzadas**
- **Transfer Learning**: Usar modelos pre-entrenados
- **Ensemble Methods**: Combinar múltiples modelos
- **Fine-tuning**: Optimización específica del dominio

### **Funcionalidades Adicionales**
- **Detección de Objetos**: Localizar daños específicos
- **Clasificación Multiclase**: Tipos de daños
- **Análisis Temporal**: Seguimiento de deterioro

### **Despliegue Producción**
- **API REST**: Integración con sistemas existentes
- **Containerización**: Docker para despliegue
- **Cloud Deployment**: AWS, Azure, GCP

## 📞 **Soporte y Contacto**

### **Troubleshooting Común**
- **Dataset**: Mínimo 50 imágenes por clase
- **Memoria**: Requiere 8GB+ RAM para entrenamiento
- **GPU**: Recomendada para entrenamiento rápido

### **Referencias**
- **Documentación TensorFlow**: https://tensorflow.org
- **Guía Keras**: https://keras.io
- **Streamlit Docs**: https://docs.streamlit.io

---

**🏭 Proyecto 1: Automatización en Logística**  
*Laboratorio 2.3: Redes Neuronales Avanzadas y Aplicaciones Prácticas*

**🎯 Versión: 1.0 - Última Actualización: 2026**

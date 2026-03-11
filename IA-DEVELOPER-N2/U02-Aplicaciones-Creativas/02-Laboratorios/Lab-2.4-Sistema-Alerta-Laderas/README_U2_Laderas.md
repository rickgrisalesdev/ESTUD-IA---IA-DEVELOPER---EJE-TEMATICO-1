# 🏔️ Sistema de Alerta Temprana de Deslizamientos (Siata-AI)

## 📋 **Contexto del Problema**

Medellín, por su topografía montañosa, necesita monitorear constantemente las laderas para prevenir deslizamientos de tierra que pueden causar pérdidas humanas y materiales. El sistema SIATA (Sistema de Alerta Temprana de Medellín) requiere modelos de IA capaces de procesar datos de múltiples sensores en tiempo real para identificar señales críticas antes de que ocurra un evento catastrófico.

## 🧠 **Arquitectura del Modelo**

### **Componentes Principales:**

1. **Capa de Atención Personalizada**: Identifica qué sensores y variables son más importantes en cada momento
2. **Normalización Personalizada**: Maneja datos ruidosos de sensores en condiciones extremas
3. **Red LSTM Secuencial**: Procesa series temporales para detectar patrones de riesgo
4. **Visualización con GradientTape**: Muestra qué variables pesan más en la predicción

## 🔧 **Técnicas de Regularización para Datos Ruidosos**

### **1. Normalización por Capas Personalizadas**
```python
class NormalizacionPersonalizada(layers.Layer):
    def __init__(self, epsilon=1e-3):
        super(NormalizacionPersonalizada, self).__init__()
        self.epsilon = epsilon  # Evita división por cero
        
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        var = tf.reduce_variance(inputs, axis=-1, keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(var + self.epsilon)
        return self.gamma * normalized + self.beta
```

**¿Por qué es importante?**
- Los sensores en terreno generan datos con ruido y valores extremos
- La normalización estándar puede fallar con datos atípicos
- Permite al modelo aprender patrones reales ignorando artefactos

### **2. Dropout Estratégico**
```python
self.dropout = layers.Dropout(0.4)  # Alto dropout para evitar overfitting
```

**Beneficios:**
- Previene sobreajuste a patrones espurios del ruido
- Fuerza al modelo a aprender características robustas
- Mejora generalización a condiciones no vistas

### **3. Atención con Regularización**
```python
class AtencionPersonalizada(layers.Layer):
    def call(self, inputs):
        score = tf.nn.tanh(self.W(inputs))  # Tanh limita valores extremos
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
```

**Ventajas:**
- El mecanismo de atención filtra automáticamente el ruido
- Softmax asegura que los pesos estén normalizados
- Tanh limita la influencia de valores extremos

## 📊 **Manejo de Datasets Ruidosos**

### **Características del Ruido en Sensores:**

1. **Ruido Gaussiano**: Errores de medición aleatorios
2. **Valores Atípicos**: Fallas temporales de sensores
3. **Deriva**: Cambios graduales en calibración
4. **Pérdida de Datos**: Comunicación intermitente

### **Estrategias de Mitigación:**

#### **A. Preprocesamiento**
- **Interpolación**: Para datos faltantes
- **Filtros de Mediana**: Eliminan valores atípicos
- **Suavizado Exponencial**: Reduce ruido de alta frecuencia

#### **B. Data Augmentation**
- **Adición de Ruido Controlado**: Hace el modelo robusto
- **Variaciones Temporales**: Simula diferentes condiciones
- **Mezcla de Sensores**: Combina datos de múltiples fuentes

#### **C. Validación Cruzada Temporal**
- **Respeto Secuencial**: No mezclar futuro con pasado
- **Ventanas Deslizantes**: Simula predicción en tiempo real
- **Validación Walk-Forward**: Prueba en datos no vistos

## 🎯 **Métricas de Evaluación para Datos Ruidosos**

### **1. Métricas Robustas**
- **F1-Score**: Balancea precisión y recall
- **AUC-ROC**: Resistente a desbalance de clases
- **Matriz de Confusión**: Visualiza errores específicos

### **2. Métricas Temporales**
- **Tiempo de Detección**: Qué tan rápido se detecta el riesgo
- **Falsas Alarmas**: Costo de alertas incorrectas
- **Precisión Temporal**: Alineación con eventos reales

## 🚀 **Implementación Práctica**

### **Flujo de Datos:**
1. **Adquisición**: Sensores → Base de datos
2. **Preprocesamiento**: Limpieza → Normalización
3. **Inferencia**: Modelo → Predicción de riesgo
4. **Postprocesamiento**: Umbral → Alerta

### **Niveles de Alerta:**
- **🟢 Verde**: Riesgo bajo (< 0.3)
- **🟡 Amarillo**: Riesgo medio (0.3-0.7)
- **🔴 Rojo**: Riesgo alto (> 0.7)

## 📈 **Resultados Esperados**

### **Precisión del Modelo:**
- **Sensibilidad**: > 85% para eventos críticos
- **Especificidad**: > 90% para condiciones estables
- **Tiempo de Detección**: < 30 minutos antes del evento

### **Impacto Social:**
- **Reducción de Víctimas**: Prevención temprana
- **Evacuaciones Ordenadas**: Tiempo suficiente para actuar
- **Confianza Pública**: Sistema confiable y transparente

## 🔍 **Interpretabilidad del Modelo**

### **Visualización de Atención:**
- **Mapa de Calor**: Muestra sensores importantes
- **Gradientes**: Identifica variables críticas
- **Series Temporales**: Visualiza evolución del riesgo

### **Explicabilidad para Operadores:**
- **Justificación de Alertas**: Por qué se activó el sistema
- **Confianza de Predicción**: Nivel de certeza del modelo
- **Factores Contribuyentes**: Variables que dispararon la alerta

## 🛠️ **Mejoras Futuras**

### **Técnicas Avanzadas:**
- **Transformers**: Para secuencias más largas
- **Ensemble Learning**: Múltiples modelos voting
- **Transfer Learning**: Adaptación a nuevas geologías

### **Integración de Datos:**
- **Satelital**: Imágenes de cambios topográficos
- **Climático**: Lluvias y condiciones meteorológicas
- **Geológico**: Propiedades del suelo

## 📚 **Referencias Técnicas**

1. **Regularización en Deep Learning**: Srivastava et al., 2014
2. **Attention Mechanisms**: Vaswani et al., 2017
3. **Time Series Classification**: Fawaz et al., 2019
4. **Early Warning Systems**: UNDRR Guidelines, 2022

---

**🎯 Objetivo Principal**: Desarrollar un sistema de IA robusto que pueda operar con datos imperfectos del mundo real y proporcionar alertas confiables para proteger vidas humanas en Medellín.

**💡 Innovación**: Combinar técnicas avanzadas de regularización con mecanismos de atención para crear un sistema interpretable y confiable para toma de decisiones críticas.

## 🔧 **Dependencias del Sistema**

### **TensorFlow (`tensorflow`)**
- **Función**: Framework principal para construir y entrenar redes neuronales
- **Ejecución**: 
  - Define las capas LSTM para procesamiento de secuencias temporales
  - Implementa mecanismos de atención personalizada
  - Compila y entrena el modelo con optimizador Adam
  - Realiza inferencia y cálculos de gradientes

### **Keras (`tensorflow.keras`)**
- **Función**: API de alto nivel para construir arquitecturas de redes neuronales
- **Ejecución**:
  - `layers.Layer`: Base para capas personalizadas (AtencionPersonalizada, NormalizacionPersonalizada)
  - `Model`: Clase base para el modelo ModeloLaderas
  - `layers.Dense`: Capas completamente conectadas
  - `layers.LSTM`: Redes neuronales recurrentes para datos secuenciales
  - `layers.Dropout`: Regularización para evitar overfitting

### **NumPy (`numpy`)**
- **Función**: Computación numérica y manipulación de arrays
- **Ejecución**:
  - Genera datos sintéticos de sensores con distribuciones estadísticas
  - Realiza operaciones matemáticas en los datos de entrada
  - Convierte datos entre formatos (numpy a TensorFlow)
  - Calcula estadísticas para etiquetado automático

### **Matplotlib (`matplotlib.pyplot`)**
- **Función**: Visualización de datos y resultados
- **Ejecución**:
  - Crea gráficos de pesos de atención por sensor
  - Visualiza importancia de variables mediante gradientes
  - Muestra datos de series temporales como mapas de calor
  - Genera barras de predicción de riesgo con probabilidades

---

**🚀 Implementación Completa**: El sistema combina procesamiento de datos de sensores en tiempo real, mecanismos de atención para identificar patrones críticos, y visualizaciones interpretables para toma de decisiones en el SIATA de Medellín.

## 🚀 **Procedimiento de Reproducción y Despliegue**

### **📋 Requisitos Previos**
```bash
# Python 3.8+ recomendado
python --version

# Instalar dependencias
pip install tensorflow numpy matplotlib
```

### **🔧 Pasos de Ejecución**

#### **1. Configuración del Entorno**
```bash
# Clonar repositorio (si aplica)
git clone [repositorio]

# Activar entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

#### **2. Ejecución del Modelo**
```bash
# Ejecutar script principal
python modelo_laderas.py

# Salida esperada:
# 🏔️  Sistema de Alerta Temprana de Deslizamientos (Siata-AI)
# 📊 Generando datos de sensores...
# 🧠 Construyendo modelo de IA...
# 🏋️  Iniciando entrenamiento...
# 📊 Evaluando modelo...
# 🔍 Analizando predicciones con atención...
```

#### **3. Despliegue en Producción**
```python
# Guardar modelo entrenado
modelo.save('modelo_laderas_siata.h5')

# Cargar para predicción
import tensorflow as tf
modelo_cargado = tf.keras.models.load_model('modelo_laderas_siata.h5')
```

### **🌐 Integración con SIATA**
```python
# Conexión con APIs del SIATA
import requests

def obtener_datos_sensores_siata():
    """Obtener datos en tiempo real del SIATA"""
    response = requests.get('https://siata.gov.co/api/sensores/laderas')
    return response.json()

def enviar_alerta_siata(nivel_riesgo, ubicacion):
    """Enviar alerta al sistema SIATA"""
    alerta = {
        'nivel': nivel_riesgo,
        'ubicacion': ubicacion,
        'timestamp': datetime.now()
    }
    requests.post('https://siata.gov.co/api/alertas', json=alerta)
```

---

## 🏗️ **Aplicando los 4 Pilares del Desarrollo Profesional**

### **1. PEP 8: Código Limpio y Profesional**

#### **✅ Cumplimiento en el Código:**
- **Sangría**: 4 espacios consistentes en todo el código
- **Nomenclatura**: 
  - Clases: `ModeloLaderas`, `AtencionPersonalizada` (PascalCase)
  - Funciones: `generar_datos_sinteticos`, `visualizar_atencion` (snake_case)
  - Variables: `num_features`, `attention_weights` (snake_case)
- **Longitud**: Líneas bajo 79 caracteres
- **Espacios**: `func(argumento)` sin espacios extra

#### **📝 Ejemplo de Aplicación:**
```python
# ✅ Correcto (PEP 8)
class AtencionPersonalizada(layers.Layer):
    def __init__(self, unidades: int):
        super(AtencionPersonalizada, self).__init__()
        self.unidades = unidades

# ❌ Incorrecto
class atencionPersonalizada(layers.Layer):
    def __init__(self,unidades):
        super(atencionPersonalizada,self).__init__()
        self.unidades=unidades
```

### **2. Type Hinting: Código Sin Ambigüedad**

#### **🎯 Implementación:**
```python
from typing import Tuple, Optional
import numpy as np

def generar_datos_sinteticos(
    n_samples: int = 1000, 
    time_steps: int = 24, 
    n_features: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera datos sintéticos con tipos definidos."""
    pass

class ModeloLaderas(Model):
    def __init__(self, num_features: int, num_sensores: int = 10):
        """Constructor con tipos explícitos."""
        pass
    
    def call(
        self, 
        inputs: tf.Tensor, 
        training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Método con tipos de retorno definidos."""
        pass
```

### **3. Testing: Código Robusto y Confiable**

#### **🧪 Pruebas Unitarias:**
```python
import unittest
import numpy as np
from modelo_laderas import ModeloLaderas, generar_datos_sinteticos

class TestModeloLaderas(unittest.TestCase):
    def setUp(self):
        """Configuración inicial de pruebas."""
        self.modelo = ModeloLaderas(num_features=8)
        self.X_test, self.y_test = generar_datos_sinteticos(100)
    
    def test_generacion_datos(self):
        """Verificar generación correcta de datos."""
        self.assertEqual(self.X_test.shape, (100, 24, 8))
        self.assertEqual(len(self.y_test), 100)
    
    def test_prediccion_modelo(self):
        """Verificar que el modelo genera predicciones válidas."""
        predictions, attention = self.modelo(self.X_test[:1])
        self.assertEqual(predictions.shape, (1, 3))
        self.assertTrue(np.allclose(tf.reduce_sum(predictions, axis=1), 1.0))

if __name__ == '__main__':
    unittest.main()
```

#### **🔍 Integración Continua:**
```yaml
# .github/workflows/test.yml
name: Tests Modelo Laderas
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: python -m pytest tests/
```

### **4. Documentación: Código Autodocumentado**

#### **📚 Docstrings y Comentarios:**
```python
class AtencionPersonalizada(layers.Layer):
    """
    Capa de atención personalizada para identificar señales críticas.
    
    Esta capa implementa un mecanismo de atención que pondera la importancia
    de diferentes sensores y momentos temporales para detectar patrones
    de deslizamiento.
    
    Args:
        unidades (int): Número de unidades para la capa de atención.
        
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Vector de contexto y pesos de atención.
    """
    
    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Ejecuta el mecanismo de atención sobre los datos de entrada.
        
        Args:
            inputs (tf.Tensor): Datos de sensores con forma 
                (batch_size, time_steps, features).
                
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: 
                - context_vector: Vector contextual ponderado
                - attention_weights: Pesos de atención por sensor/tiempo
        """
        # Calcular puntuaciones de atención
        score = tf.nn.tanh(self.W(inputs))
        
        # Normalizar pesos con softmax
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # Aplicar pesos a entrada
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
```

#### **📖 README Completo:**
- Instalación y configuración
- Ejemplos de uso
- Arquitectura del modelo
- Métricas de rendimiento
- Contribución y licencia

---

**🏆 Resultado**: Código production-ready que cumple con estándares empresariales, es mantenible, testeable y documentado profesionalmente.

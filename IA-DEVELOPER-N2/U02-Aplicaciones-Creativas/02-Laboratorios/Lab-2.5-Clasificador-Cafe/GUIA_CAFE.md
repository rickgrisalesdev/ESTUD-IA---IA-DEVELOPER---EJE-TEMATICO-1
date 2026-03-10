# ☕ Guía de Manejo de Datasets Desbalanceados en Agricultura

## 🌾 **Contexto del Problema**

En el sector agrícola, especialmente en la clasificación de calidad de café, es común encontrarse con **datasets desbalanceados**. Esto ocurre naturalmente porque:

- **Menos muestras de alta calidad**: El café premium es escaso
- **Más muestras de calidad media**: La mayoría de producción es estándar
- **Variable cantidad de baja calidad**: Depende de condiciones climáticas y cosecha

Este desbalance puede hacer que los modelos de IA se sesguen hacia las clases mayoritarias, perdiendo capacidad para detectar productos de alta calidad.

## 🎯 **Impacto del Desbalance en Modelos Agrícolas**

### **Problemas Comunes:**
1. **Sesgo hacia clases mayoritarias**: El modelo aprende a predecir siempre "Media Calidad"
2. **Pérdida de sensibilidad**: No detecta café premium (crucial para exportación)
3. **Métricas engañosas**: Alta precisión general pero pobre en clases importantes
4. **Decisiones económicas erróneas**: Mal clasificación afecta precios de venta

### **Consecuencias Reales:**
- **Pérdidas económicas**: Café de alta calidad vendido como estándar
- **Oportunidades perdidas**: No identificar lotes premium para mercados especiales
- **Ineficiencia**: Recursos mal asignados en procesos de poscosecha

## 🔧 **Estrategias para Manejar Datasets Desbalanceados**

### **1. Técnicas de Muestreo**

#### **A. Oversampling (Sobre-muestreo)**
```python
# Aumentar muestras de clases minoritarias
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**Ventajas:**
- Mantiene toda la información original
- Balancea perfectamente las clases
- Funciona bien con datasets pequeños

**Desventajas:**
- Puede crear muestras sintéticas poco realistas
- Riesgo de overfitting en clases minoritarias

#### **B. Undersampling (Sub-muestreo)**
```python
# Reducir muestras de clases mayoritarias
from imblearn.under_sampling import RandomUnderSampler

undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
```

**Ventajas:**
- Reduce tiempo de entrenamiento
- Elimina datos ruidosos o redundantes
- Simple de implementar

**Desventajas:**
- Pierde información valiosa
- Puede eliminar muestras importantes

#### **C. Combinación Híbrida**
```python
# SMOTE + Tomek Links
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
```

### **2. Técnicas de Ponderación de Clases**

#### **A. Class Weights en Keras**
```python
# Calcular pesos automáticamente
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# En el entrenamiento
model.fit(X_train, y_train, class_weight=class_weight_dict)
```

#### **B. Pesos Personalizados para Agricultura**
```python
# Asignar importancia económica
class_weights = {
    0: 1.0,    # Baja calidad - peso normal
    1: 1.5,    # Media calidad - importancia moderada
    2: 3.0     # Alta calidad - máxima importancia (premium)
}
```

**Razonamiento:**
- El café de alta calidad tiene mayor valor económico
- Error en clasificación premium es más costoso
- Priorizar detección de productos especiales

### **3. Técnicas de Ensemble**

#### **A. Balanced Random Forest**
```python
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=42,
    sampling_strategy='auto'
)
```

#### **B. EasyEnsemble**
```python
from imblearn.ensemble import EasyEnsembleClassifier

eec = EasyEnsembleClassifier(
    n_estimators=10,
    random_state=42
)
```

### **4. Métricas Apropiadas para Datos Desbalanceados**

#### **A. Métricas de Clasificación**
```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

# Reporte completo por clase
print(classification_report(y_test, y_pred, target_names=class_names))

# F1-Score por clase (muy importante)
f1_per_class = f1_score(y_test, y_pred, average=None)

# AUC-ROC para cada clase (One-vs-Rest)
auc_per_class = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr')
```

#### **B. Métricas Específicas para Agricultura**
```python
# Sensibilidad para café premium (recall de clase alta)
premium_recall = recall_score(y_test, y_pred, labels=[2], average='micro')[0]

# Precisión económica (ponderada por valor)
economic_weights = [1.0, 2.0, 5.0]  # Baja, Media, Alta
economic_precision = np.average(precision_per_class, weights=economic_weights)
```

## 🚀 **Implementación Práctica en Café Expert**

### **Pipeline Completo:**
```python
class CafeDesbalanceadoHandler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        self.class_weights = None
        
    def preparar_datos(self, X, y):
        # 1. Normalizar características
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Aplicar SMOTE para balancear
        X_balanced, y_balanced = self.smote.fit_resample(X_scaled, y)
        
        # 3. Calcular pesos para entrenamiento original
        self.class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        
        return X_balanced, y_balanced
    
    def entrenar_con_pesos(self, model, X_original, y_original):
        # Entrenar con pesos en datos originales
        class_weight_dict = dict(enumerate(self.class_weights))
        
        history = model.fit(
            X_original, y_original,
            class_weight=class_weight_dict,
            validation_split=0.2,
            epochs=50,
            batch_size=32
        )
        
        return history
```

### **Evaluación Especializada:**
```python
def evaluar_modelo_agricola(model, X_test, y_test, class_names):
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Métricas estándar
    print("📊 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Análisis económico
    analizar_impacto_economico(cm, class_names)
    
    # Métricas por clase
    metricas_por_clase(cm, class_names)

def analizar_impacto_economico(cm, class_names):
    """Analiza el impacto económico de las clasificaciones."""
    # Precios por kg de café (USD)
    precios = {'Baja Calidad': 2.5, 'Media Calidad': 5.0, 'Alta Calidad': 12.0}
    
    # Calcular pérdidas/ganancias por clasificación incorrecta
    perdidas = 0
    ganancias_perdidas = 0
    
    for i, clase_real in enumerate(class_names):
        for j, clase_predicha in enumerate(class_names):
            if i != j:  # Clasificación incorrecta
                cantidad = cm[i, j]
                precio_real = precios[clase_real]
                precio_predicho = precios[clase_predicha]
                
                perdida_por_unidad = precio_real - precio_predicho
                if perdida_por_unidad > 0:
                    perdidas += cantidad * perdida_por_unidad
                else:
                    ganancias_perdidas += abs(cantidad * perdida_por_unidad)
    
    print(f"\n💰 Análisis Económico:")
    print(f"   Pérdidas por subclasificación: ${perdidas:,.2f}")
    print(f"   Ganancias no realizadas: ${ganancias_perdidas:,.2f}")
    print(f"   Impacto económico total: ${perdidas + ganancias_perdidas:,.2f}")
```

## 📈 **Resultados Esperados**

### **Mejoras en Clasificación:**
- **Recall Alta Calidad**: > 85% (detectar café premium)
- **Precisión Media**: > 80% (evitar falsos positivos)
- **F1-Score Balanceado**: > 75% general

### **Impacto Económico:**
- **Reducción de pérdidas**: 30-40% menos café premium mal clasificado
- **Mejora en ingresos**: 15-20% aumento en ventas de café especial
- **Optimización de procesos**: Mejor asignación de recursos de poscosecha

## 🔄 **Validación Cruzada para Datos Desbalanceados**

### **Stratified K-Fold:**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Aplicar técnicas de balanceamiento solo en entrenamiento
    X_train_balanced, y_train_balanced = aplicar_balanceamiento(X_train, y_train)
    
    # Entrenar y evaluar
    model.fit(X_train_balanced, y_train_balanced)
    scores.append(model.evaluate(X_val, y_val))
```

## 🎯 **Mejores Prácticas para Agricultura**

### **1. Entendimiento del Dominio:**
- Conocer el valor económico de cada clase
- Identificar qué errores son más costosos
- Considerar factores estacionales y regionales

### **2. Monitoreo Continuo:**
- Evaluar el rendimiento por separación geográfica
- Monitorear cambios estacionales en los datos
- Actualizar modelos con nuevas cosechas

### **3. Explicabilidad:**
- Justificar por qué una muestra es clasificada como premium
- Identificar características clave para cada calidad
- Proporcionar confianza en las predicciones

## 📚 **Referencias y Recursos**

1. **Imbalanced-learn Library**: Documentación oficial
2. **Agricultural AI Papers**: Casos de estudio en clasificación de cultivos
3. **Economic Impact Analysis**: Métodos para cuantificar valor del modelo
4. **SMOTE Algorithm**: Chawla et al., 2002

---

**🎯 Objetivo Principal**: Desarrollar sistemas de IA que no solo sean técnicamente precisos, sino también económicamente valiosos para el sector agrícola, especialmente en la detección de productos de alta calidad.

**💡 Innovación**: Combinar técnicas de balanceamiento de datos con análisis económico para crear modelos que maximicen el retorno de inversión para agricultores y exportadores.

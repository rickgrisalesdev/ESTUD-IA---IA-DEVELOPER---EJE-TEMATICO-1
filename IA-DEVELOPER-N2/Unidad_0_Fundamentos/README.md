# Unidad 0: Fundamentos y Buenas Prácticas del AI Developer

## 🎯 **Propósito de la Unidad**

Esta unidad consolida los cimientos fundamentales que todo AI Developer profesional debe dominar antes de escribir una sola línea de código de IA. Aquí aprenderás a pensar como un Ingeniero de Software especializado en Inteligencia Artificial.

---

## 📚 **Marco Teórico: Los 4 Pilares del Desarrollo Profesional**

### **1. PEP 8: El Manual de Estilo de Python**
**Documentación Oficial:** [PEP 8 -- Style Guide for Python Code](https://peps.python.org/pep-0008/)

**Justificación:** PEP 8 es la constitución del código Python limpio. No es opcional en empresas de élite.

#### **Reglas Esenciales:**
- **Sangría:** Siempre 4 espacios (nunca tabs)
- **Longitud de línea:** Máximo 79 caracteres
- **Nomenclatura:**
  - Variables/funciones: `snake_case`
  - Clases: `PascalCase`
  - Constantes: `UPPER_CASE`
- **Espacios:** `func(argumento)` no `func( argumento )`

### **2. Type Hinting: El Fin de las Adivinanzas**
**Documentación Oficial:** [Python Type Hinting](https://docs.python.org/3/library/typing.html)

**Justificación:** En IA, donde los datos pueden ser tensores, listas, o diccionarios, el Type Hinting evita errores catastróficos.

#### **Ejemplo Práctico:**
```python
# Sin Type Hinting (Deuda técnica)
def procesar_datos(datos):
    return datos * 2

# Con Type Hinting (Código Profesional)
def procesar_datos(datos: list[float]) -> list[float]:
    """Duplica cada elemento de una lista de flotantes."""
    return [d * 2 for d in datos]
```

### **3. SOLID Principles: Arquitectura Robusta**
**Justificación:** Estos 5 principios evitan que arreglar un error rompa tres cosas más.

| Sigla | Principio | Aplicación en IA |
|-------|------------|------------------|
| S | Single Responsibility | Un script para cargar datos, otro para entrenar |
| O | Open/Closed | Fácil agregar nuevos modelos sin modificar los existentes |
| L | Liskov Substitution | Un modelo personalizado debe poder reemplazar a uno estándar |
| I | Interface Segregation | Interfaces pequeñas para cada tipo de procesamiento |
| D | Dependency Inversion | Tu código no debe depender de una base de datos específica |

### **4. Docstrings: La Voz del Código**
**Documentación Oficial:** [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

**Justificación:** El código dice "cómo", el Docstring dice "qué" y "por qué".

#### **Estructura Profesional:**
```python
def entrenar_modelo(epochs: int, lr: float, datos: np.ndarray) -> dict:
    """
    Entrena un modelo de clasificación de imágenes.
    
    Args:
        epochs (int): Número de iteraciones completas sobre el dataset.
        lr (float): Tasa de aprendizaje (Learning Rate).
        datos (np.ndarray): Dataset de entrenamiento con forma (N, H, W, C).
    
    Returns:
        dict: Diccionario con métricas {'loss': float, 'accuracy': float}.
        
    Raises:
        ValueError: Si lr <= 0 o epochs <= 0.
    """
```

---

## 🛠️ **Estructura de Carpetas Profesional**

### **Planimetría Estándar para Proyectos de IA:**
```
proyecto_ia/
├── .env                    # Variables de entorno (API Keys, secrets)
├── .gitignore              # Archivos que Git debe ignorar
├── requirements.txt        # Dependencias exactas del proyecto
├── README.md              # Documentación principal
├── setup.py               # Configuración de instalación (opcional)
│
├── data/                  # Datos del proyecto
│   ├── raw/              # Datos crudos, sin procesar
│   ├── processed/         # Datos limpios y listos para usar
│   └── external/         # Datos de fuentes externas
│
├── src/                  # Código fuente principal
│   ├── __init__.py
│   ├── data/             # Módulos de procesamiento de datos
│   ├── models/           # Definiciones de modelos y arquitecturas
│   ├── training/         # Scripts de entrenamiento
│   ├── inference/        # Scripts de inferencia/predicción
│   └── utils/           # Funciones auxiliares (logging, métricas)
│
├── notebooks/            # Jupyter notebooks para experimentación
├── tests/                # Pruebas unitarias y de integración
├── scripts/              # Scripts de automatización y utilidades
├── configs/              # Archivos de configuración (YAML, JSON)
├── outputs/              # Resultados: modelos entrenados, gráficos
│   ├── models/
│   ├── logs/
│   └── figures/
└── docs/                 # Documentación técnica adicional
```

---

## 🐍 **Entornos Virtuales: Aislamiento Profesional**

### **¿Por qué son Indispensables?**
1. **Reproducibilidad:** El mismo código funciona en cualquier máquina
2. **Aislamiento:** Evita conflictos entre proyectos
3. **Control de Versiones:** Cada proyecto con sus dependencias específicas

### **Procedimiento Estándar:**
```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar (Windows)
.venv\Scripts\activate

# 3. Activar (Mac/Linux)
source .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Desactivar
deactivate
```

---

## 📋 **Checklist del AI Developer Profesional**

### **Antes de Escribir Código:**
- [ ] Entorno virtual creado y activado
- [ ] requirements.txt actualizado
- [ ] .gitignore configurado
- [ ] Estructura de carpetas definida

### **Durante el Desarrollo:**
- [ ] Type Hinting en todas las funciones
- [ ] Docstrings siguiendo estándar Google
- [ ] Nombres descriptivos (PEP 8)
- [ ] Principios SOLID aplicados

### **Antes del Commit:**
- [ ] Código formateado (Black/autopep8)
- [ ] Pruebas ejecutadas (pytest)
- [ ] Sin hardcoded values
- [ ] Variables sensibles en .env

---

## 🚀 **Laboratorios Prácticos**

### **Lab 0.1: Configuración del Entorno Profesional**
**Objetivo:** Crear tu primer proyecto siguiendo todas las buenas prácticas.

**Archivos a crear:**
1. `setup_project.py` - Script de automatización
2. `requirements_dev.txt` - Dependencias de desarrollo
3. `.gitignore` - Configuración de Git

### **Lab 0.2: Código Limpio y Modular**
**Objetivo:** Refactorizar código "espagueti" a código profesional.

**Ejercicios:**
1. Aplicar Type Hinting a funciones existentes
2. Convertir monolito a módulos (separar datos, modelos, entrenamiento)
3. Escribir Docstrings para todo el código

---

## 📖 **Material de Referencia**

### **Documentación Oficial Esencial:**
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Keras Documentation](https://keras.io/)
- [Python Type Hinting](https://docs.python.org/3/library/typing.html)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)

### **Herramientas Indispensables:**
- **Formateo:** Black, autopep8
- **Linting:** Flake8, Pylint
- **Testing:** pytest
- **Entornos:** venv, conda
- **Documentación:** Sphinx

---

## 🎯 **Evaluación de Competencias**

Al finalizar esta unidad, deberás ser capaz de:

1. **Explicar** por qué cada principio SOLID es crucial en IA
2. **Crear** un entorno virtual y gestionar dependencias
3. **Escribir** código Python siguiendo PEP 8 perfectamente
4. **Diseñar** una arquitectura modular para un proyecto de IA
5. **Documentar** código con Docstrings de nivel profesional
6. **Evitar** la deuda técnica desde el primer día

---

## ⚡ **Próximos Pasos**

Una vez dominados estos fundamentos, estarás listo para:
- **Unidad 1:** Introducción a TensorFlow/Keras
- **Unidad 2:** Programación Avanzada con Modelos Personalizados
- **Unidad 3:** Automatización de Flujos de Trabajo

**Recuerda:** Un buen código no solo funciona, es comprensible, mantenible y escalable.

# 📁 Estructura del Repositorio - Explicación Detallada

## 🐍 **¿Qué es `site-packages`?**

### **📍 Definición Oficial:**
`site-packages` es una carpeta **automática** del sistema Python que contiene **todos los paquetes de terceros** instalados.

### **🎯 Función Principal:**
```
site-packages/
├── tensorflow/           # Librería TensorFlow
├── numpy/               # Librería NumPy  
├── pandas/              # Librería Pandas
├── matplotlib/          # Librería Matplotlib
├── fastapi/             # Librería FastAPI
└── ...                  # Otras dependencias
```

### **🔍 Características Clave:**
- ✅ **Se genera automáticamente** al ejecutar `pip install`
- ✅ **No está en el repositorio** (excluida por `.gitignore`)
- ✅ **Es específica del entorno** (cada entorno virtual tiene la suya)
- ✅ **Se reconstruye fácilmente** con `requirements.txt`

### **📍 Ubicación Típica:**
```bash
# Windows (entorno virtual)
.venv/Lib/site-packages/

# Linux/Mac (entorno virtual)  
.venv/lib/python3.x/site-packages/

# Windows (sistema)
C:/Python39/Lib/site-packages/

# Linux/Mac (sistema)
/usr/lib/python3.9/site-packages/
```

### **⚠️ Por qué NO la ves en el repositorio:**
1. **`.gitignore` la excluye** - No se sube al control de versiones
2. **No es código fuente** - Son dependencias externas
3. **Ocupa mucho espacio** - Sería ineficiente subirla
4. **Se genera sola** - No necesita estar en el repo

---

## 🎓 **¿Dónde está la "Unidad 1"?**

### **📋 Estructura Actual del Bootcamp:**

```
Lab-2.3-Redes-Avanzadas/
├── 📖 Unidad_0_Fundamentos/          # ✅ Fundamentos y Buenas Prácticas
├── 🏗️ proyecto1_logistica/           # ✅ Unidad 1.1: Logística (CNN)
├── 🏥 proyecto2_salud/               # ✅ Unidad 1.2: Salud (Transformer)  
├── 🛍️ proyecto3_retail/              # ✅ Unidad 1.3: Retail (Híbrido)
├── 🔄 scripts/                       # ✅ Unidad 2: Automatización
└── 📚 README_*.md                    # ✅ Documentación
```

### **🎯 Explicación:**
**No existe una carpeta "Unidad_1"** porque los **3 proyectos integradores SON la Unidad 1**:

- **🏗️ `proyecto1_logistica/`** = **Unidad 1.1** - CNN para Logística
- **🏥 `proyecto2_salud/`** = **Unidad 1.2** - Transformer para Salud  
- **🛍️ `proyecto3_retail/`** = **Unidad 1.3** - Red Híbrida para Retail

### **📚 Estructura Pedagógica Real:**

```
🎓 BOOTCAMP IA DEVELOPER
├── 📖 Unidad 0: Fundamentos Profesionales
│   ├── 📄 README.md (teoría PEP 8, Type Hinting, SOLID)
│   ├── 🛠️ setup_project.py (automatización)
│   ├── 🔧 lab_0_2_codigo_limpio.py (refactorización)
│   └── 📋 requirements.txt (herramientas)
│
├── 📖 Unidad 1: Aplicaciones Prácticas de IA
│   ├── 🏗️ 1.1: Logística - Detección de Daños (CNN)
│   │   └── 📁 proyecto1_logistica/scripts/train_cnn.py
│   ├── 🏥 1.2: Salud - Análisis Médico (Transformer)
│   │   └── 📁 proyecto2_salud/scripts/train_transformer_refactored.py
│   └── 🛍️ 1.3: Retail - Recomendación (Red Híbrida)
│       └── 📁 proyecto3_retail/scripts/train_hybrid_refactored.py
│
└── 📖 Unidad 2: Automatización y Deploy
    ├── 🔄 pipeline_automatizado.py (orquestación)
    ├── 🌐 api_server.py (API FastAPI)
    └── 📦 requirements_unificado.txt (dependencias)
```

---

## 🔍 **Verificación de Estructura**

### **✅ Carpetas que SÍ existen:**
- `Unidad_0_Fundamentos/` - Fundamentos y buenas prácticas
- `proyecto1_logistica/` - Aplicación CNN (Unidad 1.1)
- `proyecto2_salud/` - Aplicación Transformer (Unidad 1.2)
- `proyecto3_retail/` - Aplicación Híbrida (Unidad 1.3)
- `scripts/` - Automatización y API (Unidad 2)

### **❌ Carpetas que NO existen (y por qué):**
- `site-packages/` - Se genera automáticamente, no va en el repo
- `Unidad_1/` - Los proyectos YA SON la Unidad 1
- `__pycache__/` - Excluida por `.gitignore`
- `.venv/` - Excluida por `.gitignore`

---

## 🚀 **Cómo Funciona en Práctica**

### **📦 Instalación de Dependencias:**
```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar entorno
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Instalar dependencias (crea site-packages)
pip install -r requirements_unificado.txt

# 4. site-packages se crea automáticamente con:
#    - tensorflow/
#    - numpy/
#    - pandas/
#    - fastapi/
#    - etc.
```

### **🎯 Flujo de Trabajo:**
```
1. Clonas el repo (sin site-packages)
2. Creas entorno virtual  
3. Ejecutas pip install (crea site-packages)
4. Ejecutas los proyectos (usan site-packages)
5. Commiteas cambios (site-packages se ignora)
```

---

## 📋 **Resumen Claro**

| Concepto | ¿Existe? | ¿Por qué? |
|----------|----------|-----------|
| `site-packages/` | ❌ No en repo | Se genera con `pip install` |
| `Unidad_0_Fundamentos/` | ✅ Sí | Fundamentos del bootcamp |
| `Unidad_1/` | ❌ No existe | Los proyectos YA SON Unidad 1 |
| `proyecto1_logistica/` | ✅ Sí | Es Unidad 1.1 |
| `proyecto2_salud/` | ✅ Sí | Es Unidad 1.2 |
| `proyecto3_retail/` | ✅ Sí | Es Unidad 1.3 |
| `scripts/` | ✅ Sí | Es Unidad 2 |

---

## 🎯 **Conclusión**

1. **`site-packages`** es una carpeta **automática** de Python con las librerías instaladas, **no pertenece al repositorio**.

2. **"Unidad 1"** no existe como carpeta porque **los 3 proyectos integradores SON la Unidad 1**, organizados por aplicación práctica.

3. La estructura está **diseñada profesionalmente** para ser modular, escalable y seguir las mejores prácticas de desarrollo de software.

**🚀 Tu bootcamp está completo y listo para usarse profesionalmente.**

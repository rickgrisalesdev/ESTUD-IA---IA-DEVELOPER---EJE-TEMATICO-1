# 🚀 Bootcamp IA Developer - Repositorio Unificado

## 📋 **Estructura del Repositorio**

Este repositorio ha sido reestructurado siguiendo las **mejores prácticas de desarrollo de software** y la **metodología de marco lógico** para garantizar un aprendizaje profesional sin deuda técnica.

```
Lab-2.3-Redes-Avanzadas/
├── 📁 Unidad_0_Fundamentos/          # Buenas prácticas y cimientos
├── 📁 proyecto1_logistica/           # Proyecto 1: Detección de daños
├── 📁 proyecto2_salud/               # Proyecto 2: Diagnóstico médico
├── 📁 proyecto3_retail/              # Proyecto 3: Recomendación retail
├── 📄 README_UNIFICADO.md            # Este archivo
├── 📄 requirements_unificado.txt      # Dependencias unificadas
└── 📄 README.md                     # Documentación original
```

---

## 🎯 **Unidad 0: Fundamentos y Buenas Prácticas**

### **📁 Contenido:**
- `README.md` - Marco teórico completo
- `requirements.txt` - Herramientas de desarrollo
- `setup_project.py` - Script automatizador de proyectos
- `lab_0_2_codigo_limpio.py` - Ejercicio de refactorización
- `.gitignore` - Configuración estándar para Git

### **🎓 Objetivos de Aprendizaje:**
1. **PEP 8** - Estilo de código Python profesional
2. **Type Hinting** - Tipado estático para evitar errores
3. **SOLID Principles** - Arquitectura robusta y escalable
4. **Docstrings** - Documentación técnica estándar
5. **Entornos Virtuales** - Aislamiento y reproducibilidad

### **🛠️ Laboratorios Prácticos:**

#### **Lab 0.1: Configuración Profesional**
```bash
# Ejecutar el script automatizador
python Unidad_0_Fundamentos/setup_project.py
```
- Crea estructura de carpetas profesional
- Configura entorno virtual automáticamente
- Genera archivos base (README, requirements, etc.)

#### **Lab 0.2: Refactorización a Código Limpio**
```bash
# Comparar código espagueti vs profesional
python Unidad_0_Fundamentos/lab_0_2_codigo_limpio.py
```
- Demuestra transformación de código malo a bueno
- Aplica todos los principios de diseño
- Incluye logging, manejo de errores y testing

---

## 📦 **Dependencias Unificadas**

### **🎯 Características:**
- **Sin conflictos de versiones** entre proyectos
- **Compatible con Python 3.9-3.12**
- **Optimizado para producción**
- **Modular** (comentar/descomentar según necesidades)

### **🔧 Instalación:**
```bash
# Instalación básica
pip install -r requirements_unificado.txt

# Para desarrollo completo
pip install -r Unidad_0_Fundamentos/requirements.txt
```

### **📊 Categorías de Dependencias:**
1. **Core ML/DL** - TensorFlow, Keras, PyTorch, Scikit-learn
2. **Data Processing** - NumPy, Pandas, SciPy
3. **Computer Vision** - OpenCV, Pillow, Scikit-image
4. **NLP** - Transformers, Tokenizers, NLTK
5. **API/Deploy** - FastAPI, Uvicorn, Pydantic
6. **Development** - pytest, black, flake8

---

## 🏗️ **Estructura de Proyectos Integradores**

### **📦 Proyecto 1: Logística (Detección de Daños)**
- **Redes:** CNN, RNN, GANs, CapsNet
- **Tecnologías:** OpenCV, TensorFlow, Keras
- **Caso de Uso:** Automatización en centros de distribución

### **🏥 Proyecto 2: Salud (Diagnóstico Médico)**
- **Redes:** CNN, Transformers, Redes de Grafos
- **Tecnologías:** Medical Imaging, NLP
- **Caso de Uso:** Triaje y diagnóstico asistido

### **🛍️ Proyecto 3: Retail (Recomendación)**
- **Redes:** Híbridas (CNN+RNN), Memory Networks, Neuro-Simbólicas
- **Tecnologías:** Sistemas de recomendación, Procesamiento multimodal
- **Caso de Uso:** Personalización de experiencia de compra

---

## 🔄 **Flujo de Trabajo Profesional**

### **📋 Ciclo de Vida del Desarrollo:**

1. **🔧 Configuración del Entorno**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

2. **📦 Instalación de Dependencias**
   ```bash
   pip install -r requirements_unificado.txt
   ```

3. **🧪 Desarrollo y Testing**
   ```bash
   # Formateo de código
   black src/ tests/
   
   # Linting
   flake8 src/ tests/
   
   # Testing
   pytest tests/ -v --cov=src
   ```

4. **🚀 Ejecución de Proyectos**
   ```bash
   # Proyecto 1
   cd proyecto1_logistica/scripts
   python train_cnn.py
   
   # Proyecto 2
   cd proyecto2_salud/scripts
   python train_transformer.py
   
   # Proyecto 3
   cd proyecto3_retail/scripts
   python train_hybrid.py
   ```

---

## 📚 **Marco Teórico por Unidad**

### **🏛️ Unidad 0: Fundamentos**
- **PEP 8:** Guía de estilo oficial de Python
- **Type Hinting:** `def func(x: int) -> str:`
- **SOLID:** 5 principios de diseño robusto
- **Docstrings:** Estándar Google para documentación

### **🧠 Unidad 1: TensorFlow Intermedio**
- **API Funcional:** Modelos complejos y personalizados
- **Capas Custom:** Crear componentes propios
- **Métricas:** Accuracy, Precision, Recall, F1-Score

### **⚡ Unidad 2: Programación Avanzada**
- **12 Tipos de Redes:** CNN, RNN, Transformers, GANs, etc.
- **Optimización:** Hiperparámetros y regularización
- **Datasets Complejos:** Manejo de datos a gran escala

### **🤖 Unidad 3: Automatización de Flujos**
- **Pipelines:** ETL automatizado con tf.data
- **Scripts Python:** Orquestación de procesos
- **Integración APIs:** Conexión con servicios externos

---

## 🎯 **Competencias del AI Developer**

### **🔧 Técnicas (Hard Skills):**
1. **Arquitectura de Software** - Diseño de sistemas escalables
2. **Machine Learning Engineering** - MLOps y producción
3. **Optimización de Modelos** - Performance y eficiencia
4. **Integración de Sistemas** - APIs y microservicios

### **💡 Metodológicas (Soft Skills):**
1. **Pensamiento Crítico** - Análisis de problemas complejos
2. **Comunicación Técnica** - Documentación y presentaciones
3. **Trabajo en Equipo** - Colaboración con Git y metodologías ágiles
4. **Aprendizaje Continuo** - Adaptación a nuevas tecnologías

---

## 🚀 **Próximos Pasos**

### **📈 Ruta de Aprendizaje Sugerida:**

1. **Dominar Unidad 0** - Fundamentos sólidos
2. **Proyecto 1 (Logística)** - Aplicar CNN y visión artificial
3. **Proyecto 2 (Salud)** - Integrar Transformers y NLP
4. **Proyecto 3 (Retail)** - Construir sistemas híbridos
5. **Integración Final** - Unir todos los proyectos

### **🎯 Metas de Carrera:**
- **Junior Developer:** Dominar Unidad 0 + Proyecto 1
- **Mid-Level Developer:** Completar Proyectos 1 y 2
- **Senior AI Developer:** Dominar todos los proyectos + Unidad 3

---

## 📖 **Material de Referencia**

### **📚 Documentación Oficial:**
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Keras Documentation](https://keras.io/)
- [Python Type Hinting](https://docs.python.org/3/library/typing.html)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)

### **🛠️ Herramientas Esenciales:**
- **IDE:** VS Code, PyCharm, Windsurf
- **Version Control:** Git + GitHub
- **Testing:** pytest, coverage
- **CI/CD:** GitHub Actions, GitLab CI

---

## 🏆 **Evaluación y Certificación**

### **📋 Criterios de Evaluación:**
- [ ] Código sigue PEP 8 perfectamente
- [ ] Type Hinting en todas las funciones
- [ ] Principios SOLID aplicados correctamente
- [ ] Documentación completa con Docstrings
- [ ] Tests unitarios con >80% cobertura
- [ ] Proyectos funcionales y desplegables

### **🎖️ Certificación Final:**
Completar todos los proyectos con:
- **Código production-ready**
- **Documentación técnica completa**
- **Deploy funcional**
- **Presentación de resultados**

---

## 👥 **Comunidad y Soporte**

### **💬 Canales de Comunicación:**
- **GitHub Issues:** Reporte de bugs y preguntas técnicas
- **Discussions:** Dudas generales y colaboración
- **Wiki:** Documentación adicional y tutoriales

### **🤝 Contribución:**
1. **Fork** del repositorio
2. **Branch** con nombre descriptivo
3. **Commit** siguiendo convenciones
4. **Pull Request** con descripción detallada

---

## 📄 Licencia

MIT License - Ver archivo LICENSE para detalles

---

**🚀 ¡Bienvenido al Bootcamp IA Developer!**

*Este repositorio está diseñado para llevarte de principiante a profesional en Inteligencia Artificial, aplicando las mejores prácticas de la industria desde el primer día.*

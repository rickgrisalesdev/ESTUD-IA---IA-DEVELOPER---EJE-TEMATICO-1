# 💻 Lab 1.2: Fundamentos de Programación con Gemini

## 🎯 **Objetivo del Laboratorio**

Consolidar los **fundamentos de programación** basados en los principios y proyectos de **Gemini AI**, aplicando las **buenas prácticas** aprendidas en la Unidad 0.

---

## 📋 **Contenido del Laboratorio**

### **🎯 Objetivos de Aprendizaje:**
- **Estructuras de Datos**: Listas, diccionarios, tuplas, sets
- **Algoritmos Básicos**: Búsqueda, ordenamiento, recursión
- **Programación Orientada a Objetos**: Clases, herencia, polimorfismo
- **Patrones de Diseño**: Singleton, Factory, Observer
- **Buenas Prácticas**: PEP 8, Type Hinting, SOLID, Docstrings

### **🔧 Tecnologías Utilizadas:**
- **Python 3.9+**: Lenguaje principal
- **Type Hinting**: Tipado estático
- **pytest**: Testing unitario
- **Black**: Formateo de código
- **Jupyter**: Experimentación interactiva

---

## 🏗️ **Estructura del Laboratorio**

```
Lab-1.2-Programacion-Basica/
├── 📁 exercises/                     # Ejercicios prácticos
│   ├── 📁 01_data_structures/        # Estructuras de datos
│   │   ├── 📄 lists_and_tuples.py     # Listas y tuplas
│   │   ├── 📄 dictionaries.py        # Diccionarios
│   │   └── 📄 sets_and_booleans.py    # Sets y booleanos
│   ├── 📁 02_algorithms/             # Algoritmos básicos
│   │   ├── 📄 searching.py            # Algoritmos de búsqueda
│   │   ├── 📄 sorting.py              # Algoritmos de ordenamiento
│   │   └── 📄 recursion.py            # Recursión
│   ├── 📁 03_oop_concepts/           # Programación orientada a objetos
│   │   ├── 📄 classes_and_objects.py  # Clases y objetos
│   │   ├── 📄 inheritance.py          # Herencia
│   │   └── 📄 polymorphism.py         # Polimorfismo
│   └── 📁 04_design_patterns/        # Patrones de diseño
│       ├── 📄 singleton.py             # Patrón Singleton
│       ├── 📄 factory.py               # Patrón Factory
│       └── 📄 observer.py              # Patrón Observer
│
├── 📁 projects/                      # Proyectos de Gemini
│   ├── 📁 gemini_calculator/         # Calculadora básica
│   ├── 📁 gemini_todo_list/          # Lista de tareas
│   ├── 📁 gemini_weather_app/        # App del clima
│   └── 📁 gemini_chatbot/            # Chatbot básico
│
├── 📁 solutions/                      # Soluciones completas
│   ├── 📄 exercise_solutions.py       # Soluciones de ejercicios
│   └── 📄 project_solutions/        # Soluciones de proyectos
│
├── 📁 notebooks/                     # Jupyter notebooks
│   ├── 📄 01_python_basics.ipynb     # Fundamentos de Python
│   ├── 📄 02_data_structures.ipynb   # Estructuras de datos
│   ├── 📄 03_algorithms.ipynb        # Algoritmos
│   └── 📄 04_oop_concepts.ipynb       # POO
│
├── 📁 tests/                         # Pruebas unitarias
│   ├── 📄 test_exercises.py           # Tests de ejercicios
│   └── 📄 test_projects.py            # Tests de proyectos
│
└── 📄 requirements.txt                # Dependencias
```

---

## 📚 **Temas Fundamentales**

### **🔤 1. Python Básico**
- **Variables y Tipos**: Números, strings, booleanos
- **Operadores**: Aritméticos, lógicos, de comparación
- **Control de Flujo**: if/else, for, while
- **Funciones**: Definición, parámetros, retorno
- **Módulos**: Importación, creación de módulos

### **📊 2. Estructuras de Datos**
- **Listas**: Creación, manipulación, métodos
- **Tuplas**: Inmutabilidad, unpacking
- **Diccionarios**: Claves, valores, métodos
- **Sets**: Eliminación de duplicados, operaciones

### **⚡ 3. Algoritmos Básicos**
- **Búsqueda**: Lineal, binaria
- **Ordenamiento**: Bubble, selection, insertion, quick, merge
- **Recursión**: Conceptos, casos base, ejemplos
- **Complejidad**: Notación Big O

### **🏗️ 4. Programación Orientada a Objetos**
- **Clases y Objetos**: Definición, instanciación
- **Atributos y Métodos**: Instancia, clase, estáticos
- **Herencia**: Simple, múltiple, super()
- **Polimorfismo**: Duck typing, interfaces
- **Encapsulación**: Privacidad, properties

### **🎨 5. Patrones de Diseño**
- **Creacionales**: Singleton, Factory, Builder
- **Estructurales**: Adapter, Decorator, Facade
- **Comportamiento**: Observer, Strategy, Command

---

## 🚀 **Flujo de Trabajo**

### **📋 Paso 1: Configuración del Entorno**
```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### **📋 Paso 2: Ejercicios Fundamentales**
```bash
# Ejecutar ejercicios de estructuras de datos
python exercises/01_data_structures/lists_and_tuples.py

# Ejecutar ejercicios de algoritmos
python exercises/02_algorithms/searching.py

# Ejecutar tests
pytest tests/test_exercises.py -v
```

### **📋 Paso 3: Proyectos Prácticos**
```bash
# Ejecutar calculadora
python projects/gemini_calculator/main.py

# Ejecutar lista de tareas
python projects/gemini_todo_list/main.py

# Ejecutar chatbot
python projects/gemini_chatbot/main.py
```

### **📋 Paso 4: Experimentación con Notebooks**
```bash
# Iniciar Jupyter
jupyter notebook

# Abrir notebooks de aprendizaje
# notebooks/01_python_basics.ipynb
# notebooks/02_data_structures.ipynb
# etc.
```

---

## 💻 **Ejercicios Prácticos**

### **🔤 Nivel 1: Python Básico**
```python
# Ejercicio: Calculadora de IMC
def calcular_imc(peso: float, altura: float) -> float:
    """Calcular Índice de Masa Corporal."""
    return peso / (altura ** 2)

# Ejercicio: Validador de email
def es_email_valido(email: str) -> bool:
    """Validar formato de email."""
    return '@' in email and '.' in email.split('@')[-1]
```

### **📊 Nivel 2: Estructuras de Datos**
```python
# Ejercicio: Gestor de inventario
class GestorInventario:
    def __init__(self) -> None:
        self.inventario: Dict[str, int] = {}
    
    def agregar_producto(self, nombre: str, cantidad: int) -> None:
        """Agregar producto al inventario."""
        self.inventario[nombre] = self.inventario.get(nombre, 0) + cantidad
    
    def obtener_stock(self, nombre: str) -> int:
        """Obtener stock de un producto."""
        return self.inventario.get(nombre, 0)
```

### **⚡ Nivel 3: Algoritmos**
```python
# Ejercicio: Búsqueda binaria
def busqueda_binaria(arr: List[int], objetivo: int) -> int:
    """Búsqueda binaria recursiva."""
    if not arr:
        return -1
    
    medio = len(arr) // 2
    
    if arr[medio] == objetivo:
        return medio
    elif arr[medio] > objetivo:
        return busqueda_binaria(arr[:medio], objetivo)
    else:
        resultado = busqueda_binaria(arr[medio + 1:], objetivo)
        return resultado + medio + 1 if resultado != -1 else -1
```

---

## 🏗️ **Proyectos de Gemini**

### **🧮 Proyecto 1: Calculadora Científica**
- **Funcionalidades**: Operaciones básicas, funciones trigonométricas
- **Características**: Historial, validación de entrada, modo gráfico
- **Habilidades**: POO, manejo de excepciones, GUI

### **📝 Proyecto 2: Lista de Tareas (Todo List)**
- **Funcionalidades**: CRUD de tareas, prioridades, fechas
- **Características**: Persistencia en JSON, filtrado, búsqueda
- **Habilidades**: Estructuras de datos, file I/O, validación

### **🌤️ Proyecto 3: App del Clima**
- **Funcionalidades**: Consulta del clima por ciudad, pronóstico
- **Características**: API integration, caching, GUI
- **Habilidades**: Requests, JSON parsing, manejo de errores

### **🤖 Proyecto 4: Chatbot Básico**
- **Funcionalidades**: Conversación simple, respuestas predefinidas
- **Características**: NLU básico, aprendizaje, persistencia
- **Habilidades**: NLP, patrones de diseño, state management

---

## 🧪 **Testing y Validación**

### **📋 Testing Unitario**
```python
# Ejemplo de test
import pytest
from exercises.lists_and_tuples import encontrar_maximo

def test_encontrar_maximo():
    """Test de función encontrar_maximo."""
    assert encontrar_maximo([1, 2, 3, 4, 5]) == 5
    assert encontrar_maximo([-1, -5, -3]) == -1
    assert encontrar_maximo([10]) == 10
```

### **📊 Métricas de Calidad**
- **Coverage**: > 80% de código cubierto
- **Complejidad**: Ciclomática < 10 por función
- **Maintainability**: Índice > 70
- **Duplicación**: < 3% de código duplicado

---

## 🎯 **Buenas Prácticas Aplicadas**

### **🐍 PEP 8 Compliance**
- **Nomenclatura**: `snake_case` para variables/funciones
- **Indentación**: 4 espacios consistentes
- **Longitud**: Máximo 79 caracteres por línea
- **Imports**: Ordenados y agrupados

### **📝 Type Hinting Completo**
```python
from typing import List, Dict, Optional, Union

def procesar_datos(
    datos: List[Dict[str, Union[str, int]]],
    filtro: Optional[str] = None
) -> Dict[str, List[str]]:
    """Procesar lista de diccionarios con filtro opcional."""
    pass
```

### **🏗️ Principios SOLID**
- **S**: Cada clase una responsabilidad
- **O**: Abierto para extensión, cerrado para modificación
- **L**: Subtipos reemplazables
- **I**: Interfaces específicas
- **D**: Dependencias invertidas

### **📖 Docstrings Estándar**
```python
def calcular_factorial(n: int) -> int:
    """
    Calcular el factorial de un número.
    
    Args:
        n (int): Número entero no negativo.
        
    Returns:
        int: Factorial de n.
        
    Raises:
        ValueError: Si n es negativo.
    """
    if n < 0:
        raise ValueError("El factorial no está definido para números negativos")
    return 1 if n == 0 else n * calcular_factorial(n - 1)
```

---

## 📈 **Progresión de Aprendizaje**

### **🥉 Nivel Básico (Semanas 1-2)**
- Python básico y estructuras de datos
- Ejercicios simples y proyectos pequeños
- Introducción al testing

### **🥈 Nivel Intermedio (Semanas 3-4)**
- Algoritmos y complejidad
- POO y patrones de diseño
- Proyectos más complejos

### **🥇 Nivel Avanzado (Semanas 5-6)**
- Optimización y rendimiento
- Patrones avanzados
- Proyecto integrador final

---

## 🚀 **Evaluación del Laboratorio**

### **📋 Criterios de Éxito:**
- [ ] **Ejercicios Completados**: Todos los ejercicios funcionales
- [ ] **Proyectos Terminados**: 4 proyectos de Gemini completos
- [ ] **Testing**: > 80% de coverage
- [ ] **Código Limpio**: PEP 8, Type Hinting, SOLID
- [ ] **Documentación**: Docstrings completos y READMEs

### **🏆 Niveles de Logro:**
- **🥉 Básico**: Ejercicios y 2 proyectos
- **🥈 Intermedio**: Todos los ejercicios y 3 proyectos
- **🥇 Avanzado**: Todo + proyecto integrador personal

---

## 📚 **Recursos Adicionales**

### **📖 Documentación:**
- [Python Documentation](https://docs.python.org/3/)
- [Real Python](https://realpython.com/)
- [Python Type Hinting](https://docs.python.org/3/library/typing.html)

### **🎥 Tutoriales:**
- [Python for Everybody](https://www.py4e.com/)
- [Automate the Boring Stuff](https://automatetheboringstuff.com/)
- [Design Patterns in Python](https://refactoring.guru/design-patterns/python)

### **📚 Libros Recomendados:**
- "Python Crash Course" - Eric Matthes
- "Fluent Python" - Luciano Ramalho
- "Design Patterns" - Gang of Four
- "Clean Code" - Robert C. Martin

---

## 🎯 **Consejos de Éxito**

### **💡 Para los Ejercicios:**
- Comienza con los problemas más simples
- Escribe código limpio desde el principio
- Testea cada función individualmente
- Revisa las soluciones después de intentar

### **💡 Para los Proyectos:**
- Planifica antes de codificar
- Divide los problemas grandes en pequeños
- Aplica los principios de la Unidad 0
- Documenta tus decisiones

### **💡 Para el Aprendizaje:**
- Practica todos los días
- Explica conceptos a otros
- Construye proyectos personales
- Participa en comunidades

---

**🚀 ¡Domina los fundamentos de la programación con Gemini!**

*Este laboratorio es tu base sólida para convertirte en un desarrollador Python profesional.*

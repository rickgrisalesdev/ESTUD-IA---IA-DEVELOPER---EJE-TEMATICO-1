#!/usr/bin/env python3
"""
Unidad 0 - Laboratorio 0.1: Configuración del Entorno Profesional
Script para automatizar la creación de un proyecto de IA con buenas prácticas.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


class ProjectSetup:
    """
    Clase para configurar un proyecto de IA con estructura profesional.
    
    Aplica principios de ingeniería de software para evitar deuda técnica.
    """
    
    def __init__(self, project_name: str, author_name: str = "AI Developer"):
        """
        Inicializa la configuración del proyecto.
        
        Args:
            project_name (str): Nombre del proyecto a crear.
            author_name (str): Nombre del autor/desarrollador.
        """
        self.project_name = project_name
        self.author_name = author_name
        self.base_path = Path.cwd() / project_name
        
        # Estructura de directorios estándar
        self.directories = [
            "data/raw",
            "data/processed", 
            "data/external",
            "src/data",
            "src/models",
            "src/training",
            "src/inference", 
            "src/utils",
            "notebooks",
            "tests",
            "scripts",
            "configs",
            "outputs/models",
            "outputs/logs",
            "outputs/figures",
            "docs"
        ]
        
        # Archivos de configuración
        self.config_files = {
            ".gitignore": self._get_gitignore_content(),
            ".env.example": self._get_env_example(),
            "README.md": self._get_readme_content(),
            "requirements.txt": self._get_requirements_content(),
            "setup.py": self._get_setup_content()
        }
    
    def create_project_structure(self) -> bool:
        """
        Crea la estructura completa de directorios del proyecto.
        
        Returns:
            bool: True si la creación fue exitosa, False en caso contrario.
        """
        try:
            print(f"🏗️ Creando estructura para proyecto: {self.project_name}")
            
            # Crear directorio base
            self.base_path.mkdir(exist_ok=True)
            
            # Crear subdirectorios
            for directory in self.directories:
                dir_path = self.base_path / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Directorio creado: {directory}")
            
            # Crear archivos de configuración
            for filename, content in self.config_files.items():
                file_path = self.base_path / filename
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  📄 Archivo creado: {filename}")
            
            # Crear __init__.py files para paquetes Python
            init_files = [
                "src/__init__.py",
                "src/data/__init__.py", 
                "src/models/__init__.py",
                "src/training/__init__.py",
                "src/inference/__init__.py",
                "src/utils/__init__.py"
            ]
            
            for init_file in init_files:
                file_path = self.base_path / init_file
                file_path.touch()
                print(f"  🐍 __init__.py creado: {init_file}")
            
            print(f"\n✅ Proyecto '{self.project_name}' creado exitosamente!")
            return True
            
        except Exception as e:
            print(f"❌ Error creando el proyecto: {str(e)}")
            return False
    
    def setup_virtual_environment(self) -> bool:
        """
        Crea y activa un entorno virtual para el proyecto.
        
        Returns:
            bool: True si la configuración fue exitosa.
        """
        try:
            print(f"\n🐍 Configurando entorno virtual para {self.project_name}")
            
            # Crear entorno virtual
            venv_path = self.base_path / ".venv"
            result = subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Error creando entorno virtual: {result.stderr}")
                return False
            
            print("  ✅ Entorno virtual creado")
            
            # Instalar dependencias básicas
            if os.name == "nt":  # Windows
                pip_path = venv_path / "Scripts" / "pip"
                activate_cmd = f".venv\\Scripts\\activate"
            else:  # Unix
                pip_path = venv_path / "bin" / "pip"
                activate_cmd = "source .venv/bin/activate"
            
            print(f"  📦 Instalando dependencias...")
            install_result = subprocess.run([
                str(pip_path), "install", "-r", 
                str(self.base_path / "requirements.txt")
            ], capture_output=True, text=True)
            
            if install_result.returncode != 0:
                print(f"⚠️ Advertencia instalando dependencias: {install_result.stderr}")
            else:
                print("  ✅ Dependencias instaladas")
            
            print(f"\n🚀 Para activar el entorno, ejecuta:")
            print(f"   cd {self.project_name}")
            print(f"   {activate_cmd}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error configurando entorno virtual: {str(e)}")
            return False
    
    def _get_gitignore_content(self) -> str:
        """Genera el contenido del archivo .gitignore."""
        return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Entorno Virtual
.venv/
venv/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Datos y Modelos
data/raw/*
data/processed/*
outputs/models/*
outputs/logs/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!outputs/models/.gitkeep
!outputs/logs/.gitkeep

# Variables de Entorno
.env
.env.local
.env.production

# Jupyter
.ipynb_checkpoints

# Testing
.coverage
.pytest_cache/
htmlcov/

# Documentación
docs/_build/

# Sistema Operativo
.DS_Store
Thumbs.db
"""
    
    def _get_env_example(self) -> str:
        """Genera el contenido del archivo .env.example."""
        return """# Variables de Entorno - Copiar a .env
# =====================================

# API Keys (Ejemplos)
OPENAI_API_KEY=tu_api_key_aqui
HUGGINGFACE_API_KEY=tu_huggingface_key_aqui

# Configuración de Base de Datos
DATABASE_URL=postgresql://user:password@localhost/dbname

# Configuración del Modelo
MODEL_PATH=outputs/models/
LOG_LEVEL=INFO

# Configuración de Servidor
HOST=localhost
PORT=8000
"""
    
    def _get_readme_content(self) -> str:
        """Genera el contenido del README.md."""
        return f"""# {self.project_name}

## 🎯 Descripción

Breve descripción del proyecto y su propósito.

## 🏗️ Estructura del Proyecto

```
{self.project_name}/
├── data/                  # Datos del proyecto
│   ├── raw/              # Datos crudos
│   ├── processed/         # Datos procesados
│   └── external/         # Datos externos
├── src/                  # Código fuente
│   ├── data/             # Procesamiento de datos
│   ├── models/           # Definiciones de modelos
│   ├── training/         # Scripts de entrenamiento
│   ├── inference/        # Scripts de inferencia
│   └── utils/           # Utilidades
├── notebooks/            # Jupyter notebooks
├── tests/                # Pruebas unitarias
├── scripts/              # Scripts de automatización
├── configs/              # Configuraciones
├── outputs/              # Resultados
└── docs/                 # Documentación
```

## 🚀 Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone <repository-url>
   cd {self.project_name}
   ```

2. **Crear y activar entorno virtual:**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\\Scripts\\activate
   
   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno:**
   ```bash
   cp .env.example .env
   # Editar .env con tus configuraciones
   ```

## 🧪 Ejecución

### Entrenamiento
```bash
python src/training/train.py
```

### Inferencia
```bash
python src/inference/predict.py
```

## 🧪 Testing

```bash
pytest tests/ -v --cov=src
```

## 📝 Desarrollo

### Formateo de código
```bash
black src/ tests/
```

### Linting
```bash
flake8 src/ tests/
```

## 📊 Monitoreo

Los logs se guardan en `outputs/logs/`.

## 👥 Autores

- {self.author_name}

## 📄 Licencia

MIT License
"""
    
    def _get_requirements_content(self) -> str:
        """Genera el contenido básico de requirements.txt."""
        return """# Dependencias del Proyecto
# =====================

# Core ML/DL
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualización
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilidades
python-dotenv>=1.0.0
tqdm>=4.65.0
requests>=2.31.0
pyyaml>=6.0.0

# Desarrollo
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
"""
    
    def _get_setup_content(self) -> str:
        """Genera el contenido del setup.py."""
        return f'''"""
Setup configuration for {self.project_name}.
"""

from setuptools import setup, find_packages

setup(
    name="{self.project_name.lower().replace(' ', '_')}",
    version="0.1.0",
    author="{self.author_name}",
    description="AI project following best practices",
    packages=find_packages(where="src"),
    package_dir={{"": "src"}},
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.13.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={{
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    }},
)
'''


def main():
    """
    Función principal para ejecutar la configuración del proyecto.
    """
    print("🚀 Iniciando Configuración Profesional de Proyecto IA")
    print("=" * 60)
    
    # Solicitar información del usuario
    project_name = input("📝 Nombre del proyecto: ").strip()
    if not project_name:
        print("❌ El nombre del proyecto no puede estar vacío")
        return
    
    author_name = input("👤 Nombre del autor (AI Developer): ").strip()
    if not author_name:
        author_name = "AI Developer"
    
    # Crear instancia y configurar
    setup = ProjectSetup(project_name, author_name)
    
    # Crear estructura
    if setup.create_project_structure():
        # Configurar entorno virtual
        setup.setup_virtual_environment()
        
        print(f"\n✨ ¡Proyecto '{project_name}' configurado profesionalmente!")
        print("📖 Consulta el README.md para próximos pasos")
    else:
        print("\n❌ No se pudo completar la configuración del proyecto")


if __name__ == "__main__":
    main()

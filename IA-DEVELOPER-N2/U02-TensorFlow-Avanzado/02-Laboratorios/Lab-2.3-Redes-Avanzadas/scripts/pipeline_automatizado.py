#!/usr/bin/env python3
"""
Pipeline Automatizado para Proyectos de IA
Ejecución orquestada de todos los proyectos del bootcamp

Aplicando buenas prácticas: PEP 8, Type Hinting, SOLID, Docstrings
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import warnings

# Configuración
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orquestador principal para ejecutar pipelines de IA automatizados.
    
    Aplica el Principio de Responsabilidad Única: Solo se encarga de
    orquestar la ejecución de proyectos de IA.
    
    Attributes:
        projects_config (Dict[str, Dict]): Configuración de proyectos.
        results (Dict[str, Any]): Resultados de ejecución.
        start_time (datetime): Tiempo de inicio del pipeline.
    """
    
    def __init__(self) -> None:
        """Inicializar el orquestador de pipelines."""
        self.projects_config: Dict[str, Dict] = self._load_projects_config()
        self.results: Dict[str, Any] = {}
        self.start_time: datetime = datetime.now()
        
        # Crear directorios necesarios
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Crear directorios necesarios para el pipeline."""
        directories = ['logs', 'outputs', 'reports', 'data/processed']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info("Directorios del pipeline creados")
    
    def _load_projects_config(self) -> Dict[str, Dict]:
        """
        Cargar configuración de proyectos.
        
        Returns:
            Dict[str, Dict]: Configuración de los proyectos.
        """
        return {
            "unidad_0": {
                "name": "Fundamentos y Buenas Prácticas",
                "script": "Unidad_0_Fundamentos/setup_project.py",
                "description": "Configuración profesional de proyectos",
                "dependencies": ["python", "pathlib", "logging"],
                "priority": 1,
                "timeout": 300  # 5 minutos
            },
            "proyecto_1": {
                "name": "Logística - Detección de Daños",
                "script": "proyecto1_logistica/scripts/train_cnn.py",
                "description": "CNN para detección de daños en paquetes",
                "dependencies": ["tensorflow", "opencv-python", "matplotlib"],
                "priority": 2,
                "timeout": 1800  # 30 minutos
            },
            "proyecto_2": {
                "name": "Salud - Análisis Médico",
                "script": "proyecto2_salud/scripts/train_transformer_refactored.py",
                "description": "Transformer para análisis de informes médicos",
                "dependencies": ["transformers", "tensorflow", "scikit-learn"],
                "priority": 3,
                "timeout": 2400  # 40 minutos
            },
            "proyecto_3": {
                "name": "Retail - Recomendación Híbrida",
                "script": "proyecto3_retail/scripts/train_hybrid_refactored.py",
                "description": "Red híbrida CNN+RNN para recomendación",
                "dependencies": ["tensorflow", "pandas", "numpy"],
                "priority": 4,
                "timeout": 2100  # 35 minutos
            }
        }
    
    def check_dependencies(self, project_key: str) -> bool:
        """
        Verificar dependencias de un proyecto.
        
        Args:
            project_key (str): Clave del proyecto.
            
        Returns:
            bool: True si las dependencias están disponibles.
        """
        config = self.projects_config[project_key]
        dependencies = config.get("dependencies", [])
        
        logger.info(f"Verificando dependencias para {config['name']}...")
        
        for dep in dependencies:
            try:
                __import__(dep)
                logger.info(f"  ✅ {dep}")
            except ImportError:
                logger.error(f"  ❌ {dep} - No encontrado")
                return False
        
        return True
    
    def execute_script(self, script_path: str, timeout: int = 3600) -> Tuple[bool, str, str]:
        """
        Ejecutar un script Python y capturar salida.
        
        Args:
            script_path (str): Ruta del script a ejecutar.
            timeout (int): Tiempo máximo de ejecución en segundos.
            
        Returns:
            Tuple[bool, str, str]: (éxito, stdout, stderr).
        """
        logger.info(f"Ejecutando script: {script_path}")
        
        try:
            # Ejecutar script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd()
            )
            
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
            
            if success:
                logger.info(f"✅ Script ejecutado exitosamente")
            else:
                logger.error(f"❌ Error ejecutando script: {stderr}")
            
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout ejecutando {script_path}")
            return False, "", "Script execution timeout"
        except Exception as e:
            logger.error(f"❌ Error ejecutando script: {str(e)}")
            return False, "", str(e)
    
    def run_project(self, project_key: str) -> Dict[str, Any]:
        """
        Ejecutar un proyecto completo.
        
        Args:
            project_key (str): Clave del proyecto.
            
        Returns:
            Dict[str, Any]: Resultados de la ejecución.
        """
        config = self.projects_config[project_key]
        script_path = config["script"]
        
        logger.info(f"🚀 Iniciando proyecto: {config['name']}")
        logger.info(f"📝 Descripción: {config['description']}")
        
        # Verificar que el script existe
        if not Path(script_path).exists():
            error_msg = f"Script no encontrado: {script_path}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration": 0
            }
        
        # Verificar dependencias
        if not self.check_dependencies(project_key):
            error_msg = "Dependencias faltantes"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration": 0
            }
        
        # Ejecutar script
        start_time = datetime.now()
        success, stdout, stderr = self.execute_script(script_path, config["timeout"])
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Guardar resultados
        result = {
            "success": success,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": duration,
            "stdout": stdout,
            "stderr": stderr,
            "config": config
        }
        
        # Guardar salida en archivos
        output_dir = Path("outputs") / project_key
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "stdout.log", "w", encoding="utf-8") as f:
            f.write(stdout)
        
        if stderr:
            with open(output_dir / "stderr.log", "w", encoding="utf-8") as f:
                f.write(stderr)
        
        logger.info(f"📊 Resultados guardados en: {output_dir}")
        
        return result
    
    def run_full_pipeline(self, projects: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Ejecutar el pipeline completo de proyectos.
        
        Args:
            projects (Optional[List[str]]): Lista de proyectos a ejecutar.
                                           Si es None, ejecuta todos.
            
        Returns:
            Dict[str, Any]: Resultados completos del pipeline.
        """
        logger.info("🚀 Iniciando Pipeline Completo de Bootcamp IA Developer")
        logger.info("=" * 80)
        
        # Determinar proyectos a ejecutar
        if projects is None:
            projects = list(self.projects_config.keys())
        
        # Ordenar por prioridad
        sorted_projects = sorted(
            projects,
            key=lambda x: self.projects_config[x]["priority"]
        )
        
        logger.info(f"📋 Proyectos a ejecutar: {[self.projects_config[p]['name'] for p in sorted_projects]}")
        
        # Ejecutar cada proyecto
        total_success = True
        for project_key in sorted_projects:
            logger.info(f"\n{'='*20} {project_key.upper()} {'='*20}")
            
            result = self.run_project(project_key)
            self.results[project_key] = result
            
            if not result["success"]:
                total_success = False
                logger.error(f"❌ Proyecto {project_key} falló")
            else:
                logger.info(f"✅ Proyecto {project_key} completado")
        
        # Generar reporte final
        pipeline_end_time = datetime.now()
        total_duration = (pipeline_end_time - self.start_time).total_seconds()
        
        pipeline_results = {
            "success": total_success,
            "start_time": self.start_time.isoformat(),
            "end_time": pipeline_end_time.isoformat(),
            "total_duration": total_duration,
            "projects_executed": len(projects),
            "successful_projects": sum(1 for r in self.results.values() if r["success"]),
            "failed_projects": sum(1 for r in self.results.values() if not r["success"]),
            "project_results": self.results
        }
        
        # Guardar reporte
        self._generate_report(pipeline_results)
        
        logger.info("\n" + "=" * 80)
        if total_success:
            logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        else:
            logger.info("⚠️ PIPELINE COMPLETADO CON ERRORES")
        logger.info("=" * 80)
        
        return pipeline_results
    
    def _generate_report(self, results: Dict[str, Any]) -> None:
        """
        Generar reporte del pipeline.
        
        Args:
            results (Dict[str, Any]): Resultados del pipeline.
        """
        logger.info("📊 Generando reporte del pipeline...")
        
        report = {
            "pipeline_summary": {
                "success": results["success"],
                "start_time": results["start_time"],
                "end_time": results["end_time"],
                "total_duration_minutes": results["total_duration"] / 60,
                "projects_executed": results["projects_executed"],
                "successful_projects": results["successful_projects"],
                "failed_projects": results["failed_projects"]
            },
            "project_details": {}
        }
        
        # Detalles de cada proyecto
        for project_key, result in results["project_results"].items():
            config = result["config"]
            
            report["project_details"][project_key] = {
                "name": config["name"],
                "description": config["description"],
                "success": result["success"],
                "duration_minutes": result["duration"] / 60,
                "error": result.get("error", None)
            }
        
        # Guardar reporte JSON
        report_path = Path("reports") / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generar reporte en texto
        text_report_path = Path("reports") / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(text_report_path, "w", encoding="utf-8") as f:
            f.write("REPORTE DE PIPELINE - BOOTCAMP IA DEVELOPER\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Éxito General: {'✅' if results['success'] else '❌'}\n")
            f.write(f"Duración Total: {results['total_duration'] / 60:.2f} minutos\n")
            f.write(f"Proyectos Ejecutados: {results['projects_executed']}\n")
            f.write(f"Proyectos Exitosos: {results['successful_projects']}\n")
            f.write(f"Proyectos Fallidos: {results['failed_projects']}\n\n")
            
            f.write("DETALLES POR PROYECTO:\n")
            f.write("-" * 40 + "\n")
            
            for project_key, result in results["project_results"].items():
                config = result["config"]
                status = "✅" if result["success"] else "❌"
                f.write(f"{status} {config['name']}\n")
                f.write(f"   Descripción: {config['description']}\n")
                f.write(f"   Duración: {result['duration'] / 60:.2f} minutos\n")
                if result.get("error"):
                    f.write(f"   Error: {result['error']}\n")
                f.write("\n")
        
        logger.info(f"📄 Reportes guardados:")
        logger.info(f"   • {report_path}")
        logger.info(f"   • {text_report_path}")
    
    def get_project_status(self, project_key: str) -> Dict[str, Any]:
        """
        Obtener estado de un proyecto específico.
        
        Args:
            project_key (str): Clave del proyecto.
            
        Returns:
            Dict[str, Any]: Estado del proyecto.
        """
        if project_key not in self.projects_config:
            return {"error": f"Proyecto {project_key} no encontrado"}
        
        if project_key not in self.results:
            return {"status": "not_executed", "config": self.projects_config[project_key]}
        
        result = self.results[project_key]
        return {
            "status": "completed" if result["success"] else "failed",
            "result": result
        }


def main() -> None:
    """
    Función principal del pipeline.
    
    Ejecuta todos los proyectos del bootcamp en orden de prioridad.
    """
    logger.info("🚀 INICIANDO PIPELINE AUTOMATIZADO - BOOTCAMP IA DEVELOPER")
    logger.info("Hora de inicio: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    # Crear orquestador
    orchestrator = PipelineOrchestrator()
    
    # Ejecutar pipeline completo
    try:
        results = orchestrator.run_full_pipeline()
        
        # Resumen final
        logger.info("\n📊 RESUMEN FINAL DEL PIPELINE:")
        logger.info(f"   • Proyectos ejecutados: {results['projects_executed']}")
        logger.info(f"   • Proyectos exitosos: {results['successful_projects']}")
        logger.info(f"   • Proyectos fallidos: {results['failed_projects']}")
        logger.info(f"   • Duración total: {results['total_duration'] / 60:.2f} minutos")
        
        if results["success"]:
            logger.info("\n🎉 ¡TODOS LOS PROYECTOS COMPLETADOS EXITOSAMENTE!")
            logger.info("📁 Revisa las carpetas 'outputs/', 'reports/' y 'logs/' para resultados detallados")
        else:
            logger.info("\n⚠️ Algunos proyectos fallaron. Revisa los logs para más detalles.")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Pipeline interrumpido por el usuario")
    except Exception as e:
        logger.error(f"\n❌ Error en el pipeline: {str(e)}")
    
    logger.info("Pipeline finalizado")


if __name__ == "__main__":
    # Ejecutar pipeline completo
    main()

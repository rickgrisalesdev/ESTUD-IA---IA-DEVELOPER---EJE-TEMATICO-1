#!/usr/bin/env python3
"""
Unidad 0 - Laboratorio 0.2: Refactorización a Código Limpio y Modular

Este laboratorio demuestra cómo transformar código "espagueti" en código 
profesional aplicando PEP 8, Type Hinting, SOLID y Docstrings.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod
import logging
from pathlib import Path


# Configuración de logging (mejor que print())
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== ANTES: CÓDIGO ESPAGUETI ====================

def procesar_datos_antiguo(datos, umbral=0.5):
    """Versión antigua sin buenas prácticas."""
    resultado = []
    for item in datos:
        if item > umbral:
            resultado.append(item * 2)
        else:
            resultado.append(item / 2)
    return resultado


def analizar_datos_antiguo(datos):
    """Versión antigua que hace múltiples cosas."""
    # Calcula promedio
    promedio = sum(datos) / len(datos)
    
    # Calcula máximo
    maximo = max(datos)
    
    # Filtra datos
    filtrados = [x for x in datos if x > promedio]
    
    # Genera reporte
    reporte = {
        'promedio': promedio,
        'maximo': maximo,
        'filtrados': filtrados,
        'total': len(filtrados)
    }
    
    return reporte


# ==================== DESPUÉS: CÓDIGO PROFESIONAL ====================

class DataProcessor(ABC):
    """
    Clase abstracta base para procesadores de datos.
    
    Aplica el Principio de Responsabilidad Única (S) y Abierto/Cerrado (O).
    """
    
    def __init__(self, config: Dict[str, Union[float, str]]):
        """
        Inicializa el procesador con configuración.
        
        Args:
            config (Dict[str, Union[float, str]]): Configuración del procesador.
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Método abstracto para procesar datos.
        
        Args:
            data (np.ndarray): Datos de entrada.
            
        Returns:
            np.ndarray: Datos procesados.
        """
        pass
    
    def _validate_config(self) -> None:
        """
        Valida la configuración del procesador.
        
        Raises:
            ValueError: Si la configuración es inválida.
        """
        required_keys = ['threshold', 'operation']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Config missing required key: {key}")


class ThresholdProcessor(DataProcessor):
    """
    Procesador que aplica operaciones basadas en umbral.
    
    Aplica Type Hinting y Docstrings de nivel profesional.
    """
    
    def __init__(self, threshold: float = 0.5, operation: str = 'multiply'):
        """
        Inicializa el procesador de umbral.
        
        Args:
            threshold (float): Umbral para procesamiento (default: 0.5).
            operation (str): Tipo de operación ('multiply' o 'divide').
        """
        config = {
            'threshold': threshold,
            'operation': operation
        }
        super().__init__(config)
        
        self.threshold: float = threshold
        self.operation: str = operation
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Procesa datos aplicando operación basada en umbral.
        
        Args:
            data (np.ndarray): Array de datos de entrada.
            
        Returns:
            np.ndarray: Datos procesados.
            
        Raises:
            ValueError: Si la operación no es válida.
        """
        logger.info(f"Procesando {len(data)} elementos con umbral {self.threshold}")
        
        if self.operation not in ['multiply', 'divide']:
            raise ValueError(f"Operación inválida: {self.operation}")
        
        # Vectorización con NumPy (más eficiente)
        mask = data > self.threshold
        
        if self.operation == 'multiply':
            result = np.where(mask, data * 2, data / 2)
        else:  # divide
            result = np.where(mask, data * 2, data / 2)
        
        return result


class DataAnalyzer:
    """
    Clase para análisis estadístico de datos.
    
    Aplica Principio de Responsabilidad Única: Solo analiza, no procesa.
    """
    
    def __init__(self):
        """Inicializa el analizador de datos."""
        self.metrics: Dict[str, float] = {}
    
    def calculate_basic_stats(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calcula estadísticas básicas de los datos.
        
        Args:
            data (np.ndarray): Datos a analizar.
            
        Returns:
            Dict[str, float]: Diccionario con estadísticas.
        """
        logger.info("Calculando estadísticas básicas")
        
        self.metrics = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'count': len(data)
        }
        
        return self.metrics
    
    def filter_by_threshold(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """
        Filtra datos por umbral.
        
        Args:
            data (np.ndarray): Datos a filtrar.
            threshold (float): Umbral de filtrado.
            
        Returns:
            np.ndarray: Datos filtrados.
        """
        filtered = data[data > threshold]
        logger.info(f"Filtrados {len(filtered)} de {len(data)} elementos")
        return filtered
    
    def generate_report(self, data: np.ndarray, filtered_data: np.ndarray) -> Dict[str, Union[float, int, List[float]]]:
        """
        Genera un reporte completo del análisis.
        
        Args:
            data (np.ndarray): Datos originales.
            filtered_data (np.ndarray): Datos filtrados.
            
        Returns:
            Dict[str, Union[float, int, List[float]]]: Reporte estructurado.
        """
        stats = self.calculate_basic_stats(data)
        
        report = {
            'estadisticas': stats,
            'datos_filtrados': filtered_data.tolist(),
            'cantidad_filtrados': len(filtered_data),
            'porcentaje_filtrado': (len(filtered_data) / len(data)) * 100
        }
        
        return report


class DataManager:
    """
    Gestor de datos que coordina procesamiento y análisis.
    
    Aplica Principio de Inversión de Dependencias (D).
    """
    
    def __init__(self, processor: DataProcessor, analyzer: DataAnalyzer):
        """
        Inicializa el gestor de datos.
        
        Args:
            processor (DataProcessor): Estrategia de procesamiento.
            analyzer (DataAnalyzer): Analizador de datos.
        """
        self.processor = processor
        self.analyzer = analyzer
    
    def execute_pipeline(self, data: np.ndarray) -> Dict[str, Union[float, int, List[float]]]:
        """
        Ejecuta el pipeline completo de procesamiento y análisis.
        
        Args:
            data (np.ndarray): Datos de entrada.
            
        Returns:
            Dict[str, Union[float, int, List[float]]]: Resultados completos.
        """
        logger.info("Iniciando pipeline de procesamiento")
        
        # Paso 1: Procesar datos
        processed_data = self.processor.process(data)
        
        # Paso 2: Filtrar datos procesados
        threshold = self.processor.config.get('threshold', 0.5)
        filtered_data = self.analyzer.filter_by_threshold(processed_data, threshold)
        
        # Paso 3: Generar reporte
        report = self.analyzer.generate_report(processed_data, filtered_data)
        
        logger.info("Pipeline completado exitosamente")
        return report


# ==================== FUNCIONES AUXILIARES ====================

def load_sample_data() -> np.ndarray:
    """
    Carga datos de ejemplo para demostración.
    
    Returns:
        np.ndarray: Array de datos de ejemplo.
    """
    logger.info("Cargando datos de ejemplo")
    
    # Generar datos simulados
    np.random.seed(42)
    data = np.random.normal(0.5, 0.2, 100)
    
    return data


def save_results_to_file(results: Dict, filename: str = "analysis_results.json") -> None:
    """
    Guarda resultados en archivo JSON.
    
    Args:
        results (Dict): Resultados a guardar.
        filename (str): Nombre del archivo de salida.
    """
    import json
    
    output_path = Path("outputs") / filename
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resultados guardados en: {output_path}")


# ==================== FUNCIÓN PRINCIPAL ====================

def main() -> None:
    """
    Función principal que demuestra la refactorización.
    
    Compara código antiguo vs código profesional.
    """
    print("🔧 Laboratorio 0.2: Refactorización a Código Limpio")
    print("=" * 60)
    
    # Cargar datos
    data = load_sample_data()
    
    print("\n📊 ANÁLISIS COMPARATIVO")
    print("-" * 30)
    
    # Versión antigua
    print("\n❌ VERSIÓN ANTIGUA (Código Espagueti):")
    try:
        resultado_antiguo = procesar_datos_antiguo(data[:10])
        analisis_antiguo = analizar_datos_antiguo(resultado_antiguo)
        print(f"   Resultado: {resultado_antiguo[:5]}...")
        print(f"   Análisis: {analisis_antiguo}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Versión profesional
    print("\n✅ VERSIÓN PROFESIONAL (Código Limpio):")
    try:
        # Crear componentes (inyección de dependencias)
        processor = ThresholdProcessor(threshold=0.5, operation='multiply')
        analyzer = DataAnalyzer()
        manager = DataManager(processor, analyzer)
        
        # Ejecutar pipeline
        resultado_profesional = manager.execute_pipeline(data[:10])
        
        print(f"   Estadísticas: {{'mean': {resultado_profesional['estadisticas']['mean']:.3f}, 'count': {resultado_profesional['estadisticas']['count']}}}")
        print(f"   Datos filtrados: {len(resultado_profesional['datos_filtrados'])} elementos")
        print(f"   Porcentaje filtrado: {resultado_profesional['porcentaje_filtrado']:.1f}%")
        
        # Guardar resultados
        save_results_to_file(resultado_profesional)
        
    except Exception as e:
        logger.error(f"Error en versión profesional: {e}")
    
    print("\n🎯 BENEFICIOS DEL CÓDIGO PROFESIONAL:")
    print("   ✅ Type Hinting: El IDE detecta errores antes de ejecutar")
    print("   ✅ Docstrings: Documentación clara y autogenerada")
    print("   ✅ SOLID: Código modular, extensible y mantenible")
    print("   ✅ Logging: Mejor que print() para producción")
    print("   ✅ Manejo de errores: Excepciones específicas")
    print("   ✅ Testing: Cada clase puede probarse independientemente")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Lab 1.2 - Proyecto Gemini: Calculadora Científica
Calculadora con operaciones básicas y funciones científicas

Aplicando buenas prácticas: PEP 8, Type Hinting, SOLID, Docstrings
"""

import math
import logging
from typing import Union, Optional, Dict, Any
from abc import ABC, abstractmethod

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Operation(ABC):
    """
    Clase abstracta base para operaciones matemáticas.
    
    Aplica el Principio de Abstracción: Define la interfaz
    común para todas las operaciones matemáticas.
    """
    
    @abstractmethod
    def execute(self, a: float, b: Optional[float] = None) -> float:
        """
        Ejecutar la operación matemática.
        
        Args:
            a (float): Primer operando.
            b (Optional[float]): Segundo operando (opcional).
            
        Returns:
            float: Resultado de la operación.
        """
        pass
    
    @abstractmethod
    def get_symbol(self) -> str:
        """
        Obtener el símbolo de la operación.
        
        Returns:
            str: Símbolo matemático.
        """
        pass


class Addition(Operation):
    """Operación de suma."""
    
    def execute(self, a: float, b: Optional[float] = None) -> float:
        return a + b if b is not None else a
    
    def get_symbol(self) -> str:
        return "+"


class Subtraction(Operation):
    """Operación de resta."""
    
    def execute(self, a: float, b: Optional[float] = None) -> float:
        return a - b if b is not None else a
    
    def get_symbol(self) -> str:
        return "-"


class Multiplication(Operation):
    """Operación de multiplicación."""
    
    def execute(self, a: float, b: Optional[float] = None) -> float:
        return a * b if b is not None else a
    
    def get_symbol(self) -> str:
        return "*"


class Division(Operation):
    """Operación de división."""
    
    def execute(self, a: float, b: Optional[float] = None) -> float:
        if b is None:
            return a
        if b == 0:
            raise ValueError("No se puede dividir por cero")
        return a / b
    
    def get_symbol(self) -> str:
        return "/"


class Power(Operation):
    """Operación de potencia."""
    
    def execute(self, a: float, b: Optional[float] = None) -> float:
        if b is None:
            return a
        return a ** b
    
    def get_symbol(self) -> str:
        return "^"


class SquareRoot(Operation):
    """Operación de raíz cuadrada."""
    
    def execute(self, a: float, b: Optional[float] = None) -> float:
        if a < 0:
            raise ValueError("No se puede calcular raíz cuadrada de número negativo")
        return math.sqrt(a)
    
    def get_symbol(self) -> str:
        return "√"


class Sine(Operation):
    """Operación seno (en grados)."""
    
    def execute(self, a: float, b: Optional[float] = None) -> float:
        return math.sin(math.radians(a))
    
    def get_symbol(self) -> str:
        return "sin"


class Cosine(Operation):
    """Operación coseno (en grados)."""
    
    def execute(self, a: float, b: Optional[float] = None) -> float:
        return math.cos(math.radians(a))
    
    def get_symbol(self) -> str:
        return "cos"


class Tangent(Operation):
    """Operación tangente (en grados)."""
    
    def execute(self, a: float, b: Optional[float] = None) -> float:
        # Evitar tangente de 90° + n*180°
        if a % 180 == 90:
            raise ValueError("La tangente no está definida para este ángulo")
        return math.tan(math.radians(a))
    
    def get_symbol(self) -> str:
        return "tan"


class Calculator:
    """
    Clase principal de la calculadora científica.
    
    Aplica el Principio de Responsabilidad Única: Solo se encarga de
    coordinar las operaciones matemáticas.
    
    Attributes:
        operations (Dict[str, Operation]): Diccionario de operaciones disponibles.
        history (List[str]): Historial de cálculos realizados.
    """
    
    def __init__(self) -> None:
        """Inicializar la calculadora con operaciones básicas y científicas."""
        self.operations: Dict[str, Operation] = {
            '+': Addition(),
            '-': Subtraction(),
            '*': Multiplication(),
            '/': Division(),
            '^': Power(),
            'sqrt': SquareRoot(),
            'sin': Sine(),
            'cos': Cosine(),
            'tan': Tangent()
        }
        self.history: List[str] = []
    
    def get_available_operations(self) -> Dict[str, str]:
        """
        Obtener operaciones disponibles con descripción.
        
        Returns:
            Dict[str, str]: Diccionario con símbolo y descripción.
        """
        return {
            symbol: f"{op.__class__.__name__} ({op.get_symbol()})"
            for symbol, op in self.operations.items()
        }
    
    def calculate(self, operation: str, a: float, b: Optional[float] = None) -> float:
        """
        Realizar un cálculo.
        
        Args:
            operation (str): Símbolo de la operación.
            a (float): Primer operando.
            b (Optional[float]): Segundo operando.
            
        Returns:
            float: Resultado del cálculo.
            
        Raises:
            ValueError: Si la operación no existe o hay error en el cálculo.
        """
        if operation not in self.operations:
            raise ValueError(f"Operación '{operation}' no disponible")
        
        try:
            op = self.operations[operation]
            result = op.execute(a, b)
            
            # Agregar al historial
            if b is not None:
                history_entry = f"{a} {op.get_symbol()} {b} = {result}"
            else:
                history_entry = f"{op.get_symbol()}({a}) = {result}"
            
            self.history.append(history_entry)
            logger.info(f"Cálculo realizado: {history_entry}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en cálculo: {str(e)}")
            raise
    
    def get_history(self) -> List[str]:
        """
        Obtener historial de cálculos.
        
        Returns:
            List[str]: Lista de cálculos realizados.
        """
        return self.history.copy()
    
    def clear_history(self) -> None:
        """Limpiar el historial de cálculos."""
        self.history.clear()
        logger.info("Historial limpiado")
    
    def save_history_to_file(self, filename: str = "calculator_history.txt") -> None:
        """
        Guardar historial en archivo.
        
        Args:
            filename (str): Nombre del archivo.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=== HISTORIAL DE CALCULADORA ===\n")
                f.write(f"Total de cálculos: {len(self.history)}\n\n")
                
                for i, entry in enumerate(self.history, 1):
                    f.write(f"{i:3d}. {entry}\n")
            
            logger.info(f"Historial guardado en: {filename}")
            
        except Exception as e:
            logger.error(f"Error guardando historial: {str(e)}")


class CalculatorUI:
    """
    Interfaz de usuario para la calculadora.
    
    Aplica el Principio de Responsabilidad Única: Solo se encarga de
    la interacción con el usuario.
    """
    
    def __init__(self, calculator: Calculator) -> None:
        """
        Inicializar la interfaz de usuario.
        
        Args:
            calculator (Calculator): Instancia de la calculadora.
        """
        self.calculator = calculator
    
    def display_menu(self) -> None:
        """Mostrar menú de operaciones disponibles."""
        print("\n" + "="*60)
        print("🧮 CALCULADORA CIENTÍFICA GEMINI")
        print("="*60)
        print("\n📋 Operaciones Disponibles:")
        
        operations = self.calculator.get_available_operations()
        
        for symbol, description in operations.items():
            print(f"  {symbol:6} - {description}")
        
        print("\n📋 Comandos Especiales:")
        print("  history - Mostrar historial")
        print("  clear   - Limpiar historial")
        print("  save    - Guardar historial")
        print("  help    - Mostrar ayuda")
        print("  exit    - Salir")
        print("="*60)
    
    def get_user_input(self) -> tuple[str, float, Optional[float]]:
        """
        Obtener entrada del usuario.
        
        Returns:
            tuple[str, float, Optional[float]]: (operación, operando1, operando2).
        """
        while True:
            try:
                user_input = input("\n🎯 Ingresa operación (ej: 5 + 3, sin 45): ").strip()
                
                # Comandos especiales
                if user_input.lower() in ['exit', 'help', 'history', 'clear', 'save']:
                    return user_input.lower(), 0.0, None
                
                # Parsear entrada
                parts = user_input.split()
                
                if len(parts) == 2:
                    # Operación unaria (sin 45, sqrt 16)
                    op = parts[0]
                    a = float(parts[1])
                    return op, a, None
                
                elif len(parts) == 3:
                    # Operación binaria (5 + 3, 2 ^ 8)
                    a = float(parts[0])
                    op = parts[1]
                    b = float(parts[2])
                    return op, a, b
                
                else:
                    print("❌ Formato incorrecto. Ejemplos:")
                    print("   5 + 3    (binaria)")
                    print("   sin 45    (unaria)")
                    print("   sqrt 16   (unaria)")
                
            except ValueError as e:
                print(f"❌ Error en los números: {e}")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def display_result(self, result: float) -> None:
        """
        Mostrar resultado formateado.
        
        Args:
            result (float): Resultado a mostrar.
        """
        # Formatear resultado para mejor visualización
        if result.is_integer():
            formatted_result = int(result)
        elif abs(result) < 0.001 or abs(result) > 1000000:
            formatted_result = f"{result:.2e}"
        else:
            formatted_result = f"{result:.6f}".rstrip('0').rstrip('.')
        
        print(f"✅ Resultado: {formatted_result}")
    
    def display_history(self) -> None:
        """Mostrar historial de cálculos."""
        history = self.calculator.get_history()
        
        if not history:
            print("📝 No hay cálculos en el historial")
            return
        
        print("\n📝 HISTORIAL DE CÁLCULOS:")
        print("-" * 40)
        
        for i, entry in enumerate(history, 1):
            print(f"{i:3d}. {entry}")
        
        print("-" * 40)
        print(f"Total: {len(history)} cálculos")
    
    def run(self) -> None:
        """
        Ejecutar la interfaz de usuario.
        
        Bucle principal de la calculadora.
        """
        logger.info("🧮 Iniciando Calculadora Científica Gemini")
        
        while True:
            try:
                self.display_menu()
                operation, a, b = self.get_user_input()
                
                # Manejar comandos especiales
                if operation == 'exit':
                    print("👋 ¡Gracias por usar la Calculadora Gemini!")
                    break
                
                elif operation == 'help':
                    print("\n📖 AYUDA:")
                    print("• Operaciones binarias: número operador número")
                    print("  Ejemplos: 5 + 3, 10 / 2, 2 ^ 8")
                    print("• Operaciones unarias: operador número")
                    print("  Ejemplos: sin 45, sqrt 16, cos 60")
                    print("• Los ángulos trigonométricos están en grados")
                    continue
                
                elif operation == 'history':
                    self.display_history()
                    continue
                
                elif operation == 'clear':
                    self.calculator.clear_history()
                    print("🗑️ Historial limpiado")
                    continue
                
                elif operation == 'save':
                    self.calculator.save_history_to_file()
                    continue
                
                # Realizar cálculo
                result = self.calculator.calculate(operation, a, b)
                self.display_result(result)
                
            except KeyboardInterrupt:
                print("\n\n⏹️ Calculadora interrumpida por el usuario")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Error en interfaz: {str(e)}")


def main() -> None:
    """
    Función principal del proyecto.
    
    Crea e inicia la calculadora científica.
    """
    logger.info("🚀 Iniciando Proyecto Gemini: Calculadora Científica")
    
    # Crear instancias
    calculator = Calculator()
    ui = CalculatorUI(calculator)
    
    # Ejecutar interfaz
    ui.run()


if __name__ == "__main__":
    main()

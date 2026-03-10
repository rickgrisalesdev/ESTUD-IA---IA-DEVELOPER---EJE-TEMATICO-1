#!/usr/bin/env python3
"""
Lab 1.2 - Ejercicios de Estructuras de Datos: Listas y Tuplas
Ejercicios prácticos para dominar listas y tuplas en Python

Aplicando buenas prácticas: PEP 8, Type Hinting, SOLID, Docstrings
"""

from typing import List, Tuple, Any, Optional
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ListOperations:
    """
    Clase para operaciones con listas.
    
    Aplica el Principio de Responsabilidad Única: Solo se encarga de
    operaciones relacionadas con listas.
    """
    
    @staticmethod
    def encontrar_maximo(numeros: List[int]) -> int:
        """
        Encontrar el número máximo en una lista.
        
        Args:
            numeros (List[int]): Lista de números enteros.
            
        Returns:
            int: El número máximo encontrado.
            
        Raises:
            ValueError: Si la lista está vacía.
        """
        if not numeros:
            raise ValueError("La lista no puede estar vacía")
        
        maximo = numeros[0]
        for numero in numeros[1:]:
            if numero > maximo:
                maximo = numero
        
        return maximo
    
    @staticmethod
    def eliminar_duplicados(lista: List[Any]) -> List[Any]:
        """
        Eliminar elementos duplicados manteniendo el orden.
        
        Args:
            lista (List[Any]): Lista con posibles duplicados.
            
        Returns:
            List[Any]: Lista sin duplicados manteniendo el orden original.
        """
        vista = set()
        resultado = []
        
        for elemento in lista:
            if elemento not in vista:
                vista.add(elemento)
                resultado.append(elemento)
        
        return resultado
    
    @staticmethod
    def rotar_lista(lista: List[Any], posiciones: int) -> List[Any]:
        """
        Rotar una lista N posiciones hacia la derecha.
        
        Args:
            lista (List[Any]): Lista a rotar.
            posiciones (int): Número de posiciones a rotar.
            
        Returns:
            List[Any]: Lista rotada.
        """
        if not lista:
            return lista
        
        posiciones = posiciones % len(lista)
        return lista[-posiciones:] + lista[:-posiciones]
    
    @staticmethod
    def intercalar_listas(lista1: List[Any], lista2: List[Any]) -> List[Any]:
        """
        Intercalar dos listas alternando elementos.
        
        Args:
            lista1 (List[Any]): Primera lista.
            lista2 (List[Any]): Segunda lista.
            
        Returns:
            List[Any]: Lista con elementos intercalados.
        """
        resultado = []
        min_len = min(len(lista1), len(lista2))
        
        for i in range(min_len):
            resultado.append(lista1[i])
            resultado.append(lista2[i])
        
        # Agregar elementos restantes
        resultado.extend(lista1[min_len:])
        resultado.extend(lista2[min_len:])
        
        return resultado


class TupleOperations:
    """
    Clase para operaciones con tuplas.
    
    Aplica el Principio de Responsabilidad Única: Solo se encarga de
    operaciones relacionadas con tuplas.
    """
    
    @staticmethod
    def desempaquetar_coordenadas(coordenadas: Tuple[float, float]) -> Tuple[float, float]:
        """
        Desempaquetar coordenadas x, y.
        
        Args:
            coordenadas (Tuple[float, float]): Tupla con coordenadas (x, y).
            
        Returns:
            Tuple[float, float]: Tupla con x e y separados.
        """
        if len(coordenadas) != 2:
            raise ValueError("Las coordenadas deben tener exactamente 2 elementos")
        
        x, y = coordenadas
        return x, y
    
    @staticmethod
    def calcular_distancia(punto1: Tuple[float, float], punto2: Tuple[float, float]) -> float:
        """
        Calcular distancia euclidiana entre dos puntos.
        
        Args:
            punto1 (Tuple[float, float]): Primer punto (x, y).
            punto2 (Tuple[float, float]): Segundo punto (x, y).
            
        Returns:
            float: Distancia euclidiana entre los puntos.
        """
        x1, y1 = punto1
        x2, y2 = punto2
        
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    @staticmethod
    def convertir_a_dict(tupla_clave_valor: List[Tuple[str, Any]]) -> dict:
        """
        Convertir lista de tuplas (clave, valor) a diccionario.
        
        Args:
            tupla_clave_valor (List[Tuple[str, Any]]): Lista de tuplas.
            
        Returns:
            dict: Diccionario resultante.
        """
        return dict(tupla_clave_valor)
    
    @staticmethod
    def encontrar_extremos(numeros: Tuple[int, ...]) -> Tuple[int, int]:
        """
        Encontrar el mínimo y máximo en una tupla de números.
        
        Args:
            numeros (Tuple[int, ...]): Tupla de números.
            
        Returns:
            Tuple[int, int]: Tupla con (mínimo, máximo).
            
        Raises:
            ValueError: Si la tupla está vacía.
        """
        if not numeros:
            raise ValueError("La tupla no puede estar vacía")
        
        return min(numeros), max(numeros)


def demostrar_operaciones_listas() -> None:
    """Demostrar operaciones con listas."""
    logger.info("=== DEMOSTRACIÓN DE OPERACIONES CON LISTAS ===")
    
    # Crear instancia
    list_ops = ListOperations()
    
    # Ejemplo 1: Encontrar máximo
    numeros = [3, 7, 2, 9, 1, 5]
    maximo = list_ops.encontrar_maximo(numeros)
    logger.info(f"Lista: {numeros}")
    logger.info(f"Máximo: {maximo}")
    
    # Ejemplo 2: Eliminar duplicados
    con_duplicados = [1, 2, 2, 3, 4, 4, 5, 1]
    sin_duplicados = list_ops.eliminar_duplicados(con_duplicados)
    logger.info(f"\nCon duplicados: {con_duplicados}")
    logger.info(f"Sin duplicados: {sin_duplicados}")
    
    # Ejemplo 3: Rotar lista
    original = [1, 2, 3, 4, 5]
    rotada = list_ops.rotar_lista(original, 2)
    logger.info(f"\nOriginal: {original}")
    logger.info(f"Rotada 2 posiciones: {rotada}")
    
    # Ejemplo 4: Intercalar listas
    lista1 = [1, 3, 5]
    lista2 = [2, 4, 6, 8, 10]
    intercalada = list_ops.intercalar_listas(lista1, lista2)
    logger.info(f"\nLista1: {lista1}")
    logger.info(f"Lista2: {lista2}")
    logger.info(f"Intercalada: {intercalada}")


def demostrar_operaciones_tuplas() -> None:
    """Demostrar operaciones con tuplas."""
    logger.info("\n=== DEMOSTRACIÓN DE OPERACIONES CON TUPLAS ===")
    
    # Crear instancia
    tuple_ops = TupleOperations()
    
    # Ejemplo 1: Desempaquetar coordenadas
    coordenadas = (10.5, 20.3)
    x, y = tuple_ops.desempaquetar_coordenadas(coordenadas)
    logger.info(f"Coordenadas: {coordenadas}")
    logger.info(f"X: {x}, Y: {y}")
    
    # Ejemplo 2: Calcular distancia
    punto1 = (0, 0)
    punto2 = (3, 4)
    distancia = tuple_ops.calcular_distancia(punto1, punto2)
    logger.info(f"\nPunto1: {punto1}")
    logger.info(f"Punto2: {punto2}")
    logger.info(f"Distancia: {distancia}")
    
    # Ejemplo 3: Convertir a diccionario
    tuplas = [("nombre", "Juan"), ("edad", 25), ("ciudad", "Madrid")]
    diccionario = tuple_ops.convertir_a_dict(tuplas)
    logger.info(f"\nTuplas: {tuplas}")
    logger.info(f"Diccionario: {diccionario}")
    
    # Ejemplo 4: Encontrar extremos
    numeros = (5, 2, 8, 1, 9, 3)
    minimo, maximo = tuple_ops.encontrar_extremos(numeros)
    logger.info(f"\nNúmeros: {numeros}")
    logger.info(f"Mínimo: {minimo}, Máximo: {maximo}")


def ejercicio_interactivo() -> None:
    """Ejercicio interactivo para practicar."""
    logger.info("\n=== EJERCICIO INTERACTIVO ===")
    
    list_ops = ListOperations()
    
    while True:
        print("\n🎯 Menú de Ejercicios:")
        print("1. Encontrar máximo de una lista")
        print("2. Eliminar duplicados de una lista")
        print("3. Rotar una lista")
        print("4. Intercalar dos listas")
        print("5. Salir")
        
        opcion = input("\nSelecciona una opción (1-5): ").strip()
        
        if opcion == '1':
            try:
                entrada = input("Ingresa números separados por comas: ").strip()
                numeros = [int(x.strip()) for x in entrada.split(',')]
                resultado = list_ops.encontrar_maximo(numeros)
                print(f"✅ Máximo: {resultado}")
            except ValueError as e:
                print(f"❌ Error: {e}")
        
        elif opcion == '2':
            try:
                entrada = input("Ingresa elementos separados por comas: ").strip()
                elementos = [x.strip() for x in entrada.split(',')]
                resultado = list_ops.eliminar_duplicados(elementos)
                print(f"✅ Sin duplicados: {resultado}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif opcion == '3':
            try:
                entrada = input("Ingresa elementos separados por comas: ").strip()
                elementos = [x.strip() for x in entrada.split(',')]
                posiciones = int(input("Número de posiciones a rotar: "))
                resultado = list_ops.rotar_lista(elementos, posiciones)
                print(f"✅ Rotada: {resultado}")
            except ValueError as e:
                print(f"❌ Error: {e}")
        
        elif opcion == '4':
            try:
                print("Primera lista:")
                entrada1 = input("  Elementos separados por comas: ").strip()
                lista1 = [x.strip() for x in entrada1.split(',')]
                
                print("Segunda lista:")
                entrada2 = input("  Elementos separados por comas: ").strip()
                lista2 = [x.strip() for x in entrada2.split(',')]
                
                resultado = list_ops.intercalar_listas(lista1, lista2)
                print(f"✅ Intercalada: {resultado}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif opcion == '5':
            logger.info("👋 Saliendo del ejercicio interactivo...")
            break
        
        else:
            print("❌ Opción no válida. Intenta de nuevo.")


def main() -> None:
    """
    Función principal del script.
    
    Ejecuta demostraciones y permite práctica interactiva.
    """
    logger.info("🐍 Iniciando Ejercicios de Listas y Tuplas")
    logger.info("=" * 50)
    
    # Demostraciones
    demostrar_operaciones_listas()
    demostrar_operaciones_tuplas()
    
    # Ejercicio interactivo
    ejercicio_interactivo()
    
    logger.info("\n✅ Ejercicios completados exitosamente")


if __name__ == "__main__":
    main()

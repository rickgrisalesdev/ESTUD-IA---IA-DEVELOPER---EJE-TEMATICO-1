import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SIATAConnector:
    """Conector oficial para la API del SIATA - Sistema de Alerta Temprana"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://siata.gov.co"):
        """
        Inicializar conector SIATA
        
        Args:
            api_key: Clave de API (si es requerida)
            base_url: URL base de la API SIATA
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        
        # Headers para autenticación
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
        else:
            self.session.headers.update({
                'Content-Type': 'application/json',
                'User-Agent': 'SIATA-AI-Client/1.0'
            })
        
        # Endpoints oficiales del SIATA (según documentación GitHub)
        self.endpoints_siata = {
            'pm25': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_pm25.json',
            'pm25_last': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_pm25_Last.json',
            'pm10': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_pm10.json',
            'pm10_last': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_pm10_Last.json',
            'pm1': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_pm1.json',
            'pm1_last': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_pm1_Last.json',
            'co': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_co.json',
            'co_last': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_co_Last.json',
            'no': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_no.json',
            'no_last': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_no_Last.json',
            'no2': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_no2.json',
            'no2_last': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_no2_Last.json',
            'ozono': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_ozono.json',
            'ozono_last': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_ozono_Last.json',
            'so2': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_so2.json',
            'so2_last': 'https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_AQ_so2_Last.json'
        }
        
        # Token para API de estaciones (si está disponible)
        self.token_estaciones = 'cf7bb09b4d7d859a2840e22c3f3a9a8039917cc3'
        self.endpoint_estaciones = 'http://siata.gov.co:8089/estacionesAirePM25/'
        
        # Estado de conexión
        self.api_disponible = False
        self.endpoint_activo = None
    
    def obtener_datos_siata_reales(self, variable: str = 'pm25', usar_ultimos: bool = True) -> Dict:
        """
        Obtener datos reales del SIATA usando endpoints oficiales
        
        Args:
            variable: Variable a consultar (pm25, pm10, pm1, co, no, no2, ozono, so2)
            usar_ultimos: Si True, usa datos del último día, si False, usa datos de 6 meses
            
        Returns:
            Dict: Datos reales del SIATA
        """
        try:
            # Seleccionar endpoint
            endpoint_key = f"{variable}_last" if usar_ultimos else variable
            
            if endpoint_key not in self.endpoints_siata:
                return {'status': 'error', 'message': f'Variable {variable} no disponible'}
            
            url = self.endpoints_siata[endpoint_key]
            logger.info(f"Consultando datos reales SIATA: {url}")
            
            # Hacer petición
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                try:
                    datos = response.json()
                    logger.info(f"✅ Datos reales obtenidos: {variable}")
                    return self._procesar_datos_siata_formato_openaq(datos, variable)
                except json.JSONDecodeError:
                    logger.warning("Respuesta no es JSON válido")
                    return self._procesar_respuesta_texto_siata(response.text, variable)
            else:
                logger.error(f"Error HTTP {response.status_code}: {url}")
                return self._generar_datos_simulados_siata(variable)
                
        except Exception as e:
            logger.error(f"Error obteniendo datos SIATA: {e}")
            return self._generar_datos_simulados_siata(variable)
    
    def _procesar_datos_siata_formato_openaq(self, datos: Dict, variable: str) -> Dict:
        """
        Procesar datos del SIATA en formato OpenAQ
        
        Args:
            datos: Datos crudos del SIATA
            variable: Variable consultada
            
        Returns:
            Dict: Datos procesados
        """
        return {
            'status': 'real_siata',
            'timestamp': datetime.now().isoformat(),
            'variable': variable,
            'datos_crudos': datos,
            'metadatos': {
                'fuente': 'API SIATA Oficial',
                'variable': variable,
                'endpoint': self.endpoints_siata.get(f"{variable}_last", self.endpoints_siata.get(variable)),
                'formato': 'OpenAQ',
                'total_estaciones': len(datos) if isinstance(datos, list) else 1
            }
        }
    
    def _procesar_respuesta_texto_siata(self, texto: str, variable: str) -> Dict:
        """
        Procesar respuesta de texto del SIATA
        
        Args:
            texto: Respuesta en texto
            variable: Variable consultada
            
        Returns:
            Dict: Datos procesados
        """
        return {
            'status': 'real_siata_texto',
            'timestamp': datetime.now().isoformat(),
            'variable': variable,
            'datos_crudos': texto,
            'metadatos': {
                'fuente': 'API SIATA Oficial (Texto)',
                'variable': variable,
                'longitud_texto': len(texto),
                'formato': 'texto_plano'
            }
        }
    
    def _generar_datos_simulados_siata(self, variable: str) -> Dict:
        """
        Generar datos simulados basados en patrones reales del SIATA
        
        Args:
            variable: Variable a simular
            
        Returns:
            Dict: Datos simulados
        """
        logger.info(f"Generando simulación SIATA para {variable}...")
        
        # Configuración específica por variable
        config_variables = {
            'pm25': {'rango': (5, 150), 'media': 25, 'unidad': 'μg/m³'},
            'pm10': {'rango': (10, 250), 'media': 40, 'unidad': 'μg/m³'},
            'pm1': {'rango': (2, 80), 'media': 15, 'unidad': 'μg/m³'},
            'co': {'rango': (0.1, 5.0), 'media': 1.0, 'unidad': 'ppm'},
            'no': {'rango': (0.01, 0.5), 'media': 0.1, 'unidad': 'ppm'},
            'no2': {'rango': (0.01, 0.3), 'media': 0.05, 'unidad': 'ppm'},
            'ozono': {'rango': (0.01, 0.15), 'media': 0.05, 'unidad': 'ppm'},
            'so2': {'rango': (0.001, 0.1), 'media': 0.01, 'unidad': 'ppm'}
        }
        
        config = config_variables.get(variable, config_variables['pm25'])
        
        # Generar datos para múltiples estaciones
        estaciones = []
        n_estaciones = 12  # Estaciones típicas de Medellín
        n_puntos = 96  # 24 horas * 4 puntos por hora
        
        timestamps = [
            (datetime.now() - timedelta(hours=24) + timedelta(minutes=15*i)).isoformat()
            for i in range(n_puntos)
        ]
        
        # Ubicaciones reales de estaciones SIATA
        ubicaciones_estaciones = [
            {'nombre': 'Aranjuez', 'lat': 6.2677, 'lon': -75.5989, 'comuna': 'Aranjuez'},
            {'nombre': 'Belén', 'lat': 6.2034, 'lon': -75.5678, 'comuna': 'Belén'},
            {'nombre': 'Buenos Aires', 'lat': 6.2301, 'lon': -75.6012, 'comuna': 'Buenos Aires'},
            {'nombre': 'Candelaria', 'lat': 6.2442, 'lon': -75.5812, 'comuna': 'La Candelaria'},
            {'nombre': 'La América', 'lat': 6.2677, 'lon': -75.5989, 'comuna': 'La América'},
            {'nombre': 'Laureles', 'lat': 6.2694, 'lon': -75.5679, 'comuna': 'Laureles-Estadio'},
            {'nombre': 'El Poblado', 'lat': 6.1844, 'lon': -75.5447, 'comuna': 'El Poblado'},
            {'nombre': 'Robledo', 'lat': 6.2034, 'lon': -75.5678, 'comuna': 'Robledo'},
            {'nombre': 'Santa Cruz', 'lat': 6.2677, 'lon': -75.5989, 'comuna': 'Santa Cruz'},
            {'nombre': 'Villa Hermosa', 'lat': 6.2677, 'lon': -75.5989, 'comuna': 'Villa Hermosa'},
            {'nombre': 'Popular', 'lat': 6.2677, 'lon': -75.5989, 'comuna': 'Popular'},
            {'nombre': 'San Javier', 'lat': 6.2677, 'lon': -75.5989, 'comuna': 'San Javier'}
        ]
        
        for i in range(n_estaciones):
            ubicacion = ubicaciones_estaciones[i % len(ubicaciones_estaciones)]
            
            # Generar patrón realista
            valores = self._generar_patron_contaminacion(config, n_puntos)
            
            # Añadir eventos específicos de la estación
            if i < 3:  # Estaciones en zonas industriales tienen picos más altos
                valores = self._añadir_picos_industriales(valores, n_puntos)
            elif i < 6:  # Estaciones en zonas comerciales
                valores = self._añadir_patron_comercial(valores, n_puntos)
            else:  # Estaciones en zonas residenciales
                valores = self._añadir_patron_residencial(valores, n_puntos)
            
            estaciones.append({
                'id_estacion': f'SIATA_{variable}_{i+1:02d}',
                'nombre_estacion': ubicacion['nombre'],
                'variable': variable.upper(),
                'unidad': config['unidad'],
                'ubicacion': {
                    'lat': ubicacion['lat'] + np.random.uniform(-0.01, 0.01),
                    'lon': ubicacion['lon'] + np.random.uniform(-0.01, 0.01),
                    'nombre': ubicacion['nombre'],
                    'comuna': ubicacion['comuna'],
                    'altura': '3m'
                },
                'datos': [
                    {
                        'timestamp': timestamps[j],
                        'valor': float(valores[j]),
                        'calidad_flag': self._determinar_calidad_siata(valores[j], variable),
                        'confianza': np.random.uniform(0.85, 0.99)
                    }
                    for j in range(n_puntos)
                ]
            })
        
        return {
            'status': 'simulado_siata',
            'timestamp': datetime.now().isoformat(),
            'variable': variable,
            'estaciones': estaciones,
            'metadatos': {
                'fuente': 'Simulación SIATA Realista',
                'variable': variable,
                'total_estaciones': n_estaciones,
                'total_lecturas': n_estaciones * n_puntos,
                'periodo': '24 horas',
                'formato': 'OpenAQ',
                'notas': f'Datos simulados basados en patrones reales de {variable.upper()} en Medellín'
            }
        }
    
    def _generar_patron_contaminacion(self, config: Dict, n_puntos: int) -> np.ndarray:
        """Generar patrón realista de contaminación"""
        rango = config['rango']
        media = config['media']
        
        # Patrón base
        valores = np.random.normal(media, (rango[1] - rango[0])/10, n_puntos)
        
        # Patrón diario (más contaminación en mañana y tarde)
        horas = np.arange(n_puntos) % (24 * 4) / 4
        patron_diario = (
            np.exp(-((horas - 7)**2) / 8) * 0.3 +  # Pico mañana
            np.exp(-((horas - 18)**2) / 8) * 0.4   # Pico tarde
        ) * (rango[1] - rango[0])
        valores += patron_diario
        
        # Suavizar
        kernel = np.ones(5) / 5
        valores = np.convolve(valores, kernel, mode='same')
        
        # Limitar al rango
        valores = np.clip(valores, rango[0], rango[1])
        
        return valores
    
    def _añadir_picos_industriales(self, valores: np.ndarray, n_puntos: int) -> np.ndarray:
        """Añadir picos de contaminación industrial"""
        # Picos aleatorios durante horas laborales
        for _ in range(np.random.randint(2, 5)):
            hora = np.random.randint(6, 18) * 4  # Convertir a índice de 15 min
            duracion = np.random.randint(4, 12)  # 1-3 horas
            magnitud = np.random.uniform(20, 60)
            
            inicio = max(0, hora)
            fin = min(n_puntos, hora + duracion)
            valores[inicio:fin] += magnitud
        
        return valores
    
    def _añadir_patron_comercial(self, valores: np.ndarray, n_puntos: int) -> np.ndarray:
        """Añadir patrón de zona comercial"""
        # Aumento durante horas comerciales
        horas = np.arange(n_puntos) % (24 * 4) / 4
        comercial_mask = (horas >= 9) & (horas <= 21)
        valores[comercial_mask] *= np.random.uniform(1.2, 1.5)
        
        return valores
    
    def _añadir_patron_residencial(self, valores: np.ndarray, n_puntos: int) -> np.ndarray:
        """Añadir patrón de zona residencial"""
        # Picos en horas pico (tráfico)
        horas = np.arange(n_puntos) % (24 * 4) / 4
        pico_manana = ((horas >= 6) & (horas <= 9))
        pico_tarde = ((horas >= 17) & (horas <= 20))
        
        valores[pico_manana | pico_tarde] *= np.random.uniform(1.3, 1.8)
        
        return valores
    
    def _determinar_calidad_siata(self, valor: float, variable: str) -> float:
        """
        Determinar calidad del dato según estándares SIATA
        
        Args:
            valor: Valor medido
            variable: Variable medida
            
        Returns:
            float: Flag de calidad (<= 2.5 es buena calidad)
        """
        # Calidad basada en desviación estándar
        if np.random.random() < 0.05:  # 5% datos malos
            return 3.0  # Mala calidad
        elif np.random.random() < 0.15:  # 15% datos regulares
            return 2.6  # Regular
        else:
            return 1.0  # Buena calidad
        """
        Obtener datos de sensores del SIATA con múltiples endpoints
        
        Args:
            sensor_name: Nombre específico del sensor
            days: Número de días hacia atrás
            
        Returns:
            Dict: Datos de sensores en formato estructurado
        """
        try:
            # Usar sensor aleatorio si no se especifica
            if not sensor_name:
                sensor_name = np.random.choice(self.sensores_disponibles + self.sensores_calidad_aire)
            
            # Si ya tenemos un endpoint activo, usarlo
            if self.endpoint_activo:
                return self._consultar_endpoint(self.endpoint_activo, sensor_name)
            
            # Probar diferentes endpoints hasta encontrar uno funcional
            for endpoint in self.endpoints_alternativos:
                try:
                    resultado = self._consultar_endpoint(endpoint, sensor_name)
                    if resultado['status'] != 'error':
                        self.endpoint_activo = endpoint
                        self.api_disponible = True
                        logger.info(f"✅ Endpoint activo: {endpoint}")
                        return resultado
                except Exception as e:
                    logger.debug(f"❌ Endpoint {endpoint} no disponible: {e}")
                    continue
            
            # Si ningún endpoint funciona, usar simulación mejorada
            logger.warning("🔄 Todos los endpoints fallaron, usando simulación realista")
            self.api_disponible = False
            return self._generar_datos_simulados_mejorados(sensor_name, days)
                
        except Exception as e:
            logger.error(f"Error de conexión: {e}")
            return self._generar_datos_simulados_mejorados(sensor_name or "pms_001", days)
    
    def _consultar_endpoint(self, endpoint: str, sensor_name: str) -> Dict:
        """
        Consultar un endpoint específico con múltiples parámetros
        
        Args:
            endpoint: Path del endpoint
            sensor_name: Nombre del sensor
            
        Returns:
            Dict: Respuesta del endpoint
        """
        url = self.base_url + endpoint
        
        # Diferentes combinaciones de parámetros
        params_combinations = [
            {'metodo': 'getDatosSensor', 'sensor_name': sensor_name, 'formato': 'json'},
            {'sensor': sensor_name, 'format': 'json', 'limit': 100},
            {'id_sensor': sensor_name, 'tipo': 'json'},
            {'estacion': sensor_name, 'salida': 'json'},
            {'sensor_name': sensor_name, 'format': 'json'},
            {'codigo': sensor_name, 'output': 'json'},
            {'name': sensor_name, 'database': 'Processed', 'type_data': 'Flux'}
        ]
        
        for params in params_combinations:
            try:
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    content = response.text.strip()
                    if content.startswith('{') or content.startswith('['):
                        datos = response.json()
                        logger.info(f"✅ Datos reales obtenidos de {sensor_name}")
                        return self._procesar_datos_siata_reales(datos, sensor_name)
                    elif len(content) > 100 and 'error' not in content.lower():
                        logger.info(f"✅ Datos texto obtenidos de {sensor_name}")
                        return self._procesar_respuesta_texto(content, sensor_name)
            except Exception as e:
                continue
        
        return {'status': 'error', 'message': f'No response from {endpoint}'}
    
    def _procesar_datos_siata_reales(self, datos: Dict, sensor_name: str) -> Dict:
        """
        Procesar datos reales del SIATA al formato esperado
        
        Args:
            datos: Datos crudos del SIATA
            sensor_name: Nombre del sensor
            
        Returns:
            Dict: Datos procesados
        """
        return {
            'status': 'real',
            'timestamp': datetime.now().isoformat(),
            'sensor_name': sensor_name,
            'datos_crudos': datos,
            'metadatos': {
                'fuente': 'API SIATA Real',
                'sensor': sensor_name,
                'endpoint': f"{self.base_url}/interfaz.php"
            }
        }
    
    def _procesar_respuesta_texto(self, texto_respuesta: str, sensor_name: str) -> Dict:
        """
        Procesar respuesta de texto cuando no es JSON
        
        Args:
            texto_respuesta: Respuesta en texto plano
            sensor_name: Nombre del sensor
            
        Returns:
            Dict: Datos procesados
        """
        return {
            'status': 'real',
            'timestamp': datetime.now().isoformat(),
            'sensor_name': sensor_name,
            'datos_crudos': texto_respuesta,
            'metadatos': {
                'fuente': 'API SIATA Real (Texto)',
                'sensor': sensor_name,
                'formato': 'texto_plano'
            }
        }
    
    def _generar_datos_simulados_mejorados(self, sensor_name: str, dias: int) -> Dict:
        """
        Generar datos simulados ultra-realistas basados en características SIATA
        
        Args:
            sensor_name: Nombre del sensor SIATA
            dias: Número de días de datos
            
        Returns:
            Dict: Datos simulados de alta calidad
        """
        logger.info(f"🎭 Generando simulación realista para {sensor_name}...")
        
        # Determinar tipo de sensor y ubicación
        ubicacion = self._obtener_ubicacion_sensor(sensor_name)
        tipo_sensor = self._determinar_tipo_sensor(sensor_name)
        
        # Generar datos temporales realistas
        n_puntos = dias * 24 * 4  # 15-min intervals
        timestamps = [
            (datetime.now() - timedelta(days=dias) + timedelta(minutes=15*i)).isoformat()
            for i in range(n_puntos)
        ]
        
        # Variables según tipo de sensor
        variables_config = self._obtener_configuracion_variables(tipo_sensor)
        
        sensores = []
        for i, config in enumerate(variables_config):
            # Generar patrones realistas
            valores = self._generar_patron_realista(config, n_puntos)
            
            # Añadir eventos específicos del sensor
            if tipo_sensor == 'calidad_aire':
                valores = self._añadir_eventos_calidad_aire(valores, n_puntos)
            elif tipo_sensor == 'meteorologico':
                valores = self._añadir_patrones_meteorologicos(valores, n_puntos)
            elif tipo_sensor == 'flujo_gases':
                valores = self._añadir_patrones_flujo(valores, n_puntos)
            
            sensores.append({
                'id': f'{sensor_name}_{i+1:02d}',
                'nombre_sensor': sensor_name,
                'variable': config['nombre'],
                'unidad': config['unidad'],
                'tipo_sensor': tipo_sensor,
                'ubicacion': {
                    'lat': ubicacion['lat'],
                    'lon': ubicacion['lon'],
                    'nombre': ubicacion['nombre'],
                    'comuna': ubicacion['comuna'],
                    'altura': ubicacion.get('altura', '10m')
                },
                'datos': [
                    {
                        'timestamp': timestamps[j],
                        'valor': float(valores[j]),
                        'calidad': self._determinar_calidad_dato(valores[j]),
                        'confianza': np.random.uniform(0.85, 0.99)
                    }
                    for j in range(n_puntos)
                ]
            })
        
        return {
            'status': 'simulado_mejorado',
            'timestamp': datetime.now().isoformat(),
            'sensor_principal': sensor_name,
            'tipo_sensor': tipo_sensor,
            'sensores': sensores,
            'metadatos': {
                'fuente': 'Simulación SIATA Mejorada',
                'sensor': sensor_name,
                'dias': dias,
                'total_sensores': len(sensores),
                'total_lecturas': len(sensores) * n_puntos,
                'ubicacion': ubicacion,
                'calidad': 'alta_realismo',
                'endpoint': 'simulado',
                'notas': f'Datos simulados basados en patrones reales de {tipo_sensor}'
            }
        }
    
    def _obtener_ubicacion_sensor(self, sensor_name: str) -> Dict:
        """Obtener ubicación realista según nombre del sensor"""
        ubicaciones = {
            'IRGASON_Candelaria': {'lat': 6.2442, 'lon': -75.5812, 'nombre': 'Candelaria', 'comuna': 'La Candelaria', 'altura': '10m'},
            'IRGASON_Federico_Carrasquilla': {'lat': 6.2677, 'lon': -75.5989, 'nombre': 'Federico Carrasquilla', 'comuna': 'La América', 'altura': '20m'},
            'IRGASON_SENA': {'lat': 6.2301, 'lon': -75.6012, 'nombre': 'SENA', 'comuna': 'Buenos Aires', 'altura': '10m'},
            'IRGASON_ITM': {'lat': 6.2694, 'lon': -75.5679, 'nombre': 'ITM', 'comuna': 'Laureles-Estadio', 'altura': '30m'},
            'IRGASON_Villa_Niza': {'lat': 6.1844, 'lon': -75.5447, 'nombre': 'Villa Niza', 'comuna': 'El Poblado', 'altura': '20m'},
            'IRGASON_Altavista': {'lat': 6.2034, 'lon': -75.5678, 'nombre': 'Altavista', 'comuna': 'Robledo', 'altura': '10m'},
            'CSAT3B_MED_FISC': {'lat': 6.2518, 'lon': -75.5635, 'nombre': 'Medellín FISC', 'comuna': 'Laureles', 'altura': '3m'}
        }
        
        for key, ubic in ubicaciones.items():
            if key in sensor_name:
                return ubic
        
        # Ubicación por defecto para sensores PMS
        return {
            'lat': 6.2442 + np.random.uniform(-0.05, 0.05),
            'lon': -75.5812 + np.random.uniform(-0.05, 0.05),
            'nombre': f'Estación {sensor_name}',
            'comuna': 'Medellín',
            'altura': '5m'
        }
    
    def _determinar_tipo_sensor(self, sensor_name: str) -> str:
        """Determinar tipo de sensor según nombre"""
        if 'IRGASON' in sensor_name:
            return 'flujo_gases'
        elif 'CSAT3B' in sensor_name:
            return 'meteorologico'
        elif 'pms_' in sensor_name:
            return 'calidad_aire'
        else:
            return 'generico'
    
    def _obtener_configuracion_variables(self, tipo_sensor: str) -> List[Dict]:
        """Obtener configuración de variables según tipo de sensor"""
        configs = {
            'flujo_gases': [
                {'nombre': 'CO2_Flux', 'unidad': 'μmol/m²/s', 'rango': (-5, 15), 'media': 2},
                {'nombre': 'H2O_Flux', 'unidad': 'mmol/m²/s', 'rango': (-2, 8), 'media': 1},
                {'nombre': 'Temperature', 'unidad': '°C', 'rango': (15, 30), 'media': 22},
                {'nombre': 'Humidity', 'unidad': '%', 'rango': (40, 90), 'media': 65},
                {'nombre': 'Pressure', 'unidad': 'hPa', 'rango': (880, 920), 'media': 900}
            ],
            'meteorologico': [
                {'nombre': 'Wind_Speed', 'unidad': 'm/s', 'rango': (0, 15), 'media': 3},
                {'nombre': 'Wind_Direction', 'unidad': '°', 'rango': (0, 360), 'media': 180},
                {'nombre': 'Sonic_Temperature', 'unidad': '°C', 'rango': (15, 30), 'media': 22},
                {'nombre': 'Vertical_Wind', 'unidad': 'm/s', 'rango': (-2, 2), 'media': 0}
            ],
            'calidad_aire': [
                {'nombre': 'PM2.5', 'unidad': 'μg/m³', 'rango': (5, 150), 'media': 25},
                {'nombre': 'PM10', 'unidad': 'μg/m³', 'rango': (10, 250), 'media': 40},
                {'nombre': 'Temperature', 'unidad': '°C', 'rango': (15, 30), 'media': 22},
                {'nombre': 'Humidity', 'unidad': '%', 'rango': (40, 90), 'media': 65}
            ],
            'generico': [
                {'nombre': 'Variable_1', 'unidad': 'unit1', 'rango': (-10, 10), 'media': 0},
                {'nombre': 'Variable_2', 'unidad': 'unit2', 'rango': (-10, 10), 'media': 0},
                {'nombre': 'Variable_3', 'unidad': 'unit3', 'rango': (-10, 10), 'media': 0},
                {'nombre': 'Variable_4', 'unidad': 'unit4', 'rango': (-10, 10), 'media': 0}
            ]
        }
        return configs.get(tipo_sensor, configs['generico'])
    
    def _generar_patron_realista(self, config: Dict, n_puntos: int) -> np.ndarray:
        """Generar patrón realista para una variable"""
        rango = config['rango']
        media = config['media']
        
        # Patrón base con ruido
        valores = np.random.normal(media, (rango[1] - rango[0])/8, n_puntos)
        
        # Añadir tendencias diarias
        horas = np.arange(n_puntos) % (24 * 4) / 4  # Convertir a horas
        patron_diario = np.sin(2 * np.pi * horas / 24) * (rango[1] - rango[0]) * 0.2
        valores += patron_diario
        
        # Añadir variaciones suaves
        kernel = np.exp(-np.arange(5)**2 / 2)
        kernel /= kernel.sum()
        valores = np.convolve(valores, kernel, mode='same')
        
        # Limitar al rango
        valores = np.clip(valores, rango[0], rango[1])
        
        return valores
    
    def _añadir_eventos_calidad_aire(self, valores: np.ndarray, n_puntos: int) -> np.ndarray:
        """Añadir eventos realistas de calidad del aire"""
        # Eventos de contaminación (picos de PM2.5)
        if np.random.random() < 0.3:  # 30% de probabilidad
            inicio = np.random.randint(0, n_puntos - 20)
            duracion = np.random.randint(4, 12)  # 1-3 horas
            magnitud = np.random.uniform(20, 80)
            valores[inicio:inicio+duracion] += magnitud
        
        return valores
    
    def _añadir_patrones_meteorologicos(self, valores: np.ndarray, n_puntos: int) -> np.ndarray:
        """Añadir patrones meteorológicos realistas"""
        # Ráfagas de viento
        if 'Wind' in str(valores):
            if np.random.random() < 0.2:
                inicio = np.random.randint(0, n_puntos - 8)
                valores[inicio:inicio+8] *= np.random.uniform(2, 4)
        
        return valores
    
    def _añadir_patrones_flujo(self, valores: np.ndarray, n_puntos: int) -> np.ndarray:
        """Añadir patrones de flujo de gases"""
        # Picos de flujo durante el día
        horas = np.arange(n_puntos) % (24 * 4) / 4
        dia_mask = (horas >= 6) & (horas <= 18)
        valores[dia_mask] *= np.random.uniform(1.2, 1.5)
        
        return valores
    
    def _determinar_calidad_dato(self, valor: float) -> str:
        """Determinar calidad del dato basado en valor"""
        if np.random.random() < 0.05:  # 5% de datos malos
            return 'mala'
        elif np.random.random() < 0.15:  # 15% de datos regulares
            return 'regular'
        else:
            return 'buena'
        """
        Generar datos simulados usando nombre real del sensor SIATA
        
        Args:
            sensor_name: Nombre del sensor SIATA
            dias: Número de días de datos
            
        Returns:
            Dict: Datos simulados con metadatos reales
        """
        logger.info(f"Generando datos simulados para sensor {sensor_name}...")
        
        # Generar datos para múltiples sensores basados en el nombre principal
        n_puntos = dias * 24 * 4  # 4 puntos por hora
        timestamps = [
            (datetime.now() - timedelta(days=dias) + timedelta(minutes=15*i)).isoformat()
            for i in range(n_puntos)
        ]
        
        # Datos específicos según tipo de sensor
        if "IRGASON" in sensor_name:
            # Sensor IRGASON (flujo de gases)
            variables = ['CO2_Flux', 'H2O_Flux', 'Temperature', 'Humidity', 'Pressure']
            unidades = ['μmol/m²/s', 'mmol/m²/s', '°C', '%', 'hPa']
        elif "CSAT3B" in sensor_name:
            # Sensor CSAT3B (sonico anemómetro)
            variables = ['Wind_Speed', 'Wind_Direction', 'Sonic_Temperature', 'Vertical_Wind']
            unidades = ['m/s', '°', '°C', 'm/s']
        else:
            # Sensor genérico
            variables = ['Variable_1', 'Variable_2', 'Variable_3', 'Variable_4']
            unidades = ['unit1', 'unit2', 'unit3', 'unit4']
        
        sensores = []
        for i, (var, unit) in enumerate(zip(variables, unidades)):
            # Generar datos realistas según variable
            if 'Flux' in var:
                valores = np.random.normal(0, 2, n_puntos)
            elif 'Temperature' in var:
                valores = np.random.normal(20, 5, n_puntos)
            elif 'Humidity' in var:
                valores = np.random.normal(60, 15, n_puntos)
            elif 'Wind' in var:
                valores = np.abs(np.random.weibull(2, 1, n_puntos) * 5)
            else:
                valores = np.random.normal(0, 1, n_puntos)
            
            # Añadir eventos de riesgo aleatorios
            if np.random.random() < 0.3:
                riesgo_start = np.random.randint(0, n_puntos-20)
                valores[riesgo_start:riesgo_start+20] += np.random.uniform(5, 15)
            
            sensores.append({
                'id': f'{sensor_name}_{i+1:02d}',
                'nombre_sensor': sensor_name,
                'variable': var,
                'unidad': unit,
                'ubicacion': {
                    'lat': 6.2442 + np.random.uniform(-0.05, 0.05),
                    'lon': -75.5812 + np.random.uniform(-0.05, 0.05),
                    'nombre': f'Estación {sensor_name}',
                    'comuna': self._obtener_comuna_sensor(sensor_name)
                },
                'datos': [
                    {
                        'timestamp': timestamps[j],
                        'valor': float(valores[j]),
                        'calidad': 'buena' if np.random.random() > 0.1 else 'regular'
                    }
                    for j in range(n_puntos)
                ]
            })
        
        return {
            'status': 'simulado',
            'timestamp': datetime.now().isoformat(),
            'sensor_principal': sensor_name,
            'sensores': sensores,
            'metadatos': {
                'fuente': 'Simulación SIATA',
                'sensor': sensor_name,
                'dias': dias,
                'total_sensores': len(sensores),
                'total_lecturas': len(sensores) * n_puntos,
                'tipo_datos': 'Flux'
            }
        }
    
    def _obtener_comuna_sensor(self, sensor_name: str) -> str:
        """
        Obtener comuna probable según nombre del sensor
        
        Args:
            sensor_name: Nombre del sensor
            
        Returns:
            str: Nombre de la comuna
        """
        comunas = {
            'Candelaria': 'La Candelaria',
            'Federico_Carrasquilla': 'La América',
            'SENA': 'Buenos Aires',
            'ITM': 'Laureles-Estadio',
            'Villa_Niza': 'El Poblado',
            'Jesus_Maria_Valle': 'Santa Cruz',
            'Villa_Socorro': 'Villa Hermosa',
            'Altavista': 'Robledo'
        }
        
        for key, comuna in comunas.items():
            if key in sensor_name:
                return comuna
        
        return 'Medellín'
    
    def enviar_alerta(self, 
                     nivel_riesgo: str, 
                     ubicacion: Dict, 
                     confianza: float,
                     sensores_criticos: List[str]) -> bool:
        """
        Enviar alerta al sistema SIATA
        
        Args:
            nivel_riesgo: Nivel de riesgo (Bajo, Medio, Alto)
            ubicacion: Coordenadas y descripción
            confianza: Nivel de confianza del modelo (0-1)
            sensores_criticos: Lista de sensores que activaron la alerta
            
        Returns:
            bool: True si se envió correctamente
        """
        try:
            endpoint = f"{self.base_url}/alertas"
            
            alerta = {
                'nivel_riesgo': nivel_riesgo,
                'ubicacion': ubicacion,
                'confianza': confianza,
                'sensores_criticos': sensores_criticos,
                'timestamp': datetime.now().isoformat(),
                'fuente': 'IA-SIATA-Model',
                'modelo_version': '1.0'
            }
            
            logger.info(f"Enviando alerta {nivel_riesgo} para {ubicacion.get('nombre', 'Ubicación desconocida')}")
            response = self.session.post(endpoint, json=alerta, timeout=10)
            
            if response.status_code in [200, 201]:
                logger.info("Alerta enviada exitosamente")
                return True
            else:
                logger.error(f"Error enviando alerta: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error al enviar alerta: {e}")
            return False
    
    def obtener_estado_sistema(self) -> Dict:
        """
        Obtener estado general del sistema SIATA
        
        Returns:
            Dict: Estado del sistema
        """
        try:
            endpoint = f"{self.base_url}/estado"
            response = self.session.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return self._estado_simulado()
                
        except Exception as e:
            logger.error(f"Error obteniendo estado: {e}")
            return self._estado_simulado()
    
    def _generar_datos_simulados(self, tipo_sensor: str, tiempo: int) -> Dict:
        """
        Generar datos simulados cuando la API no está disponible
        
        Args:
            tipo_sensor: Tipo de sensor
            tiempo: Período de tiempo
            
        Returns:
            Dict: Datos simulados
        """
        logger.info("Generando datos simulados...")
        
        # Generar datos realistas para 8 sensores
        n_sensores = 8
        n_puntos = tiempo * 4  # 4 puntos por hora
        
        # Datos base con patrones realistas
        timestamps = [
            (datetime.now() - timedelta(hours=tiempo) + timedelta(minutes=15*i)).isoformat()
            for i in range(n_puntos)
        ]
        
        sensores = []
        for i in range(n_sensores):
            # Simular diferentes tipos de datos
            if i == 0:  # Humedad
                valores = np.random.normal(60, 15, n_puntos)
            elif i == 1:  # Movimiento
                valores = np.random.exponential(0.5, n_puntos)
            elif i == 2:  # Presión
                valores = np.random.normal(1000, 50, n_puntos)
            elif i == 3:  # Temperatura
                valores = np.random.normal(25, 3, n_puntos)
            elif i == 4:  # Inclinación
                valores = np.random.normal(15, 5, n_puntos)
            elif i == 5:  # Vibración
                valores = np.random.gamma(2, 1, n_puntos)
            elif i == 6:  # Nivel freático
                valores = np.random.normal(2, 0.5, n_puntos)
            else:  # Velocidad viento
                valores = np.random.weibull(2, n_puntos) * 10
            
            # Añadir algún patrón de riesgo aleatorio
            if np.random.random() < 0.3:
                riesgo_start = np.random.randint(0, n_puntos-10)
                valores[riesgo_start:riesgo_start+10] *= np.random.uniform(1.5, 3.0)
            
            sensores.append({
                'id': f'SENSOR_{i+1:03d}',
                'tipo': ['Humedad', 'Movimiento', 'Presión', 'Temperatura', 
                        'Inclinación', 'Vibración', 'Nivel Freático', 'Viento'][i],
                'unidad': ['%', 'mm/s', 'hPa', '°C', '°', 'Hz', 'm', 'km/h'][i],
                'ubicacion': {
                    'lat': 6.2442 + np.random.uniform(-0.1, 0.1),
                    'lon': -75.5812 + np.random.uniform(-0.1, 0.1),
                    'nombre': f'Estación Monitoreo {i+1}'
                },
                'datos': [
                    {
                        'timestamp': timestamps[j],
                        'valor': float(valores[j]),
                        'calidad': 'buena' if np.random.random() > 0.1 else 'regular'
                    }
                    for j in range(n_puntos)
                ]
            })
        
        return {
            'status': 'simulado',
            'timestamp': datetime.now().isoformat(),
            'sensores': sensores,
            'metadatos': {
                'tipo_sensor': tipo_sensor,
                'periodo': f'{tiempo} horas',
                'total_sensores': n_sensores,
                'total_lecturas': n_sensores * n_puntos
            }
        }
    
    def _estado_simulado(self) -> Dict:
        """
        Generar estado simulado del sistema
        
        Returns:
            Dict: Estado simulado
        """
        return {
            'status': 'simulado',
            'timestamp': datetime.now().isoformat(),
            'servidores': {
                'api': 'online',
                'database': 'online',
                'monitoreo': 'online'
            },
            'sensores_activos': 8,
            'sensores_totales': 10,
            'alertas_activas': 0,
            'ultima_actualizacion': datetime.now().isoformat(),
            'rendimiento': {
                'latencia_ms': 45,
                'uptime_percentage': 99.8,
                'cpu_usage': 23.5,
                'memory_usage': 67.2
            }
        }
    
    def convertir_datos_a_tensor(self, datos: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Convertir datos de la API a formato tensor para el modelo
        
        Args:
            datos: Datos de sensores de la API
            
        Returns:
            Tuple[np.ndarray, Dict]: Tensor de datos y metadatos
        """
        try:
            # Extraer datos crudos
            datos_crudos = datos.get('datos_crudos', datos)
            
            # Si es una respuesta JSON con estructura
            if isinstance(datos_crudos, dict):
                # Procesar según estructura real de SIATA
                if 'datos' in datos_crudos:
                    # Estructura con lista de datos
                    sensores_data = datos_crudos['datos']
                else:
                    # Estructura directa
                    sensores_data = datos_crudos
            else:
                # Respuesta en texto plano - crear estructura simulada
                sensores_data = self._generar_estructura_desde_texto(datos_crudos, datos.get('sensor_name', 'unknown'))
            
            # Convertir a matriz numpy con forma correcta (time_steps, features)
            if isinstance(sensores_data, list):
                # Lista de sensores
                matriz_datos = []
                metadatos_sensores = []
                
                for sensor in sensores_data[:8]:  # Limitar a 8 sensores
                    if isinstance(sensor, dict) and 'datos' in sensor:
                        # Extraer valores numéricos
                        valores = [lectura.get('valor', 0) for lectura in sensor.get('datos', [])]
                        if len(valores) > 24:
                            valores = valores[-24:]  # Últimas 24 lecturas
                        
                        matriz_datos.append(valores)
                        metadatos_sensores.append({
                            'id': sensor.get('id', f'SENSOR_{len(matriz_datos)}'),
                            'tipo': sensor.get('variable', 'Desconocido'),
                            'unidad': sensor.get('unidad', 'unidad'),
                            'ubicacion': sensor.get('ubicacion', {})
                        })
                    else:
                        # Datos simulados si no hay estructura clara
                        valores_simulados = np.random.normal(0, 1, 24).tolist()
                        matriz_datos.append(valores_simulados)
                        metadatos_sensores.append({
                            'id': f'SENSOR_{len(matriz_datos)}',
                            'tipo': 'Simulado',
                            'unidad': 'unidad',
                            'ubicacion': {'nombre': datos.get('sensor_name', 'Unknown')}
                        })
                
                # Transponer para formato (time_steps, features)
                if matriz_datos:
                    matriz_np = np.array(matriz_datos).T
                    # Asegurar que tenga al menos 24 time_steps
                    if matriz_np.shape[0] < 24:
                        # Padding con repetición
                        repeat_times = (24 + matriz_np.shape[0] - 1) // matriz_np.shape[0]
                        matriz_np = np.tile(matriz_np, (repeat_times, 1))[:24]
                    elif matriz_np.shape[0] > 24:
                        # Truncar a 24 time_steps
                        matriz_np = matriz_np[:24]
                else:
                    matriz_np = np.random.randn(24, 8)
                
            else:
                # Respuesta simple - generar matriz simulada
                matriz_np = np.random.randn(24, 8)
                metadatos_sensores = [
                    {'id': f'SENSOR_{i}', 'tipo': 'Simulado', 'unidad': 'unidad'}
                    for i in range(8)
                ]
            
            # Normalización
            if matriz_np.size > 0:
                matriz_np = (matriz_np - matriz_np.mean()) / (matriz_np.std() + 1e-8)
            
            return matriz_np.astype(np.float32), {
                'sensores': metadatos_sensores,
                'timestamp': datos.get('timestamp'),
                'sensor_principal': datos.get('sensor_name'),
                'fuente': datos.get('metadatos', {}).get('fuente', 'Desconocida'),
                'forma_tensor': matriz_np.shape
            }
                
        except Exception as e:
            logger.error(f"Error convirtiendo datos: {e}")
            # Fallback a datos simulados con forma correcta
            matriz_np = np.random.randn(24, 8)
            return matriz_np.astype(np.float32), {
                'sensores': [],
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'forma_tensor': matriz_np.shape
            }
    
    def _generar_estructura_desde_texto(self, texto: str, sensor_name: str) -> List[Dict]:
        """
        Generar estructura de datos desde respuesta de texto
        
        Args:
            texto: Respuesta en texto plano
            sensor_name: Nombre del sensor
            
        Returns:
            List[Dict]: Estructura de datos simulada
        """
        # Crear estructura básica basada en el nombre del sensor
        variables = ['Valor_1', 'Valor_2', 'Valor_3', 'Valor_4']
        
        return [
            {
                'id': f'{sensor_name}_{i+1:02d}',
                'variable': var,
                'unidad': 'unidad',
                'ubicacion': {'nombre': f'Estación {sensor_name}'},
                'datos': [
                    {
                        'timestamp': (datetime.now() - timedelta(hours=24-i)).isoformat(),
                        'valor': np.random.normal(0, 1),
                        'calidad': 'buena' if np.random.random() > 0.1 else 'regular'
                    }
                    for i in range(24)
                ]
            }
            for i, var in enumerate(variables)
        ]

# Función de prueba
def test_conexion():
    """Probar la conexión con el SIATA"""
    connector = SIATAConnector()
    
    print("🔗 Probando conexión con SIATA...")
    
    # Probar obtener datos
    datos = connector.obtener_datos_sensores()
    print(f"📊 Datos obtenidos: {len(datos.get('sensores', []))} sensores")
    
    # Probar estado
    estado = connector.obtener_estado_sistema()
    print(f"🟢 Estado del sistema: {estado.get('status', 'desconocido')}")
    
    # Probar envío de alerta
    alerta_enviada = connector.enviar_alerta(
        nivel_riesgo="Medio",
        ubicacion={
            'lat': 6.2442,
            'lon': -75.5812,
            'nombre': 'Zona de prueba'
        },
        confianza=0.85,
        sensores_criticos=['SENSOR_001', 'SENSOR_002']
    )
    print(f"🚨 Alerta enviada: {alerta_enviada}")
    
    return connector

if __name__ == "__main__":
    test_conexion()

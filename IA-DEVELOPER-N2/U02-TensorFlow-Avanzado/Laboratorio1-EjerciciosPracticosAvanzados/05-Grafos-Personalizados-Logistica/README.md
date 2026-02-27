# Ejercicio 5: Modelo con Capas Personalizadas para Datos de Grafos

## 🗺️ Caso de Uso: Optimización de Rutas en Logística

### Contexto Empresarial
En el sector logístico, optimizar rutas de entrega usando grafos de carreteras y restricciones (ej. tráfico, peajes, ventanas de tiempo) puede reducir costos operativos, mejorar tiempos de entrega y aumentar la satisfacción del cliente.

### Problema a Resolver
- **Optimización**: Encontrar la ruta más eficiente entre múltiples puntos
- **Restricciones**: Considerar tráfico, horarios, capacidad de vehículos
- **Costos**: Minimizar distancia, tiempo y costos operativos
- **ROI**: 10% reducción en costos de transporte

## 🎯 Objetivos de Aprendizaje

1. **Modelado de grafos complejos**:
   - Construir grafos con atributos en nodos (ej. capacidad de almacenes) y aristas (ej. distancia, tráfico)
   - Implementar **Graph Attention Networks (GAT)** desde cero

2. **Capas personalizadas para grafos**:
   - Implementar **Graph Attention Networks (GAT)** personalizadas
   - Integrar lógica de negocio específica

3. **Integración con sistemas de geolocalización**:
   - Usar APIs como **Google Maps** o **OpenStreetMap** para datos reales
   - Implementar optimización con OR-Tools de Google

## 📊 Dataset

### Opción 1: Datos Reales - OpenStreetMap
- **Descripción**: Datos de carreteras y nodos de ciudades reales
- **Descarga**: [GeoFabrik](https://download.geofabrik.de/)
- **Características**: Coordenadas, tipos de carreteras, límites de velocidad

### Opción 2: Dataset Sintético (para desarrollo)
- **Generación**: Simular red logística con almacenes y clientes
- **Ventajas**: Control sobre distribución y características

## 🛠️ Implementación

### Paso 1: Configuración del Entorno
```python
import tensorflow as tf
import spektral as sp
from spektral.layers import GraphConv
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import requests
import json
import osmnx as ox
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
```

### Paso 2: Capa GAT Personalizada
```python
class GraphAttentionLayer(sp.layers.GraphBase):
    """Capa de Graph Attention Network personalizada"""
    
    def __init__(self, units, num_heads=8, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Capas de transformación
        self.W_query = None
        self.W_key = None
        self.W_value = None
        
        # Capa de salida
        self.W_out = None
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def build(self, input_shape):
        # Inicializar pesos para Query, Key, Value
        self.W_query = self.add_weight(
            shape=(input_shape[0][-1], self.units * self.num_heads),
            initializer='glorot_uniform',
            name='W_query'
        )
        self.W_key = self.add_weight(
            shape=(input_shape[0][-1], self.units * self.num_heads),
            initializer='glorot_uniform',
            name='W_key'
        )
        self.W_value = self.add_weight(
            shape=(input_shape[0][-1], self.units * self.num_heads),
            initializer='glorot_uniform',
            name='W_value'
        )
        
        # Capa de salida
        self.W_out = self.add_weight(
            shape=(self.units * self.num_heads, self.units),
            initializer='glorot_uniform',
            name='W_out'
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        """
        Forward pass de GAT
        
        Args:
            inputs: Tupla (X, A) donde X son features de nodos y A es matriz de adyacencia
        """
        X, A = inputs
        
        batch_size = tf.shape(X)[0]
        num_nodes = tf.shape(X)[1]
        
        # Calcular Query, Key, Value
        Q = tf.matmul(X, self.W_query)  # (batch_size, num_nodes, units * num_heads)
        K = tf.matmul(X, self.W_key)    # (batch_size, num_nodes, units * num_heads)
        V = tf.matmul(X, self.W_value)  # (batch_size, num_nodes, units * num_heads)
        
        # Reshape para multi-head attention
        Q = tf.reshape(Q, (batch_size, num_nodes, self.num_heads, self.units))
        K = tf.reshape(K, (batch_size, num_nodes, self.num_heads, self.units))
        V = tf.reshape(V, (batch_size, num_nodes, self.num_heads, self.units))
        
        # Transponer para atención: (batch_size, num_heads, num_nodes, units)
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])
        
        # Calcular scores de atención
        scores = tf.matmul(Q, K, transpose_b=True)  # (batch_size, num_heads, num_nodes, num_nodes)
        scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))
        
        # Aplicar máscara de adyacencia
        A_expanded = tf.expand_dims(A, axis=1)  # (batch_size, 1, num_nodes, num_nodes)
        A_expanded = tf.tile(A_expanded, [1, self.num_heads, 1, 1])
        scores = scores + (A_expanded * -1e9)
        
        # Softmax para obtener pesos de atención
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Aplicar pesos a los valores
        context = tf.matmul(attention_weights, V)  # (batch_size, num_heads, num_nodes, units)
        
        # Concatenar heads
        context = tf.transpose(context, perm=[0, 2, 1, 3])  # (batch_size, num_nodes, num_heads, units)
        context = tf.reshape(context, (batch_size, num_nodes, self.num_heads * self.units))
        
        # Capa de salida
        output = tf.matmul(context, self.W_out)  # (batch_size, num_nodes, units)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return config

class CustomGNN(tf.keras.Model):
    """Modelo GNN personalizado para optimización de rutas"""
    
    def __init__(self, hidden_units=[128, 64], num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Capas GAT
        self.gat_layers = []
        for i, units in enumerate(hidden_units):
            self.gat_layers.append(GraphAttentionLayer(units, num_heads, dropout_rate))
        
        # Capas de clasificación/regresión
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense_layers = []
        for i, units in enumerate([32, 16]):
            self.dense_layers.append(tf.keras.layers.Dense(units, activation='relu'))
        
        # Capa de salida (predicción de costos de ruta)
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')
    
    def call(self, inputs, training=False):
        """
        Forward pass del modelo
        
        Args:
            inputs: Tupla (X, A) donde X son features de nodos y A es matriz de adyacencia
            training: Booleano para modo de entrenamiento
        """
        X, A = inputs
        
        # Pasar por capas GAT
        for i, gat_layer in enumerate(self.gat_layers):
            X = gat_layer([X, A])
            if training:
                X = self.dropout(X)
        
        # Global pooling
        X = tf.reduce_mean(X, axis=1)  # Global average pooling
        
        # Capas densas
        for dense_layer in self.dense_layers:
            X = dense_layer(X)
            if training:
                X = self.dropout(X)
        
        # Capa de salida
        output = self.output_layer(X)
        
        return output
```

### Paso 3: Generación de Grafo Logístico
```python
def generate_logistics_graph(num_warehouses=10, num_customers=50, num_routes=200):
    """Generar grafo logístico sintético"""
    
    # Crear grafo vacío
    G = nx.Graph()
    
    # Añadir nodos de almacenes
    warehouse_locations = []
    for i in range(num_warehouses):
        # Ubicación aleatoria (simulada)
        lat = np.random.uniform(40.0, 41.0)  # Latitud (ej: Madrid área)
        lon = np.random.uniform(-3.8, -3.5)  # Longitud
        
        capacity = np.random.randint(1000, 5000)  # Capacidad del almacén
        operating_cost = np.random.uniform(0.5, 2.0)  # Costo operativo por unidad
        
        G.add_node(f"warehouse_{i}", 
                  node_type='warehouse',
                  lat=lat,
                  lon=lon,
                  capacity=capacity,
                  operating_cost=operating_cost,
                  current_stock=np.random.randint(100, capacity))
        
        warehouse_locations.append((lat, lon))
    
    # Añadir nodos de clientes
    customer_locations = []
    for i in range(num_customers):
        lat = np.random.uniform(40.0, 41.0)
        lon = np.random.uniform(-3.8, -3.5)
        
        demand = np.random.randint(10, 200)  # Demanda del cliente
        priority = np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
        
        G.add_node(f"customer_{i}",
                  node_type='customer',
                  lat=lat,
                  lon=lon,
                  demand=demand,
                  priority=priority)
        
        customer_locations.append((lat, lon))
    
    # Añadir aristas (rutas)
    for i in range(num_routes):
        # Seleccionar nodos aleatorios
        node1 = np.random.choice(list(G.nodes()))
        node2 = np.random.choice(list(G.nodes()))
        
        while node2 == node1:
            node2 = np.random.choice(list(G.nodes()))
        
        # Calcular distancia (simulada)
        lat1, lon1 = G.nodes[node1]['lat'], G.nodes[node1]['lon']
        lat2, lon2 = G.nodes[node2]['lat'], G.nodes[node2]['lon']
        
        # Distancia euclidiana simplificada
        distance = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111000  # Aproximación a metros
        
        # Características de la ruta
        traffic_factor = np.random.uniform(0.8, 2.0)  # Factor de tráfico
        road_type = np.random.choice(['highway', 'arterial', 'local'], p=[0.2, 0.3, 0.5])
        travel_time = distance / (1000 / traffic_factor)  # Tiempo en minutos
        
        # Costo de la ruta
        if road_type == 'highway':
            cost_per_km = 0.5
        elif road_type == 'arterial':
            cost_per_km = 0.3
        else:
            cost_per_km = 0.2
        
        route_cost = (distance / 1000) * cost_per_km * traffic_factor
        
        G.add_edge(node1, node2,
                  distance=distance,
                  travel_time=travel_time,
                  route_cost=route_cost,
                  traffic_factor=traffic_factor,
                  road_type=road_type,
                  capacity=np.random.randint(10, 100))  # Capacidad de la ruta
    
    return G

def extract_graph_features_for_routing(G):
    """Extraer características del grafo para optimización de rutas"""
    
    # Features de nodos
    node_features = []
    node_ids = []
    
    for node in G.nodes():
        features = [
            G.nodes[node].get('lat', 0),
            G.nodes[node].get('lon', 0),
            G.nodes[node].get('capacity', 0) if G.nodes[node].get('node_type') == 'warehouse' else 0,
            G.nodes[node].get('demand', 0) if G.nodes[node].get('node_type') == 'customer' else 0,
            1 if G.nodes[node].get('node_type') == 'warehouse' else 0,
            1 if G.nodes[node].get('node_type') == 'customer' else 0,
            G.degree(node),  # Número de conexiones
            nx.betweenness_centrality(G)[node] if G.number_of_nodes() > 1 else 0,
        ]
        node_features.append(features)
        node_ids.append(node)
    
    # Matriz de adyacencia
    A = nx.adjacency_matrix(G).toarray()
    
    # Normalizar matriz de adyacencia
    A = sp.utils.normalized_adjacency_matrix(A)
    
    return np.array(node_features), A, node_ids, G

def create_routing_dataset(G, num_samples=1000):
    """Crear dataset de entrenamiento para optimización de rutas"""
    
    node_features, A, node_ids, graph = extract_graph_features_for_routing(G)
    
    # Generar pares de nodos (origen-destino)
    samples = []
    labels = []
    
    for _ in range(num_samples):
        # Seleccionar origen y destino aleatorios
        origin = np.random.choice(node_ids)
        destination = np.random.choice(node_ids)
        
        while destination == origin:
            destination = np.random.choice(node_ids)
        
        # Calcular características del par
        origin_idx = node_ids.index(origin)
        dest_idx = node_ids.index(destination)
        
        # Features del par
        pair_features = np.concatenate([
            node_features[origin_idx],
            node_features[dest_idx]
        ])
        
        # Calcular etiqueta (costo óptimo estimado)
        if graph.has_edge(origin, destination):
            optimal_cost = graph[origin][destination]['route_cost']
        else:
            # Si no hay conexión directa, estimar costo
            try:
                shortest_path = nx.shortest_path(graph, origin, destination, weight='route_cost')
                optimal_cost = sum(graph[shortest_path[i]][shortest_path[i+1]]['route_cost'] 
                                 for i in range(len(shortest_path)-1))
            except nx.NetworkXNoPath:
                optimal_cost = np.inf
        
        # Solo incluir si hay ruta válida
        if optimal_cost != np.inf:
            samples.append(pair_features)
            labels.append(optimal_cost)
    
    return np.array(samples), np.array(labels), node_features, A, node_ids
```

### Paso 4: Integración con OpenStreetMap
```python
def load_real_world_graph(city_name="Madrid, Spain", network_type="drive"):
    """Cargar grafo real de OpenStreetMap"""
    
    try:
        # Descargar grafo de la ciudad
        G = ox.graph_from_place(city_name, network_type=network_type)
        
        # Añadir información adicional a los nodos
        for node in G.nodes():
            G.nodes[node]['node_type'] = 'intersection'
            G.nodes[node]['lat'] = G.nodes[node]['y']
            G.nodes[node]['lon'] = G.nodes[node]['x']
        
        # Añadir información a las aristas
        for edge in G.edges(data=True):
            # Calcular distancia real
            edge_data = edge[2]
            edge_data['distance'] = edge_data.get('length', 0)
            edge_data['travel_time'] = edge_data.get('travel_time', edge_data['distance'] / 1000 * 60)  # Aproximación
            edge_data['route_cost'] = edge_data['distance'] / 1000 * 0.3  # Costo por km
            
            # Tipo de carretera
            edge_data['road_type'] = edge_data.get('highway', 'unclassified')
            
            # Límite de velocidad
            edge_data['speed_limit'] = edge_data.get('maxspeed', '50')
        
        print(f"Grafo cargado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
        return G
        
    except Exception as e:
        print(f"Error cargando grafo de OpenStreetMap: {e}")
        print("Usando grafo sintético en su lugar")
        return generate_logistics_graph()

def get_traffic_data_from_api(lat, lon, radius=1000):
    """Obtener datos de tráfico de una API (simulado)"""
    
    # En producción, usar APIs reales como:
    # - Google Maps Traffic Layer API
    # - Here Traffic API
    # - TomTom Traffic API
    
    # Simulación de datos de tráfico
    hour = pd.Timestamp.now().hour
    
    if 7 <= hour <= 9 or 17 <= hour <= 19:  # Horas pico
        traffic_factor = np.random.uniform(1.5, 2.5)
    elif 22 <= hour or hour <= 6:  # Horas valle
        traffic_factor = np.random.uniform(0.5, 0.8)
    else:  # Horas normales
        traffic_factor = np.random.uniform(0.9, 1.2)
    
    return {
        'traffic_factor': traffic_factor,
        'congestion_level': 'high' if traffic_factor > 1.5 else ('medium' if traffic_factor > 1.0 else 'low'),
        'estimated_delay': (traffic_factor - 1.0) * 10  # minutos
    }
```

### Paso 5: Optimización con OR-Tools
```python
class RouteOptimizer:
    """Optimizador de rutas usando OR-Tools"""
    
    def __init__(self, graph):
        self.graph = graph
        self.distance_matrix = self.create_distance_matrix()
        self.node_ids = list(graph.nodes())
    
    def create_distance_matrix(self):
        """Crear matriz de distancias desde el grafo"""
        
        n = len(self.graph.nodes())
        distance_matrix = np.zeros((n, n))
        
        for i, node1 in enumerate(self.node_ids):
            for j, node2 in enumerate(self.node_ids):
                if i == j:
                    distance_matrix[i][j] = 0
                elif self.graph.has_edge(node1, node2):
                    distance_matrix[i][j] = self.graph[node1][node2]['distance']
                else:
                    try:
                        path = nx.shortest_path(self.graph, node1, node2, weight='distance')
                        distance = sum(self.graph[path[k]][path[k+1]]['distance'] 
                                     for k in range(len(path)-1))
                        distance_matrix[i][j] = distance
                    except nx.NetworkXNoPath:
                        distance_matrix[i][j] = np.inf
        
        return distance_matrix
    
    def solve_tsp(self, start_node_idx=0):
        """Resolver problema del viajante (TSP)"""
        
        # Crear el gestor de rutas
        manager = pywrapcp.RoutingIndexManager(len(self.distance_matrix), 1, 0)
        
        # Crear el modelo de enrutamiento
        routing = pywrapcp.RoutingModel(manager)
        
        # Definir callback de distancia
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.distance_matrix[from_node][to_node])
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # Definir costo de arista
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Configurar parámetros de búsqueda
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(30)
        
        # Resolver el problema
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            # Extraer ruta
            index = routing.Start(0)
            route = []
            route_distance = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += solution.Value(routing.GetArcCostForVehicle(previous_index, index, 0))
            
            # Añadir nodo de regreso
            route.append(manager.IndexToNode(index))
            
            return route, route_distance
        else:
            return None, None
    
    def solve_vrp(self, depot_idx=0, num_vehicles=5):
        """Resolver problema de enrutamiento de vehículos (VRP)"""
        
        # Crear el gestor de rutas
        manager = pywrapcp.RoutingIndexManager(len(self.distance_matrix), num_vehicles, depot_idx)
        
        # Crear el modelo de enrutamiento
        routing = pywrapcp.RoutingModel(manager)
        
        # Definir callback de distancia
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.distance_matrix[from_node][to_node])
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # Definir costo de arista
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Añadir constraint de capacidad
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return self.graph.nodes[from_node].get('demand', 0)
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, 1000, True, 'Capacity')
        
        # Configurar parámetros de búsqueda
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(60)
        
        # Resolver el problema
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            # Extraer rutas para cada vehículo
            routes = []
            total_distance = 0
            
            for vehicle_id in range(num_vehicles):
                index = routing.Start(vehicle_id)
                route = []
                route_distance = 0
                
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route.append(node_index)
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += solution.Value(routing.GetArcCostForVehicle(previous_index, index, vehicle_id))
                
                routes.append(route)
                total_distance += route_distance
            
            return routes, total_distance
        else:
            return None, None
```

## 🧪 Ejecución Completa

```python
def main():
    """Función principal para ejecutar el ejercicio completo"""
    
    print("🗺️ EJERCICIO 5: OPTIMIZACIÓN DE RUTAS CON GAT PERSONALIZADO")
    print("=" * 70)
    
    # 1. Cargar grafo logístico
    print("📊 Cargando grafo logístico...")
    try:
        G = load_real_world_graph("Madrid, Spain", "drive")
    except:
        print("   Usando grafo sintético (OpenStreetMap no disponible)")
        G = generate_logistics_graph(num_warehouses=5, num_customers=20, num_routes=50)
    
    print(f"   Nodos: {G.number_of_nodes()}")
    print(f"   Aristas: {G.number_of_edges()}")
    
    # 2. Visualizar grafo
    print("\n🎨 Visualizando grafo logístico...")
    plt.figure(figsize=(15, 10))
    
    # Extraer posiciones para visualización
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Colorear nodos por tipo
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if G.nodes[node].get('node_type') == 'warehouse':
            node_colors.append('red')
            node_sizes.append(200)
        elif G.nodes[node].get('node_type') == 'customer':
            node_colors.append('blue')
            node_sizes.append(100)
        else:
            node_colors.append('green')
            node_sizes.append(50)
    
    # Colorear aristas por tipo de carretera
    edge_colors = []
    edge_widths = []
    for edge in G.edges(data=True):
        road_type = edge[2].get('road_type', 'unclassified')
        if road_type == 'highway':
            edge_colors.append('red')
            edge_widths.append(2)
        elif road_type == 'arterial':
            edge_colors.append('orange')
            edge_widths.append(1.5)
        else:
            edge_colors.append('gray')
            edge_widths.append(0.5)
    
    # Dibujar grafo
    nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors,
            width=edge_widths, with_labels=False, node_size=node_sizes, alpha=0.7)
    
    # Añadir leyenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Almacén'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label='Cliente'),
        Line2D([0], [0], color='red', linewidth=2, label='Autopista'),
        Line2D([0], [0], color='orange', linewidth=2, label='Arterial'),
        Line2D([0], [0], color='gray', linewidth=2, label='Local')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title('Grafo Logístico - Madrid')
    plt.show()
    
    # 3. Preparar datos para entrenamiento
    print("\n🔄 Preparando datos para entrenamiento...")
    X_train, y_train, node_features, A, node_ids = create_routing_dataset(G, num_samples=500)
    
    print(f"   Muestras de entrenamiento: {len(X_train)}")
    print(f"   Features de nodos: {node_features.shape}")
    print(f"   Matriz de adyacencia: {A.shape}")
    
    # 4. Construir y entrenar modelo GAT
    print("\n🏗️ Construyendo modelo GAT personalizado...")
    
    # Normalizar features
    scaler = StandardScaler()
    node_features_scaled = scaler.fit_transform(node_features)
    
    # Construir modelo
    gat_model = CustomGNN(hidden_units=[128, 64], num_heads=8, dropout_rate=0.1)
    
    # Compilar modelo
    gat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Entrenar modelo
    print("🚀 Entrenando modelo GAT...")
    history = gat_model.fit(
        [node_features_scaled, A],
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ],
        verbose=1
    )
    
    # 5. Optimización de rutas con OR-Tools
    print("\n🚀 Optimizando rutas con OR-Tools...")
    optimizer = RouteOptimizer(G)
    
    # Resolver TSP para una ruta de ejemplo
    print("   Resolviendo TSP (ruta óptima)...")
    tsp_route, tsp_distance = optimizer.solve_tsp(start_node_idx=0)
    
    if tsp_route:
        print(f"   Ruta TSP encontrada: {len(tsp_route)} nodos")
        print(f"   Distancia total: {tsp_distance:.2f} metros")
        print(f"   Nodos visitados: {[optimizer.node_ids[i] for i in tsp_route[:5]]}...")
    
    # 6. Evaluar modelo GAT
    print("\n📈 Evaluando modelo GAT...")
    
    # Predecir costos para pares de nodos
    test_pairs = []
    true_costs = []
    predicted_costs = []
    
    for _ in range(100):
        # Seleccionar par aleatorio
        origin_idx = np.random.randint(0, len(node_ids))
        dest_idx = np.random.randint(0, len(node_ids))
        
        while dest_idx == origin_idx:
            dest_idx = np.random.randint(0, len(node_ids))
        
        # Features del par
        pair_features = np.concatenate([
            node_features_scaled[origin_idx],
            node_features_scaled[dest_idx]
        ]).reshape(1, -1)
        
        # Predecir costo
        predicted_cost = gat_model.predict([node_features_scaled, A], verbose=0)[0][0]
        
        # Obtener costo real
        origin_node = node_ids[origin_idx]
        dest_node = node_ids[dest_idx]
        
        if G.has_edge(origin_node, dest_node):
            true_cost = G[origin_node][dest_node]['route_cost']
        else:
            true_cost = np.inf
        
        if true_cost != np.inf:
            test_pairs.append((origin_node, dest_node))
            true_costs.append(true_cost)
            predicted_costs.append(predicted_cost[0])
    
    # Calcular métricas
    if len(true_costs) > 0:
        mae = np.mean(np.abs(np.array(true_costs) - np.array(predicted_costs)))
        rmse = np.sqrt(np.mean((np.array(true_costs) - np.array(predicted_costs))**2))
        
        print(f"   MAE en predicción de costos: {mae:.2f}")
        print(f"   RMSE en predicción de costos: {rmse:.2f}")
        
        # Visualizar predicciones vs reales
        plt.figure(figsize=(10, 6))
        plt.scatter(true_costs, predicted_costs, alpha=0.6)
        plt.plot([0, max(true_costs)], [0, max(true_costs)], 'r--', alpha=0.8)
        plt.xlabel('Costo Real')
        plt.ylabel('Costo Predicho')
        plt.title('Predicción de Costos de Ruta')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # 7. Guardar modelo y artefactos
    print("\n💾 Guardando modelo y artefactos...")
    gat_model.save('gat_routing_model.h5')
    
    # Guardar scaler
    import joblib
    joblib.dump(scaler, 'routing_scaler.pkl')
    
    # Guardar grafo
    nx.write_gpickle(G, 'logistics_graph.gpickle')
    
    print("\n✅ EJERCICIO COMPLETADO")
    print("=" * 70)
    print("🎯 RESULTADOS:")
    print(f"   • MAE en predicción de costos: {mae:.2f}")
    print(f"   • RMSE en predicción de costos: {rmse:.2f}")
    print(f"   • Modelo guardado: gat_routing_model.h5")
    print(f"   • Grafo guardado: logistics_graph.gpickle")
    print(f"   • ROI estimado: 10% reducción en costos de transporte")
    print("=" * 70)

if __name__ == "__main__":
    main()
```

## 📊 Métricas de Evaluación

### Métricas de Predicción de Costos
- **MAE (Mean Absolute Error)**: Error absoluto medio en predicción de costos
- **RMSE (Root Mean Square Error)**: Raíz del error cuadrático medio
- **R² Score**: Coeficiente de determinación
- **Correlación**: Correlación entre costos predichos y reales

### Métricas de Optimización
- **Reducción de distancia**: Porcentaje de reducción en distancia total
- **Reducción de tiempo**: Porcentaje de reducción en tiempo de viaje
- **Eficiencia computacional**: Tiempo de convergencia del optimizador

## 🚀 Desafíos Adicionales

### Desafío 1: Optimización Multi-Objetivo
- Optimizar simultáneamente distancia, tiempo y costo
- Usar algoritmos genéticos o NSGA-II
- Implementar Pareto front

### Desafío 2: Rutas Dinámicas
- Actualizar rutas en tiempo real basado en tráfico
- Implementar recálculo dinámico de rutas
- Usar streaming de datos de tráfico

### Desafío 3: Integración Real
- Conectar con sistemas de gestión de flota
- Implementar dashboard de monitoreo
- Añadir sistema de alertas de retrasos

## 📝 Entrega Requerida

1. **Código fuente** completo y funcional
2. **Informe técnico** con:
   - Análisis del grafo logístico
   - Arquitectura del modelo GAT
   - Resultados de optimización
3. **Sistema de optimización** funcionando
4. **Dashboard** de visualización de rutas

## 🔗 Recursos Adicionales

- [OR-Tools Documentation](https://developers.google.com/optimization/routing)
- [OpenStreetMap](https://www.openstreetmap.org/)
- [OSMnx Library](https://osmnx.readthedocs.io/)
- [Graph Attention Networks Paper](https://arxiv.org/abs/1710.10903)

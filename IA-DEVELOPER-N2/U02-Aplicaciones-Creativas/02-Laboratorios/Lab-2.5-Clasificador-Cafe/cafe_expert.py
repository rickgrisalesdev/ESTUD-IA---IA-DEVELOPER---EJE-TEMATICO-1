import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class ClasificadorCafeMultiInput(Model):
    """Modelo multi-input para clasificación de calidad de café."""
    
    def __init__(self, num_quimicas=8, image_shape=(224, 224, 3), num_classes=3):
        super(ClasificadorCafeMultiInput, self).__init__()
        self.num_quimicas = num_quimicas
        self.image_shape = image_shape
        self.num_classes = num_classes
        
        # Rama para datos químicos (tabulares)
        self.build_quimica_branch()
        
        # Rama para imágenes de granos
        self build_image_branch()
        
        # Capa de fusión y clasificación final
        self.build_fusion_layer()
        
    def build_quimica_branch(self):
        """Construye la rama para datos químicos tabulares."""
        self.quimica_input = Input(shape=(self.num_quimicas,), name='input_quimico')
        
        # Capas densas para procesar datos químicos
        self.dense_q1 = layers.Dense(64, activation='relu', name='dense_q1')
        self.batch_q1 = layers.BatchNormalization(name='batch_q1')
        self.dropout_q1 = layers.Dropout(0.3, name='dropout_q1')
        
        self.dense_q2 = layers.Dense(32, activation='relu', name='dense_q2')
        self.batch_q2 = layers.BatchNormalization(name='batch_q2')
        self.dropout_q2 = layers.Dropout(0.3, name='dropout_q2')
        
        self.dense_q3 = layers.Dense(16, activation='relu', name='dense_q3')
        
    def build_image_branch(self):
        """Construye la rama para imágenes de granos de café."""
        self.image_input = Input(shape=self.image_shape, name='input_imagen')
        
        # Usar ResNet50 pre-entrenada como base (transfer learning)
        self.base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.image_shape,
            name='resnet_base'
        )
        
        # Congelar las capas base inicialmente
        for layer in self.base_model.layers:
            layer.trainable = False
            
        # Capas adicionales para procesar características visuales
        self.global_avg_pool = layers.GlobalAveragePooling2D(name='global_avg_pool')
        self.dense_i1 = layers.Dense(128, activation='relu', name='dense_i1')
        self.batch_i1 = layers.BatchNormalization(name='batch_i1')
        self.dropout_i1 = layers.Dropout(0.3, name='dropout_i1')
        
        self.dense_i2 = layers.Dense(64, activation='relu', name='dense_i2')
        
    def build_fusion_layer(self):
        """Construye la capa de fusión de ambas ramas."""
        # Concatenar características de ambas ramas
        self.concatenated = layers.Concatenate(name='concatenated')([
            self.dense_q3.output, 
            self.dense_i2.output
        ])
        
        # Capas de fusión
        self.dense_f1 = layers.Dense(64, activation='relu', name='dense_f1')
        self.batch_f1 = layers.BatchNormalization(name='batch_f1')
        self.dropout_f1 = layers.Dropout(0.4, name='dropout_f1')
        
        self.dense_f2 = layers.Dense(32, activation='relu', name='dense_f2')
        self.batch_f2 = layers.BatchNormalization(name='batch_f2')
        self.dropout_f2 = layers.Dropout(0.3, name='dropout_f2')
        
        # Capa de salida
        self.output_layer = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='output'
        )
        
    def call(self, inputs, training=None):
        quimica_input, image_input = inputs
        
        # Procesar rama química
        x_q = self.dense_q1(quimica_input)
        x_q = self.batch_q1(x_q, training=training)
        x_q = self.dropout_q1(x_q, training=training)
        
        x_q = self.dense_q2(x_q)
        x_q = self.batch_q2(x_q, training=training)
        x_q = self.dropout_q2(x_q, training=training)
        
        x_q = self.dense_q3(x_q)
        
        # Procesar rama de imagen
        x_i = self.base_model(image_input, training=training)
        x_i = self.global_avg_pool(x_i)
        x_i = self.dense_i1(x_i)
        x_i = self.batch_i1(x_i, training=training)
        x_i = self.dropout_i1(x_i, training=training)
        
        x_i = self.dense_i2(x_i)
        
        # Fusionar ambas ramas
        x_f = layers.Concatenate()([x_q, x_i])
        x_f = self.dense_f1(x_f)
        x_f = self.batch_f1(x_f, training=training)
        x_f = self.dropout_f1(x_f, training=training)
        
        x_f = self.dense_f2(x_f)
        x_f = self.batch_f2(x_f, training=training)
        x_f = self.dropout_f2(x_f, training=training)
        
        # Salida final
        output = self.output_layer(x_f)
        
        return output
    
    def build_model(self):
        """Construye el modelo completo con Functional API."""
        # Definir entradas
        quimica_input = Input(shape=(self.num_quimicas,), name='input_quimico')
        image_input = Input(shape=self.image_shape, name='input_imagen')
        
        # Procesar rama química
        x_q = self.dense_q1(quimica_input)
        x_q = self.batch_q1(x_q)
        x_q = self.dropout_q1(x_q)
        
        x_q = self.dense_q2(x_q)
        x_q = self.batch_q2(x_q)
        x_q = self.dropout_q2(x_q)
        
        x_q = self.dense_q3(x_q)
        
        # Procesar rama de imagen
        x_i = self.base_model(image_input)
        x_i = self.global_avg_pool(x_i)
        x_i = self.dense_i1(x_i)
        x_i = self.batch_i1(x_i)
        x_i = self.dropout_i1(x_i)
        
        x_i = self.dense_i2(x_i)
        
        # Fusionar ambas ramas
        x_f = layers.Concatenate()([x_q, x_i])
        x_f = self.dense_f1(x_f)
        x_f = self.batch_f1(x_f)
        x_f = self.dropout_f1(x_f)
        
        x_f = self.dense_f2(x_f)
        x_f = self.batch_f2(x_f)
        x_f = self.dropout_f2(x_f)
        
        # Salida final
        output = self.output_layer(x_f)
        
        # Crear modelo
        model = Model(
            inputs=[quimica_input, image_input],
            outputs=output,
            name='ClasificadorCafeMultiInput'
        )
        
        return model
    
    def fine_tune_unfreeze(self, unfreeze_from=100):
        """Descongela capas para fine-tuning."""
        # Descongelar capas desde una capa específica
        for layer in self.base_model.layers[unfreeze_from:]:
            layer.trainable = True
            
        print(f"🔧 Descongeladas {len(self.base_model.layers) - unfreeze_from} capas para fine-tuning")

def generar_datos_sinteticos(n_samples=500):
    """Genera datos sintéticos para café del suroeste antioqueño."""
    np.random.seed(42)
    
    # Datos químicos simulados
    # Variables: pH, Acidez, Cuerpo, Aroma, Sabor, Dulzor, Amargura, Aftertaste
    datos_quimicos = []
    
    for i in range(n_samples):
        # Asignar clase aleatoriamente (0: Baja, 1: Media, 2: Alta calidad)
        calidad = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
        
        if calidad == 0:  # Baja calidad
            ph = np.random.normal(5.8, 0.3)
            acidez = np.random.normal(6.5, 0.8)
            cuerpo = np.random.normal(5.0, 0.7)
            aroma = np.random.normal(5.5, 0.9)
            sabor = np.random.normal(5.2, 0.8)
            dulzor = np.random.normal(4.8, 0.6)
            amargura = np.random.normal(7.5, 1.0)
            aftertaste = np.random.normal(5.0, 0.7)
            
        elif calidad == 1:  # Media calidad
            ph = np.random.normal(6.2, 0.2)
            acidez = np.random.normal(7.2, 0.5)
            cuerpo = np.random.normal(6.8, 0.5)
            aroma = np.random.normal(7.0, 0.6)
            sabor = np.random.normal(7.1, 0.5)
            dulzor = np.random.normal(6.5, 0.4)
            amargura = np.random.normal(5.5, 0.7)
            aftertaste = np.random.normal(6.8, 0.5)
            
        else:  # Alta calidad
            ph = np.random.normal(6.5, 0.15)
            acidez = np.random.normal(8.2, 0.3)
            cuerpo = np.random.normal(8.5, 0.3)
            aroma = np.random.normal(8.8, 0.3)
            sabor = np.random.normal(8.9, 0.2)
            dulzor = np.random.normal(8.2, 0.3)
            amargura = np.random.normal(3.5, 0.4)
            aftertaste = np.random.normal(8.7, 0.2)
        
        datos_quimicos.append([ph, acidez, cuerpo, aroma, sabor, dulzor, amargura, aftertaste])
    
    # Generar imágenes sintéticas (simuladas como arrays aleatorios)
    imagenes = np.random.rand(n_samples, 224, 224, 3) * 255
    
    # Añadir patrones visuales según la calidad
    for i in range(n_samples):
        calidad = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
        if calidad == 2:  # Alta calidad - granos más uniformes
            imagenes[i] += np.random.rand(224, 224, 3) * 20
        elif calidad == 0:  # Baja calidad - más variación
            imagenes[i] += np.random.rand(224, 224, 3) * 60
    
    imagenes = np.clip(imagenes, 0, 255).astype(np.uint8)
    
    return np.array(datos_quimicos), imagenes

def cargar_datos_reales():
    """Carga datos desde archivos (simulación)."""
    try:
        # Intentar cargar datos reales si existen
        datos_quimicos = pd.read_csv('muestras_lab.txt', sep='|')
        print("✅ Datos reales cargados desde muestras_lab.txt")
        return datos_quimicos.values, None  # Imágenes se cargarían por separado
    except FileNotFoundError:
        print("📊 Usando datos sintéticos (muestras_lab.txt no encontrado)")
        return generar_datos_sinteticos()

def preprocesar_datos(datos_quimicos, imagenes):
    """Preprocesa los datos para el modelo."""
    # Normalizar datos químicos
    scaler = StandardScaler()
    quimicos_normalizados = scaler.fit_transform(datos_quimicos)
    
    # Normalizar imágenes
    imagenes_normalizadas = imagenes / 255.0
    
    return quimicos_normalizados, imagenes_normalizadas, scaler

def visualizar_resultados(history, y_true, y_pred, class_names):
    """Visualiza métricas y resultados del modelo."""
    plt.figure(figsize=(15, 10))
    
    # 1. Curvas de entrenamiento
    plt.subplot(2, 3, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    # 2. Matriz de confusión
    plt.subplot(2, 3, 3)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    
    # 3. Distribución de características químicas
    plt.subplot(2, 3, 4)
    feature_names = ['pH', 'Acidez', 'Cuerpo', 'Aroma', 'Sabor', 'Dulzor', 'Amargura', 'Aftertaste']
    for i in range(min(4, len(feature_names))):
        plt.hist(datos_quimicos[:, i], alpha=0.7, label=feature_names[i], bins=20)
    plt.title('Distribución de Características Químicas')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.legend()
    
    # 4. Muestras de imágenes por clase
    plt.subplot(2, 3, 5)
    if imagenes is not None:
        # Mostrar 3 imágenes aleatorias
        indices = np.random.choice(len(imagenes), 3, replace=False)
        for i, idx in enumerate(indices):
            plt.subplot(1, 3, i+1)
            plt.imshow(imagenes[idx])
            plt.title(f'Muestra {idx}')
            plt.axis('off')
    
    # 5. Importancia de características (simulada)
    plt.subplot(2, 3, 6)
    importancia = np.random.rand(len(feature_names))
    plt.barh(feature_names, importancia)
    plt.title('Importancia de Características')
    plt.xlabel('Importancia')
    
    plt.tight_layout()
    plt.show()

def main():
    """Función principal del clasificador de café."""
    print("☕ Clasificador de Calidad de Café (AgroTech Antioquia)")
    print("=" * 60)
    
    # Cargar datos
    print("📊 Cargando datos de muestras de café...")
    datos_quimicos, imagenes = cargar_datos_reales()
    
    if imagenes is None:
        datos_quimicos, imagenes = generar_datos_sinteticos(500)
    
    print(f"📈 Datos químicos: {datos_quimicos.shape}")
    print(f"🖼️  Imágenes: {imagenes.shape}")
    
    # Preprocesar datos
    print("🔧 Preprocesando datos...")
    X_quim, X_img, scaler = preprocesar_datos(datos_quimicos, imagenes)
    
    # Generar etiquetas (simuladas basadas en características)
    y = np.array([1 if x[1] > 7.0 else (2 if x[1] > 6.0 else 0) for x in datos_quimicos])
    
    # Dividir datos
    X_quim_train, X_quim_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
        X_quim, X_img, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🎯 Distribución de clases - Train: {np.bincount(y_train)}")
    print(f"🎯 Distribución de clases - Test: {np.bincount(y_test)}")
    
    # Crear modelo
    print("🧠 Construyendo modelo multi-input...")
    clasificador = ClasificadorCafeMultiInput(
        num_quimicas=X_quim.shape[1],
        image_shape=X_img.shape[1:],
        num_classes=3
    )
    
    model = clasificador.build_model()
    
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Resumen del modelo
    print("\n📋 Arquitectura del Modelo:")
    model.summary()
    
    # Entrenamiento inicial
    print("\n🏋️  Iniciando entrenamiento inicial...")
    history = model.fit(
        [X_quim_train, X_img_train], y_train,
        validation_data=([X_quim_test, X_img_test], y_test),
        epochs=10,
        batch_size=16,
        verbose=1
    )
    
    # Fine-tuning
    print("\n🔧 Iniciando fine-tuning...")
    clasificador.fine_tune_unfreeze(unfreeze_from=100)
    
    # Recompilar con learning rate más bajo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continuar entrenamiento
    history_fine = model.fit(
        [X_quim_train, X_img_train], y_train,
        validation_data=([X_quim_test, X_img_test], y_test),
        epochs=5,
        batch_size=16,
        verbose=1
    )
    
    # Evaluar modelo
    print("\n📊 Evaluando modelo final...")
    test_loss, test_acc = model.evaluate([X_quim_test, X_img_test], y_test, verbose=0)
    print(f"🎯 Precisión final: {test_acc:.4f}")
    
    # Predicciones
    y_pred_probs = model.predict([X_quim_test, X_img_test])
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Reporte de clasificación
    class_names = ['Baja Calidad', 'Media Calidad', 'Alta Calidad']
    print("\n📋 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Visualizar resultados
    print("\n📈 Generando visualizaciones...")
    visualizar_resultados(history_fine, y_test, y_pred, class_names)
    
    # Ejemplo de predicción
    print("\n🔍 Ejemplo de predicción:")
    sample_idx = np.random.choice(len(X_quim_test))
    sample_quim = np.expand_dims(X_quim_test[sample_idx], axis=0)
    sample_img = np.expand_dims(X_img_test[sample_idx], axis=0)
    
    pred = model.predict([sample_quim, sample_img], verbose=0)
    pred_class = np.argmax(pred[0])
    confidence = pred[0][pred_class]
    
    print(f"🎯 Muestra {sample_idx}:")
    print(f"   Calidad real: {class_names[y_test[sample_idx]]}")
    print(f"   Calidad predicha: {class_names[pred_class]}")
    print(f"   Confianza: {confidence:.3f}")
    
    print("\n✅ Clasificador de café implementado exitosamente!")
    print("🚀 Listo para despliegue en cooperativas del suroeste antioqueño")

if __name__ == "__main__":
    main()

"""
Sistema de Alerta Temprana - SIATA (Versión Simplificada)
Interfaz web para el Centinela del Valle
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from modelo_laderas import ModeloLaderas, generar_datos_sinteticos
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Alerta Temprana - SIATA",
    page_icon="🏔️",
    layout="wide"
)

# Título principal
st.title("🏔️ Sistema de Alerta Temprana de Deslizamientos")
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("🎯 Panel de Control")
opcion = st.sidebar.selectbox(
    "Seleccionar Módulo:",
    ["📊 Monitoreo en Tiempo Real", "🤖 Modelo IA", "📈 Análisis de Datos", "⚙️ Configuración"]
)

if opcion == "📊 Monitoreo en Tiempo Real":
    st.header("📊 Monitoreo de Sensores en Tiempo Real")
    
    # Generar datos simulados
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Sensores de Humedad")
        humedad_data = np.random.normal(0.5, 0.2, 100)
        fig_humedad = px.line(
            x=list(range(100)),
            y=humedad_data,
            title="Niveles de Humedad (%)",
            labels={"x": "Tiempo", "y": "Humedad (%)"}
        )
        st.plotly_chart(fig_humedad, width="stretch")
        
    with col2:
        st.subheader("📐 Sensores de Inclinación")
        inclinacion_data = np.random.normal(0.3, 0.1, 100)
        fig_inclinacion = px.line(
            x=list(range(100)),
            y=inclinacion_data,
            title="Ángulos de Inclinación (°)",
            labels={"x": "Tiempo", "y": "Inclinación (°)"}
        )
        st.plotly_chart(fig_inclinacion, width="stretch")
    
    # Alertas
    st.subheader("🚨 Estado de Alertas")
    
    # Simular estados de alerta
    alerta_humedad = np.random.choice(["Normal", "Precaución", "Crítico"])
    alerta_inclinacion = np.random.choice(["Normal", "Precaución", "Crítico"])
    
    col_alert1, col_alert2 = st.columns(2)
    
    with col_alert1:
        if alerta_humedad == "Crítico":
            st.error("🚨 ¡ALERTA CRÍTICA DE HUMEDAD!")
        elif alerta_humedad == "Precaución":
            st.warning("⚠️ Nivel de humedad elevado")
        else:
            st.success("✅ Nivel de humedad normal")
            
    with col_alert2:
        if alerta_inclinacion == "Crítico":
            st.error("🚨 ¡ALERTA CRÍTICA DE INCLINACIÓN!")
        elif alerta_inclinacion == "Precaución":
            st.warning("⚠️ Inclinación elevada")
        else:
            st.success("✅ Inclinación normal")

elif opcion == "🤖 Modelo IA":
    st.header("🤖 Modelo de Inteligencia Artificial - Centinela del Valle")
    
    # Cargar y entrenar modelo
    if st.button("🚀 Entrenar Modelo"):
        with st.spinner("Entrenando modelo..."):
            # Generar datos
            X_train, y_train = generar_datos_sinteticos(500)
            X_test, y_test = generar_datos_sinteticos(100)
            
            # Convertir etiquetas
            y_train_onehot = tf.keras.utils.to_categorical(y_train, 3)
            y_test_onehot = tf.keras.utils.to_categorical(y_test, 3)
            
            # Crear y entrenar modelo
            modelo = ModeloLaderas(num_features=X_train.shape[2])
            modelo.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Entrenamiento rápido
            history = modelo.fit(
                X_train, y_train_onehot,
                epochs=5,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            st.success("✅ Modelo entrenado exitosamente!")
            
            # Mostrar métricas
            loss, accuracy = modelo.evaluate(X_test, y_test_onehot, verbose=0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📉 Pérdida", f"{loss:.4f}")
            with col2:
                st.metric("🎯 Precisión", f"{accuracy:.4f}")
            
            # Visualizar predicciones
            st.subheader("📊 Predicciones del Modelo")
            
            # Hacer predicciones
            predictions = modelo.predict(X_test[:5])
            
            for i in range(5):
                pred_class = np.argmax(predictions[i])
                true_class = y_test[i]
                
                class_names = ['Bajo', 'Medio', 'Alto']
                
                col_pred, col_true = st.columns(2)
                with col_pred:
                    st.write(f"**Predicción:** {class_names[pred_class]}")
                with col_true:
                    st.write(f"**Real:** {class_names[true_class]}")
                
                # Barra de probabilidad
                proba = predictions[i]
                fig = go.Figure(data=[
                    go.Bar(x=class_names, y=proba, marker_color=['green', 'yellow', 'red'])
                ])
                fig.update_layout(title=f"Probabilidades - Muestra {i+1}")
                st.plotly_chart(fig, width="stretch")

elif opcion == "📈 Análisis de Datos":
    st.header("📈 Análisis Histórico de Datos")
    
    # Generar datos históricos simulados
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
    humedad_hist = np.random.normal(0.5, 0.2, len(dates))
    inclinacion_hist = np.random.normal(0.3, 0.1, len(dates))
    
    # Crear DataFrame
    df_hist = pd.DataFrame({
        'Fecha': dates,
        'Humedad': humedad_hist,
        'Inclinación': inclinacion_hist
    })
    
    # Gráfico de series temporales
    st.subheader("📊 Tendencias Históricas")
    fig_hist = px.line(
        df_hist, 
        x='Fecha', 
        y=['Humedad', 'Inclinación'],
        title="Evolución de Sensores - Enero 2024",
        labels={"value": "Valor", "variable": "Sensor"}
    )
    st.plotly_chart(fig_hist, width="stretch")
    
    # Estadísticas
    st.subheader("📈 Estadísticas Descriptivas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Humedad**")
        st.write(f"- Promedio: {np.mean(humedad_hist):.3f}")
        st.write(f"- Máximo: {np.max(humedad_hist):.3f}")
        st.write(f"- Mínimo: {np.min(humedad_hist):.3f}")
        st.write(f"- Desviación: {np.std(humedad_hist):.3f}")
        
    with col2:
        st.write("**Inclinación**")
        st.write(f"- Promedio: {np.mean(inclinacion_hist):.3f}")
        st.write(f"- Máximo: {np.max(inclinacion_hist):.3f}")
        st.write(f"- Mínimo: {np.min(inclinacion_hist):.3f}")
        st.write(f"- Desviación: {np.std(inclinacion_hist):.3f}")

elif opcion == "⚙️ Configuración":
    st.header("⚙️ Configuración del Sistema")
    
    # Parámetros configurables
    st.subheader("🎛️ Umbrales de Alerta")
    
    umbral_humedad = st.slider(
        "Umbral de Humedad (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )
    
    umbral_inclinacion = st.slider(
        "Umbral de Inclinación (°)",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05
    )
    
    st.subheader("🔄 Frecuencia de Actualización")
    
    frecuencia = st.selectbox(
        "Frecuencia de monitoreo:",
        ["Tiempo Real", "Cada 5 segundos", "Cada 30 segundos", "Cada minuto"]
    )
    
    # Guardar configuración
    if st.button("💾 Guardar Configuración"):
        config = {
            "umbral_humedad": umbral_humedad,
            "umbral_inclinacion": umbral_inclinacion,
            "frecuencia": frecuencia,
            "fecha_actualizacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.success("✅ Configuración guardada exitosamente!")
        st.json(config)

# Footer
st.markdown("---")
st.markdown("""
**🏔️ Sistema de Alerta Temprana - SIATA**  
*Centinela del Valle: Sistema inteligente para detección temprana de deslizamientos*  

📍 **Ubicación:** Medellín, Colombia  
🔧 **Tecnología:** TensorFlow + Streamlit  
📊 **Datos:** Sensores de humedad, inclinación y vibración  
🚨 **Alertas:** Detección automática de condiciones críticas
""")

st.markdown("---")
st.markdown("*Desarrollado como parte del Ejercicio 1: El Centinela del Valle*")

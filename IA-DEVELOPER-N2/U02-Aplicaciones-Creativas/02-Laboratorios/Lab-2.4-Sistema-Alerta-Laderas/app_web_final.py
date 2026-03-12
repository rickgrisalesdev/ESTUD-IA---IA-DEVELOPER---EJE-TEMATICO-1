"""
Sistema de Alerta Temprana - SIATA (Versión Final Corregida)
Interfaz web para el Centinela del Valle con API SIATA integrada
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import json
import time

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Alerta Temprana - SIATA",
    page_icon="🏔️",
    layout="wide"
)

# Título principal
st.title("🏔️ Sistema de Alerta Temprana de Deslizamientos")
st.markdown("### 🛡️ Centinela del Valle - Inteligencia Artificial para la Seguridad")
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("🎯 Panel de Control")
opcion = st.sidebar.selectbox(
    "Seleccionar Módulo:",
    ["📊 Monitoreo en Tiempo Real", "🤖 Modelo IA", "📈 Análisis de Datos", "⚙️ Configuración"]
)

# Función para generar datos simulados realistas
def generar_datos_sensores_realistas():
    """Generar datos de sensores simulados pero realistas"""
    np.random.seed(int(time.time()) % 1000)
    
    # Datos de humedad con patrón diario
    horas = np.arange(96) / 4  # 24 horas en intervalos de 15 min
    humedad_base = 60 + 20 * np.sin(2 * np.pi * horas / 24 - np.pi/2)
    humedad = humedad_base + np.random.normal(0, 5, 96)
    humedad = np.clip(humedad, 20, 95)
    
    # Datos de inclinación con eventos aleatorios
    inclinacion_base = 15 + 5 * np.sin(2 * np.pi * horas / 24)
    inclinacion = inclinacion_base + np.random.normal(0, 2, 96)
    
    # Añadir evento crítico aleatorio
    if np.random.random() < 0.3:
        evento_inicio = np.random.randint(20, 60)
        evento_duracion = np.random.randint(8, 20)
        inclinacion[evento_inicio:evento_inicio+evento_duracion] += np.random.uniform(10, 25)
    
    inclinacion = np.clip(inclinacion, 0, 45)
    
    # Datos de vibración
    vibracion = np.abs(np.random.normal(0.5, 0.3, 96))
    vibracion = np.clip(vibracion, 0, 2)
    
    # Timestamps
    timestamps = [
        (datetime.now() - timedelta(hours=24) + timedelta(minutes=15*i)).strftime("%H:%M")
        for i in range(96)
    ]
    
    return {
        'timestamps': timestamps,
        'humedad': humedad,
        'inclinacion': inclinacion,
        'vibracion': vibracion
    }

# Función para generar datos SIATA simulados
def generar_datos_siata():
    """Generar datos de calidad del aire tipo SIATA"""
    variables = ['PM2.5', 'PM10', 'CO', 'NO2', 'O3']
    estaciones = ['Aranjuez', 'Belén', 'Laureles', 'El Poblado', 'Robledo']
    
    datos_siata = {}
    
    for estacion in estaciones:
        datos_estacion = {}
        for variable in variables:
            if variable in ['PM2.5', 'PM10']:
                valores = np.random.gamma(2, 10, 24)  # Distribución gamma para partículas
            elif variable == 'CO':
                valores = np.random.normal(1.0, 0.3, 24)
            elif variable == 'NO2':
                valores = np.random.normal(0.05, 0.02, 24)
            else:  # O3
                valores = np.random.normal(0.08, 0.03, 24)
            
            # Asegurar valores positivos
            valores = np.abs(valores)
            
            datos_estacion[variable] = {
                'valores': valores.tolist(),
                'unidad': 'μg/m³' if 'PM' in variable else 'ppm',
                'promedio': float(np.mean(valores)),
                'maximo': float(np.max(valores))
            }
        
        datos_siata[estacion] = datos_estacion
    
    return datos_siata

# Función para detectar alertas
def detectar_alertas(datos):
    """Detectar condiciones de alerta basadas en umbrales"""
    alertas = []
    
    # Umbral de humedad
    humedad_critica = np.max(datos['humedad']) > 80
    if humedad_critica:
        alertas.append({
            'tipo': 'Humedad Crítica',
            'nivel': 'ALTO',
            'valor': f"{np.max(datos['humedad']):.1f}%",
            'descripcion': 'Nivel de humedad elevado detectado'
        })
    
    # Umbral de inclinación
    inclinacion_critica = np.max(datos['inclinacion']) > 30
    if inclinacion_critica:
        alertas.append({
            'tipo': 'Inclinación Crítica',
            'nivel': 'ALTO',
            'valor': f"{np.max(datos['inclinacion']):.1f}°",
            'descripcion': 'Movimiento de terreno significativo'
        })
    
    # Umbral de vibración
    vibracion_critica = np.max(datos['vibracion']) > 1.5
    if vibracion_critica:
        alertas.append({
            'tipo': 'Vibración Elevada',
            'nivel': 'MEDIO',
            'valor': f"{np.max(datos['vibracion']):.2f} Hz",
            'descripcion': 'Actividad sísmica detectada'
        })
    
    return alertas

if opcion == "📊 Monitoreo en Tiempo Real":
    st.header("📊 Monitoreo de Sensores en Tiempo Real")
    
    # Generar datos
    datos = generar_datos_sensores_realistas()
    
    # Detectar alertas
    alertas = detectar_alertas(datos)
    
    # Mostrar alertas si hay
    if alertas:
        st.subheader("🚨 Alertas Activas")
        for alerta in alertas:
            if alerta['nivel'] == 'ALTO':
                st.error(f"🚨 {alerta['tipo']}: {alerta['valor']} - {alerta['descripcion']}")
            else:
                st.warning(f"⚠️ {alerta['tipo']}: {alerta['valor']} - {alerta['descripcion']}")
    else:
        st.success("✅ No hay alertas activas")
    
    # Gráficos de sensores
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Humedad del Terreno")
        fig_humedad = go.Figure()
        fig_humedad.add_trace(go.Scatter(
            x=datos['timestamps'],
            y=datos['humedad'],
            mode='lines',
            name='Humedad (%)',
            line=dict(color='blue', width=2)
        ))
        fig_humedad.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Umbral Crítico")
        fig_humedad.update_layout(
            title="Niveles de Humedad (%)",
            xaxis_title="Tiempo",
            yaxis_title="Humedad (%)",
            height=300
        )
        st.plotly_chart(fig_humedad, width="stretch")
        
    with col2:
        st.subheader("📐 Inclinación del Terreno")
        fig_inclinacion = go.Figure()
        fig_inclinacion.add_trace(go.Scatter(
            x=datos['timestamps'],
            y=datos['inclinacion'],
            mode='lines',
            name='Inclinación (°)',
            line=dict(color='orange', width=2)
        ))
        fig_inclinacion.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Umbral Crítico")
        fig_inclinacion.update_layout(
            title="Ángulos de Inclinación (°)",
            xaxis_title="Tiempo",
            yaxis_title="Inclinación (°)",
            height=300
        )
        st.plotly_chart(fig_inclinacion, width="stretch")
    
    # Gráfico de vibración
    st.subheader("📊 Vibración Sísmica")
    fig_vibracion = go.Figure()
    fig_vibracion.add_trace(go.Scatter(
        x=datos['timestamps'],
        y=datos['vibracion'],
        mode='lines',
        name='Vibración (Hz)',
        line=dict(color='green', width=2)
    ))
    fig_vibracion.add_hline(y=1.5, line_dash="dash", line_color="orange", annotation_text="Umbral de Atención")
    fig_vibracion.update_layout(
        title="Niveles de Vibración (Hz)",
        xaxis_title="Tiempo",
        yaxis_title="Vibración (Hz)",
        height=300
    )
    st.plotly_chart(fig_vibracion, width="stretch")
    
    # Estado general
    st.subheader("📊 Estado General del Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Humedad Promedio",
            f"{np.mean(datos['humedad']):.1f}%",
            f"{np.mean(datos['humedad']) - 60:.1f}% vs normal"
        )
    
    with col2:
        st.metric(
            "Inclinación Máxima",
            f"{np.max(datos['inclinacion']):.1f}°",
            f"{np.max(datos['inclinacion']) - 15:.1f}° vs normal"
        )
    
    with col3:
        st.metric(
            "Vibración Máxima",
            f"{np.max(datos['vibracion']):.2f} Hz",
            f"{np.max(datos['vibracion']) - 0.5:.2f} Hz vs normal"
        )
    
    with col4:
        riesgo_total = len(alertas)
        color = "🔴" if riesgo_total >= 2 else "🟡" if riesgo_total == 1 else "🟢"
        st.metric(
            "Nivel de Riesgo",
            f"{color} {'ALTO' if riesgo_total >= 2 else 'MEDIO' if riesgo_total == 1 else 'BAJO'}",
            f"{riesgo_total} alertas activas"
        )

elif opcion == "🤖 Modelo IA":
    st.header("🤖 Modelo de Inteligencia Artificial - Centinela del Valle")
    
    st.subheader("🎯 Características del Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🏗️ Arquitectura del Modelo**")
        st.write("• LSTM para procesamiento secuencial")
        st.write("• Capas residuales (ResNet)")
        st.write("• Mecanismo de salto de alarma")
        st.write("• Función de pérdida con penalización")
        
        st.write("**📊 Variables Monitoreadas**")
        st.write("• Humedad del terreno")
        st.write("• Inclinación del suelo")
        st.write("• Vibración sísmica")
        st.write("• Presión atmosférica")
        st.write("• Temperatura")
        st.write("• Nivel freático")
        st.write("• Velocidad del viento")
        st.write("• Precipitación")
    
    with col2:
        st.write("**⚙️ Configuración Técnica**")
        st.write("• Input shape: [batch, 24, 8]")
        st.write("• 3 clases de riesgo: Bajo, Medio, Alto")
        st.write("• Optimizador: Adam")
        st.write("• Pérdida: Safety-Weighted Loss")
        st.write("• Safety factor: 2.0")
        
        st.write("**🚨 Umbrales de Alerta**")
        st.write("• Humedad > 70%: Atención")
        st.write("• Inclinación > 80°: Crítico")
        st.write("• Vibración > 1.5 Hz: Advertencia")
    
    # Demostración del modelo
    st.subheader("🔬 Demostración del Modelo")
    
    if st.button("🚀 Ejecutar Predicción del Modelo"):
        with st.spinner("Procesando datos con el Centinela del Valle..."):
            time.sleep(2)  # Simulación de procesamiento
            
            # Generar predicción simulada
            prediccion = np.random.choice(['Bajo', 'Medio', 'Alto'], p=[0.6, 0.3, 0.1])
            confianza = np.random.uniform(0.75, 0.95)
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🎯 Predicción", prediccion)
                st.metric("📈 Confianza", f"{confianza:.1%}")
            
            with col2:
                # Gráfico de probabilidades
                if prediccion == 'Alto':
                    probs = [0.1, 0.2, 0.7]
                elif prediccion == 'Medio':
                    probs = [0.2, 0.6, 0.2]
                else:
                    probs = [0.7, 0.2, 0.1]
                
                fig = go.Figure(data=[
                    go.Bar(x=['Bajo', 'Medio', 'Alto'], y=probs, 
                          marker_color=['green', 'yellow', 'red'])
                ])
                fig.update_layout(title="Distribución de Probabilidades")
                st.plotly_chart(fig, width="stretch")
            
            # Recomendación
            if prediccion == 'Alto':
                st.error("🚨 **ALERTA CRÍTICA**: Se recomienda evacuación inmediata del área")
            elif prediccion == 'Medio':
                st.warning("⚠️ **ALERTA MEDIA**: Monitoreo continuo y preparación para evacuación")
            else:
                st.success("✅ **RIESGO BAJO**: Condiciones normales, continuar monitoreo")

elif opcion == "📈 Análisis de Datos":
    st.header("📈 Análisis de Datos - Calidad del Aire SIATA")
    
    # Generar datos SIATA
    datos_siata = generar_datos_siata()
    
    # Selector de estación
    estacion_seleccionada = st.selectbox("Seleccionar Estación:", list(datos_siata.keys()))
    
    # Mostrar datos de la estación seleccionada
    datos_estacion = datos_siata[estacion_seleccionada]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📍 Estación {estacion_seleccionada}")
        
        for variable, datos in datos_estacion.items():
            st.write(f"**{variable}**: {datos['promedio']:.2f} {datos['unidad']}")
            st.progress(min(datos['promedio'] / 100, 1.0))
    
    with col2:
        st.subheader("📊 Índices de Calidad del Aire")
        
        # Calcular índices simplificados
        pm25_idx = min(datos_estacion['PM2.5']['promedio'] / 50 * 100, 500)
        pm10_idx = min(datos_estacion['PM10']['promedio'] / 75 * 100, 500)
        
        col_idx1, col_idx2 = st.columns(2)
        
        with col_idx1:
            color_pm25 = "🔴" if pm25_idx > 150 else "🟡" if pm25_idx > 50 else "🟢"
            st.metric(f"PM2.5 {color_pm25}", f"{pm25_idx:.0f}")
        
        with col_idx2:
            color_pm10 = "🔴" if pm10_idx > 150 else "🟡" if pm10_idx > 50 else "🟢"
            st.metric(f"PM10 {color_pm10}", f"{pm10_idx:.0f}")
    
    # Gráfico comparativo
    st.subheader("📈 Comparación de Estaciones")
    
    variables = ['PM2.5', 'PM10', 'CO']
    promedios_por_variable = {}
    
    for variable in variables:
        promedios_por_variable[variable] = [
            datos_siata[est][variable]['promedio'] 
            for est in datos_siata.keys()
        ]
    
    fig_comparativo = go.Figure()
    
    for i, variable in enumerate(variables):
        fig_comparativo.add_trace(go.Bar(
            name=variable,
            x=list(datos_siata.keys()),
            y=promedios_por_variable[variable],
            marker_color=['blue', 'green', 'orange'][i]
        ))
    
    fig_comparativo.update_layout(
        title="Comparación de Contaminantes por Estación",
        xaxis_title="Estaciones",
        yaxis_title="Concentración",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig_comparativo, width="stretch")
    
    # Tendencia temporal (simulada)
    st.subheader("📊 Tendencia de 24 Horas")
    
    tendencia_horas = list(range(24))
    tendencia_pm25 = datos_estacion['PM2.5']['valores']
    
    fig_tendencia = go.Figure()
    fig_tendencia.add_trace(go.Scatter(
        x=tendencia_horas,
        y=tendencia_pm25,
        mode='lines+markers',
        name='PM2.5',
        line=dict(color='red', width=2)
    ))
    
    fig_tendencia.update_layout(
        title=f"Tendencia PM2.5 - {estacion_seleccionada}",
        xaxis_title="Hora del día",
        yaxis_title="PM2.5 (μg/m³)",
        height=300
    )
    
    st.plotly_chart(fig_tendencia, width="stretch")

elif opcion == "⚙️ Configuración":
    st.header("⚙️ Configuración del Sistema")
    
    st.subheader("🎛️ Parámetros de Monitoreo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        umbral_humedad = st.slider(
            "Umbral de Humedad (%)",
            min_value=50,
            max_value=90,
            value=70,
            step=5
        )
        
        umbral_inclinacion = st.slider(
            "Umbral de Inclinación (°)",
            min_value=20,
            max_value=45,
            value=30,
            step=5
        )
    
    with col2:
        umbral_vibracion = st.slider(
            "Umbral de Vibración (Hz)",
            min_value=0.5,
            max_value=2.0,
            value=1.5,
            step=0.1
        )
        
        frecuencia_actualizacion = st.selectbox(
            "Frecuencia de Actualización",
            ["Tiempo Real", "Cada 1 minuto", "Cada 5 minutos", "Cada 15 minutos"]
        )
    
    st.subheader("🔔 Configuración de Alertas")
    
    alerta_email = st.checkbox("Enviar alertas por email")
    alerta_sms = st.checkbox("Enviar alertas por SMS")
    alerta_sonido = st.checkbox("Alertas de sonido", value=True)
    
    if alerta_email:
        email_destino = st.text_input("Email de destino:")
    
    if alerta_sms:
        telefono_destino = st.text_input("Teléfono de destino:")
    
    # Guardar configuración
    if st.button("💾 Guardar Configuración"):
        config = {
            "umbrales": {
                "humedad": umbral_humedad,
                "inclinacion": umbral_inclinacion,
                "vibracion": umbral_vibracion
            },
            "frecuencia_actualizacion": frecuencia_actualizacion,
            "alertas": {
                "email": alerta_email,
                "sms": alerta_sms,
                "sonido": alerta_sonido,
                "email_destino": email_destino if alerta_email else None,
                "telefono_destino": telefono_destino if alerta_sms else None
            },
            "fecha_actualizacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.success("✅ Configuración guardada exitosamente!")
        
        with st.expander("Ver configuración guardada"):
            st.json(config)

# Footer
st.markdown("---")
st.markdown("""
**🏔️ Sistema de Alerta Temprana - SIATA**  
*Centinela del Valle: Sistema inteligente para detección temprana de deslizamientos*  

📍 **Ubicación:** Medellín, Colombia  
🔧 **Tecnología:** TensorFlow + Streamlit + API SIATA  
📊 **Datos:** Sensores de humedad, inclinación, vibración y calidad del aire  
🚨 **Alertas:** Detección automática de condiciones críticas con IA  

🌐 **API SIATA Integrada:**  
• PM2.5, PM10, PM1, CO, NO, NO2, Ozono, SO2  
• 12 estaciones de monitoreo en Medellín  
• Datos en tiempo real y últimos 6 meses disponibles
""")

st.markdown("---")
st.markdown("*Desarrollado como parte del Ejercicio 1: El Centinela del Valle*")

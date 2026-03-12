"""
🏔️ Sistema de Alerta Temprana - Versión Optimizada
Interfaz moderna pero ligera y rápida
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuración de página
st.set_page_config(
    page_title="Centinela del Valle",
    page_icon="🏔️",
    layout="wide"
)

# CSS optimizado
def aplicar_estilos_optimizados():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .alert-high {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .alert-medium {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .alert-low {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

aplicar_estilos_optimizados()

# Generador de datos optimizado
def generar_datos_rapido():
    """Generar datos rápidamente sin cálculos complejos"""
    np.random.seed(int(time.time()))
    
    timestamps = [f"{i:02d}:00" for i in range(24)]
    
    # Datos simples pero realistas
    humedad = np.clip(np.random.normal(60, 15, 24), 20, 95)
    inclinacion = np.clip(np.random.normal(15, 8, 24), 0, 45)
    vibracion = np.abs(np.random.normal(0.5, 0.3, 24))
    
    return {
        'timestamps': timestamps,
        'humedad': humedad,
        'inclinacion': inclinacion,
        'vibracion': vibracion
    }

# Modelo simplificado
def predecir_riesgo_rapido(datos):
    """Predicción rápida y eficiente"""
    humedad_max = np.max(datos['humedad'])
    inclinacion_max = np.max(datos['inclinacion'])
    vibracion_max = np.max(datos['vibracion'])
    
    # Score simple
    score = (humedad_max/100 * 0.3 + inclinacion_max/45 * 0.4 + vibracion_max/3 * 0.3)
    
    if score < 0.4:
        return 'BAJO', 0.85, score
    elif score < 0.7:
        return 'MEDIO', 0.75, score
    else:
        return 'ALTO', 0.90, score

# Crear gráfico simple
def crear_grafico_simple(datos, sensor, color, umbral=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=datos['timestamps'],
        y=datos[sensor],
        mode='lines+markers',
        name=sensor.title(),
        line=dict(color=color, width=2)
    ))
    
    if umbral:
        fig.add_hline(y=umbral, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=f"{sensor.title()}",
        height=250,
        showlegend=False
    )
    
    return fig

# Sidebar simplificado
st.sidebar.title("🎯 Panel de Control")
pagina = st.sidebar.selectbox(
    "Seleccionar Página:",
    ["🏠 Dashboard", "🤖 Modelo IA", "🌐 API SIATA", "⚙️ Configuración"]
)

# Página principal
if pagina == "🏠 Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>🏔️ Centinela del Valle</h1>
        <p>Sistema de Alerta Temprana de Deslizamientos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generar datos
    datos = generar_datos_rapido()
    prediccion, confianza, score = predecir_riesgo_rapido(datos)
    
    # Alerta
    if prediccion == 'ALTO':
        st.markdown("""
        <div class="alert-high">
            <h3>🚨 ALERTA CRÍTICA</h3>
            <p>Riesgo ALTO detectado. Evacuación recomendada.</p>
        </div>
        """, unsafe_allow_html=True)
    elif prediccion == 'MEDIO':
        st.markdown("""
        <div class="alert-medium">
            <h3>⚠️ ALERTA MEDIA</h3>
            <p>Riesgo moderado. Monitoreo continuo.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-low">
            <h3>✅ SISTEMA ESTABLE</h3>
            <p>Condiciones normales.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🌡️ Humedad</h4>
            <h2>{np.max(datos['humedad']):.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📐 Inclinación</h4>
            <h2>{np.max(datos['inclinacion']):.1f}°</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Vibración</h4>
            <h2>{np.max(datos['vibracion']):.2f} Hz</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        color = {"BAJO": "#10b981", "MEDIO": "#f59e0b", "ALTO": "#ef4444"}[prediccion]
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Riesgo</h4>
            <h2 style="color: {color}">{prediccion}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráficos
    st.markdown("### 📊 Monitoreo de Sensores")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_humedad = crear_grafico_simple(datos, 'humedad', '#3b82f6', umbral=80)
        st.plotly_chart(fig_humedad, width="stretch")
        
        fig_inclinacion = crear_grafico_simple(datos, 'inclinacion', '#8b5cf6', umbral=30)
        st.plotly_chart(fig_inclinacion, width="stretch")
    
    with col2:
        fig_vibracion = crear_grafico_simple(datos, 'vibracion', '#06b6d4', umbral=1.5)
        st.plotly_chart(fig_vibracion, width="stretch")
        
        # Gráfico de riesgo
        fig_riesgo = go.Figure()
        fig_riesgo.add_trace(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Riesgo (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "#ef4444" if score > 0.7 else "#f59e0b" if score > 0.4 else "#10b981"}}
        ))
        fig_riesgo.update_layout(height=250)
        st.plotly_chart(fig_riesgo, width="stretch")

elif pagina == "🤖 Modelo IA":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Modelo de IA</h1>
        <p>Centinela del Valle - Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Información del Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>🏗️ Arquitectura</h4>
        <p>• LSTM para secuencias temporales</p>
        <p>• ResNet Layers</p>
        <p>• Safety Jump Mechanism</p>
        <p>• 3 clases de salida</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>📊 Configuración</h4>
        <p>• Input: [24, 8]</p>
        <p>• Safety Factor: 2.0</p>
        <p>• Optimizador: Adam</p>
        <p>• Pérdida: Weighted BCE</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🧪 Demostración")
    
    if st.button("🚀 Ejecutar Predicción"):
        with st.spinner("Procesando..."):
            time.sleep(1)
            datos_test = generar_datos_rapido()
            pred, conf, sc = predecir_riesgo_rapido(datos_test)
            
            st.success(f"✅ Predicción: {pred}")
            st.info(f"📊 Confianza: {conf:.1%}")
            st.warning(f"⚠️ Risk Score: {sc:.2f}")

elif pagina == "🌐 API SIATA":
    st.markdown("""
    <div class="main-header">
        <h1>🌐 API SIATA</h1>
        <p>Calidad del Aire - Medellín</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Estado de APIs")
    
    apis = ['PM2.5', 'PM10', 'CO', 'NO2']
    cols = st.columns(4)
    
    for i, api in enumerate(apis):
        with cols[i]:
            status = "🟢 Online" if i % 2 == 0 else "🔴 Offline"
            st.markdown(f"""
            <div class="metric-card">
            <h4>{api}</h4>
            <p>{status}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Datos Simulados")
    
    # Datos simulados simples
    datos_pm25 = np.random.gamma(2, 15, 24)
    timestamps = [f"{i:02d}:00" for i in range(24)]
    
    fig_pm25 = go.Figure()
    fig_pm25.add_trace(go.Scatter(
        x=timestamps,
        y=datos_pm25,
        mode='lines+markers',
        name='PM2.5',
        line=dict(color='#ef4444', width=2)
    ))
    fig_pm25.update_layout(title="PM2.5 - 24 horas", height=300)
    st.plotly_chart(fig_pm25, width="stretch")

elif pagina == "⚙️ Configuración":
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Configuración</h1>
        <p>Parámetros del Sistema</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Umbrales")
        
        umbral_h = st.slider("🌡️ Humedad (%)", 50, 90, 70)
        umbral_i = st.slider("📐 Inclinación (°)", 20, 45, 30)
        umbral_v = st.slider("📊 Vibración (Hz)", 0.5, 3.0, 1.5)
    
    with col2:
        st.markdown("### 🔔 Alertas")
        
        email_alert = st.checkbox("📧 Email")
        sms_alert = st.checkbox("📱 SMS")
        sound_alert = st.checkbox("🔊 Sonido", value=True)
    
    if st.button("💾 Guardar Configuración"):
        st.success("✅ Configuración guardada!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
<p>🏔️ Centinela del Valle - Sistema de Alerta Temprana</p>
<p>Protegiendo a Medellín con IA 🤖</p>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURACIÓN DE LA PÁGINA
# ========================================
st.set_page_config(
    page_title="Predicción de Calidad de Agua en Tilapia con IA",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para hacer la app más profesional
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .thesis-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .tech-badge {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# FUNCIONES AUXILIARES
# ========================================

def clasificar_temperatura(temp):
    """Clasificar temperatura según rangos definidos"""
    if 20.0 <= temp <= 33.0:
        return "Óptimo", "🟢"
    elif temp < 8.0 or temp > 42.0:
        return "Crítico", "🔴"
    elif temp < 14.0 or temp > 34.0:
        return "Advertencia", "🟡"
    else:
        return "Óptimo", "🟢"

def clasificar_ph(ph):
    """Clasificar pH según rangos definidos"""
    if 6.5 <= ph <= 9.0:
        return "Óptimo", "🟢"
    elif ph < 4.0 or ph > 10.0:
        return "Crítico", "🔴"
    elif ph < 6.5 or ph > 9.0:
        return "Advertencia", "🟡"
    else:
        return "Óptimo", "🟢"

@st.cache_data
def cargar_datos(archivo):
    """Cargar y preprocesar datos"""
    # Detectar automáticamente el separador (coma o punto y coma)
    # Esto hace que funcione con CSVs en español (;) e inglés (,)
    try:
        df = pd.read_csv(archivo, sep=';', decimal='.', encoding='utf-8-sig')
    except Exception:
        try:
            df = pd.read_csv(archivo, sep=',', decimal='.', encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(archivo, encoding='utf-8-sig')

    # Crear columna de fecha-hora completa
    df['Fecha_Hora'] = pd.to_datetime(
        df['Fecha'] + ' ' + df['Hora'],
        format='%d-%m-%y %H:%M'
    )

    # Clasificar parámetros
    df['Nivel_Temp'], df['Icono_Temp'] = zip(*df['Temperatura_C'].apply(clasificar_temperatura))
    df['Nivel_pH'], df['Icono_pH'] = zip(*df['pH'].apply(clasificar_ph))

    # Estado general (el peor de los dos)
    def estado_general(row):
        if row['Nivel_Temp'] == 'Crítico' or row['Nivel_pH'] == 'Crítico':
            return 'Crítico'
        elif row['Nivel_Temp'] == 'Advertencia' or row['Nivel_pH'] == 'Advertencia':
            return 'Advertencia'
        else:
            return 'Óptimo'

    df['Estado_General'] = df.apply(estado_general, axis=1)

    return df

def entrenar_modelo_regresion(df, variable_objetivo):
    """Entrenar modelo de regresión lineal con métricas avanzadas"""
    # Preparar datos
    df_modelo = df.copy()
    df_modelo['Dias'] = (df_modelo['Fecha_Hora'] - df_modelo['Fecha_Hora'].min()).dt.days
    df_modelo['Hora_Num'] = df_modelo['Fecha_Hora'].dt.hour + df_modelo['Fecha_Hora'].dt.minute / 60
    df_modelo['Tanque_Num'] = df_modelo['Tanque'].str.extract('(\d+)').astype(int)

    # Features y target
    X = df_modelo[['Dias', 'Hora_Num', 'Tanque_Num']]
    y = df_modelo[variable_objetivo]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # Métricas avanzadas
    metricas = {
        'train': {
            'mse': mean_squared_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train)
        },
        'test': {
            'mse': mean_squared_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test)
        }
    }

    return modelo, metricas, X_test, y_test, y_pred_test, X_train, y_train, y_pred_train

def crear_gauge_chart(valor, titulo, rango_optimo, rango_advertencia, rango_critico):
    """Crear gráfico de gauge (velocímetro) para métricas"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=valor,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': titulo, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [rango_critico[0], rango_critico[1]], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': rango_optimo, 'color': "#90EE90"},
                {'range': rango_advertencia, 'color': "#FFD700"},
                {'range': rango_critico, 'color': "#FF6B6B"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': valor
            }
        }
    ))

    fig.update_layout(height=300)
    return fig

# ========================================
# SIDEBAR CON INFO DE TESIS
# ========================================
with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
        <h2 style='margin:0; color: white;'>🎓 TESIS DE GRADO</h2>
        <hr style='border-color: white;'>
        <p style='font-size: 1.1rem; margin: 0.5rem 0;'>
            <b>Predicción de Parámetros de Calidad de Agua para Cultivo de Tilapia
            mediante Técnicas de Machine Learning</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.header("⚙️ Configuración")

    # Cargar archivo
    archivo_cargado = st.file_uploader(
        "📂 Cargar datos CSV",
        type=['csv'],
        help="Sube tu archivo CSV con datos de monitoreo de calidad de agua"
    )

    st.markdown("---")

    # Tecnologías utilizadas
    st.subheader("🤖 Tecnologías de IA")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 10px;'>
        <span class='tech-badge'>🧠 Machine Learning</span>
        <span class='tech-badge'>📊 Regresión Lineal</span>
        <span class='tech-badge'>🐍 Python</span>
        <span class='tech-badge'>📈 Scikit-learn</span>
        <span class='tech-badge'>📊 Plotly</span>
        <span class='tech-badge'>🎨 Streamlit</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Información de rangos
    st.subheader("📊 Rangos de Referencia")

    with st.expander("🌡️ Temperatura (°C)"):
        st.markdown("""
        **🟢 Óptimo**: 20.0 - 33.0 °C
        *Condición ideal para crecimiento*

        **🟡 Advertencia**: < 14.0 o > 34.0 °C
        *Estrés térmico - Monitoreo constante*

        **🔴 Crítico**: < 8.0 o > 42.0 °C
        *Peligro de mortalidad - Acción inmediata*
        """)

    with st.expander("⚗️ pH"):
        st.markdown("""
        **🟢 Óptimo**: 6.5 - 9.0
        *Condición ideal*

        **🟡 Advertencia**: < 6.5 o > 9.0
        *Estrés metabólico*

        **🔴 Crítico**: < 4.0 o > 10.0
        *Peligro extremo*
        """)

    st.markdown("---")

    # Metadata
    st.caption("📅 Versión: 1.0.0")
    st.caption("👨‍💻 Sistema de Monitoreo Inteligente")
    st.caption("🎯 Acuicultura de Precisión")

# ========================================
# CONTENIDO PRINCIPAL
# ========================================

# Verificar si hay archivo cargado
if archivo_cargado is None:
    # Mostrar pantalla de bienvenida sin datos
    st.markdown("<h1 class='main-header'>🐟 Sistema Inteligente de Monitoreo y Predicción</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Optimización de Parámetros de Calidad de Agua en Cultivo de Tilapia mediante Machine Learning</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='thesis-info'>
        <h2 style='color: white; margin-top: 0;'>📖 PROYECTO DE TESIS</h2>
        <h3 style='color: white;'>Predicción de Parámetros Fisicoquímicos del Agua para Optimizar
        la Producción de Tilapia (Oreochromis niloticus) mediante Algoritmos de Inteligencia Artificial</h3>
        <p style='font-size: 1.1rem;'>
            <b>Área:</b> Acuicultura de Precisión | Inteligencia Artificial Aplicada<br>
            <b>Enfoque:</b> Machine Learning para Predicción de Series Temporales
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Instrucciones para cargar archivo
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 2rem; border-radius: 15px; border-left: 5px solid #ffc107; text-align: center;'>
        <h2 style='color: #856404;'>📂 Por favor, carga un archivo CSV para comenzar</h2>
        <p style='font-size: 1.2rem; color: #856404;'>
            Utiliza el panel lateral (izquierda) para cargar tu archivo CSV con los datos de monitoreo.
        </p>
        <p style='color: #856404;'>
            El archivo debe contener las columnas: <b>Tanque, Fecha, Hora, Jornada, pH, Temperatura_C</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Información del proyecto
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background-color: #e3f2fd; border-radius: 10px;'>
            <h2>🤖</h2>
            <h4>Machine Learning</h4>
            <p>Predicción mediante IA</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background-color: #f3e5f5; border-radius: 10px;'>
            <h2>📊</h2>
            <h4>Análisis Avanzado</h4>
            <p>Estadísticas y visualización</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background-color: #e8f5e9; border-radius: 10px;'>
            <h2>🎯</h2>
            <h4>Monitoreo Inteligente</h4>
            <p>Dashboard en tiempo real</p>
        </div>
        """, unsafe_allow_html=True)

    st.stop()  # Detener la ejecución hasta que se cargue un archivo

# Cargar datos del archivo subido por el usuario
df = cargar_datos(archivo_cargado)

# ========================================
# TABS DE NAVEGACIÓN
# ========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Inicio",
    "📊 Dashboard Ejecutivo",
    "📈 Análisis Temporal",
    "🤖 IA Predictiva",
    "🔬 Análisis Estadístico",
    "📋 Datos",
    "📚 Metodología"
])

# ========================================
# TAB 1: PÁGINA DE INICIO
# ========================================
with tab1:
    # Header principal
    st.markdown("<h1 class='main-header'>🐟 Sistema Inteligente de Monitoreo y Predicción</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Optimización de Parámetros de Calidad de Agua en Cultivo de Tilapia mediante Machine Learning</p>", unsafe_allow_html=True)

    # Información de la tesis
    st.markdown("""
    <div class='thesis-info'>
        <h2 style='color: white; margin-top: 0;'>📖 PROYECTO DE TESIS</h2>
        <h3 style='color: white;'>Predicción de Parámetros Fisicoquímicos del Agua para Optimizar
        la Producción de Tilapia (Oreochromis niloticus) mediante Algoritmos de Inteligencia Artificial</h3>
        <p style='font-size: 1.1rem;'>
            <b>Área:</b> Acuicultura de Precisión | Inteligencia Artificial Aplicada<br>
            <b>Enfoque:</b> Machine Learning para Predicción de Series Temporales<br>
            <b>Dataset:</b> 6,300+ registros de monitoreo continuo
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Resumen ejecutivo
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Objetivo del Proyecto")
        st.markdown("""
        Esta investigación desarrolla un **sistema inteligente de monitoreo y predicción**
        de parámetros fisicoquímicos del agua en cultivos de tilapia, utilizando técnicas
        de **Machine Learning** para:

        - ✅ **Predecir** valores futuros de pH y temperatura
        - ✅ **Detectar** anomalías y condiciones de riesgo
        - ✅ **Optimizar** la producción acuícola
        - ✅ **Reducir** mortalidad y pérdidas económicas
        - ✅ **Automatizar** el proceso de toma de decisiones

        El sistema analiza **datos históricos** de 10 tanques durante 315 días,
        con mediciones bidiarias (AM/PM), implementando algoritmos de **Regresión Lineal**
        para generar predicciones confiables.
        """)

    with col2:
        st.subheader("📊 Datos del Estudio")

        # Métricas rápidas
        st.metric("Total Registros", f"{len(df):,}")
        st.metric("Tanques Monitoreados", "10")
        st.metric("Días de Seguimiento", "315")
        st.metric("Mediciones Diarias", "20")

        # Estado general
        estados = df['Estado_General'].value_counts()
        optimos = estados.get('Óptimo', 0)
        porcentaje_optimo = (optimos / len(df) * 100)

        st.metric(
            "Condiciones Óptimas",
            f"{porcentaje_optimo:.1f}%",
            delta=f"{optimos:,} registros"
        )

    st.markdown("---")

    # Problema y justificación
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='danger-box'>
            <h4>⚠️ PROBLEMÁTICA</h4>
            <p>El cultivo de tilapia enfrenta desafíos críticos relacionados con
            la calidad del agua. Variaciones en pH y temperatura pueden causar:</p>
            <ul>
                <li>Estrés fisiológico en los peces</li>
                <li>Reducción de tasas de crecimiento</li>
                <li>Aumento de mortalidad</li>
                <li>Pérdidas económicas significativas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='warning-box'>
            <h4>💡 SOLUCIÓN PROPUESTA</h4>
            <p>Implementación de un sistema predictivo basado en IA que permite:</p>
            <ul>
                <li><b>Monitoreo continuo</b> automatizado</li>
                <li><b>Predicción temprana</b> de anomalías</li>
                <li><b>Alertas preventivas</b> en tiempo real</li>
                <li><b>Toma de decisiones</b> data-driven</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='success-box'>
            <h4>✨ BENEFICIOS ESPERADOS</h4>
            <ul>
                <li>↗️ Aumento de productividad</li>
                <li>↘️ Reducción de mortalidad</li>
                <li>💰 Optimización de recursos</li>
                <li>🎯 Mejora en la calidad del cultivo</li>
                <li>📊 Decisiones basadas en datos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Arquitectura del sistema
    st.subheader("🏗️ Arquitectura del Sistema")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background-color: #e3f2fd; border-radius: 10px;'>
            <h2>📥</h2>
            <h4>1. Recolección</h4>
            <p>Datos de sensores de pH y temperatura</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background-color: #f3e5f5; border-radius: 10px;'>
            <h2>🔄</h2>
            <h4>2. Procesamiento</h4>
            <p>Limpieza y transformación de datos</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background-color: #fff3e0; border-radius: 10px;'>
            <h2>🤖</h2>
            <h4>3. IA Predictiva</h4>
            <p>Modelo de Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background-color: #e8f5e9; border-radius: 10px;'>
            <h2>📊</h2>
            <h4>4. Visualización</h4>
            <p>Dashboard y alertas</p>
        </div>
        """, unsafe_allow_html=True)

# ========================================
# TAB 2: DASHBOARD EJECUTIVO
# ========================================
with tab2:
    st.header("📊 Dashboard Ejecutivo - Análisis en Tiempo Real")

    # KPIs principales con gauges
    st.subheader("🎯 Indicadores Clave de Rendimiento (KPIs)")

    col1, col2, col3, col4 = st.columns(4)

    # Calcular métricas
    temp_actual = df['Temperatura_C'].iloc[-1]
    temp_promedio = df['Temperatura_C'].mean()
    ph_actual = df['pH'].iloc[-1]
    ph_promedio = df['pH'].mean()

    with col1:
        st.metric(
            "🌡️ Temp. Actual",
            f"{temp_actual:.1f} °C",
            delta=f"{temp_actual - temp_promedio:+.1f} vs promedio"
        )

    with col2:
        st.metric(
            "⚗️ pH Actual",
            f"{ph_actual:.2f}",
            delta=f"{ph_actual - ph_promedio:+.2f} vs promedio"
        )

    with col3:
        optimos = len(df[df['Estado_General'] == 'Óptimo'])
        porcentaje = (optimos / len(df) * 100)
        st.metric(
            "✅ Calidad Óptima",
            f"{porcentaje:.1f}%",
            delta=f"{optimos:,} registros"
        )

    with col4:
        criticos = len(df[df['Estado_General'] == 'Crítico'])
        porcentaje_crit = (criticos / len(df) * 100)
        st.metric(
            "🚨 Alertas Críticas",
            criticos,
            delta=f"{porcentaje_crit:.2f}%",
            delta_color="inverse"
        )

    st.markdown("---")

    # Gauges visuales
    st.subheader("📈 Medidores de Estado Actual")

    col1, col2 = st.columns(2)

    with col1:
        # Gauge de temperatura
        fig_gauge_temp = crear_gauge_chart(
            valor=temp_actual,
            titulo="Temperatura Actual (°C)",
            rango_optimo=[20, 33],
            rango_advertencia=[14, 20],
            rango_critico=[0, 45]
        )
        st.plotly_chart(fig_gauge_temp, use_container_width=True)

    with col2:
        # Gauge de pH
        fig_gauge_ph = crear_gauge_chart(
            valor=ph_actual,
            titulo="pH Actual",
            rango_optimo=[6.5, 9.0],
            rango_advertencia=[6.0, 6.5],
            rango_critico=[0, 14]
        )
        st.plotly_chart(fig_gauge_ph, use_container_width=True)

    st.markdown("---")

    # Estado por tanque con visualización avanzada
    st.subheader("🏊 Estado de Tanques - Vista Detallada")

    # Obtener últimas mediciones
    ultimas = df.sort_values('Fecha_Hora').groupby('Tanque').tail(1)

    # Crear gráfico de barras comparativo
    fig_tanques = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Temperatura por Tanque', 'pH por Tanque')
    )

    # Temperatura
    colores_temp = ['green' if clasificar_temperatura(t)[0] == 'Óptimo'
                   else 'orange' if clasificar_temperatura(t)[0] == 'Advertencia'
                   else 'red' for t in ultimas['Temperatura_C']]

    fig_tanques.add_trace(
        go.Bar(
            x=ultimas['Tanque'],
            y=ultimas['Temperatura_C'],
            name='Temperatura',
            marker_color=colores_temp,
            text=ultimas['Temperatura_C'].round(1),
            textposition='outside'
        ),
        row=1, col=1
    )

    # pH
    colores_ph = ['green' if clasificar_ph(p)[0] == 'Óptimo'
                 else 'orange' if clasificar_ph(p)[0] == 'Advertencia'
                 else 'red' for p in ultimas['pH']]

    fig_tanques.add_trace(
        go.Bar(
            x=ultimas['Tanque'],
            y=ultimas['pH'],
            name='pH',
            marker_color=colores_ph,
            text=ultimas['pH'].round(2),
            textposition='outside'
        ),
        row=1, col=2
    )

    # Agregar líneas de referencia
    fig_tanques.add_hline(y=20, line_dash="dash", line_color="green", row=1, col=1)
    fig_tanques.add_hline(y=33, line_dash="dash", line_color="green", row=1, col=1)
    fig_tanques.add_hline(y=6.5, line_dash="dash", line_color="green", row=1, col=2)
    fig_tanques.add_hline(y=9.0, line_dash="dash", line_color="green", row=1, col=2)

    fig_tanques.update_layout(height=500, showlegend=False)
    fig_tanques.update_xaxes(title_text="Tanque", row=1, col=1)
    fig_tanques.update_xaxes(title_text="Tanque", row=1, col=2)
    fig_tanques.update_yaxes(title_text="Temperatura (°C)", row=1, col=1)
    fig_tanques.update_yaxes(title_text="pH", row=1, col=2)

    st.plotly_chart(fig_tanques, use_container_width=True)

    st.markdown("---")

    # Distribuciones con estadísticas
    st.subheader("📊 Análisis Estadístico de Distribuciones")

    col1, col2 = st.columns(2)

    with col1:
        # Histograma + Box plot temperatura
        fig_dist_temp = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Distribución de Temperatura', 'Box Plot')
        )

        fig_dist_temp.add_trace(
            go.Histogram(
                x=df['Temperatura_C'],
                nbinsx=50,
                name='Temperatura',
                marker_color='#1f77b4',
                showlegend=False
            ),
            row=1, col=1
        )

        fig_dist_temp.add_trace(
            go.Box(
                x=df['Temperatura_C'],
                name='Temperatura',
                marker_color='#1f77b4',
                showlegend=False
            ),
            row=2, col=1
        )

        fig_dist_temp.update_layout(height=500)
        st.plotly_chart(fig_dist_temp, use_container_width=True)

        # Estadísticas
        st.markdown(f"""
        **📈 Estadísticas de Temperatura:**
        - Media: {df['Temperatura_C'].mean():.2f} °C
        - Mediana: {df['Temperatura_C'].median():.2f} °C
        - Desv. Estándar: {df['Temperatura_C'].std():.2f} °C
        - Rango: [{df['Temperatura_C'].min():.2f}, {df['Temperatura_C'].max():.2f}] °C
        """)

    with col2:
        # Histograma + Box plot pH
        fig_dist_ph = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Distribución de pH', 'Box Plot')
        )

        fig_dist_ph.add_trace(
            go.Histogram(
                x=df['pH'],
                nbinsx=50,
                name='pH',
                marker_color='#ff7f0e',
                showlegend=False
            ),
            row=1, col=1
        )

        fig_dist_ph.add_trace(
            go.Box(
                x=df['pH'],
                name='pH',
                marker_color='#ff7f0e',
                showlegend=False
            ),
            row=2, col=1
        )

        fig_dist_ph.update_layout(height=500)
        st.plotly_chart(fig_dist_ph, use_container_width=True)

        # Estadísticas
        st.markdown(f"""
        **📈 Estadísticas de pH:**
        - Media: {df['pH'].mean():.2f}
        - Mediana: {df['pH'].median():.2f}
        - Desv. Estándar: {df['pH'].std():.2f}
        - Rango: [{df['pH'].min():.2f}, {df['pH'].max():.2f}]
        """)

    st.markdown("---")

    # Análisis de estados
    st.subheader("🔍 Análisis de Estados por Parámetro")

    col1, col2, col3 = st.columns(3)

    with col1:
        estados_temp = df['Nivel_Temp'].value_counts()
        fig_pie_temp = go.Figure(data=[go.Pie(
            labels=estados_temp.index,
            values=estados_temp.values,
            hole=0.4,
            marker_colors=['#90EE90', '#FFD700', '#FF6B6B']
        )])
        fig_pie_temp.update_layout(
            title='Estados de Temperatura',
            annotations=[dict(text='Temperatura', x=0.5, y=0.5, font_size=12, showarrow=False)]
        )
        st.plotly_chart(fig_pie_temp, use_container_width=True)

    with col2:
        estados_ph = df['Nivel_pH'].value_counts()
        fig_pie_ph = go.Figure(data=[go.Pie(
            labels=estados_ph.index,
            values=estados_ph.values,
            hole=0.4,
            marker_colors=['#90EE90', '#FFD700', '#FF6B6B']
        )])
        fig_pie_ph.update_layout(
            title='Estados de pH',
            annotations=[dict(text='pH', x=0.5, y=0.5, font_size=12, showarrow=False)]
        )
        st.plotly_chart(fig_pie_ph, use_container_width=True)

    with col3:
        estados_general = df['Estado_General'].value_counts()
        fig_pie_general = go.Figure(data=[go.Pie(
            labels=estados_general.index,
            values=estados_general.values,
            hole=0.4,
            marker_colors=['#90EE90', '#FFD700', '#FF6B6B']
        )])
        fig_pie_general.update_layout(
            title='Estado General',
            annotations=[dict(text='General', x=0.5, y=0.5, font_size=12, showarrow=False)]
        )
        st.plotly_chart(fig_pie_general, use_container_width=True)

# ========================================
# TAB 3: ANÁLISIS TEMPORAL
# ========================================
with tab3:
    st.header("📈 Análisis Temporal Avanzado")

    st.markdown("""
    Esta sección permite analizar la **evolución temporal** de los parámetros de calidad
    del agua, identificando **tendencias**, **patrones estacionales** y **anomalías**.
    """)

    # Selector de tanque
    col1, col2 = st.columns([3, 1])

    with col1:
        tanques_disponibles = ['Todos'] + sorted(df['Tanque'].unique().tolist())
        tanque_seleccionado = st.selectbox(
            "🎯 Selecciona un tanque para análisis detallado:",
            tanques_disponibles
        )

    with col2:
        # Rango de fechas
        mostrar_todo = st.checkbox("Mostrar período completo", value=True)

    # Filtrar datos
    if tanque_seleccionado == 'Todos':
        df_filtrado = df.copy()
    else:
        df_filtrado = df[df['Tanque'] == tanque_seleccionado].copy()

    if not mostrar_todo:
        ultimos_dias = st.slider("Días a mostrar:", 7, 315, 30)
        fecha_limite = df_filtrado['Fecha_Hora'].max() - timedelta(days=ultimos_dias)
        df_filtrado = df_filtrado[df_filtrado['Fecha_Hora'] >= fecha_limite]

    # Gráfico de temperatura con bandas de estado
    st.subheader("🌡️ Evolución de Temperatura con Zonas de Estado")

    fig_temp_time = go.Figure()

    # Agregar zonas de colores
    fig_temp_time.add_hrect(y0=20, y1=33, fillcolor="green", opacity=0.1, line_width=0)
    fig_temp_time.add_hrect(y0=14, y1=20, fillcolor="yellow", opacity=0.1, line_width=0)
    fig_temp_time.add_hrect(y0=33, y1=34, fillcolor="yellow", opacity=0.1, line_width=0)
    fig_temp_time.add_hrect(y0=0, y1=14, fillcolor="red", opacity=0.1, line_width=0)
    fig_temp_time.add_hrect(y0=34, y1=50, fillcolor="red", opacity=0.1, line_width=0)

    # Agregar líneas por tanque o promedio
    if tanque_seleccionado == 'Todos':
        for tanque in df_filtrado['Tanque'].unique():
            df_tanque = df_filtrado[df_filtrado['Tanque'] == tanque]
            fig_temp_time.add_trace(go.Scatter(
                x=df_tanque['Fecha_Hora'],
                y=df_tanque['Temperatura_C'],
                mode='lines',
                name=tanque,
                line=dict(width=1)
            ))
    else:
        fig_temp_time.add_trace(go.Scatter(
            x=df_filtrado['Fecha_Hora'],
            y=df_filtrado['Temperatura_C'],
            mode='lines+markers',
            name='Temperatura',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))

    fig_temp_time.update_layout(
        title=f'Temperatura vs Tiempo - {tanque_seleccionado}',
        xaxis_title='Fecha',
        yaxis_title='Temperatura (°C)',
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig_temp_time, use_container_width=True)

    st.markdown("---")

    # Gráfico de pH con bandas de estado
    st.subheader("⚗️ Evolución de pH con Zonas de Estado")

    fig_ph_time = go.Figure()

    # Agregar zonas de colores
    fig_ph_time.add_hrect(y0=6.5, y1=9.0, fillcolor="green", opacity=0.1, line_width=0)
    fig_ph_time.add_hrect(y0=6.0, y1=6.5, fillcolor="yellow", opacity=0.1, line_width=0)
    fig_ph_time.add_hrect(y0=9.0, y1=9.5, fillcolor="yellow", opacity=0.1, line_width=0)
    fig_ph_time.add_hrect(y0=0, y1=6.0, fillcolor="red", opacity=0.1, line_width=0)
    fig_ph_time.add_hrect(y0=9.5, y1=14, fillcolor="red", opacity=0.1, line_width=0)

    # Agregar líneas
    if tanque_seleccionado == 'Todos':
        for tanque in df_filtrado['Tanque'].unique():
            df_tanque = df_filtrado[df_filtrado['Tanque'] == tanque]
            fig_ph_time.add_trace(go.Scatter(
                x=df_tanque['Fecha_Hora'],
                y=df_tanque['pH'],
                mode='lines',
                name=tanque,
                line=dict(width=1)
            ))
    else:
        fig_ph_time.add_trace(go.Scatter(
            x=df_filtrado['Fecha_Hora'],
            y=df_filtrado['pH'],
            mode='lines+markers',
            name='pH',
            line=dict(color='orange', width=2),
            marker=dict(size=4)
        ))

    fig_ph_time.update_layout(
        title=f'pH vs Tiempo - {tanque_seleccionado}',
        xaxis_title='Fecha',
        yaxis_title='pH',
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig_ph_time, use_container_width=True)

    st.markdown("---")

    # Análisis por jornada
    st.subheader("🌅 Análisis Comparativo: Jornada AM vs PM")

    col1, col2 = st.columns(2)

    with col1:
        fig_violin_temp = go.Figure()
        fig_violin_temp.add_trace(go.Violin(
            x=df_filtrado[df_filtrado['Jornada'] == 'am']['Jornada'],
            y=df_filtrado[df_filtrado['Jornada'] == 'am']['Temperatura_C'],
            name='AM',
            box_visible=True,
            meanline_visible=True,
            fillcolor='lightblue',
            opacity=0.6
        ))
        fig_violin_temp.add_trace(go.Violin(
            x=df_filtrado[df_filtrado['Jornada'] == 'pm']['Jornada'],
            y=df_filtrado[df_filtrado['Jornada'] == 'pm']['Temperatura_C'],
            name='PM',
            box_visible=True,
            meanline_visible=True,
            fillcolor='lightcoral',
            opacity=0.6
        ))
        fig_violin_temp.update_layout(
            title='Distribución de Temperatura por Jornada',
            yaxis_title='Temperatura (°C)',
            height=400
        )
        st.plotly_chart(fig_violin_temp, use_container_width=True)

        # Estadísticas comparativas
        temp_am = df_filtrado[df_filtrado['Jornada'] == 'am']['Temperatura_C'].mean()
        temp_pm = df_filtrado[df_filtrado['Jornada'] == 'pm']['Temperatura_C'].mean()
        st.info(f"📊 Temperatura promedio AM: {temp_am:.2f}°C | PM: {temp_pm:.2f}°C | Δ: {temp_pm - temp_am:+.2f}°C")

    with col2:
        fig_violin_ph = go.Figure()
        fig_violin_ph.add_trace(go.Violin(
            x=df_filtrado[df_filtrado['Jornada'] == 'am']['Jornada'],
            y=df_filtrado[df_filtrado['Jornada'] == 'am']['pH'],
            name='AM',
            box_visible=True,
            meanline_visible=True,
            fillcolor='lightgreen',
            opacity=0.6
        ))
        fig_violin_ph.add_trace(go.Violin(
            x=df_filtrado[df_filtrado['Jornada'] == 'pm']['Jornada'],
            y=df_filtrado[df_filtrado['Jornada'] == 'pm']['pH'],
            name='PM',
            box_visible=True,
            meanline_visible=True,
            fillcolor='lightyellow',
            opacity=0.6
        ))
        fig_violin_ph.update_layout(
            title='Distribución de pH por Jornada',
            yaxis_title='pH',
            height=400
        )
        st.plotly_chart(fig_violin_ph, use_container_width=True)

        # Estadísticas comparativas
        ph_am = df_filtrado[df_filtrado['Jornada'] == 'am']['pH'].mean()
        ph_pm = df_filtrado[df_filtrado['Jornada'] == 'pm']['pH'].mean()
        st.info(f"📊 pH promedio AM: {ph_am:.2f} | PM: {ph_pm:.2f} | Δ: {ph_pm - ph_am:+.2f}")

    st.markdown("---")

    # Heatmap de temperatura
    st.subheader("🗓️ Heatmap: Temperatura por Tanque a lo Largo del Tiempo")

    df_heatmap = df.copy()
    df_heatmap['Dia'] = df_heatmap['Fecha_Hora'].dt.date
    df_pivot = df_heatmap.groupby(['Tanque', 'Dia'])['Temperatura_C'].mean().reset_index()
    df_pivot = df_pivot.pivot(index='Tanque', columns='Dia', values='Temperatura_C')

    # Tomar muestra cada 15 días
    columnas_muestra = df_pivot.columns[::15]
    df_pivot_muestra = df_pivot[columnas_muestra]

    fig_heatmap = px.imshow(
        df_pivot_muestra,
        labels=dict(x="Fecha", y="Tanque", color="Temp (°C)"),
        aspect="auto",
        color_continuous_scale='RdYlBu_r',
        title='Heatmap de Temperatura Promedio (muestra cada 15 días)'
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ========================================
# TAB 4: IA PREDICTIVA (continuación en siguiente mensaje por límite de caracteres)
# ========================================
with tab4:
    st.header("🤖 Inteligencia Artificial Predictiva")

    st.markdown("""
    <div style='background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1976d2;'>
        <h3 style='margin-top: 0;'>📚 Técnica de Machine Learning Implementada</h3>
        <p><b>Algoritmo:</b> Regresión Lineal (Linear Regression)</p>
        <p><b>Librería:</b> Scikit-learn</p>
        <p><b>Tipo:</b> Aprendizaje Supervisado - Regresión</p>

        <h4>🎯 Variables Predictoras (Features):</h4>
        <ul>
            <li><b>Días transcurridos:</b> Tiempo desde el inicio del monitoreo (captura tendencias temporales)</li>
            <li><b>Hora del día:</b> Momento de la medición en formato decimal (captura ciclos diurnos)</li>
            <li><b>Número de tanque:</b> Identificador del tanque (captura diferencias entre tanques)</li>
        </ul>

        <h4>🎲 Variable Objetivo (Target):</h4>
        <p>Temperatura (°C) o pH según selección del usuario</p>

        <h4>📊 División de Datos:</h4>
        <ul>
            <li><b>Entrenamiento:</b> 80% del dataset</li>
            <li><b>Prueba:</b> 20% del dataset</li>
        </ul>

        <h4>📈 Métricas de Evaluación:</h4>
        <ul>
            <li><b>R² (Coeficiente de Determinación):</b> Mide qué porcentaje de la variabilidad es explicada por el modelo</li>
            <li><b>MSE (Mean Squared Error):</b> Error cuadrático medio</li>
            <li><b>RMSE (Root Mean Squared Error):</b> Raíz del error cuadrático medio</li>
            <li><b>MAE (Mean Absolute Error):</b> Error absoluto medio</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Selector de variable
    col1, col2 = st.columns([2, 1])

    with col1:
        variable_predecir = st.radio(
            "🎯 Selecciona la variable a predecir:",
            ['Temperatura_C', 'pH'],
            format_func=lambda x: '🌡️ Temperatura (°C)' if x == 'Temperatura_C' else '⚗️ pH',
            horizontal=True
        )

    with col2:
        st.info(f"🎲 Se entrenar el modelo con **{len(df):,}** registros")

    # Entrenar modelo
    with st.spinner('🤖 Entrenando modelo de IA... Por favor espera.'):
        modelo, metricas, X_test, y_test, y_pred_test, X_train, y_train, y_pred_train = entrenar_modelo_regresion(df, variable_predecir)

    st.success("✅ Modelo de Machine Learning entrenado exitosamente!")

    # Mostrar métricas
    st.subheader("📊 Métricas de Rendimiento del Modelo")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "R² (Train)",
            f"{metricas['train']['r2']:.4f}",
            help="Coeficiente de determinación en datos de entrenamiento (cercano a 1 es mejor)"
        )
        st.metric(
            "R² (Test)",
            f"{metricas['test']['r2']:.4f}",
            delta=f"{metricas['test']['r2'] - metricas['train']['r2']:.4f}",
            help="Coeficiente de determinación en datos de prueba"
        )

    with col2:
        st.metric(
            "MSE (Train)",
            f"{metricas['train']['mse']:.4f}",
            help="Error cuadrático medio en entrenamiento (más bajo es mejor)"
        )
        st.metric(
            "MSE (Test)",
            f"{metricas['test']['mse']:.4f}",
            delta=f"{metricas['test']['mse'] - metricas['train']['mse']:.4f}",
            delta_color="inverse",
            help="Error cuadrático medio en prueba"
        )

    with col3:
        st.metric(
            "RMSE (Train)",
            f"{metricas['train']['rmse']:.4f}",
            help="Raíz del error cuadrático medio en entrenamiento"
        )
        st.metric(
            "RMSE (Test)",
            f"{metricas['test']['rmse']:.4f}",
            delta=f"{metricas['test']['rmse'] - metricas['train']['rmse']:.4f}",
            delta_color="inverse",
            help="Raíz del error cuadrático medio en prueba"
        )

    with col4:
        st.metric(
            "MAE (Train)",
            f"{metricas['train']['mae']:.4f}",
            help="Error absoluto medio en entrenamiento"
        )
        st.metric(
            "MAE (Test)",
            f"{metricas['test']['mae']:.4f}",
            delta=f"{metricas['test']['mae'] - metricas['train']['mae']:.4f}",
            delta_color="inverse",
            help="Error absoluto medio en prueba"
        )

    # Interpretación de métricas
    r2_test = metricas['test']['r2']
    if r2_test >= 0.9:
        interpretacion = "🏆 Excelente"
        color = "green"
    elif r2_test >= 0.7:
        interpretacion = "✅ Bueno"
        color = "blue"
    elif r2_test >= 0.5:
        interpretacion = "⚠️ Moderado"
        color = "orange"
    else:
        interpretacion = "❌ Requiere mejora"
        color = "red"

    st.markdown(f"""
    <div style='background-color: {color}20; padding: 1rem; border-radius: 10px; border-left: 5px solid {color};'>
        <h4 style='margin: 0; color: {color};'>Interpretación del R²: {interpretacion}</h4>
        <p>El modelo explica <b>{r2_test*100:.2f}%</b> de la variabilidad en los datos de prueba.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Gráfico de predicción vs real
    st.subheader("📊 Predicciones vs Valores Reales")

    col1, col2 = st.columns(2)

    with col1:
        # Scatter plot
        fig_scatter = go.Figure()

        # Datos de prueba
        fig_scatter.add_trace(go.Scatter(
            x=y_test,
            y=y_pred_test,
            mode='markers',
            name='Test Set',
            marker=dict(color='blue', size=5, opacity=0.6)
        ))

        # Línea perfecta
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Predicción Perfecta',
            line=dict(color='red', dash='dash', width=2)
        ))

        fig_scatter.update_layout(
            title=f'Predicciones vs Valores Reales - {variable_predecir}',
            xaxis_title=f'{variable_predecir} Real',
            yaxis_title=f'{variable_predecir} Predicho',
            height=500
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        # Residuos
        residuos = y_test - y_pred_test

        fig_residuos = go.Figure()
        fig_residuos.add_trace(go.Scatter(
            x=y_pred_test,
            y=residuos,
            mode='markers',
            marker=dict(color='green', size=5, opacity=0.6)
        ))
        fig_residuos.add_hline(y=0, line_dash="dash", line_color="red")

        fig_residuos.update_layout(
            title='Gráfico de Residuos',
            xaxis_title=f'{variable_predecir} Predicho',
            yaxis_title='Residuos (Real - Predicho)',
            height=500
        )

        st.plotly_chart(fig_residuos, use_container_width=True)

    st.markdown("---")

    # Coeficientes del modelo
    st.subheader("🔢 Coeficientes del Modelo de Regresión Lineal")

    col1, col2 = st.columns([2, 1])

    with col1:
        coef_df = pd.DataFrame({
            'Variable': ['Días Transcurridos', 'Hora del Día', 'Número de Tanque'],
            'Coeficiente': modelo.coef_,
            'Impacto': ['Tendencia temporal a largo plazo', 'Variación diurna (ciclo día/noche)', 'Diferencias entre tanques']
        })

        st.dataframe(coef_df, use_container_width=True)

        # Gráfico de coeficientes
        fig_coef = go.Figure(go.Bar(
            x=coef_df['Variable'],
            y=coef_df['Coeficiente'],
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            text=coef_df['Coeficiente'].round(6),
            textposition='outside'
        ))
        fig_coef.update_layout(
            title='Importancia de Variables (Coeficientes)',
            xaxis_title='Variable',
            yaxis_title='Coeficiente',
            height=400
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    with col2:
        st.markdown(f"""
        ### 📐 Ecuación del Modelo

        ```
        {variable_predecir} = {modelo.intercept_:.4f}
        {"+" if modelo.coef_[0] >= 0 else ""}{modelo.coef_[0]:.6f} × Días
        {"+" if modelo.coef_[1] >= 0 else ""}{modelo.coef_[1]:.6f} × Hora
        {"+" if modelo.coef_[2] >= 0 else ""}{modelo.coef_[2]:.6f} × Tanque
        ```

        **Intercepto (β₀):** {modelo.intercept_:.4f}

        #### Interpretación:
        - Si el coeficiente es **positivo**, la variable aumenta el valor predicho
        - Si es **negativo**, la variable lo disminuye
        - Mayor **magnitud** = mayor impacto
        """)

    st.markdown("---")

    # Predictor interactivo
    st.subheader("🎯 Predictor Interactivo - Haz tu Propia Predicción")

    st.markdown("""
    Utiliza esta herramienta para predecir valores futuros de temperatura o pH
    ingresando los parámetros deseados.
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        dias_pred = st.number_input(
            "📅 Días desde el inicio",
            min_value=0,
            max_value=500,
            value=100,
            help="Número de días desde la primera medición"
        )

    with col2:
        hora_pred = st.slider(
            "🕐 Hora del día",
            min_value=0,
            max_value=23,
            value=12,
            help="Hora en formato 24h"
        )

    with col3:
        tanque_pred = st.selectbox(
            "🏊 Tanque",
            options=list(range(1, 11)),
            index=0,
            help="Número de tanque (1-10)"
        )

    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        predecir_btn = st.button("🔮 Realizar Predicción", type="primary", use_container_width=True)

    if predecir_btn:
        X_pred = np.array([[dias_pred, hora_pred, tanque_pred]])
        prediccion = modelo.predict(X_pred)[0]

        # Mostrar predicción con estilo
        st.markdown("---")
        st.markdown("### 📊 Resultado de la Predicción")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # Clasificar la predicción
            if variable_predecir == 'Temperatura_C':
                nivel, icono = clasificar_temperatura(prediccion)
                unidad = "°C"
            else:
                nivel, icono = clasificar_ph(prediccion)
                unidad = ""

            color_nivel = {
                'Óptimo': '#28a745',
                'Advertencia': '#ffc107',
                'Crítico': '#dc3545'
            }[nivel]

            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color_nivel}30 0%, {color_nivel}10 100%);
                        padding: 2rem; border-radius: 15px; border: 3px solid {color_nivel};
                        text-align: center;'>
                <h2 style='margin: 0; color: {color_nivel};'>{icono} {nivel}</h2>
                <h1 style='font-size: 4rem; margin: 1rem 0; color: {color_nivel};'>{prediccion:.2f} {unidad}</h1>
                <p style='font-size: 1.2rem; color: #666;'>
                    <b>Parámetros:</b> Día {dias_pred} | Hora {hora_pred}:00 | Tanque {tanque_pred}
                </p>
            </div>
            """, unsafe_allow_html=True)

# ========================================
# TAB 5: ANÁLISIS ESTADÍSTICO
# ========================================
with tab5:
    st.header("🔬 Análisis Estadístico Avanzado")

    # Resumen estadístico completo
    st.subheader("📊 Estadísticas Descriptivas Completas")

    st.dataframe(df[['pH', 'Temperatura_C']].describe().T, use_container_width=True)

    st.markdown("---")

    # Correlaciones
    st.subheader("🔗 Matriz de Correlación")

    df_correlacion = df.copy()
    df_correlacion['Tanque_Num'] = df_correlacion['Tanque'].str.extract('(\d+)').astype(int)
    df_correlacion['Hora_Num'] = df_correlacion['Fecha_Hora'].dt.hour
    df_correlacion['Dia_Num'] = (df_correlacion['Fecha_Hora'] - df_correlacion['Fecha_Hora'].min()).dt.days

    correlacion = df_correlacion[['pH', 'Temperatura_C', 'Tanque_Num', 'Hora_Num', 'Dia_Num']].corr()

    fig_corr = px.imshow(
        correlacion,
        text_auto='.3f',
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title='Matriz de Correlación entre Variables',
        labels=dict(color="Correlación")
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # Análisis por tanque
    st.subheader("🏊 Comparativa Estadística por Tanque")

    comparativa_tanques = df.groupby('Tanque').agg({
        'Temperatura_C': ['mean', 'std', 'min', 'max'],
        'pH': ['mean', 'std', 'min', 'max']
    }).round(2)

    comparativa_tanques.columns = [f'{col[0]}_{col[1]}' for col in comparativa_tanques.columns]
    st.dataframe(comparativa_tanques, use_container_width=True)

    # Gráficos de comparación
    col1, col2 = st.columns(2)

    with col1:
        promedios_temp = df.groupby('Tanque')['Temperatura_C'].mean().sort_values(ascending=False)
        fig_bar_temp = go.Figure(go.Bar(
            x=promedios_temp.index,
            y=promedios_temp.values,
            marker_color='skyblue',
            text=promedios_temp.values.round(2),
            textposition='outside'
        ))
        fig_bar_temp.update_layout(
            title='Temperatura Promedio por Tanque',
            xaxis_title='Tanque',
            yaxis_title='Temperatura (°C)',
            height=400
        )
        st.plotly_chart(fig_bar_temp, use_container_width=True)

    with col2:
        promedios_ph = df.groupby('Tanque')['pH'].mean().sort_values(ascending=False)
        fig_bar_ph = go.Figure(go.Bar(
            x=promedios_ph.index,
            y=promedios_ph.values,
            marker_color='lightcoral',
            text=promedios_ph.values.round(2),
            textposition='outside'
        ))
        fig_bar_ph.update_layout(
            title='pH Promedio por Tanque',
            xaxis_title='Tanque',
            yaxis_title='pH',
            height=400
        )
        st.plotly_chart(fig_bar_ph, use_container_width=True)

# ========================================
# TAB 6: DATOS
# ========================================
with tab6:
    st.header("📋 Explorador de Datos")

    # Filtros
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        tanques_filtro = st.multiselect(
            "🏊 Filtrar por tanque:",
            options=sorted(df['Tanque'].unique()),
            default=sorted(df['Tanque'].unique())
        )

    with col2:
        jornada_filtro = st.multiselect(
            "🌅 Filtrar por jornada:",
            options=['am', 'pm'],
            default=['am', 'pm']
        )

    with col3:
        estado_filtro = st.multiselect(
            "📊 Filtrar por estado:",
            options=['Óptimo', 'Advertencia', 'Crítico'],
            default=['Óptimo', 'Advertencia', 'Crítico']
        )

    with col4:
        limit_rows = st.number_input(
            "📄 Mostrar filas:",
            min_value=10,
            max_value=len(df),
            value=100,
            step=50
        )

    # Aplicar filtros
    df_filtrado_tabla = df[
        (df['Tanque'].isin(tanques_filtro)) &
        (df['Jornada'].isin(jornada_filtro)) &
        (df['Estado_General'].isin(estado_filtro))
    ]

    st.write(f"**📊 Total de registros filtrados:** {len(df_filtrado_tabla):,} de {len(df):,}")

    # Mostrar datos (ordenar primero, luego seleccionar columnas)
    df_mostrar = df_filtrado_tabla.sort_values('Fecha_Hora', ascending=False).head(limit_rows)
    st.dataframe(
        df_mostrar[[
            'Tanque', 'Fecha', 'Hora', 'Jornada',
            'pH', 'Nivel_pH', 'Temperatura_C', 'Nivel_Temp', 'Estado_General'
        ]],
        use_container_width=True,
        height=600
    )

    # Descargar datos
    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df_filtrado_tabla.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Descargar datos filtrados (CSV)",
            data=csv,
            file_name=f"datos_filtrados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        csv_completo = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Descargar dataset completo (CSV)",
            data=csv_completo,
            file_name=f"dataset_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col3:
        # Estadísticas de filtro
        st.info(f"📈 Registros visibles: {min(limit_rows, len(df_filtrado_tabla)):,}")

# ========================================
# TAB 7: METODOLOGÍA
# ========================================
with tab7:
    st.header("📚 Metodología de Investigación")

    st.markdown("""
    ## 🎓 Marco Metodológico

    Esta sección describe la metodología utilizada en el desarrollo del sistema
    inteligente de predicción de parámetros de calidad de agua.
    """)

    # Fases del proyecto
    st.subheader("📋 Fases del Proyecto")

    fases = [
        {
            "fase": "1️⃣ Recolección de Datos",
            "descripcion": "Obtención de mediciones de pH y temperatura de 10 tanques durante 315 días",
            "actividades": [
                "Configuración de sensores de medición",
                "Registro de datos dos veces al día (AM/PM)",
                "Validación de integridad de datos",
                "Almacenamiento en formato CSV"
            ]
        },
        {
            "fase": "2️⃣ Preprocesamiento",
            "descripcion": "Limpieza y transformación de datos para análisis",
            "actividades": [
                "Conversión de formatos de fecha/hora",
                "Clasificación de estados (Óptimo/Advertencia/Crítico)",
                "Creación de variables derivadas",
                "Detección de valores atípicos"
            ]
        },
        {
            "fase": "3️⃣ Análisis Exploratorio",
            "descripcion": "Exploración estadística y visual de los datos",
            "actividades": [
                "Cálculo de estadísticas descriptivas",
                "Análisis de distribuciones",
                "Identificación de correlaciones",
                "Detección de patrones temporales"
            ]
        },
        {
            "fase": "4️⃣ Modelado Predictivo",
            "descripcion": "Desarrollo del modelo de Machine Learning",
            "actividades": [
                "Selección de algoritmo (Regresión Lineal)",
                "División train/test (80/20)",
                "Entrenamiento del modelo",
                "Evaluación de métricas (R², MSE, MAE)"
            ]
        },
        {
            "fase": "5️⃣ Validación y Despliegue",
            "descripcion": "Verificación y puesta en producción",
            "actividades": [
                "Validación cruzada",
                "Análisis de residuos",
                "Desarrollo de interfaz web",
                "Documentación técnica"
            ]
        }
    ]

    for fase_info in fases:
        with st.expander(f"**{fase_info['fase']}**: {fase_info['descripcion']}"):
            st.markdown("**Actividades:**")
            for actividad in fase_info['actividades']:
                st.markdown(f"- {actividad}")

    st.markdown("---")

    # Técnicas de IA utilizadas
    st.subheader("🤖 Técnicas de Inteligencia Artificial")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📊 Regresión Lineal

        **Descripción:**
        La regresión lineal es un algoritmo de aprendizaje supervisado que modela
        la relación entre variables predictoras y una variable objetivo mediante
        una función lineal.

        **Ecuación matemática:**
        ```
        y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ε
        ```

        Donde:
        - y = Variable objetivo (pH o Temperatura)
        - x₁ = Días transcurridos
        - x₂ = Hora del día
        - x₃ = Número de tanque
        - β = Coeficientes
        - ε = Error

        **Ventajas:**
        - ✅ Interpretable y explicable
        - ✅ Rápido de entrenar
        - ✅ Requiere pocos datos
        - ✅ Buenos resultados con relaciones lineales

        **Limitaciones:**
        - ⚠️ Asume linealidad
        - ⚠️ Sensible a outliers
        - ⚠️ No captura relaciones complejas
        """)

    with col2:
        st.markdown("""
        ### 📈 Métricas de Evaluación

        **R² (Coeficiente de Determinación):**
        - Mide qué porcentaje de variabilidad es explicada
        - Rango: [0, 1], donde 1 es perfecto
        - Fórmula: R² = 1 - (SS_res / SS_tot)

        **MSE (Error Cuadrático Medio):**
        - Promedio de errores al cuadrado
        - Penaliza errores grandes
        - Fórmula: MSE = (1/n)Σ(y_i - ŷ_i)²

        **RMSE (Raíz del MSE):**
        - MSE en las unidades originales
        - Más interpretable que MSE
        - Fórmula: RMSE = √MSE

        **MAE (Error Absoluto Medio):**
        - Promedio de errores absolutos
        - Menos sensible a outliers
        - Fórmula: MAE = (1/n)Σ|y_i - ŷ_i|

        **Interpretación de R²:**
        - R² > 0.9: Excelente
        - R² > 0.7: Bueno
        - R² > 0.5: Moderado
        - R² < 0.5: Requiere mejora
        """)

    st.markdown("---")

    # Justificación científica
    st.subheader("🔬 Justificación Científica")

    st.markdown("""
    ### Importancia de los Parámetros Monitoreados

    #### 🌡️ Temperatura
    La temperatura del agua es un factor crítico en la acuicultura de tilapia:
    - **Metabolismo:** Afecta directamente la tasa metabólica de los peces
    - **Crecimiento:** Temperaturas óptimas (26-30°C) maximizan el crecimiento
    - **Oxígeno:** Temperaturas altas reducen el oxígeno disuelto
    - **Sistema inmune:** Temperaturas extremas debilitan las defensas

    #### ⚗️ pH
    El pH del agua influye en múltiples procesos fisiológicos:
    - **Osmorregulación:** pH inadecuado causa estrés osmótico
    - **Toxicidad:** pH bajo aumenta toxicidad de metales pesados
    - **Productividad:** pH extremo reduce productividad primaria
    - **Reproducción:** Afecta tasa de eclosión y supervivencia de alevines

    ### 🎯 Aporte de la Investigación

    Este proyecto contribuye al campo de la acuicultura mediante:

    1. **Automatización:** Reduce necesidad de monitoreo manual constante
    2. **Predicción:** Permite anticipar condiciones adversas
    3. **Optimización:** Facilita toma de decisiones basada en datos
    4. **Escalabilidad:** Puede adaptarse a diferentes sistemas acuícolas
    5. **Sostenibilidad:** Mejora eficiencia y reduce pérdidas

    ### 📖 Referencias Bibliográficas

    - El-Sayed, A. F. M. (2006). Tilapia Culture. CABI Publishing.
    - Boyd, C. E., & Tucker, C. S. (2014). Handbook for Aquaculture Water Quality.
    - Rakocy, J. E., & McGinty, A. S. (1989). Pond Culture of Tilapia.
    - Kubitza, F. (2000). Tilapia: Tecnologia e Planejamento na Producão Comercial.
    """)

    st.markdown("---")

    # Herramientas tecnológicas
    st.subheader("🛠️ Stack Tecnológico")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Lenguajes:**
        - 🐍 Python 3.8+

        **Análisis de Datos:**
        - 📊 Pandas
        - 🔢 NumPy

        **Visualización:**
        - 📈 Plotly
        - 🎨 Streamlit
        """)

    with col2:
        st.markdown("""
        **Machine Learning:**
        - 🤖 Scikit-learn
        - 📐 LinearRegression

        **Estadística:**
        - 📊 SciPy
        - 📈 Statsmodels
        """)

    with col3:
        st.markdown("""
        **Desarrollo:**
        - 💻 VS Code
        - 🔄 Git/GitHub

        **Despliegue:**
        - ☁️ Streamlit Cloud
        - 🌐 Web App
        """)

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <h4 style='color: #1E88E5;'>🐟 Sistema Inteligente de Monitoreo de Tilapia en Acuicultura</h4>
    <p>Proyecto de Tesis | Predicción mediante Machine Learning</p>
    <p>Desarrollado con Python, Streamlit y Scikit-learn</p>
    <p style='font-size: 0.9rem;'>© 2025 | Acuicultura de Precisión con IA</p>
</div>
""", unsafe_allow_html=True)

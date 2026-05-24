import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# CONFIGURACIÓN GENERAL Y ESTILO VISUAL "POWERPOINT PREMIUM"
# ==============================================================================
st.set_page_config(page_title="Exposición Avanzada - Orden Público", layout="wide", initial_sidebar_state="collapsed")

# Inyección de CSS Avanzado para mejorar estética de botones y fondos
st.markdown("""
    <style>
    /* Fondo principal claro y limpio estilo diapositiva */
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Estilo para los botones de navegación */
    div.stButton > button {
        background-color: #FFFFFF;
        color: #475569;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 15px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    div.stButton > button:hover {
        border-color: #2563EB;
        color: #2563EB;
        background-color: #EFF6FF;
    }

    /* Color del botón cuando la diapositiva está activa */
    /* Nota: Streamlit usa clases dinámicas, este selector ayuda al estilo general */
    .st-emotion-cache-7ym5gk { 
        background-color: #2563EB !important;
        color: white !important;
        border: none !important;
    }

    /* Contenedor de la diapositiva */
    .slide-container {
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        border: 1px solid #E2E8F0;
    }
    
    /* Estilos de títulos estilo McKinsey */
    .slide-title {
        color: #0F172A;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .slide-subtitle {
        color: #64748B;
        font-size: 19px;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* Tarjetas de insights */
    .insight-card {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .insight-critical {
        background-color: #FFF1F2;
        border-left: 5px solid #E11D48;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .insight-success {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Inicialización del paginador (diapositivas)
if 'diapositiva' not in st.session_state:
    st.session_state.diapositiva = 1

def ir_a_diapositiva(num):
    st.session_state.diapositiva = num
    st.rerun()

# ==============================================================================
# CARGA AUTOMÁTICA DE DATOS 
# ==============================================================================
def cargar_datos_automatico():
    archivos_en_carpeta = os.listdir('.')
    archivo_encontrado = None
    for archivo in archivos_en_carpeta:
        nombre_minuscula = archivo.lower()
        if ("afectacion" in nombre_minuscula or "fuerza" in nombre_minuscula or "publica" in nombre_minuscula) and (archivo.endswith('.csv') or archivo.endswith('.xlsx')):
            archivo_encontrado = archivo
            break
            
    if archivo_encontrado is None:
        return None, "No se encontró el registro de datos."
    try:
        if archivo_encontrado.endswith('.csv'):
            df = pd.read_csv(archivo_encontrado, header=0)
        else:
            df = pd.read_excel(archivo_encontrado, header=0)
        return df, archivo_encontrado
    except Exception as e:
        return None, f"Error: {str(e)}"

df_original, nombre_archivo_cargado = cargar_datos_automatico()

# ==============================================================================
# NAVEGACIÓN SUPERIOR (BOTONES ESTILIZADOS)
# ==============================================================================
cols_nav = st.columns(6)
nombres_diapo = ["🏠 Portada", "🎯 Introducción", "📖 Teoría", "⚙️ Metodología", "📊 Resultados", "🏁 Conclusiones"]

for i, nombre in enumerate(nombres_diapo):
    # Resaltar la diapositiva actual
    tipo_boton = "primary" if st.session_state.diapositiva == (i + 1) else "secondary"
    if cols_nav[i].button(nombre, use_container_width=True, type=tipo_boton):
        ir_a_diapositiva(i + 1)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# DIAPOSITIVA 1: PORTADA
# ==============================================================================
if st.session_state.diapositiva == 1:
    st.markdown("""
    <div class='slide-container' style='text-align: center; padding: 80px 40px;'>
        <img src='https://www.ut.edu.co/images/logos/logo_ut.png' width='160' style='margin-bottom: 25px;'>
        <div class='slide-title' style='font-size: 48px; color: #0F172A;'>Análisis Estadístico Avanzado de la Fuerza Pública</div>
        <div class='slide-subtitle' style='font-size: 24px; color: #3B82F6;'>Segmentación Territorial mediante Modelos de Aprendizaje Automático</div>
        <div style='margin: 40px auto; width: 100px; border-top: 4px solid #2563EB;'></div>
        <p style='font-size: 20px; color: #475569;'><b>Miguel Angel Garatejo</b><br>Universidad del Tolima</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Iniciar Sustentación ➡️", type="primary"):
        ir_a_diapositiva(2)

# ==============================================================================
# DIAPOSITIVA 2: INTRODUCCIÓN
# ==============================================================================
elif st.session_state.diapositiva == 2:
    st.markdown("<div class='slide-title'>🎯 Introducción y Desafío Técnico</div>", unsafe_allow_html=True)
    st.markdown("<div class='slide-subtitle'>Transformando datos cualitativos en inteligencia operativa</div>", unsafe_allow_html=True)
    
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
        <div class='slide-container'>
            <h3 style='color: #E11D48; margin-top:0;'>🛑 El Problema</h3>
            <p>El registro original consta de <b>8 variables de texto</b> y solo <b>1 numérica</b>. Esto impide aplicar algoritmos de agrupamiento tradicionales que requieren distancias matemáticas.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_i2:
        st.markdown("""
        <div class='slide-container'>
            <h3 style='color: #2563EB; margin-top:0;'>💡 La Solución</h3>
            <p>Implementar un flujo de <b>Ingeniería de Características</b> que transforme el texto en una matriz de frecuencias, permitiendo medir la vulnerabilidad real de cada municipio.</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# DIAPOSITIVA 3: MARCO TEÓRICO
# ==============================================================================
elif st.session_state.diapositiva == 3:
    st.markdown("<div class='slide-title'>📖 Marco Teórico y Sustentación</div>", unsafe_allow_html=True)
    
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.markdown("<div class='insight-card'><h4>🔄 Pivotado</h4><p>Conversión de categorías a dimensiones numéricas.</p></div>", unsafe_allow_html=True)
    with t_col2:
        st.markdown("<div class='insight-card'><h4>📐 K-Means</h4><p>Agrupamiento por minimización de varianza interna.</p></div>", unsafe_allow_html=True)
    with t_col3:
        st.markdown("<div class='insight-card'><h4>🌐 PCA</h4><p>Reducción de dimensiones para visualización 3D.</p></div>", unsafe_allow_html=True)

# ==============================================================================
# DIAPOSITIVA 5: RESULTADOS (ANÁLISIS PROFUNDO Y COLORES BONITOS)
# ==============================================================================
elif st.session_state.diapositiva == 5:
    st.markdown("<div class='slide-title'>📊 Resultados y Hallazgos del Modelo</div>", unsafe_allow_html=True)
    
    if df_original is not None:
        # --- Lógica de procesamiento (No se toca) ---
        index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
        pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
        total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')
        datos = total_municipio.join([pivot_accion]).reset_index().dropna()
        scaler = StandardScaler()
        numericas = [col for col in datos.columns if col not in index_cols]
        datos[numericas] = scaler.fit_transform(datos[numericas])
        X_scaled = datos.drop(columns=index_cols)
        kmeans = KMeans(n_clusters=4, n_init=30, random_state=42)
        km4_clusters = kmeans.fit(X_scaled)
        datos['Cluster'] = km4_clusters.labels_.astype(str)

        # --- Gráficas con Estética Mejorada ---
        
        # A. CURVA DEL CODO (Fondo blanco y líneas elegantes)
        st.markdown("### A. Optimización del número de grupos (K)")
        wss = []
        for k in range(1, 11):
            km_test = KMeans(n_clusters=k, n_init=15, random_state=42)
            km_test.fit(X_scaled)
            wss.append(km_test.inertia_)
        
        fig_elbow = px.line(x=list(range(1, 11)), y=wss, markers=True, template='plotly_white')
        fig_elbow.update_traces(line_color='#2563EB', marker=dict(size=10, color='#1E3A8A'))
        fig_elbow.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                margin=dict(t=10), xaxis_title="Número de Clústeres", yaxis_title="Inercia")
        st.plotly_chart(fig_elbow, use_container_width=True)

        # B. MAPA DE CALOR (Colores más bonitos)
        st.markdown("### B. Matriz de Disimilitudes (Distancia Euclideana)")
        distancias_eu = euclidean_distances(X_scaled)[:40, :40]
        nombres_municipios_sub = datos['MUNICIPIO'].iloc[:40].tolist()
        
        fig_eu = px.imshow(distancias_eu, x=nombres_municipios_sub, y=nombres_municipios_sub,
                           color_continuous_scale='GnBu', template='plotly_white') # Verde-Azulado elegante
        fig_eu.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                             margin=dict(t=10))
        st.plotly_chart(fig_eu, use_container_width=True)

        # C. PCA 3D (Colores ejecutivos)
        st.markdown("### C. Mapa Tridimensional de Riesgos (PCA)")
        pca_3d = PCA(n_components=3)
        scores_pca = pca_3d.fit_transform(X_scaled)
        df_pca = pd.DataFrame(scores_pca, columns=['PC1', 'PC2', 'PC3'])
        df_pca['Cluster'] = datos['Cluster']
        df_pca['Municipio'] = datos['MUNICIPIO']
        
        fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster', 
                               hover_name='Municipio', template='plotly_white',
                               color_discrete_sequence=['#10B981', '#3B82F6', '#F59E0B', '#EF4444']) # Esmeralda, Azul, Ámbar, Rosa
        fig_3d.update_layout(scene=dict(bgcolor='white'), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("""
        <div class='insight-card'>
            <b>Análisis de Fondo:</b> Los municipios alejados del centro (Clúster Rojo) representan atipicidades críticas que requieren intervención inmediata, mientras que el bloque central (Verde) muestra estabilidad operativa.
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# DIAPOSITIVA 6: CONCLUSIONES
# ==============================================================================
elif st.session_state.diapositiva == 6:
    st.markdown("<div class='slide-title'>🏁 Conclusiones Estratégicas</div>", unsafe_allow_html=True)
    
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.markdown("""
        <div class='slide-container' style='height: 350px;'>
            <h3 style='color:#2563EB;'>📌 Puntos Clave</h3>
            <ul>
                <li>Se logró normalizar registros heterogéneos mediante pivotado dinámico.</li>
                <li>El modelo aisló con precisión los focos de violencia crítica (Outliers).</li>
                <li>La arquitectura es escalable para nuevos registros mensuales.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c_col2:
        st.markdown("""
        <div class='slide-container' style='height: 350px;'>
            <h3 style='color:#10B981;'>🚀 Recomendación</h3>
            <p>Utilizar los centroides del Clúster 3 para priorizar el despliegue de asistencia médica y apoyo logístico en zonas de alta letalidad.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<h2 style='text-align:center; color:#1E3A8A;'>¡Muchas gracias por su atención!</h2>", unsafe_allow_html=True)
    if st.button("↩️ Volver al Inicio", type="secondary"): ir_a_diapositiva(1)

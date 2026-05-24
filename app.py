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

# Inyección de CSS para máxima claridad y elegancia en botones y contenedores
st.markdown("""
    <style>
    /* Fondo principal claro (Blanco Nieve) */
    .stApp {
        background-color: #FFFFFF;
        color: #1E293B;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Contenedor de la diapositiva en blanco puro con sombra sutil */
    .slide-container {
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.03);
        margin-bottom: 25px;
        border: 1px solid #F1F5F9;
    }
    
    /* Títulos estilo consultoría */
    .slide-title {
        color: #0F172A;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .slide-subtitle {
        color: #64748B;
        font-size: 18px;
        margin-bottom: 25px;
        font-weight: 400;
    }

    /* BOTONES SUPERIORES: Claros, limpios y con texto legible */
    div.stButton > button {
        background-color: #F8FAFC !important;
        color: #0F2042 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        padding: 10px 15px !important;
    }
    div.stButton > button:hover {
        background-color: #F1F5F9 !important;
        border-color: #C5A059 !important;
    }
    /* Estilo para el botón de la página seleccionada */
    div.stButton > button[kind="primary"] {
        background-color: #0F2042 !important;
        color: #FFFFFF !important;
        border: 1px solid #0F2042 !important;
        box-shadow: 0 4px 12px rgba(15, 32, 66, 0.15) !important;
    }

    /* Tarjetas de insights */
    .insight-card {
        background-color: #F8FAFC;
        border-left: 5px solid #0F2042;
        padding: 18px;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Lógica de navegación
if 'diapositiva' not in st.session_state:
    st.session_state.diapositiva = 1

def ir_a_diapositiva(num):
    st.session_state.diapositiva = num
    st.rerun()

# Carga de datos
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
        df = pd.read_csv(archivo_encontrado) if archivo_encontrado.endswith('.csv') else pd.read_excel(archivo_encontrado)
        return df, archivo_encontrado
    except:
        return None, "Error al cargar."

df_original, _ = cargar_datos_automatico()

# Navegación Superior
cols_nav = st.columns(6)
nombres_diapo = ["🏠 1. Portada", "🎯 2. Introducción", "📖 3. Marco Teórico", "⚙️ 4. Metodología", "📊 5. Resultados", "🏁 6. Conclusiones"]

for i, nombre in enumerate(nombres_diapo):
    tipo = "primary" if st.session_state.diapositiva == (i + 1) else "secondary"
    if cols_nav[i].button(nombre, use_container_width=True, type=tipo):
        ir_a_diapositiva(i + 1)

st.markdown("---")

# ==============================================================================
# CONTENIDO DE LAS DIAPOSITIVAS
# ==============================================================================

if st.session_state.diapositiva == 1:
    st.markdown("""
    <div class='slide-container' style='text-align: center; padding: 80px 40px;'>
        <img src='https://www.ut.edu.co/images/logos/logo_ut.png' width='160'>
        <h1 style='color: #0F2042; font-size: 48px;'>Análisis Avanzado de la Fuerza Pública</h1>
        <p style='font-size: 22px; color: #64748B;'>Segmentación de Vulnerabilidad Territorial mediante Machine Learning</p>
        <p style='margin-top: 40px;'><b>Miguel Angel Garatejo</b> | <b>Prof. Yuri Saavedra</b></p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.diapositiva == 2:
    st.markdown("<div class='slide-title'>🎯 Introducción y Problema</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='slide-container'>
        <p>Los datos originales presentan <b>8 variables cualitativas</b> y solo <b>1 numérica</b>. 
        El reto es transformar el texto en geometría para poder agrupar los municipios por su nivel de riesgo real.</p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.diapositiva == 3:
    st.markdown("<div class='slide-title'>📖 Marco Teórico</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.markdown("<div class='insight-card'><h4>Reshaping</h4><p>Convertimos categorías en números.</p></div>", unsafe_allow_html=True)
    col2.markdown("<div class='insight-card'><h4>K-Means</h4><p>Agrupamos por distancias euclidianas.</p></div>", unsafe_allow_html=True)
    col3.markdown("<div class='insight-card'><h4>PCA</h4><p>Reducimos dimensiones para ver el riesgo.</p></div>", unsafe_allow_html=True)

elif st.session_state.diapositiva == 4:
    st.markdown("<div class='slide-title'>⚙️ Metodología</div>", unsafe_allow_html=True)
    st.code("scaler = StandardScaler()\nX_scaled = scaler.fit_transform(datos_pivotados)", language="python")

# ==============================================================================
# DIAPOSITIVA 5: RESULTADOS (CON GRÁFICAS ACLARADAS)
# ==============================================================================
elif st.session_state.diapositiva == 5:
    st.markdown("<div class='slide-title'>📊 Resultados y Hallazgos Estratégicos</div>", unsafe_allow_html=True)
    
    if df_original is not None:
        # Procesamiento rápido
        idx = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
        pivot = df_original.pivot_table(index=idx, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
        datos = df_original.groupby(idx)['CANTIDAD'].sum().to_frame(name='TOTAL').join(pivot).reset_index().dropna()
        
        num_cols = [c for c in datos.columns if c not in idx]
        X_scaled = StandardScaler().fit_transform(datos[num_cols])
        
        kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
        datos['Cluster'] = kmeans.fit_predict(X_scaled).astype(str)

        # --- GRÁFICA 1: MÉTODO DEL CODO (FONDO ACLARADO) ---
        wss = []
        for k in range(1, 11):
            wss.append(KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled).inertia_)
        
        fig_elbow = px.line(x=list(range(1, 11)), y=wss, markers=True, template="plotly_white")
        fig_elbow.update_layout(
            title="Optimización de Clústeres (K=4)",
            paper_bgcolor='rgba(0,0,0,0)', # Transparente
            plot_bgcolor='rgba(0,0,0,0)',  # Transparente
            xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
        )
        fig_elbow.update_traces(line_color='#0F2042', marker=dict(color='#C5A059', size=10))
        st.plotly_chart(fig_elbow, use_container_width=True)

        # --- GRÁFICA 2: MAPA DE CALOR (FONDO ACLARADO) ---
        st.markdown("### 🗺️ Matriz de Disimilitud (Primeros 40 municipios)")
        dist = euclidean_distances(X_scaled[:40, :])
        fig_hm = px.imshow(dist, x=datos['MUNICIPIO'][:40], y=datos['MUNICIPIO'][:40], 
                           color_continuous_scale='YlGnBu', template="plotly_white")
        fig_hm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # --- GRÁFICA 3: PCA 3D (MÁXIMA CLARIDAD) ---
        st.markdown("### 🌐 Proyección Tridimensional del Riesgo")
        pca = PCA(n_components=3)
        coords = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(coords, columns=['PC1', 'PC2', 'PC3'])
        df_pca['Cluster'] = datos['Cluster'].map({"0":"Estabilidad","1":"Moderado","2":"Foco Inst.","3":"Crítico"})
        df_pca['Muni'] = datos['MUNICIPIO']
        
        fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster',
                               hover_name='Muni', template="plotly_white",
                               color_discrete_sequence=['#10B981', '#3B82F6', '#F59E0B', '#EF4444'])
        
        # Aclarar el fondo del cubo 3D
        fig_3d.update_layout(
            scene = dict(
                xaxis = dict(backgroundcolor="white", gridcolor="whitesmoke", showbackground=True, zerolinecolor="white"),
                yaxis = dict(backgroundcolor="white", gridcolor="whitesmoke", showbackground=True, zerolinecolor="white"),
                zaxis = dict(backgroundcolor="white", gridcolor="whitesmoke", showbackground=True, zerolinecolor="white"),
                bgcolor = "white" # Fondo del área 3D blanco puro
            ),
            paper_bgcolor='white', # Fondo exterior blanco puro
            margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("""
        <div class='insight-card'>
            <b>Hallazgo:</b> La claridad en la separación de los puntos confirma que el modelo ha identificado con éxito 
            municipios con comportamientos <b>

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

# CONFIGURACIÓN GENERAL Y ESTILO VISUAL "POWERPOINT PREMIUM"
st.set_page_config(page_title="Exposición Avanzada - Orden Público", layout="wide", initial_sidebar_state="collapsed")

# Inyección de CSS Avanzado para simular Diapositivas de Consultoría (Fondo Claro y Elegante)
st.markdown("""
    <style>
    /* Fondo principal claro y limpio estilo diapositiva */
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Ocultar barra lateral por defecto */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    /* Contenedor de la diapositiva */
    .slide-container {
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        border: 1px solid #E2E8F0;
    }
    /* Estilos de títulos */
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
    
    /* Botones superiores */
    div.stButton > button {
        background-color: #E0F2FE !important;
        color: #0369A1 !important;
        border: 1px solid #BAE6FD !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease-in-out !important;
    }
    
    div.stButton > button:hover {
        background-color: #7DD3FC !important;
        box-shadow: 0 4px 12px rgba(3, 105, 161, 0.15) !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border: 1px solid #0284C7 !important;
    }

    .insight-card {
        background-color: #F1F5F9;
        border-left: 5px solid #38BDF8;
        padding: 18px;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 15px;
    }
    .insight-critical {
        background-color: #FEF2F2;
        border-left: 5px solid #DC2626;
        padding: 18px;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 15px;
    }
    .insight-success {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 18px;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Inicialización del paginador
if 'diapositiva' not in st.session_state:
    st.session_state.diapositiva = 1

def ir_a_diapositiva(num):
    st.session_state.diapositiva = num
    st.rerun()

# CARGA DE DATOS
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
        return None, str(e)

df_original, nombre_archivo_cargado = cargar_datos_automatico()

# NAVEGACIÓN
cols_nav = st.columns(6)
nombres_diapo = ["1. Portada", "2. Introducción", "3. Marco Teórico", "4. Metodología", "5. Resultados", "6. Conclusiones"]

for i, nombre in enumerate(nombres_diapo):
    tipo_boton = "primary" if st.session_state.diapositiva == (i + 1) else "secondary"
    if cols_nav[i].button(nombre, use_container_width=True, type=tipo_boton):
        ir_a_diapositiva(i + 1)

st.markdown("---")

# --- DIAPOSITIVAS ---

if st.session_state.diapositiva == 1:
    st.markdown("""
    <div class='slide-container' style='text-align: center; padding: 60px 40px;'>
        <img src='https://administrativos.ut.edu.co/images/Home/simbolos/logo_oficial.png' width='197' style='margin-bottom: 20px;'>
        <div class='slide-title' style='font-size: 42px; color: #1E3A8A;'>Análisis de Clústeres (K-Means) En Afectaciones a la Fuerza Pública</div>
        <div class='slide-subtitle' style='font-size: 22px;'>Segmentación Territorial de Incidentes de Orden Público Mediante Modelos de Aprendizaje no Supervisados</div>
        <div style='margin: 40px 0; border-top: 2px solid #E2E8F0;'></div>
    </div>
    """, unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("<div class='insight-card'><h4>ESTUDIANTE</h4><p><b>Miguel Angel Garatejo</b></p></div>", unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"<div class='insight-success'><h4>PROFESOR</h4><p><b>Yuri Marcela Garcia Saavedra</b></p></div>", unsafe_allow_html=True)
    if st.button("Iniciar Sustentación", type="primary", use_container_width=True):
        ir_a_diapositiva(2)

elif st.session_state.diapositiva == 2:
    st.markdown("<div class='slide-title'>Introducción</div>", unsafe_allow_html=True)
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("<div class='slide-container'><h3>El Problema</h3><p>Los datos son cualitativos y dificultan el análisis matemático directo.</p></div>", unsafe_allow_html=True)
    with col_i2:
        st.markdown("<div class='slide-container'><h3>Objetivo</h3><p>Reestructurar datos para aplicar K-Means y agrupar municipios.</p></div>", unsafe_allow_html=True)
    if st.button("Siguiente Diapositiva ➡️", type="primary"):
        ir_a_diapositiva(3)

elif st.session_state.diapositiva == 3:
    st.markdown("<div class='slide-title'>Marco Teórico</div>", unsafe_allow_html=True)
    st.markdown("<div class='slide-container'><h4>Algoritmos Utilizados</h4><p>Pivotado, K-Means y PCA (Reducción de dimensiones).</p></div>", unsafe_allow_html=True)
    if st.button("Siguiente Diapositiva ➡️", type="primary"):
        ir_a_diapositiva(4)

elif st.session_state.diapositiva == 4:
    st.markdown("<div class='slide-title'>Metodología</div>", unsafe_allow_html=True)
    st.markdown("<div class='slide-container'><p>Fases: 1. Pivotado | 2. Normalización | 3. Optimización.</p></div>", unsafe_allow_html=True)
    if st.button("Siguiente Diapositiva ➡️", type="primary"):
        ir_a_diapositiva(5)

# ==============================================================================
# DIAPOSITIVA 5: RESULTADOS (AQUÍ ESTÁN LOS CAMBIOS SOLICITADOS)
# ==============================================================================
elif st.session_state.diapositiva == 5:
    st.markdown("<div class='slide-title'>Resultados y Análisis de Clústeres</div>", unsafe_allow_html=True)
    
    if df_original is None:
        st.error("Archivo no encontrado.")
        st.stop()

    # --- Lógica de Datos ---
    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
    pivot_fuerza = df_original.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0) if 'NOMBRE_FUERZA' in df_original.columns else pd.DataFrame(index=pivot_accion.index)
    total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')
    datos = total_municipio.join([pivot_accion, pivot_fuerza]).reset_index().dropna()
    
    scaler = StandardScaler()
    numericas = [col for col in datos.columns if col not in index_cols]
    datos_originales_num = datos.copy()
    datos[numericas] = scaler.fit_transform(datos[numericas])
    X_scaled = datos.drop(columns=index_cols)
    
    kmeans = KMeans(n_clusters=4, n_init=30, random_state=42)
    km4_clusters = kmeans.fit(X_scaled)
    datos_originales_num['Cluster'] = km4_clusters.labels_
    datos['Cluster'] = km4_clusters.labels_.astype(str)

    # --- COLORES DE LA IMAGEN DE REFERENCIA ---
    COLOR_FONDO_GRAFICA = "#114B5F" # Azul turquesa profundo
    COLOR_GRID = "#1A6278"         # Líneas de cuadrícula
    PALETA_INFOGRAFIA = ['#4ADE80', '#2563EB', '#F97316', '#DC2626'] # Verde, Azul, Naranja, Rojo

    # 1. Curva del Codo
    st.markdown("### A. Validación del Número de Grupos (K)")
    wss = []
    for k in range(1, 11):
        km_test = KMeans(n_clusters=k, n_init=15, random_state=42)
        km_test.fit(X_scaled)
        wss.append(km_test.inertia_)
    
    fig_elbow = px.line(x=list(range(1, 11)), y=wss, markers=True, title="Método del Codo")
    fig_elbow.update_traces(line_color='#2563EB', marker=dict(size=10, color='#4ADE80'))
    fig_elbow.update_layout(
        paper_bgcolor=COLOR_FONDO_GRAFICA, plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), xaxis=dict(gridcolor=COLOR_GRID), yaxis=dict(gridcolor=COLOR_GRID)
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    # 2. Mapa de Calor (Matriz de Distancia)
    st.markdown("### B. Matriz de Distancia Euclideana")
    distancias_eu = euclidean_distances(X_scaled)[:50, :50]
    nombres_municipios_sub = datos['MUNICIPIO'].iloc[:50].tolist()
    
    fig_eu = px.imshow(distancias_eu, x=nombres_municipios_sub, y=nombres_municipios_sub, 
                       color_continuous_scale='Tealrose')
    fig_eu.update_layout(
        paper_bgcolor=COLOR_FONDO_GRAFICA, plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_eu, use_container_width=True)

    # 3. PCA 3D (Cambio de nombres y colores)
    st.markdown("### C. Proyección PCA 3D")
    pca_3d = PCA(n_components=3)
    scores_pca = pca_3d.fit_transform(X_scaled)
    df_pca = pd.DataFrame(scores_pca, columns=['PC1', 'PC2', 'PC3'])
    
    # NUEVOS NOMBRES SEGÚN LA INFOGRAFÍA
    nombres_infografia = {
        "0": "Infographic Sample 1 (45%)", 
        "1": "Infographic Sample 2 (50%)", 
        "2": "Infographic Sample 3 (30%)", 
        "3": "Infographic Sample 4 (25%)"
    }
    df_pca['Cluster_Name'] = [nombres_infografia[str(x)] for x in km4_clusters.labels_]
    df_pca['Municipio'] = datos['MUNICIPIO'].values

    fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster_Name', 
                           color_discrete_sequence=PALETA_INFOGRAFIA,
                           hover_name='Municipio', height=800)
    
    fig_3d.update_layout(
        paper_bgcolor=COLOR_FONDO_GRAFICA,
        scene=dict(
            xaxis=dict(backgroundcolor="#0A2530", gridcolor=COLOR_GRID, color="white"),
            yaxis=dict(backgroundcolor="#0A2530", gridcolor=COLOR_GRID, color="white"),
            zaxis=dict(backgroundcolor="#0A2530", gridcolor=COLOR_GRID, color="white"),
        ),
        legend=dict(font=dict(color="white"))
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # Interpretación
    st.markdown("<div class='insight-card'><b>Análisis:</b> Se observan los grupos claramente separados bajo la nueva paleta visual.</div>", unsafe_allow_html=True)
    if st.button("Siguiente Diapositiva ➡️", type="primary"):
        ir_a_diapositiva(6)

elif st.session_state.diapositiva == 6:
    st.markdown("<div class='slide-title'>Conclusiones</div>", unsafe_allow_html=True)
    st.markdown("<div class='slide-container'><p>El análisis permitió segmentar exitosamente el territorio nacional.</p></div>", unsafe_allow_html=True)
    if st.button("Reiniciar Exposición", type="secondary"):
        ir_a_diapositiva(1)

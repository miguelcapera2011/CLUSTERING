import pandas as pd
import numpy as np
import time
import os
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

# --- ESTILO PREMIUM PARA GRAFICAS ---
def aplicar_estilo_premium(fig):
    fig.update_layout(
        paper_bgcolor="#EAF4FF",
        plot_bgcolor="#F4F9FF",
        font=dict(color="#0F172A", size=14),
        title=dict(font=dict(size=22, color="#0F172A")),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

# --- FUNCIÓN PARA CALCULAR EL ESTADÍSTICO DE HOPKINS ---
def calcular_hopkins(X):
    X = np.array(X)
    n, d = X.shape
    m = int(0.1 * n)
    np.random.seed(42)
    vecinos = NearestNeighbors(n_neighbors=2)
    vecinos.fit(X)
    
    puntos_aleatorios = np.random.uniform(
        np.min(X, axis=0), np.max(X, axis=0), (m, d)
    )
    
    dist_aleatoria, _ = vecinos.kneighbors(puntos_aleatorios, n_neighbors=1)
    indices = np.random.choice(n, m, replace=False)
    puntos_reales = X[indices]
    dist_real, _ = vecinos.kneighbors(puntos_reales, n_neighbors=2)
    
    U = np.sum(dist_aleatoria)
    W = np.sum(dist_real[:, 1])
    H = U / (U + W)
    return H

# --- CONFIGURACIÓN GENERAL Y ESTILO VISUAL ---
st.set_page_config(
    page_title="Exposición Mineria De Datos - Orden Público en colombia", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Inyección de CSS para la interfaz estilo presentación
st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; color: #1E293B; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    .slide-container { background-color: #FFFFFF; padding: 40px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0; margin-bottom: 25px; }
    .slide-title { color: #0F172A; font-size: 36px; font-weight: 700; margin-bottom: 5px; }
    .slide-subtitle { color: #64748B; font-size: 18px; margin-bottom: 25px; }
    
    div.stButton > button { 
        background-color: #E0F2FE !important; color: #0369A1 !important; 
        border: 1px solid #BAE6FD !important; border-radius: 8px !important; 
        font-weight: 700 !important; transition: all 0.3s ease-in-out !important; 
    }
    div.stButton > button:hover { background-color: #7DD3FC !important; }
    div.stButton > button[kind="primary"] { 
        background-color: #0284C7 !important; color: #FFFFFF !important; 
    }
    
    .insight-card { background-color: #F1F5F9; border-left: 5px solid #38BDF8; padding: 18px; border-radius: 4px 12px 12px 4px; margin-bottom: 15px; }
    .insight-critical { background-color: #FEF2F2; border-left: 5px solid #DC2626; padding: 18px; border-radius: 4px 12px 12px 4px; }
    .insight-success { background-color: #F0FDF4; border-left: 5px solid #16A34A; padding: 18px; border-radius: 4px 12px 12px 4px; }
    </style>
""", unsafe_allow_html=True)

# --- LÓGICA DE NAVEGACIÓN ---
if 'diapositiva' not in st.session_state:
    st.session_state.diapositiva = 1

def ir_a_diapositiva(num):
    st.session_state.diapositiva = num
    st.rerun()

# --- CARGA AUTOMÁTICA DE DATOS ---
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
    except Exception as e:
        return None, str(e)

df_original, nombre_archivo_cargado = cargar_datos_automatico()

# --- BARRA DE NAVEGACIÓN SUPERIOR ---
cols_nav = st.columns(6)
nombres_diapo = ["1. Portada", "2. Introducción", "3. Marco Teórico", "4. Metodología", "5. Resultados", "6. Conclusiones"]
for i, nombre in enumerate(nombres_diapo):
    tipo = "primary" if st.session_state.diapositiva == (i + 1) else "secondary"
    if cols_nav[i].button(nombre, use_container_width=True, type=tipo):
        ir_a_diapositiva(i + 1)

st.markdown("---")

# --- CONTENIDO DE LAS DIAPOSITIVAS ---

# DIAPOSITIVA 1: PORTADA
if st.session_state.diapositiva == 1:
    st.markdown("""
        <div class='slide-container' style='text-align: center; padding: 60px 40px;'>
            <img src='https://administrativos.ut.edu.co/images/Home/simbolos/logo_oficial.png' width='197'>
            <div class='slide-title' style='font-size: 42px; color: #1E3A8A;'>Análisis de Clústeres (K-Means) En Afectaciones a la Fuerza Pública</div>
            <div class='slide-subtitle' style='font-size: 22px;'>Segmentación Territorial de Incidentes de Orden Público</div>
        </div>
    """, unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("<div class='insight-card'><h4>ESTUDIANTE</h4><p><b>Miguel Angel Garatejo</b><br>Matematica con Enfasis en Estadistica</p></div>", unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"<div class='insight-success'><h4>PROFESOR</h4><p><b>Yuri Marcela Garcia Saavedra</b><br>Mineria de Datos | {time.strftime('%Y')}</p></div>", unsafe_allow_html=True)
    if st.button("Iniciar Sustentación", type="primary", use_container_width=True):
        ir_a_diapositiva(2)

# DIAPOSITIVA 2: INTRODUCCIÓN
elif st.session_state.diapositiva == 2:
    st.markdown("<div class='slide-title'>Introducción y Desafío Técnico</div>", unsafe_allow_html=True)
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""<div class='slide-container'><h3 style='color: #DC2626;'>El Problema</h3>
            <p>El archivo original es un histórico de novedades cualitativo. Los algoritmos como K-Means no pueden procesar texto directo.</p></div>""", unsafe_allow_html=True)
    with col_i2:
        st.markdown("""<div class='slide-container'><h3 style='color: #0284C7;'>Objetivo</h3>
            <p>Construir un flujo automatizado para reestructurar y agrupar municipios según su vulnerabilidad real.</p></div>""", unsafe_allow_html=True)
    if st.button("Siguiente Diapositiva ➡️", type="primary"):
        ir_a_diapositiva(3)

# DIAPOSITIVA 3: MARCO TEÓRICO
elif st.session_state.diapositiva == 3:
    st.markdown("<div class='slide-title'>Fundamentos Teóricos</div>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    t1.markdown("<div class='slide-container'><h4>1. Pivotado</h4><p>Transformación de categorías en dimensiones numéricas.</p></div>", unsafe_allow_html=True)
    t2.markdown("<div class='slide-container'><h4>2. K-Means</h4><p>Partición en K grupos minimizando la inercia interna.</p></div>", unsafe_allow_html=True)
    t3.markdown("<div class='slide-container'><h4>3. PCA</h4><p>Reducción de dimensiones para visualización ortogonal.</p></div>", unsafe_allow_html=True)
    if st.button("Siguiente Diapositiva ➡️", type="primary"):
        ir_a_diapositiva(4)

# DIAPOSITIVA 4: METODOLOGÍA
elif st.session_state.diapositiva == 4:
    st.markdown("<div class='slide-title'>Arquitectura del Flujo</div>", unsafe_allow_html=True)
    with st.expander("Fase 1: Pivotado Estructurado", expanded=True):
        st.code("pivot = df.pivot_table(index=['COD_MUNI'], columns='ACCION', values='CANTIDAD', aggfunc='sum')", language="python")
    with st.expander("Fase 2: Normalización (Z-Score)", expanded=False):
        st.code("scaler = StandardScaler()\nX_scaled = scaler.fit_transform(datos[numericas])", language="python")
    if st.button("Siguiente Diapositiva ➡️", type="primary"):
        ir_a_diapositiva(5)

# DIAPOSITIVA 5: RESULTADOS (PROCESAMIENTO REAL)
elif st.session_state.diapositiva == 5:
    st.markdown("<div class='slide-title'>Análisis de Resultados</div>", unsafe_allow_html=True)
    if df_original is not None:
        # Lógica de procesamiento de datos [Indices 5-7]
        index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
        pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
        pivot_fuerza = df_original.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0)
        total_muni = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame('TOTAL_AFECTADOS')
        
        datos = total_muni.join([pivot_accion, pivot_fuerza]).reset_index().dropna()
        numericas = [c for c in datos.columns if c not in index_cols]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(datos[numericas])
        
        # Hopkins
        h_val = calcular_hopkins(X_scaled)
        st.metric("Estadístico Hopkins", f"{h_val:.3f}")
        
        # K-Means y PCA 3D
        kmeans = KMeans(n_clusters=4, n_init=30, random_state=42).fit(X_scaled)
        pca = PCA(n_components=3).fit_transform(X_scaled)
        
        df_pca = pd.DataFrame(pca, columns=['PC1', 'PC2', 'PC3'])
        df_pca['Cluster'] = kmeans.labels_.astype(str)
        df_pca['Municipio'] = datos['MUNICIPIO'].values
        
        fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster', hover_name='Municipio', title="Distribución Espacial (PCA)")
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.markdown("<div class='insight-critical'><h4>Anomalías detectadas</h4><p>El Clúster 3 agrupa municipios de Emergencia Crítica (ej. Cali, Cúcuta) por su alta letalidad.</p></div>", unsafe_allow_html=True)

    if st.button("Siguiente Diapositiva ➡️", type="primary"):
        ir_a_diapositiva(6)

# DIAPOSITIVA 6: CONCLUSIONES
elif st.session_state.diapositiva == 6:
    st.markdown("<div class='slide-title'>Conclusiones y Recomendaciones</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='slide-container'><h3>Conclusiones</h3><ul><li>Tratamiento cualitativo exitoso.</li><li>Separación clara mediante K-Means.</li></ul></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='slide-container'><h3>Futuro</h3><ul><li>Despliegue preventivo basado en centroides.</li><li>Automatización en tiempo real.</li></ul></div>", unsafe_allow_html=True)
    if st.button("↩️ Volver a Portada", type="secondary"):
        ir_a_diapositiva(1)

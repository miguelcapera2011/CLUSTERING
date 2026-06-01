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
# DIAPOSITIVA 5: RESULTADOS (CORREGIDA CON ESTILO CLARO Y LEGINILIDAD MÁXIMA)
# ==============================================================================
elif st.session_state.diapositiva == 5:
    st.markdown("<div class='slide-title'>Hallazgos, Comportamiento Estructurado y Análisis de Clústeres</div>", unsafe_allow_html=True)
    st.markdown("<div class='slide-subtitle'>Inspección profunda de patrones, métricas de separación y detección de datos atípicos</div>", unsafe_allow_html=True)
    
    if df_original is None:
        st.error("Archivo no encontrado.")
        st.stop()

    # --- Lógica de Procesamiento ---
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

    # --- CONFIGURACIÓN DE COLORES CLAROS PREMIUM ---
    FONDO_GRAFICA_CLARO = "#F8FAFC"  # Gris sutil limpio
    COLOR_TEXTO_OSCURO = "#1E293B"   # Azul pizarra de alta legibilidad
    COLOR_GRID_SUTIL = "#E2E8F0"     # Líneas de cuadrícula muy tenues
    
    # Colores vivos pero claros para los Clústeres (Estilo Infografía de Consultoría)
    PALETA_CLUSTERS_CLARA = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444'] # Verde, Azul, Naranja, Rojo

    # Metricas
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Municipios Procesados", datos.shape[0])
    with col_m2:
        st.metric("Nuevas Columnas Numéricas", datos.shape[1] - 4)

    # 1. Curva del Codo (Fondo Claro)
    st.markdown("### A. Validación del Número de Grupos (K)")
    wss = []
    for k in range(1, 11):
        km_test = KMeans(n_clusters=k, n_init=15, random_state=42)
        km_test.fit(X_scaled)
        wss.append(km_test.inertia_)
    
    fig_elbow = px.line(x=list(range(1, 11)), y=wss, markers=True, title="Evaluación de Estabilidad por Inercia Interna (WSS)")
    fig_elbow.update_traces(line_color='#0284C7', marker=dict(size=9, color='#0369A1'))
    fig_elbow.update_layout(
        paper_bgcolor=FONDO_GRAFICA_CLARO,
        plot_bgcolor='white',
        font=dict(color=COLOR_TEXTO_OSCURO, size=13),
        title_font=dict(color=COLOR_TEXTO_OSCURO, size=16, family="Arial"),
        xaxis=dict(gridcolor=COLOR_GRID_SUTIL, title="Número de Clústeres (k)", color=COLOR_TEXTO_OSCURO),
        yaxis=dict(gridcolor=COLOR_GRID_SUTIL, title="Inercia Matemática", color=COLOR_TEXTO_OSCURO)
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    # 2. Mapa de Calor (Fondo Claro y Escala Nítida)
    st.markdown("### B. Matriz Geométrica de Distancia Euclideana (Muestra de Control)")
    distancias_eu = euclidean_distances(X_scaled)[:50, :50]
    nombres_municipios_sub = datos['MUNICIPIO'].iloc[:50].tolist()
    
    fig_eu = px.imshow(distancias_eu, x=nombres_municipios_sub, y=nombres_municipios_sub, 
                       title="Mapa de Calor de Disimilitud Espacial",
                       color_continuous_scale='Blues') # Tonos azules claros y degradados muy limpios
    fig_eu.update_layout(
        paper_bgcolor=FONDO_GRAFICA_CLARO,
        plot_bgcolor='white',
        font=dict(color=COLOR_TEXTO_OSCURO, size=11),
        title_font=dict(color=COLOR_TEXTO_OSCURO, size=16),
        coloraxis_colorbar=dict(title="Distancia", tickfont=dict(color=COLOR_TEXTO_OSCURO))
    )
    st.plotly_chart(fig_eu, use_container_width=True)

    # 3. PCA 3D (Entorno Claro con Nombres Modificados)
    st.markdown("### C. Proyección Espacial Avanzada e Identificación de Datos Atípicos (PCA 3D)")
    pca_3d = PCA(n_components=3)
    scores_pca = pca_3d.fit_transform(X_scaled)
    df_pca = pd.DataFrame(scores_pca, columns=['PC1', 'PC2', 'PC3'])
    
    # Nombres de leyenda actualizados segun tu requerimiento
    nombres_infografia = {
        "0": "Infographic Sample 1 (45%)", 
        "1": "Infographic Sample 2 (50%)", 
        "2": "Infographic Sample 3 (30%)", 
        "3": "Infographic Sample 4 (25%)"
    }
    df_pca['Cluster_Name'] = [nombres_infografia[str(x)] for x in km4_clusters.labels_]
    df_pca['Municipio'] = datos['MUNICIPIO'].values
    df_pca['Depto'] = datos['DEPARTAMENTO'].values

    fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster_Name', 
                           color_discrete_sequence=PALETA_CLUSTERS_CLARA,
                           hover_name='Municipio', hover_data=['Depto'],
                           title='Dispersión Espacial de Fronteras de Vulnerabilidad',
                           height=750)
    
    fig_3d.update_layout(
        paper_bgcolor=FONDO_GRAFICA_CLARO,
        font=dict(color=COLOR_TEXTO_OSCURO, size=12),
        title_font=dict(color=COLOR_TEXTO_OSCURO, size=16),
        scene=dict(
            xaxis=dict(backgroundcolor="#F1F5F9", gridcolor=COLOR_GRID_SUTIL, color=COLOR_TEXTO_OSCURO, title="PC1"),
            yaxis=dict(backgroundcolor="#F1F5F9", gridcolor=COLOR_GRID_SUTIL, color=COLOR_TEXTO_OSCURO, title="PC2"),
            zaxis=dict(backgroundcolor="#F1F5F9", gridcolor=COLOR_GRID_SUTIL, color=COLOR_TEXTO_OSCURO, title="PC3")
        ),
        legend=dict(
            font=dict(color=COLOR_TEXTO_OSCURO, size=12),
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # Tabla e Interpretaciones
    st.markdown("### D. Perfil de Comportamiento de los Clústeres (Valores Reales Promedio)")
    tabla_perfil = datos_originales_num.groupby('Cluster').mean(numeric_only=True).round(2)
    st.dataframe(tabla_perfil, use_container_width=True)

    if st.button("Siguiente Diapositiva: Conclusiones y Recomendaciones ➡️", type="primary"):
        ir_a_diapositiva(6)

elif st.session_state.diapositiva == 6:
    st.markdown("<div class='slide-title'>Conclusiones</div>", unsafe_allow_html=True)
    st.markdown("<div class='slide-container'><p>El análisis permitió segmentar exitosamente el territorio nacional.</p></div>", unsafe_allow_html=True)
    if st.button("Reiniciar Exposición", type="secondary"):
        ir_a_diapositiva(1)

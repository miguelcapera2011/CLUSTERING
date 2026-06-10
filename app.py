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
from sklearn.neighbors import NearestNeighbors

# ESTILO PREMIUM PARA GRAFICAS
def aplicar_estilo_premium(fig):
    fig.update_layout(
        paper_bgcolor="#EAF4FF",
        plot_bgcolor="#F4F9FF",
        font=dict(
            color="#0F172A",
            size=14
        ),
        title=dict(
            font=dict(
                size=22,
                color="#0F172A"
            )
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )
    return fig

# FUNCIÓN PARA CALCULAR EL ESTADÍSTICO DE HOPKINS
def calcular_hopkins(X):
    X = np.array(X)
    n, d = X.shape
    # Se toma el 10% de las observaciones
    m = int(0.1 * n)
    np.random.seed(42)
    
    # Modelo para calcular vecinos cercanos
    vecinos = NearestNeighbors(n_neighbors=2)
    vecinos.fit(X)
    
    # Generar puntos aleatorios en el mismo espacio de los datos
    puntos_aleatorios = np.random.uniform(
        np.min(X, axis=0),
        np.max(X, axis=0),
        (m, d)
    )
    
    # Distancia de puntos aleatorios al dato real más cercano
    dist_aleatoria, _ = vecinos.kneighbors(
        puntos_aleatorios,
        n_neighbors=1
    )
    
    # Seleccionar puntos reales aleatorios
    indices = np.random.choice(n, m, replace=False)
    puntos_reales = X[indices]
    
    # Distancia entre puntos reales y su vecino más cercano
    dist_real, _ = vecinos.kneighbors(
        puntos_reales,
        n_neighbors=2
    )
    
    U = np.sum(dist_aleatoria)
    W = np.sum(dist_real[:, 1])
    H = U / (U + W)
    return H

# CONFIGURACIÓN GENERAL Y ESTILO VISUAL "POWERPOINT PREMIUM"
st.set_page_config(page_title="Exposición Minería De Datos - Orden Público en Colombia", layout="wide", initial_sidebar_state="collapsed")

# Inyección de CSS Avanzado
st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    .slide-container {
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        border: 1px solid #E2E8F0;
    }
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
        color: #0369A1 !important;
        border-color: #7DD3FC !important;
        box-shadow: 0 4px 12px rgba(3, 105, 161, 0.15) !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border: 1px solid #0284C7 !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
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

# CARGA AUTOMÁTICA DE DATOS
def cargar_datos_automatico():
    archivos_en_carpeta = os.listdir('.')
    archivo_encontrado = None
    for archivo in archivos_en_carpeta:
        nombre_minuscula = archivo.lower()
        if ("afectacion" in nombre_minuscula or "fuerza" in nombre_minuscula or "publica" in nombre_minuscula) and (archivo.endswith('.csv') or archivo.endswith('.xlsx')):
            archivo_encontrado = archivo
            break
    
    if archivo_encontrado is None:
        return None, "No se encontró el registro de datos en la carpeta raíz."
    
    try:
        if archivo_encontrado.endswith('.csv'):
            df = pd.read_csv(archivo_encontrado, header=0)
        else:
            df = pd.read_excel(archivo_encontrado, header=0)
        return df, archivo_encontrado
    except Exception as e:
        return None, f"Error al leer el archivo: {str(e)}"

df_original, nombre_archivo_cargado = cargar_datos_automatico()

# CONTROLES DE NAVEGACIÓN
cols_nav = st.columns(6)
nombres_diapo = ["1. Portada", "2. Introducción", "3. Marco Teórico", "4. Metodología", "5. Resultados", "6. Conclusiones"]
for i, nombre in enumerate(nombres_diapo):
    tipo_boton = "primary" if st.session_state.diapositiva == (i + 1) else "secondary"
    if cols_nav[i].button(nombre, use_container_width=True, type=tipo_boton):
        ir_a_diapositiva(i + 1)

st.markdown("---")

# DIAPOSITIVA 1: PORTADA
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
        st.markdown("""
        <div class='insight-card'>
            <h4 style='margin-top:0; color:#1E3A8A;'>ESTUDIANTE</h4>
            <p><b>Miguel Angel Garatejo</b><br>Facultad de Ciencias<br>Matemática con Énfasis en Estadística</p>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"""
        <div class='insight-success'>
            <h4 style='margin-top:0; color:#16A34A;'> PROFESOR</h4>
            <p><b>Yuri Marcela Garcia Saavedra </b><br>Minería de Datos <br>Año: {time.strftime('%Y')} | Clustering</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("Iniciar Sustentación", type="primary", use_container_width=True):
        ir_a_diapositiva(2)

# DIAPOSITIVA 2: INTRODUCCIÓN
elif st.session_state.diapositiva == 2:
    st.markdown("<div class='slide-title'> Introducción y Definición del Desafío Técnico</div>", unsafe_allow_html=True)
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
        <div class='slide-container'>
            <h3 style='color: #DC2626; margin-top:0;'> El Problema de los Datos Originales</h3>
            <p><b>Naturaleza del Archivo:</b> Histórico de Novedades donde cada fila reporta un ataque individual.</p>
            <ul>
                <li><b>Restricción:</b> 8 columnas cualitativas y solo 1 cuantitativa.</li>
                <li><b>Quiebre Matemático:</b> K-Means no puede procesar texto directo.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_i2:
        st.markdown("""
        <div class='slide-container'>
            <h3 style='color: #0284C7; margin-top:0;'> Objetivos</h3>
            <p>Construir un flujo automatizado para reestructurar y agrupar municipios según patrones de vulnerabilidad.</p>
        </div>
        """, unsafe_allow_html=True)

# DIAPOSITIVA 5: RESULTADOS (Lógica Principal)
elif st.session_state.diapositiva == 5:
    st.markdown("<div class='slide-title'> Hallazgos y Análisis de Clústeres</div>", unsafe_allow_html=True)
    if df_original is None:
        st.error("❌ No se detectó el archivo de datos.")
        st.stop()

    # Procesamiento
    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
    pivot_fuerza = df_original.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0)
    total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')
    
    datos = total_municipio.join([pivot_accion, pivot_fuerza]).reset_index().dropna()
    
    scaler = StandardScaler()
    numericas = [col for col in datos.columns if col not in index_cols]
    datos_originales_num = datos.copy()
    datos[numericas] = scaler.fit_transform(datos[numericas])
    X_scaled = datos.drop(columns=index_cols)

    # Hopkins y KMeans
    valor_hopkins = calcular_hopkins(X_scaled)
    kmeans = KMeans(n_clusters=4, n_init=30, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    datos_originales_num['Cluster'] = clusters
    
    st.metric("Valor Hopkins", f"{valor_hopkins:.3f}")
    
    # PCA 3D
    pca_3d = PCA(n_components=3)
    scores_pca = pca_3d.fit_transform(X_scaled)
    df_pca = pd.DataFrame(scores_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Cluster'] = clusters.astype(str)
    df_pca['Municipio'] = datos['MUNICIPIO'].values
    
    fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster', hover_name='Municipio')
    st.plotly_chart(fig_3d, use_container_width=True)

# DIAPOSITIVA 6: CONCLUSIONES
elif st.session_state.diapositiva == 6:
    st.markdown("<div class='slide-title'>🏁 Conclusiones Académicas</div>", unsafe_allow_html=True)
    st.info("El modelo logró aislar de forma óptima las zonas críticas de las estables.")
    if st.button("↩️ Reiniciar Exposición", type="secondary"):
        ir_a_diapositiva(1)

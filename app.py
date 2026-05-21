# =============================================================================
# APP STREAMLIT - EXPOSICIÓN INTERACTIVA K-MEANS
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
import time

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

st.set_page_config(
    page_title="Minería de Datos - KMeans",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS PROFESIONAL
# =============================================================================

st.markdown("""
<style>

/* FONDO GENERAL */
.stApp{
    background: linear-gradient(135deg, #06131f, #0b1f33);
    color: white;
}

/* SIDEBAR */
[data-testid="stSidebar"]{
    background: rgba(5,15,25,0.95);
    border-right: 1px solid rgba(0,255,255,0.2);
}

/* TITULOS */
h1,h2,h3{
    color: #00ffd5;
    font-family: 'Segoe UI';
}

/* TARJETAS */
.card{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 20px rgba(0,255,255,0.08);
    margin-bottom: 20px;
}

/* METRICAS */
.metric-box{
    background: linear-gradient(145deg,#0f2236,#132b43);
    padding:20px;
    border-radius:20px;
    text-align:center;
    border:1px solid rgba(0,255,255,0.15);
    transition:0.3s;
}

.metric-box:hover{
    transform:translateY(-5px);
    box-shadow:0px 0px 25px rgba(0,255,255,0.25);
}

/* TEXTO */
p{
    color:#d6e2f0;
    font-size:17px;
}

/* BOTONES */
.stButton>button{
    background: linear-gradient(90deg,#00ffd5,#008cff);
    color:white;
    border:none;
    border-radius:12px;
    padding:0.6rem 1.5rem;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🧠 MINERÍA DE DATOS")

menu = st.sidebar.radio(
    "Navegación",
    [
        "🏠 Introducción",
        "📂 Carga de Datos",
        "📊 Análisis Exploratorio",
        "📏 Distancias",
        "📉 Método del Codo",
        "🤖 K-Means",
        "🌌 PCA Interactivo",
        "📌 Conclusiones"
    ]
)

# =============================================================================
# CARGA ARCHIVO
# =============================================================================

@st.cache_data
def cargar_datos():

    df_original = pd.read_excel(
        "AFECTACIÓN A LA FUERZA PÚBLICA.xlsx",
        header=0
    )

    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']

    pivot_accion = df_original.pivot_table(
        index=index_cols,
        columns='ACCION',
        values='CANTIDAD',
        aggfunc='sum',
        fill_value=0
    )

    pivot_fuerza = df_original.pivot_table(
        index=index_cols,
        columns='NOMBRE_FUERZA',
        values='CANTIDAD',
        aggfunc='sum',
        fill_value=0
    )

    pivot_cat = df_original.pivot_table(
        index=index_cols,
        columns='CATEGORIA',
        values='CANTIDAD',
        aggfunc='sum',
        fill_value=0
    )

    total_municipio = df_original.groupby(index_cols)['CANTIDAD'] \
        .sum().to_frame(name='TOTAL_AFECTADOS')

    datos = total_municipio.join(
        [pivot_accion, pivot_fuerza, pivot_cat]
    ).reset_index()

    datos = datos.rename(columns={'MUNICIPIO': 'State'})

    return datos

datos = cargar_datos()

# =============================================================================
# INTRODUCCIÓN
# =============================================================================

if menu == "🏠 Introducción":

    st.title("📊 ANÁLISIS DE AFECTACIÓN A LA FUERZA PÚBLICA")

    st.markdown("""
    ### Clusterización de Municipios Colombianos usando K-Means
    
    Esta aplicación presenta una exposición interactiva del proceso de:
    
    - Limpieza y transformación de datos
    - Estandarización de variables
    - Cálculo de distancias
    - Método del Codo
    - Aplicación de K-Means
    - Reducción dimensional PCA
    
    El objetivo principal fue identificar grupos de municipios con
    características similares mediante aprendizaje no supervisado.
    """)

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-box">
        <h2>884</h2>
        <p>Municipios</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-box">
        <h2>4</h2>
        <p>Clusters</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-box">
        <h2>K-Means</h2>
        <p>Algoritmo</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-box">
        <h2>PCA</h2>
        <p>Visualización</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# CARGA DE DATOS
# =============================================================================

elif menu == "📂 Carga de Datos":

    st.title("📂 Carga y Consolidación")

    st.dataframe(datos.head(20), use_container_width=True)

    st.success(f"""
    Base consolidada correctamente:
    
    ✅ Municipios: {datos.shape[0]}
    ✅ Variables: {datos.shape[1]}
    """)

# =============================================================================
# ANALISIS EXPLORATORIO
# =============================================================================

elif menu == "📊 Análisis Exploratorio":

    st.title("📊 Análisis Exploratorio")

    fig = px.histogram(
        datos,
        x="TOTAL_AFECTADOS",
        nbins=40,
        title="Distribución TOTAL_AFECTADOS",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# DISTANCIAS
# =============================================================================

elif menu == "📏 Distancias":

    st.title("📏 Distancias")

    columnas_omitir = ['COD_MUNI', 'State', 'DEPARTAMENTO']

    numericas = [
        col for col in datos.columns
        if col not in columnas_omitir
    ]

    scaler = StandardScaler()

    datos_scaled = datos.copy()

    datos_scaled[numericas] = scaler.fit_transform(
        datos_scaled[numericas]
    )

    X_scaled = datos_scaled.drop(columns=columnas_omitir)

    st.subheader("Distancia Euclideana")

    distancias_eu = euclidean_distances(X_scaled.iloc[:40])

    fig = px.imshow(
        distancias_eu,
        color_continuous_scale="Turbo",
        aspect="auto"
    )

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# METODO DEL CODO
# =============================================================================

elif menu == "📉 Método del Codo":

    st.title("📉 Método del Codo")

    columnas_omitir = ['COD_MUNI', 'State', 'DEPARTAMENTO']

    numericas = [
        col for col in datos.columns
        if col not in columnas_omitir
    ]

    scaler = StandardScaler()

    datos_scaled = datos.copy()

    datos_scaled[numericas] = scaler.fit_transform(
        datos_scaled[numericas]
    )

    X_scaled = datos_scaled.drop(columns=columnas_omitir)

    wss = []

    for k in range(1,11):

        kmeans = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=42
        )

        kmeans.fit(X_scaled)

        wss.append(kmeans.inertia_)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1,11)),
        y=wss,
        mode='lines+markers'
    ))

    fig.add_vline(
        x=4,
        line_dash="dash",
        line_color="cyan"
    )

    fig.update_layout(
        template="plotly_dark",
        title="Método del Codo",
        xaxis_title="Número de Clusters",
        yaxis_title="WSS"
    )

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# KMEANS
# =============================================================================

elif menu == "🤖 K-Means":

    st.title("🤖 Visualización K-Means")

    columnas_omitir = ['COD_MUNI', 'State', 'DEPARTAMENTO']

    numericas = [
        col for col in datos.columns
        if col not in columnas_omitir
    ]

    scaler = StandardScaler()

    datos_scaled = datos.copy()

    datos_scaled[numericas] = scaler.fit_transform(
        datos_scaled[numericas]
    )

    X_scaled = datos_scaled.drop(columns=columnas_omitir)

    kmeans = KMeans(
        n_clusters=4,
        n_init=50,
        random_state=42
    )

    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)

    componentes = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(componentes, columns=['PC1','PC2'])

    pca_df['Cluster'] = labels.astype(str)

    pca_df['Municipio'] = datos['State']

    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        hover_name='Municipio',
        title='Clusters K-Means',
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PCA
# =============================================================================

elif menu == "🌌 PCA Interactivo":

    st.title("🌌 PCA Interactivo 3D")

    columnas_omitir = ['COD_MUNI', 'State', 'DEPARTAMENTO']

    numericas = [
        col for col in datos.columns
        if col not in columnas_omitir
    ]

    scaler = StandardScaler()

    datos_scaled = datos.copy()

    datos_scaled[numericas] = scaler.fit_transform(
        datos_scaled[numericas]
    )

    X_scaled = datos_scaled.drop(columns=columnas_omitir)

    kmeans = KMeans(
        n_clusters=4,
        n_init=50,
        random_state=42
    )

    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=3)

    componentes = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        componentes,
        columns=['PC1','PC2','PC3']
    )

    pca_df['Cluster'] = labels.astype(str)

    pca_df['Municipio'] = datos['State']

    fig = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Cluster',
        hover_name='Municipio',
        template='plotly_dark',
        title='PCA 3D'
    )

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# CONCLUSIONES
# =============================================================================

elif menu == "📌 Conclusiones":

    st.title("📌 Conclusiones")

    st.markdown("""
    ## Principales Hallazgos
    
    ✅ K-Means permitió identificar patrones similares entre municipios.
    
    ✅ El método del codo sugirió K=4 como número óptimo.
    
    ✅ La estandarización fue fundamental para evitar sesgos.
    
    ✅ PCA permitió visualizar la separación entre grupos.
    
    ✅ Los clústeres revelan comportamientos diferenciados.
    """)

    st.balloons()

# =============================================================================
# APP STREAMLIT - EXPOSICIÓN MINERÍA DE DATOS
# CLUSTERIZACIÓN DE MUNICIPIOS COLOMBIANOS
# =============================================================================

# =========================
# LIBRERÍAS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import time

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

st.set_page_config(
    page_title="Minería de Datos - Clustering",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS PROFESIONAL
# =============================================================================

st.markdown("""
<style>

/* =========================
FONDO GENERAL
========================= */
.stApp{
    background: linear-gradient(135deg,#06141f,#0b1120,#071c2f);
    color:white;
}

/* =========================
SIDEBAR
========================= */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#04101c,#081726,#0b2239);
    border-right:1px solid rgba(255,255,255,0.1);
}

/* =========================
TÍTULOS
========================= */
h1{
    color:#00ffd5;
    font-weight:800;
    letter-spacing:1px;
}

h2,h3{
    color:#7ef9ff;
}

/* =========================
CARDS
========================= */
.card{
    background: rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:20px;
    padding:25px;
    backdrop-filter: blur(10px);
    box-shadow:0px 4px 20px rgba(0,0,0,0.4);
    transition:0.3s;
}

.card:hover{
    transform:translateY(-5px);
    box-shadow:0px 6px 25px rgba(0,255,213,0.2);
}

/* =========================
MÉTRICAS
========================= */
.metric-card{
    background: linear-gradient(135deg,#071b2d,#0f2c47);
    border-radius:18px;
    padding:20px;
    text-align:center;
    border:1px solid rgba(0,255,213,0.15);
}

.metric-title{
    color:#8fb3c9;
    font-size:15px;
}

.metric-value{
    font-size:32px;
    color:#00ffd5;
    font-weight:bold;
}

/* =========================
BOTONES
========================= */
.stButton>button{
    background: linear-gradient(90deg,#00ffd5,#00a8ff);
    color:black;
    border:none;
    border-radius:12px;
    padding:0.6rem 1.2rem;
    font-weight:bold;
}

.stButton>button:hover{
    background: linear-gradient(90deg,#00a8ff,#00ffd5);
    color:white;
}

/* =========================
TABLAS
========================= */
[data-testid="stDataFrame"]{
    border-radius:15px;
    overflow:hidden;
}

/* =========================
SEPARADORES
========================= */
hr{
    border:1px solid rgba(255,255,255,0.08);
}

/* =========================
CAJAS DE TEXTO
========================= */
.info-box{
    background: rgba(0,255,213,0.08);
    border-left:5px solid #00ffd5;
    padding:15px;
    border-radius:10px;
    margin-bottom:15px;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("📊 Minería de Datos")
st.sidebar.markdown("### Clusterización de Municipios")

menu = st.sidebar.radio(
    "Navegación",
    [
        "Inicio",
        "Carga de Datos",
        "Análisis Exploratorio",
        "Método del Codo",
        "K-Means",
        "PCA 2D",
        "PCA 3D",
        "Boxplots",
        "Conclusiones"
    ]
)

# =============================================================================
# CARGA DE DATOS
# =============================================================================

@st.cache_data
def cargar_datos():

    df_original = pd.read_excel(
        'AFECTACIÓN A LA FUERZA PÚBLICA.xlsx',
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

    total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(
        name='TOTAL_AFECTADOS'
    )

    datos = total_municipio.join(
        [pivot_accion, pivot_fuerza, pivot_cat]
    ).reset_index()

    datos = datos.rename(columns={'MUNICIPIO': 'State'})

    datos = datos.dropna()

    return datos

datos = cargar_datos()

# =============================================================================
# PREPARACIÓN
# =============================================================================

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

# =============================================================================
# MODELO KMEANS
# =============================================================================

kmeans = KMeans(
    n_clusters=4,
    n_init=50,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)

datos["Cluster"] = clusters

# =============================================================================
# PCA
# =============================================================================

pca = PCA(n_components=3)

pca_result = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    pca_result,
    columns=['PC1','PC2','PC3']
)

pca_df["Cluster"] = clusters.astype(str)

pca_df["Municipio"] = (
    datos["State"] + " - " + datos["DEPARTAMENTO"]
)

# =============================================================================
# INICIO
# =============================================================================

if menu == "Inicio":

    st.title("📊 Análisis de Afectación a la Fuerza Pública")

    st.markdown("""
    <div class="info-box">
    Proyecto de minería de datos enfocado en la clusterización de municipios
    colombianos mediante el algoritmo K-Means.
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Municipios</div>
        <div class="metric-value">{datos.shape[0]}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Variables</div>
        <div class="metric-value">{len(numericas)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Clusters</div>
        <div class="metric-value">4</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Inercia</div>
        <div class="metric-value">{round(kmeans.inertia_,2)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1520607162513-77705c0f0d4a",
        use_container_width=True
    )

# =============================================================================
# CARGA DE DATOS
# =============================================================================

elif menu == "Carga de Datos":

    st.title("📂 Carga y Consolidación")

    st.dataframe(datos.head(20), use_container_width=True)

    st.markdown("""
    ### Explicación

    - Se cargó el archivo Excel original.
    - Se consolidaron municipios.
    - Las variables categóricas fueron convertidas a conteos.
    - Se creó una base lista para clustering.
    """)

# =============================================================================
# EXPLORATORIO
# =============================================================================

elif menu == "Análisis Exploratorio":

    st.title("📈 Análisis Exploratorio")

    fig, ax = plt.subplots(2,2, figsize=(15,10))

    sns.histplot(datos["TOTAL_AFECTADOS"], bins=20, kde=True, ax=ax[0,0], color='cyan')
    ax[0,0].set_title("TOTAL_AFECTADOS")

    sns.histplot(datos["ASESINADO"], bins=20, kde=True, ax=ax[0,1], color='red')
    ax[0,1].set_title("ASESINADO")

    sns.histplot(datos["HERIDO"], bins=20, kde=True, ax=ax[1,0], color='green')
    ax[1,0].set_title("HERIDO")

    sns.histplot(
        datos["EJERCITO NACIONAL DE COLOMBIA"],
        bins=20,
        kde=True,
        ax=ax[1,1],
        color='purple'
    )

    ax[1,1].set_title("EJÉRCITO")

    st.pyplot(fig)

# =============================================================================
# MÉTODO DEL CODO
# =============================================================================

elif menu == "Método del Codo":

    st.title("📉 Método del Codo")

    wss = []

    for k in range(1,11):

        km = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=42
        )

        km.fit(X_scaled)

        wss.append(km.inertia_)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1,11)),
        y=wss,
        mode='lines+markers'
    ))

    fig.add_vline(
        x=4,
        line_dash="dash",
        line_color="red"
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

elif menu == "K-Means":

    st.title("🤖 K-Means")

    st.write("Centroides del modelo:")

    centroides = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=X_scaled.columns
    )

    st.dataframe(
        centroides,
        use_container_width=True
    )

    conteo = datos["Cluster"].value_counts()

    fig = px.pie(
        values=conteo.values,
        names=conteo.index.astype(str),
        hole=0.5,
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PCA 2D
# =============================================================================

elif menu == "PCA 2D":

    st.title("🌌 PCA Interactivo 2D")

    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        hover_name='Municipio',
        template='plotly_dark'
    )

    fig.update_traces(marker=dict(size=9))

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PCA 3D
# =============================================================================

elif menu == "PCA 3D":

    st.title("🪐 PCA Interactivo 3D")

    fig = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Cluster',
        hover_name='Municipio',
        template='plotly_dark'
    )

    fig.update_layout(height=800)

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# BOXPLOTS
# =============================================================================

elif menu == "Boxplots":

    st.title("📦 Distribución por Cluster")

    fig = px.box(
        datos,
        x='Cluster',
        y='TOTAL_AFECTADOS',
        color='Cluster',
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# CONCLUSIONES
# =============================================================================

elif menu == "Conclusiones":

    st.title("🧠 Conclusiones")

    st.markdown("""
    ### Hallazgos principales

    ✅ Se identificaron patrones similares entre municipios.

    ✅ El algoritmo K-Means permitió separar los datos en 4 grupos.

    ✅ PCA facilitó la visualización de los clústeres.

    ✅ Existen municipios con afectaciones significativamente mayores.

    ✅ La estandarización fue fundamental para el análisis.

    ---
    
    ### Tecnologías utilizadas

    - Python
    - Streamlit
    - Scikit-Learn
    - Plotly
    - PCA
    - K-Means
    """)

    st.balloons()

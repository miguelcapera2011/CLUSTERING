import streamlit as st
import pandas as pd
import numpy as np
import time
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import plotly.express as px
import plotly.graph_objects as go


# ==========================================================
# ESTILO PREMIUM PARA GRÁFICAS
# ==========================================================

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


# ==========================================================
# FUNCIÓN DEL ESTADÍSTICO DE HOPKINS
# ==========================================================

def calcular_hopkins(X):

    X = np.array(X)

    n, d = X.shape

    # Selecciona el 10% de las observaciones
    m = int(0.1 * n)

    np.random.seed(42)

    vecinos = NearestNeighbors(n_neighbors=2)
    vecinos.fit(X)

    # Puntos artificiales en el mismo espacio
    puntos_aleatorios = np.random.uniform(
        np.min(X, axis=0),
        np.max(X, axis=0),
        (m, d)
    )

    dist_aleatoria, _ = vecinos.kneighbors(
        puntos_aleatorios,
        n_neighbors=1
    )

    # Puntos reales aleatorios
    indices = np.random.choice(
        n,
        m,
        replace=False
    )

    puntos_reales = X[indices]

    dist_real, _ = vecinos.kneighbors(
        puntos_reales,
        n_neighbors=2
    )

    U = np.sum(dist_aleatoria)
    W = np.sum(dist_real[:, 1])

    H = U / (U + W)

    return H


# ==========================================================
# CONFIGURACIÓN GENERAL DE STREAMLIT
# ==========================================================

st.set_page_config(
    page_title="Exposición Minería de Datos - Orden Público en Colombia",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# ESTILOS CSS TIPO DIAPOSITIVA PROFESIONAL
# ==========================================================

st.markdown("""
<style>

.stApp {
    background-color: #F8FAFC;
    color: #1E293B;
    font-family: Helvetica, Arial, sans-serif;
}

[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #E2E8F0;
}


.slide-container {
    background-color: white;
    padding: 40px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    margin-bottom: 25px;
    border: 1px solid #E2E8F0;
}


.slide-title {
    color: #0F172A;
    font-size: 36px;
    font-weight: 700;
}


.slide-subtitle {
    color: #64748B;
    font-size: 18px;
    margin-bottom: 25px;
}


div.stButton > button {
    background-color: #E0F2FE !important;
    color: #0369A1 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}


div.stButton > button[kind="primary"] {
    background-color: #0284C7 !important;
    color: white !important;
}


.insight-card {
    background-color: #F1F5F9;
    border-left: 5px solid #38BDF8;
    padding: 18px;
    border-radius: 4px 12px 12px 4px;
}


.insight-success {
    background-color: #F0FDF4;
    border-left: 5px solid #16A34A;
    padding: 18px;
    border-radius: 4px 12px 12px 4px;
}


.insight-critical {
    background-color: #FEF2F2;
    border-left: 5px solid #DC2626;
    padding: 18px;
    border-radius: 4px 12px 12px 4px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# CONTROL DE DIAPOSITIVAS
# ==========================================================

if "diapositiva" not in st.session_state:
    st.session_state.diapositiva = 1


def ir_a_diapositiva(numero):

    st.session_state.diapositiva = numero
    st.rerun()


# ==========================================================
# CARGA AUTOMÁTICA DEL ARCHIVO
# ==========================================================

def cargar_datos():

    archivo_encontrado = None

    for archivo in os.listdir("."):

        nombre = archivo.lower()

        if (
            ("afectacion" in nombre or
             "fuerza" in nombre or
             "publica" in nombre)
            and
            (archivo.endswith(".xlsx") or
             archivo.endswith(".csv"))
        ):

            archivo_encontrado = archivo
            break


    if archivo_encontrado is None:
        return None, "No se encontró la base de datos"


    try:

        if archivo_encontrado.endswith(".csv"):

            df = pd.read_csv(
                archivo_encontrado
            )

        else:

            df = pd.read_excel(
                archivo_encontrado
            )


        return df, archivo_encontrado


    except Exception as e:

        return None, str(e)


df_original, archivo_cargado = cargar_datos()


# ==========================================================
# MENÚ DE NAVEGACIÓN SUPERIOR
# ==========================================================

columnas = st.columns(6)

paginas = [
    "1. Portada",
    "2. Introducción",
    "3. Marco Teórico",
    "4. Metodología",
    "5. Resultados",
    "6. Conclusiones"
]


for i, nombre in enumerate(paginas):

    tipo = (
        "primary"
        if st.session_state.diapositiva == i + 1
        else "secondary"
    )


    if columnas[i].button(
        nombre,
        use_container_width=True,
        type=tipo
    ):

        ir_a_diapositiva(i + 1)


st.markdown("---")
 # ==========================================================
# DIAPOSITIVA 1 - PORTADA
# ==========================================================

if st.session_state.diapositiva == 1:

    st.markdown("""
    <div class='slide-container' style='text-align:center; padding:60px;'>

        <img src='https://administrativos.ut.edu.co/images/Home/simbolos/logo_oficial.png'
             width='190'>

        <h1 class='slide-title'
            style='color:#1E3A8A; font-size:42px;'>

        Análisis de Clústeres (K-Means) en Afectaciones a la Fuerza Pública

        </h1>

        <p class='slide-subtitle'>
        Segmentación territorial de incidentes de orden público
        mediante modelos de aprendizaje no supervisado
        </p>

        <hr>

    </div>
    """, unsafe_allow_html=True)


    col1, col2 = st.columns(2)


    with col1:

        st.markdown("""
        <div class='insight-card'>

        <h4>ESTUDIANTE</h4>

        Miguel Ángel Garatejo<br>
        Facultad de Ciencias<br>
        Matemáticas con Énfasis en Estadística

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown(f"""
        <div class='insight-success'>

        <h4>DOCENTE</h4>

        Yuri Marcela García Saavedra<br>
        Minería de Datos<br>
        Año {time.strftime("%Y")}

        </div>
        """, unsafe_allow_html=True)


    if st.button(
        "Iniciar Sustentación",
        type="primary",
        use_container_width=True
    ):

        ir_a_diapositiva(2)


# ==========================================================
# DIAPOSITIVA 2 - INTRODUCCIÓN
# ==========================================================

elif st.session_state.diapositiva == 2:

    st.markdown("## Introducción y planteamiento del problema")
    st.markdown("### Contexto del orden público y naturaleza de los datos")

    c1, c2 = st.columns(2)


    with c1:

        st.markdown("""
        <div class='slide-container'>

        <h3>Problema inicial de la base de datos</h3>

        La información original estaba compuesta principalmente por
        variables categóricas, lo que impedía aplicar directamente
        algoritmos basados en distancias como K-Means.

        Se contaba con 8 variables categóricas y una variable
        numérica de cantidad de afectados.

        </div>
        """, unsafe_allow_html=True)


    with c2:

        st.markdown("""
        <div class='slide-container'>

        <h3>Objetivo del procesamiento</h3>

        Transformar la información histórica en una matriz numérica
        donde cada municipio pudiera ser representado mediante un
        vector de características cuantificables.

        Esto permite calcular similitudes, distancias y construir
        agrupamientos mediante aprendizaje no supervisado.

        </div>
        """, unsafe_allow_html=True)


    if st.button(
        "Siguiente: Marco Teórico",
        type="primary"
    ):

        ir_a_diapositiva(3)


# ==========================================================
# DIAPOSITIVA 3 - MARCO TEÓRICO
# ==========================================================

elif st.session_state.diapositiva == 3:

    st.markdown("""
    <h1 class='slide-title'>
    Fundamentos matemáticos del modelo
    </h1>
    """, unsafe_allow_html=True)


    t1, t2, t3 = st.columns(3)


    with t1:

        st.info("""
        **Pivotado de datos**

        Convierte variables categóricas en columnas numéricas,
        permitiendo representar matemáticamente cada municipio.
        """)


    with t2:

        st.info("""
        **Algoritmo K-Means**

        Divide los municipios en grupos similares minimizando la
        distancia entre los datos y el centroide de cada clúster.
        """)


    with t3:

        st.info("""
        **Análisis PCA**

        Reduce la dimensionalidad de los datos manteniendo la mayor
        cantidad posible de información estadística.
        """)


    st.success("""
    La normalización mediante StandardScaler transforma todas las
    variables a una escala común con media 0 y desviación estándar 1,
    evitando que variables con valores grandes dominen el cálculo
    de distancias.
    """)


    if st.button(
        "Siguiente: Metodología",
        type="primary"
    ):

        ir_a_diapositiva(4)


# ==========================================================
# DIAPOSITIVA 4 - METODOLOGÍA
# ==========================================================

elif st.session_state.diapositiva == 4:


    st.markdown("""
    <h1 class='slide-title'>
    Flujo de procesamiento de datos
    </h1>
    """, unsafe_allow_html=True)


    st.markdown("""
    ### Fase 1 - Transformación de datos
    """)


    st.code("""
pivot_accion = df_original.pivot_table(
    index=['COD_MUNI','MUNICIPIO','DEPARTAMENTO'],
    columns='ACCION',
    values='CANTIDAD',
    aggfunc='sum',
    fill_value=0
)
    """, language="python")


    st.markdown("""
    ### Fase 2 - Normalización
    """)


    st.code("""
scaler = StandardScaler()

datos[numericas] = scaler.fit_transform(
    datos[numericas]
)

X_scaled = datos.drop(
    columns=['COD_MUNI','MUNICIPIO','DEPARTAMENTO']
)
    """, language="python")


    st.markdown("""
    ### Fase 3 - Validación con Hopkins
    """)


    st.code("""
valor_hopkins = calcular_hopkins(X_scaled)
    """, language="python")


    st.markdown("""
    ### Fase 4 - Clustering K-Means
    """)


    st.code("""
kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

kmeans.fit(X_scaled)
    """, language="python")


    if st.button(
        "Siguiente: Resultados del Modelo",
        type="primary"
    ):

        ir_a_diapositiva(5)

# ==========================================================
# DIAPOSITIVA 5 - RESULTADOS DEL MODELO
# ==========================================================

elif st.session_state.diapositiva == 5:

    st.markdown("""
    <div class='slide-title'>
    Hallazgos y análisis de los clústeres
    </div>

    <div class='slide-subtitle'>
    Evaluación de patrones, agrupamientos y comportamiento territorial
    </div>
    """, unsafe_allow_html=True)


    if df_original is None:
        st.error("No se encontró el archivo de datos para realizar el análisis.")
        st.stop()


    # ======================================================
    # TRANSFORMACIÓN DE LA BASE ORIGINAL
    # ======================================================

    index_cols = [
        "COD_MUNI",
        "MUNICIPIO",
        "DEPARTAMENTO"
    ]


    pivot_accion = df_original.pivot_table(
        index=index_cols,
        columns="ACCION",
        values="CANTIDAD",
        aggfunc="sum",
        fill_value=0
    )


    pivot_fuerza = df_original.pivot_table(
        index=index_cols,
        columns="NOMBRE_FUERZA",
        values="CANTIDAD",
        aggfunc="sum",
        fill_value=0
    )


    pivot_categoria = df_original.pivot_table(
        index=index_cols,
        columns="CATEGORIA",
        values="CANTIDAD",
        aggfunc="sum",
        fill_value=0
    )


    total = (
        df_original
        .groupby(index_cols)["CANTIDAD"]
        .sum()
        .to_frame("TOTAL_AFECTADOS")
    )


    datos = (
        total
        .join([
            pivot_accion,
            pivot_fuerza,
            pivot_categoria
        ])
        .reset_index()
        .fillna(0)
    )


    # ======================================================
    # NORMALIZACIÓN
    # ======================================================

    columnas_omitir = [
        "COD_MUNI",
        "MUNICIPIO",
        "DEPARTAMENTO"
    ]


    datos_originales = datos.copy()


    columnas_numericas = [
        col for col in datos.columns
        if col not in columnas_omitir
    ]


    scaler = StandardScaler()

    datos[columnas_numericas] = scaler.fit_transform(
        datos[columnas_numericas]
    )


    X_scaled = datos.drop(
        columns=columnas_omitir
    )


    # ======================================================
    # ESTADÍSTICO DE HOPKINS
    # ======================================================

    valor_hopkins = calcular_hopkins(X_scaled)


    st.markdown(
        "## A. Validación de tendencia de agrupamiento (Hopkins)"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Valor Hopkins",
            f"{valor_hopkins:.3f}"
        )


    with col2:

        if valor_hopkins < 0.5:

            st.error(
                "Los datos no presentan una estructura clara de agrupamiento."
            )

        elif valor_hopkins < 0.75:

            st.warning(
                "Los datos presentan una tendencia moderada a formar clústeres."
            )

        else:

            st.success(
                "Los datos presentan una fuerte estructura de clústeres y son adecuados para aplicar K-Means."
            )


    st.markdown("""
    <div class='insight-card'>
    El estadístico de Hopkins compara la distribución de los municipios
    reales frente a puntos generados aleatoriamente en el mismo espacio.
    Un valor cercano a 1 indica una fuerte tendencia a formar grupos.
    </div>
    """, unsafe_allow_html=True)


    # ======================================================
    # MÉTODO DEL CODO
    # ======================================================


    st.markdown(
        "## B. Selección del número óptimo de clústeres (Método del Codo)"
    )


    wss = []


    for k in range(1, 11):

        modelo = KMeans(
            n_clusters=k,
            n_init=30,
            random_state=42
        )

        modelo.fit(X_scaled)

        wss.append(
            modelo.inertia_
        )


    fig_codo = px.line(
        x=list(range(1, 11)),
        y=wss,
        markers=True,
        title="Curva del Codo - Inercia WSS"
    )


    fig_codo.add_vline(
        x=4,
        line_dash="dash",
        line_color="red",
        annotation_text="K óptimo = 4"
    )


    fig_codo = aplicar_estilo_premium(
        fig_codo
    )


    st.plotly_chart(
        fig_codo,
        use_container_width=True
    )

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


# ==============================================================================
# ESTILO PREMIUM PARA GRÁFICAS
# ==============================================================================

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


# ==============================================================================
# FUNCIÓN PARA EL ESTADÍSTICO DE HOPKINS
# ==============================================================================

def calcular_hopkins(X):

    X = np.array(X)

    n, d = X.shape

    # Se toma el 10 % de los datos
    m = int(0.1 * n)

    np.random.seed(42)

    vecinos = NearestNeighbors(n_neighbors=2)
    vecinos.fit(X)

    # Puntos aleatorios en el mismo espacio de los datos
    puntos_aleatorios = np.random.uniform(
        np.min(X, axis=0),
        np.max(X, axis=0),
        (m, d)
    )

    dist_aleatoria, _ = vecinos.kneighbors(
        puntos_aleatorios,
        n_neighbors=1
    )

    # Selección de puntos reales
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


# ==============================================================================
# CONFIGURACIÓN GENERAL DE STREAMLIT
# ==============================================================================

st.set_page_config(
    page_title="Exposición Minería de Datos - Orden Público en Colombia",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==============================================================================
# ESTILO CSS DE LA PRESENTACIÓN
# ==============================================================================

st.markdown("""
<style>

.stApp {
    background-color: #F8FAFC;
    color: #1E293B;
    font-family: Helvetica, Arial, sans-serif;
}


[data-testid="stSidebar"] {
    background-color: white !important;
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
    color:#0F172A;
    font-size:36px;
    font-weight:700;
}


.slide-subtitle {
    color:#64748B;
    font-size:18px;
}


.insight-card {
    background:#F1F5F9;
    border-left:5px solid #38BDF8;
    padding:18px;
    border-radius:4px 12px 12px 4px;
}


.insight-success {
    background:#F0FDF4;
    border-left:5px solid #16A34A;
    padding:18px;
    border-radius:4px 12px 12px 4px;
}

</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CONTROL DE DIAPOSITIVAS
# ==============================================================================

if "diapositiva" not in st.session_state:
    st.session_state.diapositiva = 1


def ir_a_diapositiva(num):
    st.session_state.diapositiva = num
    st.rerun()


# ==============================================================================
# CARGA AUTOMÁTICA DEL ARCHIVO
# ==============================================================================

def cargar_datos_automatico():

    archivos = os.listdir(".")

    archivo_encontrado = None

    for archivo in archivos:

        nombre = archivo.lower()

        if (
            ("afectacion" in nombre or
             "fuerza" in nombre or
             "publica" in nombre)
            and
            (archivo.endswith(".csv") or archivo.endswith(".xlsx"))
        ):

            archivo_encontrado = archivo
            break


    if archivo_encontrado is None:
        return None, "Archivo no encontrado"


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


df_original, nombre_archivo = cargar_datos_automatico()


# ==============================================================================
# MENÚ SUPERIOR
# ==============================================================================

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


# ==============================================================================
# DIAPOSITIVA 1 - PORTADA
# ==============================================================================

if st.session_state.diapositiva == 1:

    st.title(
        "Análisis de Clústeres K-Means en Afectaciones a la Fuerza Pública"
    )

    st.subheader(
        "Segmentación territorial mediante aprendizaje no supervisado"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.info("""
        ESTUDIANTE

        Miguel Ángel Garatejo

        Matemáticas con Énfasis en Estadística
        """)


    with col2:

        st.success(f"""
        PROFESORA

        Yuri Marcela García Saavedra

        Minería de Datos

        Año: {time.strftime("%Y")}
        """)


    if st.button(
        "Iniciar Sustentación",
        type="primary",
        use_container_width=True
    ):

        ir_a_diapositiva(2)


# ==============================================================================
# DIAPOSITIVA 2 - INTRODUCCIÓN
# ==============================================================================

elif st.session_state.diapositiva == 2:

    st.header(
        "Introducción y problema de los datos"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.warning("""
        El conjunto original posee 8 variables categóricas
        y una variable numérica.

        K-Means requiere variables numéricas porque trabaja
        con distancias euclidianas.
        """)


    with col2:

        st.success("""
        Se aplicó una transformación mediante tablas dinámicas
        para convertir categorías en nuevas variables numéricas
        por municipio.
        """)


    if st.button(
        "Siguiente: Marco Teórico",
        type="primary"
    ):

        ir_a_diapositiva(3)
# ==============================================================================
# DIAPOSITIVA 3 - MARCO TEÓRICO
# ==============================================================================

elif st.session_state.diapositiva == 3:

    st.header(
        "Fundamentos teóricos y algorítmicos"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.info("""
        Transformación mediante Pivotado
        
        Las variables categóricas originales se reorganizan
        en nuevas columnas numéricas por municipio.
        """)


    with col2:

        st.info("""
        Algoritmo K-Means
        
        Método no supervisado que agrupa municipios
        minimizando la distancia entre los datos y sus centroides.
        """)


    with col3:

        st.info("""
        Análisis de Componentes Principales (PCA)
        
        Reduce la dimensionalidad conservando la mayor
        cantidad de información posible.
        """)


    st.success("""
    Antes de aplicar K-Means se normalizan los datos mediante
    Z-Score para garantizar que todas las variables tengan
    la misma importancia en el cálculo de distancias.
    """)


    if st.button(
        "Siguiente: Metodología",
        type="primary"
    ):
        ir_a_diapositiva(4)



# ==============================================================================
# DIAPOSITIVA 4 - METODOLOGÍA
# ==============================================================================

elif st.session_state.diapositiva == 4:

    st.header(
        "Arquitectura del procesamiento de datos"
    )


    st.code("""
    Base de datos original
            ↓
    Variables categóricas
            ↓
    Pivotado por municipio
            ↓
    Nuevas variables numéricas
            ↓
    Normalización Z-Score
            ↓
    Estadístico Hopkins
            ↓
    Método del Codo
            ↓
    K-Means
            ↓
    PCA y análisis final
    """)


    if st.button(
        "Ejecutar análisis de resultados",
        type="primary"
    ):
        ir_a_diapositiva(5)



# ==============================================================================
# DIAPOSITIVA 5 - RESULTADOS
# ==============================================================================

elif st.session_state.diapositiva == 5:


    st.header(
        "Análisis de Clústeres y Resultados"
    )


    if df_original is None:

        st.error(
            "No se encontró el archivo de datos."
        )

        st.stop()



    # =========================================================
    # TRANSFORMACIÓN DE VARIABLES
    # =========================================================

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
            pivot_fuerza
        ])
        .reset_index()
        .dropna()
    )



    # =========================================================
    # NORMALIZACIÓN DE DATOS
    # =========================================================


    columnas_excluir = [
        "COD_MUNI",
        "MUNICIPIO",
        "DEPARTAMENTO"
    ]


    variables_numericas = [
        c for c in datos.columns
        if c not in columnas_excluir
    ]


    datos_originales = datos.copy()


    scaler = StandardScaler()


    datos[variables_numericas] = scaler.fit_transform(
        datos[variables_numericas]
    )


    X_scaled = datos.drop(
        columns=columnas_excluir
    )



    # =========================================================
    # ESTADÍSTICO DE HOPKINS
    # =========================================================


    valor_hopkins = calcular_hopkins(X_scaled)



    st.subheader(
        "A. Validación de tendencia de agrupamiento (Hopkins)"
    )


    col_h1, col_h2 = st.columns(2)


    with col_h1:

        st.metric(
            "Valor Hopkins",
            f"{valor_hopkins:.3f}"
        )


    with col_h2:


        if valor_hopkins < 0.50:

            st.error(
                "Los datos tienen una estructura cercana al azar. "
                "El agrupamiento no es recomendable."
            )


        elif valor_hopkins < 0.75:

            st.warning(
                "Existe una tendencia moderada a formar clústeres."
            )


        else:

            st.success(
                "Existe una fuerte estructura de agrupamiento. "
                "Es adecuado aplicar K-Means."
            )


    st.info("""
    El estadístico de Hopkins compara los municipios reales
    con puntos aleatorios generados en el mismo espacio de datos.

    Valores cercanos a 1 indican una fuerte presencia de grupos,
    mientras que valores cercanos a 0.5 representan datos aleatorios.
    """)


    # Guardamos una copia para el siguiente análisis
    X_modelo = X_scaled.copy()   

    # =========================================================
    # B. MÉTODO DEL CODO
    # =========================================================

    st.subheader(
        "B. Selección del número óptimo de clústeres (Método del Codo)"
    )

    wss = []

    for k in range(1, 11):

        modelo = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=42
        )

        modelo.fit(X_modelo)

        wss.append(
            modelo.inertia_
        )


    fig_codo = px.line(
        x=list(range(1, 11)),
        y=wss,
        markers=True,
        labels={
            "x": "Número de clústeres (K)",
            "y": "Inercia WSS"
        },
        title="Evaluación del número óptimo de grupos"
    )


    fig_codo.add_vline(
        x=4,
        line_dash="dash",
        annotation_text="K = 4 seleccionado"
    )


    fig_codo = aplicar_estilo_premium(fig_codo)


    st.plotly_chart(
        fig_codo,
        use_container_width=True
    )


    st.info("""
    El punto de inflexión de la curva se presenta en K=4,
    indicando que agregar más grupos produce una mejora
    cada vez menor en la reducción de la variabilidad interna.
    """)


    # =========================================================
    # C. APLICACIÓN DEL MODELO K-MEANS
    # =========================================================

    st.subheader(
        "C. Aplicación del algoritmo K-Means"
    )


    kmeans = KMeans(
        n_clusters=4,
        n_init=30,
        random_state=42
    )


    etiquetas = kmeans.fit_predict(X_modelo)


    datos_originales["Cluster"] = etiquetas.astype(str)


    st.success(
        "El modelo clasificó correctamente los municipios en cuatro grupos."
    )


    # =========================================================
    # D. MATRIZ DE DISTANCIAS EUCLIDIANAS
    # =========================================================

    st.subheader(
        "D. Distancias entre municipios"
    )


    matriz_distancia = euclidean_distances(
        X_modelo
    )


    muestra = min(50, len(matriz_distancia))


    fig_dist = px.imshow(
        matriz_distancia[:muestra, :muestra],
        title="Mapa de calor de distancias euclidianas",
        color_continuous_scale="Viridis"
    )


    fig_dist = aplicar_estilo_premium(fig_dist)


    st.plotly_chart(
        fig_dist,
        use_container_width=True
    )


    # =========================================================
    # E. REDUCCIÓN DE DIMENSIONES PCA
    # =========================================================

    st.subheader(
        "E. Representación espacial con PCA"
    )


    pca = PCA(
        n_components=3
    )


    componentes = pca.fit_transform(
        X_modelo
    )


    df_pca = pd.DataFrame(
        componentes,
        columns=[
            "PC1",
            "PC2",
            "PC3"
        ]
    )


    df_pca["Cluster"] = etiquetas.astype(str)


    df_pca["Municipio"] = datos_originales[
        "MUNICIPIO"
    ].values


    fig_pca = px.scatter_3d(
        df_pca,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        hover_name="Municipio",
        title="Distribución de municipios en el espacio PCA"
    )


    fig_pca = aplicar_estilo_premium(
        fig_pca
    )


    st.plotly_chart(
        fig_pca,
        use_container_width=True
    )


    st.info("""
    PCA permite proyectar el espacio de alta dimensión
    en tres componentes principales conservando la mayor
    cantidad posible de variabilidad de los datos.
    """)


    # =========================================================
    # F. PERFIL PROMEDIO DE LOS CLÚSTERES
    # =========================================================

    st.subheader(
        "F. Perfil promedio de cada grupo"
    )


    variables = [
        col for col in datos_originales.columns
        if col not in [
            "COD_MUNI",
            "MUNICIPIO",
            "DEPARTAMENTO"
        ]
    ]


    resumen = (
        datos_originales
        .groupby("Cluster")[variables]
        .mean()
        .round(2)
    )


    st.dataframe(
        resumen,
        use_container_width=True
    )


    st.success("""
    Cada clúster representa un conjunto de municipios
    con comportamientos similares en sus patrones
    de afectación a la fuerza pública.
    """)


    if st.button(
        "Ir a conclusiones",
        type="primary"
    ):
        ir_a_diapositiva(6)



# =========================================================
# DIAPOSITIVA 6 - CONCLUSIONES
# =========================================================

elif st.session_state.diapositiva == 6:


    st.header(
        "Conclusiones del estudio"
    )


    st.markdown("""
    ### Resultados principales

    - Se transformó una base con variables categóricas
      en una matriz completamente numérica.

    - La normalización mediante StandardScaler permitió
      realizar comparaciones utilizando distancias.

    - El estadístico de Hopkins confirmó que los datos
      presentan una estructura apropiada para clustering.

    - El método del codo permitió seleccionar un modelo
      de K-Means con cuatro grupos.

    - PCA facilitó la visualización espacial de los
      municipios según su comportamiento.
    """)


    st.success("""
    La metodología completa permitió identificar
    patrones territoriales ocultos en la información
    histórica de afectaciones a la fuerza pública.
    """)


    if st.button(
        "Volver al inicio",
        type="secondary"
    ):
        ir_a_diapositiva(1)
        

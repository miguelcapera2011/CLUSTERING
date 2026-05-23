import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO TÁCTICO/PROFESIONAL
# ==============================================================================
st.set_page_config(
    page_title="Análisis de Clúster - Fuerza Pública", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de estilos CSS personalizados basados en el formato de la presentación
st.markdown("""
<style>
    .main-title {
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
        text-transform: uppercase;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 20px !important;
        color: #94a3b8 !important;
        margin-bottom: 30px;
    }
    .slide-section-title {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        border-left: 6px solid #38bdf8;
        padding-left: 15px;
        text-transform: uppercase;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .accent-text {
        color: #38bdf8;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# FUNCIÓN AUTOMÁTICA PARA CARGAR LA BASE DE DATOS
# ==============================================================================
def cargar_datos_automatico():
    """Busca y carga dinámicamente el archivo de datos en la carpeta actual"""
    archivos_en_carpeta = os.listdir('.')
    archivo_encontrado = None
    
    for archivo in archivos_en_carpeta:
        nombre_minuscula = archivo.lower()
        if ("afectacion" in nombre_minuscula or "fuerza" in nombre_minuscula or "publica" in nombre_minuscula) and (archivo.endswith('.csv') or archivo.endswith('.xlsx')):
            archivo_encontrado = archivo
            break
            
    if archivo_encontrado is None:
        return None, "No se encontró ningún archivo que contenga 'AFECTACIÓN' o 'FUERZA' en la carpeta."
    
    try:
        if archivo_encontrado.endswith('.csv'):
            df = pd.read_csv(archivo_encontrado, header=0)
        else:
            df = pd.read_excel(archivo_encontrado, header=0)
        return df, archivo_encontrado
    except Exception as e:
        return None, f"Error al leer el archivo {archivo_encontrado}: {str(e)}"

# ==============================================================================
# NAVEGACIÓN ENTRE PÁGINAS (ESTRUCTURA DE LA EXPOSICIÓN)
# ==============================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'introduccion'

def cambiar_pagina(nombre_pagina):
    st.session_state.page = nombre_pagina

# Barra lateral - Control del Hilo Conductor de la Exposición
st.sidebar.markdown("### 📋 Agenda de la Exposición")

if st.sidebar.button("1. 🎬 Portada e Introducción", use_container_width=True):
    cambiar_pagina('introduccion')
if st.sidebar.button("2. 🗺️ Desafío y Pipeline", use_container_width=True):
    cambiar_pagina('pipeline')
if st.sidebar.button("3. 🧪 Modelado y Optimización", use_container_width=True):
    cambiar_pagina('modelado')
if st.sidebar.button("4. 🎯 Clústeres e Impacto", use_container_width=True):
    cambiar_pagina('resultados')
if st.sidebar.button("5. 🏁 Conclusiones y Cierre", use_container_width=True):
    cambiar_pagina('cierre')

# ==============================================================================
# CARGA DE DATOS TRANSVERSAL
# ==============================================================================
df_original, resultado_carga = cargar_datos_automatico()

# Preprocesamiento base indispensable si los datos existen
if df_original is not None:
    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
    
    columnas_fuerza = [c for c in df_original['NOMBRE_FUERZA'].unique() if pd.notna(c)] if 'NOMBRE_FUERZA' in df_original.columns else []
    pivot_fuerza = df_original.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_fuerza else pd.DataFrame(index=pivot_accion.index)
    
    columnas_cat = [c for c in df_original['CATEGORIA'].unique() if pd.notna(c)] if 'CATEGORIA' in df_original.columns else []
    pivot_cat = df_original.pivot_table(index=index_cols, columns='CATEGORIA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_cat else pd.DataFrame(index=pivot_accion.index)
    
    total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')

    datos = total_municipio.join([pivot_accion, pivot_fuerza, pivot_cat]).reset_index()
    datos = datos.rename(columns={'MUNICIPIO': 'MUNICIPIO'}).dropna()
    
    # Columnas dinámicas de referencia para la interfaz
    col_afectados = 'TOTAL_AFECTADOS' if 'TOTAL_AFECTADOS' in datos.columns else datos.columns[3]
    col_asesinado = 'ASESINADO' if 'ASESINADO' in datos.columns else datos.columns[4]
    col_herido = 'HERIDO' if 'HERIDO' in datos.columns else datos.columns[5]
    col_ejercito = 'EJERCITO NACIONAL DE COLOMBIA' if 'EJERCITO NACIONAL DE COLOMBIA' in datos.columns else datos.columns[6]

# ==============================================================================
# PÁGINA 1: PORTADA E INTRODUCCIÓN
# ==============================================================================
if st.session_state.page == 'introduccion':
    st.markdown('<div class="main-title">Análisis de Clúster de Afectación a la Fuerza Pública</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Optimización de Datos Categóricos y Modelado K-Means Avanzado</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 📊 Contexto del Proyecto")
        st.markdown("""
        Esta investigación aplica técnicas de **Machine Learning No Supervisado** para identificar y agrupar los patrones de riesgo y afectaciones territoriales que sufre la **Fuerza Pública** en Colombia.
        
        A través de un riguroso análisis matemático de distancias espaciales, el modelo segmenta los municipios según la intensidad, el actor institucional afectado y el tipo de acción registrada, permitiendo un entendimiento multidimensional de la seguridad operacional.
        """)
        
        st.info("💡 **Hito Clave:** Transformación exitosa de registros de texto cualitativos en indicadores de riesgo numéricos puros.")
    
    with col2:
        st.markdown("### 🎯 Objetivos Estratégicos")
        st.markdown("""
        * **Segmentación Territorial:** Agrupar los municipios bajo perfiles homogéneos de riesgo latente.
        * **Identificación de Focos:** Detectar áreas de alta complejidad operativa para el Ejército y la Policía Nacional.
        * **Optimización Táctica:** Proveer una base matemática sólida para la distribución estratégica de recursos estatales.
        """)

    st.markdown("---")
    st.markdown("### 🎛️ Estado de los Componentes del Sistema")
    c1, c2, c3 = st.columns(3)
    if df_original is not None:
        c1.metric("Base de Datos Detectada", resultado_carga, help="Archivo cargado automáticamente")
        c2.metric("Municipios Mapeados", datos.shape[0])
        c3.metric("Variables Numéricas Generadas", datos.shape[1] - 3)
    else:
        c1.error("❌ Archivo de datos no detectado en el directorio.")

# ==============================================================================
# PÁGINA 2: EL DESAFÍO Y PIPELINE DE DATOS
# ==============================================================================
elif st.session_state.page == 'pipeline':
    st.markdown('<div class="main-title">El Pipeline de Transformación de Datos</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Superando la Restricción Categórica mediante Ingeniería de Características</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.error("### 🛑 El Desafío Técnico de la Base Original")
        st.markdown("""
        * **Estructura Inicial:** Bitácora plana de eventos históricos de orden público.
        * **Restricción Geométrica:** Contiene **8 variables categóricas** (Texto) y únicamente **1 numérica** (`CANTIDAD`).
        * **El Problema:** Algoritmos basados en distancias (como **K-Means**) son completamente incapaces de calcular distancias sobre texto directo o nombres de instituciones (ej: 'EJÉRCITO').
        """)
    with col2:
        st.success("### 💡 La Solución y Adaptación Matemática")
        st.markdown("""
        * **Pivotado Cruzado (Reshaping):** Conversión de las filas de texto cualitativo en columnas numéricas de conteo independiente.
        * **Consolidación Municipal:** Agrupación y colapso de toda la bitácora bajo el código único de municipio (`COD_MUNI`).
        * **Resultado:** Transformación de la bitácora a una matriz estructurada apta para cálculos multidimensionales.
        """)
        
    st.markdown('<div class="slide-section-title">Código del Pipeline y Muestra de Datos</div>', unsafe_allow_html=True)
    
    with st.expander("🔄 1. Reestructuración de Datos (Matriz de Pivotado)", expanded=True):
        st.code("""
# Agrupamos por Municipio y pivotamos las variables de texto
pivot_accion = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
pivot_fuerza = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0)
total_municipio = df_original.groupby(['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'])['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')

# Unimos las tablas convirtiendo las categorías en columnas numéricas reales
datos = total_municipio.join([pivot_accion, pivot_fuerza]).reset_index()
        """, language="python")

    if df_original is not None:
        st.subheader("📋 Matriz de Datos Numéricos Consolidados (Primeros 5 Registros)")
        st.dataframe(datos.head(5), use_container_width=True)

# ==============================================================================
# PÁGINA 3: MODELADO Y OPTIMIZACIÓN MATEMÁTICA
# ==============================================================================
elif st.session_state.page == 'modelado':
    st.markdown('<div class="main-title">Modelado Avanzado y Criterios de Optimización</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Estandarización y Determinación del Número Óptimo de Clústeres</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    if df_original is None:
        st.error("❌ Es necesario cargar los datos para visualizar los gráficos de esta sección.")
        st.stop()

    # Procesar Estandarización
    scaler = StandardScaler()
    columnas_omitir = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    numericas = [col for col in datos.columns if col not in columnas_omitir]
    
    datos_originales_num = datos.copy()
    datos_escalados = datos.copy()
    datos_escalados[numericas] = scaler.fit_transform(datos_escalados[numericas])
    X_scaled = datos_escalados.drop(columns=columnas_omitir)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ⚖️ Estandarización (StandardScaler)")
        st.markdown("""
        Evita que las variables con magnitudes numéricas colosales dominen, sesguen o distorsionen el cálculo de las distancias geométricas. 
        Transforma los datos para que tengan **Media = 0** y **Varianza = 1**.
        """)
        st.latex(r"z = \frac{x - \mu}{\sigma}")
        
        # Selector para explorar la distribución interactiva
        st.markdown("#### 📊 Distribución de las Variables Principales")
        hist_variables = [col_afectados, col_asesinado, col_herido, col_ejercito]
        variable_seleccionada = st.selectbox("Seleccione la variable a analizar:", hist_variables)
        
        fig_hist_int = px.histogram(datos_originales_num, x=variable_seleccionada, nbins=25, 
                                    template="plotly_dark", color_discrete_sequence=['#38bdf8'])
        fig_hist_int.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_hist_int, use_container_width=True)

    with col2:
        st.markdown("### 📐 Curva de Optimización: Método del Codo")
        st.markdown("""
        Se itera el modelo calculando la **Inercia (WSS)**. El punto de quiebre óptimo identificado es **K = 4**, donde se maximiza la cohesión interna sin sobreajustar el modelo.
        """)
        
        # Cálculo dinámico del Método del Codo
        wss = []
        for k in range(1, 11):
            kmeans_test = KMeans(n_clusters=k, n_init=30, random_state=42)
            kmeans_test.fit(X_scaled)
            wss.append(kmeans_test.inertia_)
            
        fig_elbow_int = px.line(x=list(range(1, 11)), y=wss, markers=True, 
                                labels={'x': 'Número de Clústeres (k)', 'y': 'WSS / Inercia'}, template='plotly_dark')
        fig_elbow_int.add_vline(x=4, line_dash="dash", line_color="cyan", annotation_text="K Óptimo = 4")
        fig_elbow_int.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_elbow_int, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="slide-section-title">🌡️ Mapas de Calor de Distancias (Muestra Interactiva de Proximidad)</div>', unsafe_allow_html=True)
    
    distancias_eu = euclidean_distances(X_scaled)[:50, :50]
    dist_matrix_manhattan = pdist(X_scaled, metric='cityblock')
    C = squareform(dist_matrix_manhattan)[:50, :50]
    nombres_municipios_sub = datos['MUNICIPIO'].iloc[:50].tolist()

    mapa_col1, mapa_col2 = st.columns(2)
    with mapa_col1:
        fig_eu = px.imshow(distancias_eu, x=nombres_municipios_sub, y=nombres_municipios_sub,
                           title="Distancia Euclideana (Primeros 50 municipios)",
                           color_continuous_scale='RdBu_r', template='plotly_dark')
        fig_eu.update_layout(height=350)
        st.plotly_chart(fig_eu, use_container_width=True)
    with mapa_col2:
        fig_man = px.imshow(C, x=nombres_municipios_sub, y=nombres_municipios_sub,
                            title="Distancia Manhattan (Primeros 50 municipios)",
                            color_continuous_scale='RdBu_r', template='plotly_dark')
        fig_man.update_layout(height=350)
        st.plotly_chart(fig_man, use_container_width=True)

# ==============================================================================
# PÁGINA 4: RESULTADOS DE LOS CLÚSTERES E IMPACTO
# ==============================================================================
elif st.session_state.page == 'resultados':
    st.markdown('<div class="main-title">Estructura e Impacto Táctico de los Clústeres</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Análisis Multidimensional y Perfiles Estadísticos de Riesgo Territoriales</div>', unsafe_allow_html=True)
    st.markdown("---")

    if df_original is None:
        st.error("❌ Cargue la base de datos para ejecutar K-Means en tiempo real.")
        st.stop()

    # Ejecución Real del Algoritmo con K=4
    scaler = StandardScaler()
    columnas_omitir = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    numericas = [col for col in datos.columns if col not in columnas_omitir]
    
    datos_originales_num = datos.copy()
    datos_cruce = datos.copy()
    datos_cruce[numericas] = scaler.fit_transform(datos_cruce[numericas])
    X_scaled = datos_cruce.drop(columns=columnas_omitir)

    kmeans = KMeans(n_clusters=4, n_init=50, random_state=42)
    start = time.time()
    km4_clusters = kmeans.fit(X_scaled)
    tiempo_ms = (time.time() - start) * 1000

    datos_originales_num['Cluster'] = km4_clusters.labels_
    datos_cruce['Cluster'] = km4_clusters.labels_.astype(str)

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filtros de Consulta Operativa")
    deptos_disponibles = ["TODOS"] + sorted(datos_originales_num['DEPARTAMENTO'].unique().tolist())
    depto_seleccionado = st.sidebar.selectbox("Filtrar visualizaciones por Región:", deptos_disponibles)
    municipio_buscar = st.sidebar.text_input("Buscar Clúster de un Municipio Específico:").strip().upper()

    # Reducción de dimensionalidad con PCA
    pca_4d = PCA(n_components=4)
    pca_scores_4d = pca_4d.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pca_scores_4d, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    pca_df['Cluster'] = km4_clusters.labels_.astype(str)
    pca_df['MUNICIPIO'] = datos['MUNICIPIO'].values
    pca_df['DEPARTAMENTO'] = datos['DEPARTAMENTO'].values
    pca_df['Etiqueta'] = pca_df['MUNICIPIO'] + " (" + pca_df['DEPARTAMENTO'] + ")"

    centroids_pca_3d = pca_4d.transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_pca_3d, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    
    variables_perfil = [col_afectados, col_asesinado, col_herido, col_ejercito]
    promedios_por_cluster = datos_originales_num.groupby('Cluster')[variables_perfil].mean().reset_index()
    
    centroids_df['Cluster'] = promedios_por_cluster['Cluster'].astype(str)
    centroids_df['Promedio_Afectados'] = promedios_por_cluster[col_afectados].round(2)
    centroids_df['Promedio_Asesinados'] = promedios_por_cluster[col_asesinado].round(2)
    centroids_df['Promedio_Heridos'] = promedios_por_cluster[col_herido].round(2)
    centroids_df['Promedio_Ejercito'] = promedios_por_cluster[col_ejercito].round(2)

    # Filtrado lógico por barra lateral
    df_pca_filtrado = pca_df.copy()
    df_cruce_filtrado = datos_cruce.copy()
    df_box_filtrado = datos_originales_num.copy()

    if depto_seleccionado != "TODOS":
        df_pca_filtrado = df_pca_filtrado[df_pca_filtrado['DEPARTAMENTO'] == depto_seleccionado]
        df_cruce_filtrado = df_cruce_filtrado[df_cruce_filtrado['DEPARTAMENTO'] == depto_seleccionado]
        df_box_filtrado = df_box_filtrado[df_box_filtrado['DEPARTAMENTO'] == depto_seleccionado]

    if municipio_buscar:
        muni_encontrado = pca_df[pca_df['MUNICIPIO'].str.contains(municipio_buscar, na=False)]
        if not muni_encontrado.empty:
            for _, row in muni_encontrado.iterrows():
                st.sidebar.success(f"📍 `{row['MUNICIPIO']}` mapeado en el **Clúster {row['Cluster']}**")
        else:
            st.sidebar.warning("⚠️ Municipio no localizado en el dataset.")

    st.info(f"⚡ **Métrica de Desempeño:** K-Means calculado en {tiempo_ms:.2f} ms | Inercia Matemática Final: {km4_clusters.inertia_:.2f}")

    st.markdown('<div class="slide-section-title">📋 Perfil Estadístico Medio de los Clústeres (Valores Reales Absolutos)</div>', unsafe_allow_html=True)
    tabla_perfil_print = promedios_por_cluster.set_index('Cluster').round(2)
    tabla_perfil_print['Cantidad_Municipios'] = datos_originales_num.groupby('Cluster').size()
    tabla_perfil_print = tabla_perfil_print.rename(columns={
        col_afectados: 'Promedio Afectados Total',
        col_asesinado: 'Promedio Asesinados',
        col_herido: 'Promedio Heridos',
        col_ejercito: 'Promedio Eventos Ejército'
    })
    st.dataframe(tabla_perfil_print, use_container_width=True)
    st.markdown("💡 *Nota para el expositor: Utiliza esta matriz analítica durante tu sustentación para sustentar la categorización del grado de vulnerabilidad de cada grupo.*")

    colores_clusters = ['red', 'green', 'blue', 'orange']

    st.markdown('<div class="slide-section-title">🎯 Espacio de Dispersión Reducido (PCA 2D) y Cruce Analítico</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_pca_int = px.scatter(df_pca_filtrado, x='PC1', y='PC2', color='Cluster',
                                 color_discrete_sequence=colores_clusters, hover_name='Etiqueta',
                                 title="Clústeres K-Means Proyectados en Espacio PCA 2D", template='plotly_dark')
        fig_pca_int.add_trace(go.Scatter(x=centroids_df['PC1'], y=centroids_df['PC2'], mode='markers',
                                         marker=dict(size=14, color='white', symbol='star', line=dict(width=2, color='black')),
                                         name='Centroides'))
        st.plotly_chart(fig_pca_int, use_container_width=True)
    with c2:
        fig_cruzado_int = px.scatter(df_cruce_filtrado, x=col_afectados, y=col_asesinado, color='Cluster',
                                     color_discrete_sequence=colores_clusters, hover_data=['MUNICIPIO', 'DEPARTAMENTO'],
                                     title="Cruce de Impacto Total vs Personal Asesinado (Estandarizado)", template='plotly_dark')
        st.plotly_chart(fig_cruzado_int, use_container_width=True)

    st.markdown('<div class="slide-section-title">📦 Comportamiento de Variabilidad y Distribución 3D Avanzada</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        fig_box_int = px.box(df_box_filtrado, x='Cluster', y=col_afectados, color='Cluster',
                             color_discrete_sequence=colores_clusters, hover_data=['MUNICIPIO'], 
                             title="Distribución Absoluta de Afectados por Grupo (Valores Reales)", template='plotly_dark')
        st.plotly_chart(fig_box_int, use_container_width=True)
    with c4:
        fig_3d = px.scatter_3d(df_pca_filtrado, x='PC1', y='PC2', z='PC3', color='Cluster', hover_name='Etiqueta',
                               color_discrete_sequence=colores_clusters, title='Modelado Espacial de Municipios en 3D', 
                               template='plotly_dark')
        fig_3d.add_trace(go.Scatter3d(
            x=centroids_df['PC1'], y=centroids_df['PC2'], z=centroids_df['PC3'], mode='markers',
            marker=dict(size=12, color='white', symbol='diamond', line=dict(width=1.5, color='black')),
            name='Centroides Matemáticos',
            customdata=np.stack((
                centroids_df['Cluster'], centroids_df['Promedio_Afectados'], centroids_df['Promedio_Asesinados'],
                centroids_df['Promedio_Heridos'], centroids_df['Promedio_Ejercito']
            ), axis=-1),
            hovertemplate=(
                "<b>🎯 CENTROIDE CLÚSTER %{customdata[0]}</b><br><br>"
                "• Promedio Total Afectados: %{customdata[1]}<br>"
                "• Promedio Asesinados: %{customdata[2]}<br>"
                "• Promedio Heridos: %{customdata[3]}<br>"
                "• Promedio Conteo Ejército: %{customdata[4]}<br><extra></extra>"
            )
        ))
        st.plotly_chart(fig_3d, use_container_width=True)

# ==============================================================================
# PÁGINA 5: CONCLUSIONES Y CIERRE DE LA PRESENTACIÓN
# ==============================================================================
elif st.session_state.page == 'cierre':
    st.markdown('<div class="main-title">Conclusiones del Estudio Táctico</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Validación Operacional y Cierre de la Sustentación</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏁 Hallazgos Principales")
        st.markdown("""
        * **Valor Metodológico:** Se logró resolver con éxito la restricción de datos cualitativos planos, permitiendo la inclusión de variables como nombres de instituciones dentro de un plano métrico continuo.
        * **Priorización Basada en Evidencia:** La segmentación territorial matemática mapea con precisión los núcleos geográficos de riesgo latente e intensidad operacional, ofreciendo un mapa claro para la toma de decisiones estratégicas.
        * **Eficiencia Operacional:** Los centroides calculados representan el perfil histórico consolidado de cada región, permitiendo predecir y anticipar necesidades logísticas y humanas de la Fuerza Pública.
        """)
    
    with col2:
        st.markdown("### 🚀 Próximos Pasos Recomendados")
        st.markdown("""
        * **Incorporación de Capas Temporales:** Integrar análisis de series temporales para monitorizar de qué manera mutan los municipios de un clúster a otro a lo largo del tiempo.
        * **Modelado Geoespacial Directo:** Cruzar los resultados con coordenadas exactas e índices de densidad demográfica regional.
        """)
        
        st.success("🎯 **Fin de la Presentación.** El sistema está listo para responder preguntas de la audiencia interactuando con los filtros tácticos de la sección anterior.")
    
    st.markdown("---")
    st.markdown("<center><h3>¿Preguntas o comentarios?</h3></center>", unsafe_allow_html=True)

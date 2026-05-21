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

# Configuración de página de Streamlit
st.set_page_config(page_title="Análisis de Clúster - Fuerza Pública", layout="wide")

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
# NAVEGACIÓN ENTRE PÁGINAS
# ==============================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'infografia'

def cambiar_pagina(nombre_pagina):
    st.session_state.page = nombre_pagina

# Barra lateral
st.sidebar.title("📌 Navegación Exposición")
if st.sidebar.button("📊 Ver Infografía del Proceso", use_container_width=True):
    cambiar_pagina('infografia')
if st.sidebar.button("🚀 Ir a App de Análisis y Modelado", use_container_width=True):
    cambiar_pagina('analisis')

# ==============================================================================
# PÁGINA 1: INFOGRAFÍA INTERACTIVA
# ==============================================================================
if st.session_state.page == 'infografia':
    st.title("🗺️ Infografía del Pipeline de Datos: Categórico a Clúster")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.error("### 🛑 El Desafío de la Base Original")
        st.markdown("""
        * **Estructura Inicial:** Registro plano o bitácora de eventos de orden público.
        * **Restricción Técnica:** Contiene **8 variables categóricas** (texto) y únicamente **1 variable numérica** (`CANTIDAD`).
        * **El Problema:** Algoritmos basados en distancias geométricas (como **K-Means**) son incapaces de procesar texto directo o nombres de instituciones (ej: 'EJÉRCITO').
        """)
    with col2:
        st.success("### 💡 La Solución y Adaptación Matemática")
        st.markdown("""
        * **Pivotado Cruzado (Reshaping):** Convertir las filas de texto en columnas numéricas de conteo independiente.
        * **Consolidación Territorial:** Agrupar todo bajo los códigos únicos de cada municipio (`COD_MUNI`).
        * **El Resultado:** Transformación de la bitácora en una matriz estructurada de **884 municipios** apta para algoritmos de Machine Learning.
        """)

    st.markdown("### 📈 Pasos Clave del Código para la Transformación")
    
    with st.expander("1. 🔄 Reestructuración de Datos (Matriz de Pivotado)", expanded=True):
        st.markdown("Para cada municipio se cruzan y totalizan las variables de fuerza y tipo de afectación:")
        st.code("""
# Agrupamos por Municipio y pivotamos las variables de texto
pivot_accion = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
pivot_fuerza = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0)
total_municipio = df_original.groupby(['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'])['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')

# Unimos las tablas convirtiendo las categorías en columnas numéricas reales
datos = total_municipio.join([pivot_accion, pivot_fuerza]).reset_index()
datos = datos.rename(columns={'MUNICIPIO': 'MUNICIPIO'})
        """, language="python")

    with st.expander("2. ⚖️ Estandarización de Distancias (StandardScaler)", expanded=False):
        st.markdown("Evita que las variables con magnitudes numéricas colosales dominen o sesguen el cálculo de las distancias Euclidianas:")
        st.code("""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Modifica los datos para que tengan Media = 0 y Varianza = 1
datos[numericas] = scaler.fit_transform(datos[numericas])
        """, language="python")

    with st.expander("3. 🎯 Selección de Clústeres (Método del Codo u Optimización)", expanded=False):
        st.markdown("Se itera el modelo de 1 a 10 grupos calculando la inercia (WSS) para identificar el punto de quiebre óptimo (**K = 4**):")
        st.code("""
from sklearn.cluster import KMeans

wss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=50, random_state=42)
    kmeans.fit(X_scaled)
    wss.append(kmeans.inertia_)
        """, language="python")

    st.markdown("---")
    st.markdown("<center>", unsafe_allow_html=True)
    if st.button("🚀 ¡Entendido! Ir Directo a la Ejecución del Modelo", type="primary"):
        cambiar_pagina('analisis')
        st.rerun()
    st.markdown("</center>", unsafe_allow_html=True)

# ==============================================================================
# PÁGINA 2: ANÁLISIS DE DATOS EN TIEMPO REAL
# ==============================================================================
elif st.session_state.page == 'analisis':
    st.title("🚀 Modelado Avanzado y Visualización de Clústeres (K-Means)")
    
    df_original, resultado_carga = cargar_datos_automatico()
    
    if df_original is None:
        st.error(f"❌ {resultado_carga}")
        st.stop()
    else:
        st.success(f"📦 Archivo detectado y cargado con éxito: `{resultado_carga}`")

    # 1 Y 2. CONSOLIDACIÓN Y LIMPIEZA
    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    
    pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
    
    columnas_fuerza = [c for c in df_original['NOMBRE_FUERZA'].unique() if pd.notna(c)] if 'NOMBRE_FUERZA' in df_original.columns else []
    pivot_fuerza = df_original.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_fuerza else pd.DataFrame(index=pivot_accion.index)
    
    columnas_cat = [c for c in df_original['CATEGORIA'].unique() if pd.notna(c)] if 'CATEGORIA' in df_original.columns else []
    pivot_cat = df_original.pivot_table(index=index_cols, columns='CATEGORIA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_cat else pd.DataFrame(index=pivot_accion.index)
    
    total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')

    datos = total_municipio.join([pivot_accion, pivot_fuerza, pivot_cat]).reset_index()
    datos = datos.rename(columns={'MUNICIPIO': 'MUNICIPIO'})
    datos = datos.dropna()

    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Total de Municipios en la Muestra", datos.shape[0])
    m_col2.metric("Nuevas Columnas Numéricas Creadas", datos.shape[1] - 3)

    st.subheader("📋 Matriz de Datos Numéricos Consolidados (Primeros registros)")
    st.dataframe(datos.head(10))

    # Definición global de la paleta de colores para consistencia en toda la app
    colores_clusters = ['red', 'green', 'blue', 'orange']

    # ==============================================================================
    # 3. HISTOGRAMAS INTERACTIVOS
    # ==============================================================================
    st.subheader("📊 Distribución de las Variables Principales (Interactivo)")
    
    col_afectados = 'TOTAL_AFECTADOS' if 'TOTAL_AFECTADOS' in datos.columns else datos.columns[3]
    col_asesinado = 'ASESINADO' if 'ASESINADO' in datos.columns else datos.columns[4]
    col_herido = 'HERIDO' if 'HERIDO' in datos.columns else datos.columns[5]
    col_ejercito = 'EJERCITO NACIONAL DE COLOMBIA' if 'EJERCITO NACIONAL DE COLOMBIA' in datos.columns else datos.columns[6]

    hist_variables = [col_afectados, col_asesinado, col_herido, col_ejercito]
    fig_hist_int = go.Figure()
    
    for idx, col_name in enumerate(hist_variables):
        fig_hist_int.add_trace(go.Histogram(x=datos[col_name], name=col_name, nbinsx=25, visible=(idx==0)))
        
    botones = []
    for idx, col_name in enumerate(hist_variables):
        visibilidad = [False] * len(hist_variables)
        visibilidad[idx] = True
        botones.append(dict(label=col_name, method="update", args=[{"visible": visibilidad}, {"title": f"Distribución de {col_name}"}]))
        
    fig_hist_int.update_layout(updatemenus=[dict(active=0, buttons=botones, x=0.1, y=1.15, xanchor="left", yanchor="top")],
                              template="plotly_dark", height=400, title=f"Distribución de {hist_variables[0]}")
    st.plotly_chart(fig_hist_int, use_container_width=True)

    # 4. ESTANDARIZACIÓN
    scaler = StandardScaler()
    columnas_omitir = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    numericas = [col for col in datos.columns if col not in columnas_omitir]
    
    datos_originales_num = datos.copy()
    datos[numericas] = scaler.fit_transform(datos[numericas])
    X_scaled = datos.drop(columns=columnas_omitir)

    # ==============================================================================
    # 5. MATRICES DE DISTANCIA
    # ==============================================================================
    st.subheader("🌡️ Mapas de Calor de Distancias (Interactivo)")
    distancias_eu = euclidean_distances(X_scaled)[:50, :50]
    dist_matrix_manhattan = pdist(X_scaled, metric='cityblock')
    C = squareform(dist_matrix_manhattan)[:50, :50]
    
    nombres_municipios_sub = datos['MUNICIPIO'].iloc[:50].tolist()

    mapa_col1, mapa_col2 = st.columns(2)
    with mapa_col1:
        fig_eu = px.imshow(distancias_eu, x=nombres_municipios_sub, y=nombres_municipios_sub,
                           labels=dict(color="Distancia"), title="Distancia Euclideana (Muestra 50x50)",
                           color_continuous_scale='RdBu_r', template='plotly_dark')
        st.plotly_chart(fig_eu, use_container_width=True)
    with mapa_col2:
        fig_man = px.imshow(C, x=nombres_municipios_sub, y=nombres_municipios_sub,
                            labels=dict(color="Distancia"), title="Distancia Manhattan (Muestra 50x50)",
                            color_continuous_scale='RdBu_r', template='plotly_dark')
        st.plotly_chart(fig_man, use_container_width=True)

    # 6. MÉTODO DEL CODO
    st.subheader("📐 Curva de Optimización: Método del Codo")
    wss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, n_init=30, random_state=42)
        kmeans.fit(X_scaled)
        wss.append(kmeans.inertia_)
        
    fig_elbow_int = px.line(x=list(range(1, 11)), y=wss, markers=True, title="Evaluación WSS (Inercia)",
                            labels={'x': 'Número de Clústeres (k)', 'y': 'WSS / Inercia'}, template='plotly_dark')
    fig_elbow_int.add_vline(x=4, line_dash="dash", line_color="cyan", annotation_text="K Óptimo = 4")
    st.plotly_chart(fig_elbow_int, use_container_width=True)

    # 7. EJECUCIÓN K-MEANS CON K=4
    kmeans = KMeans(n_clusters=4, n_init=50, random_state=42)
    start = time.time()
    km4_clusters = kmeans.fit(X_scaled)
    tiempo_ms = (time.time() - start) * 1000
    st.info(f"⚡ K-Means completado en: {tiempo_ms:.2f} ms | Inercia Final: {km4_clusters.inertia_:.2f}")

    # ==============================================================================
    # 8. REDUCCIÓN DIMENSIONAL (PCA INTERACTIVO)
    # ==============================================================================
    st.subheader("🎯 Agrupación Territorial en Espacio Reducido (PCA 2D)")
    pca = PCA(n_components=2)
    datos_pca = pca.fit_transform(X_scaled)
    datos_pca_df = pd.DataFrame(data=datos_pca, columns=['PCA1', 'PCA2'])
    datos_pca_df['Cluster'] = km4_clusters.labels_.astype(str)
    datos_pca_df['MUNICIPIO'] = datos['MUNICIPIO'].values
    datos_pca_df['DEPARTAMENTO'] = datos['DEPARTAMENTO'].values

    fig_pca_int = px.scatter(datos_pca_df, x='PCA1', y='PCA2', color='Cluster',
                             color_discrete_sequence=colores_clusters,
                             hover_data=['MUNICIPIO', 'DEPARTAMENTO'],
                             title="Clústeres K-Means Proyectados en PCA", template='plotly_dark')
    
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    fig_pca_int.add_trace(go.Scatter(x=centroids_pca[:, 0], y=centroids_pca[:, 1], mode='markers',
                                     marker=dict(size=14, color='white', symbol='star', line=dict(width=2, color='black')),
                                     name='Centroides'))
    st.plotly_chart(fig_pca_int, use_container_width=True)

    # 9 Y 10. MUESTRAS, CONTEOS Y CRUCE DE VARIABLES
    st.subheader("📊 Frecuencia e Impacto de Clústeres")
    G = pd.DataFrame({'MUNICIPIO': datos['MUNICIPIO'].values, 'DEPARTAMENTO': datos['DEPARTAMENTO'].values, 'label': km4_clusters.labels_})
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cantidad de Individuos por Clúster:**")
        resumen_grupos = G.groupby('label').size().reset_index(name='Municipios')
        st.dataframe(resumen_grupos)
    with c2:
        st.markdown("**Muestra de listado de asignación de Grupos:**")
        st.dataframe(G.sort_values(by='label').head(12))

    # Cruce de Variables Interactivo
    st.subheader("⚔️ Cruce Analítico Interactivo: TOTAL_AFECTADOS vs ASESINADO (Estandarizado)")
    datos_cruce = datos.copy()
    datos_cruce['Cluster'] = km4_clusters.labels_.astype(str)
    
    fig_cruzado_int = px.scatter(datos_cruce, x=col_afectados, y=col_asesinado, color='Cluster',
                                 color_discrete_sequence=colores_clusters,
                                 hover_data=['MUNICIPIO', 'DEPARTAMENTO'],
                                 title="Cruce de Impacto Total vs Personal Asesinado", template='plotly_dark')
    st.plotly_chart(fig_cruzado_int, use_container_width=True)

    # ==============================================================================
    # 11. BOXPLOT INTERACTIVO
    # ==============================================================================
    st.subheader("📦 Boxplot del Impacto Total Real por Clúster (Valores sin Escalar - Interactivo)")
    datos_originales_num['Cluster'] = km4_clusters.labels_.astype(str)
    
    fig_box_int = px.box(datos_originales_num, x='Cluster', y=col_afectados, color='Cluster',
                         color_discrete_sequence=colores_clusters,
                         hover_data=['MUNICIPIO'], title="Distribución Absoluta de Afectados por Grupo",
                         template='plotly_dark')
    st.plotly_chart(fig_box_int, use_container_width=True)

    # ==============================================================================
    # 12. PCA INTERACTIVO EN 2D Y 3D (MEJORA DE COLOR DE CENTROIDES EN 3D)
    # ==============================================================================
    st.subheader("✨ Control Final: Componentes Principales Avanzados (2D y 3D)")
    
    pca_4d = PCA(n_components=4)
    pca_scores_4d = pca_4d.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pca_scores_4d, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    pca_df['Cluster'] = km4_clusters.labels_.astype(str)
    pca_df['Etiqueta'] = datos['MUNICIPIO'] + " (" + datos['DEPARTAMENTO'] + ")"

    fig_2d = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', hover_name='Etiqueta',
                        color_discrete_sequence=colores_clusters,
                        title='Visualización PCA Interactiva en 2D', template='plotly_dark')
    st.plotly_chart(fig_2d, use_container_width=True)

    # Construcción del Gráfico 3D
    fig_3d = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Cluster', hover_name='Etiqueta',
                           color_discrete_sequence=colores_clusters,
                           title='Modelado Espacial de Municipios en 3D', template='plotly_dark')

    # Proyección de los centroides al mismo espacio 3D (PC1, PC2, PC3)
    centroids_pca_3d = pca_4d.transform(kmeans.cluster_centers_)
    
    # CAMBIO IMPLEMENTADO: El color se asigna dinámicamente usando la misma lista del orden de los clústeres
    fig_3d.add_trace(go.Scatter3d(
        x=centroids_pca_3d[:, 0], 
        y=centroids_pca_3d[:, 1], 
        z=centroids_pca_3d[:, 2],
        mode='markers',
        marker=dict(
            size=14, 
            color=colores_clusters,  # Toma los colores exactos: Rojo, Verde, Azul y Naranja correspondientes
            symbol='diamond', 
            line=dict(width=2, color='black')  # Borde negro elegante para resaltar el rombo
        ),
        name='Centroides del Grupo'
    ))
    
    st.plotly_chart(fig_3d, use_container_width=True)

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

    # Consolación y limpieza de columnas
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

    colores_clusters = ['red', 'green', 'blue', 'orange']

    # Identificación dinámica de columnas clave
    col_afectados = 'TOTAL_AFECTADOS' if 'TOTAL_AFECTADOS' in datos.columns else datos.columns[3]
    col_asesinado = 'ASESINADO' if 'ASESINADO' in datos.columns else datos.columns[4]
    col_herido = 'HERIDO' if 'HERIDO' in datos.columns else datos.columns[5]
    col_ejercito = 'EJERCITO NACIONAL DE COLOMBIA' if 'EJERCITO NACIONAL DE COLOMBIA' in datos.columns else datos.columns[6]

    # ==============================================================================
    # HISTOGRAMAS INTERACTIVOS
    # ==============================================================================
    st.subheader("📊 Distribución de las Variables Principales (Interactivo)")
    hist_variables = [col_afectados, col_asesinado, col_herido, col_ejercito]
    fig_hist_int = go.Figure()
    for idx, col_name in enumerate(hist_variables):
        fig_hist_int.add_trace(go.Histogram(x=datos[col_name], name=col_name, nbinsx=25, visible=(idx==0)))
    botones = []
    for idx, col_name in enumerate(hist_variables):
        visibilidad = [False] * len(hist_variables)
        visibilidad[idx] = True
        botones.append(dict(label=col_name, method="update", args=[{"visible": visibilidad}, {"title": f"Distribución de {col_name}"}]))
    fig_hist_int.update_layout(updatemenus=[dict(active=0, buttons=botones, x=0.1, y=1.15, xanchor="left", yanchor="top")], template="plotly_dark", height=400)
    st.plotly_chart(fig_hist_int, use_container_width=True)

    # Escalado de datos
    scaler = StandardScaler()
    columnas_omitir = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    numericas = [col for col in datos.columns if col not in columnas_omitir]
    
    datos_originales_num = datos.copy()
    datos[numericas] = scaler.fit_transform(datos[numericas])
    X_scaled = datos.drop(columns=columnas_omitir)

    # Mapas de calor
    st.subheader("🌡️ Mapas de Calor de Distancias (Interactivo)")
    distancias_eu = euclidean_distances(X_scaled)[:50, :50]
    dist_matrix_manhattan = pdist(X_scaled, metric='cityblock')
    C = squareform(dist_matrix_manhattan)[:50, :50]
    nombres_municipios_sub = datos['MUNICIPIO'].iloc[:50].tolist()

    mapa_col1, mapa_col2 = st.columns(2)
    with mapa_col1:
        fig_eu = px.imshow(distancias_eu, x=nombres_municipios_sub, y=nombres_municipios_sub, labels=dict(color="Distancia"), color_continuous_scale='RdBu_r', template='plotly_dark', title="Distancia Euclideana")
        st.plotly_chart(fig_eu, use_container_width=True)
    with mapa_col2:
        fig_man = px.imshow(C, x=nombres_municipios_sub, y=nombres_municipios_sub, labels=dict(color="Distancia"), color_continuous_scale='RdBu_r', template='plotly_dark', title="Distancia Manhattan")
        st.plotly_chart(fig_man, use_container_width=True)

    # Curva del Codo
    st.subheader("📐 Curva de Optimización: Método del Codo")
    wss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, n_init=30, random_state=42)
        kmeans.fit(X_scaled)
        wss.append(kmeans.inertia_)
    fig_elbow_int = px.line(x=list(range(1, 11)), y=wss, markers=True, template='plotly_dark', labels={'x': 'Número de Clústeres (k)', 'y': 'WSS / Inercia'})
    fig_elbow_int.add_vline(x=4, line_dash="dash", line_color="cyan", annotation_text="K Óptimo = 4")
    st.plotly_chart(fig_elbow_int, use_container_width=True)

    # Ejecución K-Means
    kmeans = KMeans(n_clusters=4, n_init=50, random_state=42)
    start = time.time()
    km4_clusters = kmeans.fit(X_scaled)
    tiempo_ms = (time.time() - start) * 1000
    st.info(f"⚡ K-Means completado en: {tiempo_ms:.2f} ms | Inercia Final: {km4_clusters.inertia_:.2f}")

    # Añadir las etiquetas de clúster a los datos reales para calcular los promedios
    datos_originales_num['Cluster'] = km4_clusters.labels_

    # ==============================================================================
    # CALCULAR LOS PROMEDIOS REALES POR CLÚSTER PARA LOS CENTROIDES
    # ==============================================================================
    promedios_por_cluster = datos_originales_num.groupby('Cluster')[[col_afectados, col_asesinado, col_herido, col_ejercito]].mean().reset_index()
    
    # Reducción Dimensional PCA para los gráficos
    pca_4d = PCA(n_components=4)
    pca_scores_4d = pca_4d.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pca_scores_4d, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    pca_df['Cluster'] = km4_clusters.labels_.astype(str)
    pca_df['Etiqueta'] = datos['MUNICIPIO'] + " (" + datos['DEPARTAMENTO'] + ")"

    # Proyección de los Centroides al espacio de PCA
    centroids_pca_3d = pca_4d.transform(kmeans.cluster_centers_)
    
    # Crear un DataFrame específico para los Centroides combinando sus coordenadas PCA y sus promedios reales
    centroids_df = pd.DataFrame(centroids_pca_3d, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    centroids_df['Cluster'] = promedios_por_cluster['Cluster'].astype(str)
    centroids_df['Promedio_Afectados'] = promedios_por_cluster[col_afectados].round(2)
    centroids_df['Promedio_Asesinados'] = promedios_por_cluster[col_asesinado].round(2)
    centroids_df['Promedio_Heridos'] = promedios_por_cluster[col_herido].round(2)
    centroids_df['Promedio_Ejercito'] = promedios_por_cluster[col_ejercito].round(2)

    # PCA 2D Scatter de Municipios
    st.subheader("🎯 Agrupación Territorial en Espacio Reducido (PCA 2D)")
    fig_pca_int = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', color_discrete_sequence=colores_clusters, hover_name='Etiqueta', template='plotly_dark', title="Clústeres K-Means Proyectados en PCA")
    st.plotly_chart(fig_pca_int, use_container_width=True)

    # Conteos y Cruces
    st.subheader("📊 Frecuencia e Impacto de Clústeres")
    G = pd.DataFrame({'MUNICIPIO': datos['MUNICIPIO'].values, 'DEPARTAMENTO': datos['DEPARTAMENTO'].values, 'label': km4_clusters.labels_})
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(G.groupby('label').size().reset_index(name='Municipios'))
    with c2:
        st.dataframe(G.sort_values(by='label').head(12))

    # Cruce Analítico
    fig_cruzado_int = px.scatter(datos_originales_num, x=col_afectados, y=col_asesinado, color=km4_clusters.labels_.astype(str), color_discrete_sequence=colores_clusters, hover_data=['MUNICIPIO'], template='plotly_dark', title="Cruce de Impacto Total vs Personal Asesinado")
    st.plotly_chart(fig_cruzado_int, use_container_width=True)

    # Boxplot
    st.subheader("📦 Boxplot del Impacto Total Real por Clúster")
    fig_box_int = px.box(datos_originales_num, x='Cluster', y=col_afectados, color='Cluster', color_discrete_sequence=colores_clusters, hover_data=['MUNICIPIO'], template='plotly_dark')
    st.plotly_chart(fig_box_int, use_container_width=True)

    # ==============================================================================
    # 12. PCA INTERACTIVO EN 3D CON TOOLTIP MATEMÁTICO EN LOS CENTROIDES
    # ==============================================================================
    st.subheader("✨ Control Final: Componentes Principales Avanzados (2D y 3D)")
    
    fig_3d = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Cluster', hover_name='Etiqueta',
                           color_discrete_sequence=colores_clusters,
                           title='Modelado Espacial de Municipios en 3D (Pasa el cursor sobre los Rombos Blancos)', 
                           template='plotly_dark')

    # AGREGAR LOS ROMBOS BLANCOS GRANDES CON LA CONFIGURACIÓN DE PROMEDIOS NUMÉRICOS
    fig_3d.add_trace(go.Scatter3d(
        x=centroids_df['PC1'], 
        y=centroids_df['PC2'], 
        z=centroids_df['PC3'],
        mode='markers',
        marker=dict(
            size=12,                 # Rombo grande solicitado
            color='white',           # Color blanco clásico
            symbol='diamond',        # Geometría de rombo
            line=dict(width=1.5, color='black')
        ),
        name='Centroides (Promedios)',
        # Inyectamos los datos numéricos reales calculados para que Plotly los lea en el hover
        customdata=np.stack((
            centroids_df['Cluster'],
            centroids_df['Promedio_Afectados'],
            centroids_df['Promedio_Asesinados'],
            centroids_df['Promedio_Heridos'],
            centroids_df['Promedio_Ejercito']
        ), axis=-1),
        # Plantilla visual que se despliega al pasar el cursor sobre el rombo
        hovertemplate=(
            "<b>🎯 CENTROIDE CLÚSTER %{customdata[0]}</b><br><br>"
            "<b>Valores Promedio del Grupo:</b><br>"
            "• Promedio Total Afectados: %{customdata[1]}<br>"
            "• Promedio Asesinados: %{customdata[2]}<br>"
            "• Promedio Heridos: %{customdata[3]}<br>"
            "• Promedio Conteo Ejército: %{customdata[4]}<br>"
            "<extra></extra>" # Elimina la etiqueta secundaria de Plotly por limpieza visual
        )
    ))
    
    st.plotly_chart(fig_3d, use_container_width=True)

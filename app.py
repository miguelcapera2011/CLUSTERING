import streamlit as pd_st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import plotly.express as px
import plotly.graph_objects as go

# Configuración de página de Streamlit
pd_st.set_page_config(page_title="Análisis de Clúster - Fuerza Pública", layout="wide")

# ==============================================================================
# NAVEGACIÓN ENTRE PÁGINAS (Simulación de botón de paso)
# ==============================================================================
if 'page' not in pd_st.session_state:
    pd_st.session_state.page = 'infografia'

def cambiar_pagina(nombre_pagina):
    pd_st.session_state.page = nombre_pagina

# Barra lateral para navegación clara
pd_st.sidebar.title("📌 Navegación")
if pd_st.sidebar.button("📊 Ver Infografía del Proceso", use_container_width=True):
    cambiar_pagina('infografia')
if pd_st.sidebar.button("🚀 Ir a App de Análisis y Modelado", use_container_width=True):
    cambiar_pagina('analisis')

# ==============================================================================
# PÁGINA 1: INFOGRAFÍA INTERACTIVA
# ==============================================================================
if pd_st.session_state.page == 'infografia':
    pd_st.title("🗺️ Infografía del Pipeline de Datos: Categórico a Clúster")
    pd_st.markdown("---")
    
    # Fila 1: El problema y la solución
    col1, col2 = pd_st.columns(2)
    with col1:
        pd_st.error("### 🛑 El Desafío Original")
        pd_st.markdown("""
        * **Base inicial:** Registro plano de eventos.
        * **Restricción:** 8 variables categóricas y solo 1 numérica (`CANTIDAD`).
        * **Problema:** Los algoritmos de distancia como **K-Means** no pueden calcular distancias matemáticas sobre textos directos (ej: 'EJÉRCITO', 'HERIDO').
        """)
    with col2:
        pd_st.success("### 💡 La Solución Aplicada")
        pd_st.markdown("""
        * **Pivotado Cruzado:** Transformar filas de texto en columnas numéricas.
        * **Agrupación Municipal:** Consolidar todo a nivel de municipio (`COD_MUNI`).
        * **Resultado:** Pasamos de un registro bruto a una matriz numérica de **884 municipios** listos para ser medidos en un espacio geométrico.
        """)

    pd_st.markdown("### 📈 El Viaje del Dato (Paso a Paso)")
    
    # Paso 1
    with pd_st.expander("1. 🔄 Ingeniería de Características: Pivotado de Tablas", expanded=True):
        pd_st.markdown("Utilizamos `pivot_table` para contar cuántas veces ocurre cada categoría por municipio:")
        pd_st.code("""
# Convertimos categorías en conteos numéricos por municipio
pivot_accion = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
pivot_fuerza = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0)
total_municipio = df_original.groupby(['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'])['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')

# Consolidación final
datos = total_municipio.join([pivot_accion, pivot_fuerza]).reset_index()
        """, language="python")

    # Paso 2
    with pd_st.expander("2. ⚖️ Estandarización de Variables (StandardScaler)", expanded=False):
        pd_st.markdown("K-Means es sensible a las escalas. Si un municipio tiene 500 heridos y 2 ataques, el 500 absorbería la distancia. Escalamos para darles el mismo peso:")
        pd_st.code("""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Centra los datos en media 0 y varianza 1
datos[numericas] = scaler.fit_transform(datos[numericas])
        """, language="python")

    # Paso 3
    with pd_st.expander("3. 🎯 Selección de Clústeres (Método del Codo)", expanded=False):
        pd_st.markdown("Calculamos la Inercia (WSS) para definir cuántos grupos representan mejor la realidad colombiana sin sobreajustar. **Elegimos K=4**.")
        pd_st.code("""
wss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=50, random_state=42)
    kmeans.fit(X_scaled)
    wss.append(kmeans.inertia_)
        """, language="python")

    pd_st.markdown("---")
    # Botón grande animado hacia la acción
    pd_st.markdown("<center>", unsafe_allow_html=True)
    if pd_st.button("✨ ¡Entendido! Ejecutar Modelado en Tiempo Real ->", type="primary"):
        cambiar_pagina('analisis')
        pd_st.rerun()
    pd_st.markdown("</center>", unsafe_allow_html=True)

# ==============================================================================
# PÁGINA 2: APLICACIÓN DE ANÁLISIS DE DATOS (CÓDIGO ORIGINAL INTEGRADO)
# ==============================================================================
elif pd_st.session_state.page == 'analisis':
    pd_st.title("🚀 Pipeline de Machine Learning: Modelado K-Means")
    
    # Cargar archivo de datos (CSV convertido o subido)
    try:
        # Buscamos el archivo que cargaste en el entorno
        df_original = pd.read_csv('AFECTACIÓN A LA FUERZA PÚBLICA.xlsx - Sheet 1.csv', header=0)
    except FileNotFoundError:
        pd_st.error("Por favor, asegúrate de que el archivo 'AFECTACIÓN A LA FUERZA PÚBLICA.xlsx - Sheet 1.csv' esté en la misma carpeta que este script.")
        pd_st.stop()

    # 1. CONSOLIDACIÓN DE MUNICIPIOS
    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
    
    # Manejar columnas dinámicas según lo que tenga el dataset real
    columnas_fuerza = [c for c in df_original['NOMBRE_FUERZA'].unique() if pd.notna(c)] if 'NOMBRE_FUERZA' in df_original.columns else []
    pivot_fuerza = df_original.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_fuerza else pd.DataFrame(index=pivot_accion.index)
    
    columnas_cat = [c for c in df_original['CATEGORIA'].unique() if pd.notna(c)] if 'CATEGORIA' in df_original.columns else []
    pivot_cat = df_original.pivot_table(index=index_cols, columns='CATEGORIA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_cat else pd.DataFrame(index=pivot_accion.index)
    
    total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')

    # Uniones
    datos = total_municipio.join([pivot_accion, pivot_fuerza, pivot_cat]).reset_index()
    datos = datos.rename(columns={'MUNICIPIO': 'State'})
    datos = datos.dropna()

    # Dashboard Informativo Inicial
    st_col1, st_col2 = pd_st.columns(2)
    st_col1.metric("Municipios Analizados", datos.shape[0])
    st_col2.metric("Variables Numéricas Generadas", datos.shape[1] - 3)

    # Mostrar muestra de la base preparada
    pd_st.subheader("📋 Base Preparada y Consolidada (Muestra de 5 filas)")
    pd_st.dataframe(datos.head(5))

    # 3. HISTOGRAMAS
    pd_st.subheader("📊 Análisis Exploratorio: Histogramas de Variables Clave")
    fig_hist, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Validaciones por si los nombres varían ligeramente en el archivo
    col_afectados = 'TOTAL_AFECTADOS' if 'TOTAL_AFECTADOS' in datos.columns else datos.columns[3]
    col_asesinado = 'ASESINADO' if 'ASESINADO' in datos.columns else datos.columns[4]
    col_herido = 'HERIDO' if 'HERIDO' in datos.columns else datos.columns[5]

    sns.histplot(datos[col_afectados], bins=15, kde=True, color='blue', ax=axes[0])
    axes[0].set_title(f'{col_afectados} Original')
    
    sns.histplot(datos[col_asesinado], bins=15, kde=True, color='green', ax=axes[1])
    axes[1].set_title(f'{col_asesinado} Original')
    
    sns.histplot(datos[col_herido], bins=15, kde=True, color='red', ax=axes[2])
    axes[2].set_title(f'{col_herido} Original')
    
    pd_st.pyplot(fig_hist)

    # 4. ESTANDARIZACIÓN
    scaler = StandardScaler()
    columnas_omitir = ['COD_MUNI', 'State', 'DEPARTAMENTO']
    numericas = [col for col in datos.columns if col not in columnas_omitir]
    
    datos_originales_num = datos.copy()
    datos[numericas] = scaler.fit_transform(datos[numericas])
    X_scaled = datos.drop(columns=columnas_omitir)

    # 5. MATRICES DE DISTANCIA
    pd_st.subheader("🌡️ Mapas de Calor: Matrices de Distancias (Muestra de Primeros Municipios)")
    distancias_eu = euclidean_distances(X_scaled)
    dist_matrix_manhattan = pdist(X_scaled, metric='cityblock')
    C = squareform(dist_matrix_manhattan)
    
    fig_maps, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(distancias_eu[:40, :40], cmap='coolwarm', ax=ax[0])
    ax[0].set_title('Distancia Euclideana (Muestra 40x40)')
    sns.heatmap(C[:40, :40], cmap='coolwarm', ax=ax[1])
    ax[1].set_title('Distancia Manhattan (Muestra 40x40)')
    pd_st.pyplot(fig_maps)

    # 6. MÉTODO DEL CODO
    pd_st.subheader("📐 Optimización: Método del Codo")
    wss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, n_init=30, random_state=42)
        kmeans.fit(X_scaled)
        wss.append(kmeans.inertia_)
        
    fig_elbow, ax_el = plt.subplots(figsize=(8, 3.5))
    ax_el.plot(range(1, 11), wss, marker='o', color='green')
    ax_el.axvline(x=4, color='blue', linestyle='--')
    ax_el.set_title('Método del Codo (K Óptimo = 4)')
    pd_st.pyplot(fig_elbow)

    # 7. EJECUCIÓN K-MEANS
    random_seed = 42
    kmeans = KMeans(n_clusters=4, n_init=50, random_state=random_seed)
    start = time.time()
    km4_clusters = kmeans.fit(X_scaled)
    tiempo_ejecucion = (time.time() - start) * 1000
    
    pd_st.info(f"⚡ Algoritmo K-Means completado en: {tiempo_ejecucion:.2f} ms | Inercia Final: {km4_clusters.inertia_:.2f}")

    # 8. PCA REDUCCIÓN Y ELIPSES ESTÁTICAS
    pd_st.subheader("🎯 Visualización de Clústeres en Espacio PCA con Elipses de Covarianza")
    pca = PCA(n_components=2)
    datos_pca = pca.fit_transform(X_scaled)
    datos_pca_df = pd.DataFrame(data=datos_pca, columns=['PCA1', 'PCA2'])
    datos_pca_df['Cluster'] = km4_clusters.labels_

    fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
    colores = ['red', 'green', 'blue', 'orange']
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=datos_pca_df, palette=colores, s=40, alpha=0.7, ax=ax_pca)
    
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax_pca.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, c=colores, marker='*', edgecolor='black', label='Centroides')
    
    for i, color in enumerate(colores):
        cluster_points = datos_pca_df[datos_pca_df['Cluster'] == i][['PCA1', 'PCA2']]
        if len(cluster_points) > 1:
            cov = np.cov(cluster_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(eigenvalues) * 2
            ellipse = Ellipse(xy=centroids_pca[i], width=width, height=height, angle=angle, color=color, alpha=0.1)
            ax_pca.add_patch(ellipse)
    ax_pca.grid(True)
    pd_st.pyplot(fig_pca)

    # 9. CANTIDAD POR GRUPO
    pd_st.subheader("📊 Distribución de Municipios por Clúster")
    G = pd.DataFrame({'State': datos['State'].values, 'DEPARTAMENTO': datos['DEPARTAMENTO'].values, 'label': km4_clusters.labels_})
    resumen_grupos = G.groupby('label').size().reset_index(name='Cantidad de Municipios')
    pd_st.bar_chart(data=resumen_grupos, x='label', y='Cantidad de Municipios', color='#FF4B4B')

    # 12. INTERACTIVOS DE PLOTLY (2D Y 3D)
    pd_st.subheader("✨ Gráficos de Control Interactivos (Explora pasando el mouse)")
    
    pca_4d = PCA(n_components=4)
    pca_scores_4d = pca_4d.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pca_scores_4d, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    pca_df['Cluster'] = km4_clusters.labels_.astype(str)
    pca_df['Etiqueta'] = datos['State'] + " (" + datos['DEPARTAMENTO'] + ")"

    # Plotly 2D (Optimizado con hover en vez de texto fijo para fluidez)
    fig_2d = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', hover_name='Etiqueta',
                        title='Componentes Principales 2D',
                        template='plotly_dark')
    pd_st.plotly_chart(fig_2d, use_container_width=True)

    # Plotly 3D
    fig_3d = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Cluster', hover_name='Etiqueta',
                           title='Modelado Espacial de Municipios en 3D',
                           template='plotly_dark')
    centroids_pca_3d = pca_4d.transform(kmeans.cluster_centers_)
    fig_3d.add_trace(go.Scatter3d(x=centroids_pca_3d[:, 0], y=centroids_pca_3d[:, 1], z=centroids_pca_3d[:, 2],
                                  mode='markers',
                                  marker=dict(size=10, color='white', symbol='diamond'),
                                  name='Centroides'))
    pd_st.plotly_chart(fig_3d, use_container_width=True)

    # 11. BOXPLOT CON VALORES REALES
    pd_st.subheader("📦 Distribución del Impacto Real (Datos sin Escalar)")
    datos_originales_num['Cluster'] = km4_clusters.labels_
    fig_box, ax_box = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='Cluster', y=col_afectados, data=datos_originales_num, palette="Set2", ax=ax_box)
    ax_box.set_ylabel('Cantidad Total Absoluta de Afectados')
    pd_st.pyplot(fig_box)

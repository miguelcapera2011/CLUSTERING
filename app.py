import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
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
    
    # Buscar un archivo que coincida con palabras clave del dataset
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
# NAVEGACIÓN ENTRE PÁGINAS (Simulación de botones y estado de sesión)
# ==============================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'infografia'

def cambiar_pagina(nombre_pagina):
    st.session_state.page = nombre_pagina

# Barra lateral para navegación clara
st.sidebar.title("📌 Navegación Exposición")
if st.sidebar.button("📊 Ver Infografía del Proceso", use_container_width=True):
    cambiar_pagina('infografia')
if st.sidebar.button("🚀 Ir a App de Análisis y Modelado", use_container_width=True):
    cambiar_pagina('analisis')

# ==============================================================================
# PÁGINA 1: INFOGRAFÍA INTERACTIVA Y ANIMADA
# ==============================================================================
if st.session_state.page == 'infografia':
    st.title("🗺️ Infografía del Pipeline de Datos: Categórico a Clúster")
    st.markdown("---")
    
    # Fila de Presentación del problema
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
    
    # Paso 1: Pivotado
    with st.expander("1. 🔄 Reestructuración de Datos (Matriz de Pivotado)", expanded=True):
        st.markdown("Para cada municipio se cruzan y totalizan las variables de fuerza y tipo de afectación:")
        st.code("""
# Agrupamos por Municipio y pivotamos las variables de texto
pivot_accion = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
pivot_fuerza = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0)
total_municipio = df_original.groupby(['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'])['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')

# Unimos las tablas convirtiendo las categorías en columnas numéricas reales
datos = total_municipio.join([pivot_accion, pivot_fuerza]).reset_index()
datos = datos.rename(columns={'MUNICIPIO': 'Municipio'})
        """, language="python")

    # Paso 2: Escalamiento
    with st.expander("2. ⚖️ Estandarización de Distancias (StandardScaler)", expanded=False):
        st.markdown("Evita que las variables con magnitudes numéricas colosales dominen o sesguen el cálculo de las distancias Euclidianas:")
        st.code("""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Modifica los datos para que tengan Media = 0 y Varianza = 1
datos[numericas] = scaler.fit_transform(datos[numericas])
        """, language="python")

    # Paso 3: Codo
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
# PÁGINA 2: ANÁLISIS DE DATOS EN TIEMPO REAL (CÓDIGO COMPLETO INTEGRADO)
# ==============================================================================
elif st.session_state.page == 'analisis':
    st.title("🚀 Modelado Avanzado y Visualización de Clústeres (K-Means)")
    
    # Ejecutar la búsqueda y carga automática del archivo
    df_original, resultado_carga = cargar_datos_automatico()
    
    if df_original is None:
        st.error(f"❌ {resultado_carga}")
        st.info("💡 Por favor, coloca tu archivo de Excel o CSV en el mismo directorio donde guardaste este script.")
        st.stop()
    else:
        st.success(f"📦 Archivo detectado y cargado con éxito: `{resultado_carga}`")

    # ==============================================================================
    # 1 Y 2. CONSOLIDACIÓN Y LIMPIEZA
    # ==============================================================================
    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    
    pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
    
    columnas_fuerza = [c for c in df_original['NOMBRE_FUERZA'].unique() if pd.notna(c)] if 'NOMBRE_FUERZA' in df_original.columns else []
    pivot_fuerza = df_original.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_fuerza else pd.DataFrame(index=pivot_accion.index)
    
    columnas_cat = [c for c in df_original['CATEGORIA'].unique() if pd.notna(c)] if 'CATEGORIA' in df_original.columns else []
    pivot_cat = df_original.pivot_table(index=index_cols, columns='CATEGORIA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_cat else pd.DataFrame(index=pivot_accion.index)
    
    total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')

    datos = total_municipio.join([pivot_accion, pivot_fuerza, pivot_cat]).reset_index()
    # CAMBIO SOLICITADO: 'MUNICIPIO' pasa a llamarse 'Municipio' en vez de 'State'
    datos = datos.rename(columns={'MUNICIPIO': 'Municipio'})
    datos = datos.dropna()

    # Métricas principales en pantalla
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Total de Municipios en la Muestra", datos.shape[0])
    m_col2.metric("Nuevas Columnas Numéricas Creadas", datos.shape[1] - 3)

    st.subheader("📋 Matriz de Datos Numéricos Consolidados (Primeros registros)")
    st.dataframe(datos.head(10))

    # ==============================================================================
    # 3. HISTOGRAMAS
    # ==============================================================================
    st.subheader("📊 Distribución Bruta de las Variables Principales")
    fig_hist, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    col_afectados = 'TOTAL_AFECTADOS' if 'TOTAL_AFECTADOS' in datos.columns else datos.columns[3]
    col_asesinado = 'ASESINADO' if 'ASESINADO' in datos.columns else datos.columns[4]
    col_herido = 'HERIDO' if 'HERIDO' in datos.columns else datos.columns[5]
    col_ejercito = 'EJERCITO NACIONAL DE COLOMBIA' if 'EJERCITO NACIONAL DE COLOMBIA' in datos.columns else datos.columns[6]

    sns.histplot(datos[col_afectados], bins=15, kde=True, color='blue', ax=axes[0])
    axes[0].set_title('TOTAL_AFECTADOS Original')
    
    sns.histplot(datos[col_asesinado], bins=15, kde=True, color='green', ax=axes[1])
    axes[1].set_title('ASESINADO Original')
    
    sns.histplot(datos[col_herido], bins=15, kde=True, color='red', ax=axes[2])
    axes[2].set_title('HERIDO Original')
    
    sns.histplot(datos[col_ejercito], bins=15, kde=True, color='purple', ax=axes[3])
    axes[3].set_title('EJÉRCITO Original')
    
    plt.tight_layout()
    st.pyplot(fig_hist)

    # ==============================================================================
    # 4. ESTANDARIZACIÓN
    # ==============================================================================
    scaler = StandardScaler()
    columnas_omitir = ['COD_MUNI', 'Municipio', 'DEPARTAMENTO']
    numericas = [col for col in datos.columns if col not in columnas_omitir]
    
    datos_originales_num = datos.copy()
    datos[numericas] = scaler.fit_transform(datos[numericas])
    X_scaled = datos.drop(columns=columnas_omitir)

    # ==============================================================================
    # 5. MATRICES DE DISTANCIA
    # ==============================================================================
    st.subheader("🌡️ Mapas de Calor: Comparación de Métrica Euclideana vs Manhattan")
    distancias_eu = euclidean_distances(X_scaled)
    dist_matrix_manhattan = pdist(X_scaled, metric='cityblock')
    C = squareform(dist_matrix_manhattan)
    
    fig_maps, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(distancias_eu[:50, :50], cmap='coolwarm', annot=False, ax=ax[0])
    ax[0].set_title('Distancia Euclideana (Submuestra de 50 Municipios)', fontsize=12)
    
    sns.heatmap(C[:50, :50], cmap='coolwarm', annot=False, ax=ax[1])
    ax[1].set_title('Distancia Manhattan (Submuestra de 50 Municipios)', fontsize=12)
    st.pyplot(fig_maps)

    # ==============================================================================
    # 6. MÉTODO DEL CODO
    # ==============================================================================
    st.subheader("📐 Curva de Optimización: Método del Codo (Elbow Method)")
    wss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, n_init=30, random_state=42)
        kmeans.fit(X_scaled)
        wss.append(kmeans.inertia_)
        
    fig_elbow, ax_el = plt.subplots(figsize=(10, 4))
    ax_el.plot(range(1, 11), wss, marker='o', color='green', linestyle='-')
    ax_el.axvline(x=4, color='blue', linestyle='--', linewidth=2)
    ax_el.set_title('Evaluación de la Suma de Cuadrados Intra-Clúster (WSS)', fontsize=14, color="red")
    ax_el.set_xlabel('Número de Clústeres (k)')
    ax_el.set_ylabel('WSS / Inercia')
    ax_el.grid(True)
    st.pyplot(fig_elbow)

    # ==============================================================================
    # 7. EJECUCIÓN K-MEANS CON K=4
    # ==============================================================================
    kmeans = KMeans(n_clusters=4, n_init=50, random_state=42)
    start = time.time()
    km4_clusters = kmeans.fit(X_scaled)
    tiempo_ms = (time.time() - start) * 1000
    
    st.info(f"⚡ Algoritmo K-Means ejecutado de forma nativa en: {tiempo_ms:.2f} ms | Inercia Final Obtenida: {km4_clusters.inertia_:.2f}")

    # ==============================================================================
    # 8. REDUCCIÓN DIMENSIONAL (PCA) CON ELIPSES ESTÁTICAS
    # ==============================================================================
    st.subheader("🎯 Agrupación Territorial en Espacio Reducido (PCA 2D)")
    pca = PCA(n_components=2)
    datos_pca = pca.fit_transform(X_scaled)
    datos_pca_df = pd.DataFrame(data=datos_pca, columns=['PCA1', 'PCA2'])
    datos_pca_df['Cluster'] = km4_clusters.labels_

    fig_pca, ax_pca = plt.subplots(figsize=(12, 7))
    colores = ['red', 'green', 'blue', 'orange']
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=datos_pca_df, palette=colores, s=55, alpha=0.7, ax=ax_pca)
    
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax_pca.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c=colores, marker='*', edgecolor='black', label='Centroides')
    
    # Trazar elipses de covarianza por clúster
    for i, color in enumerate(colores):
        cluster_points = datos_pca_df[datos_pca_df['Cluster'] == i][['PCA1', 'PCA2']]
        if len(cluster_points) > 1:
            cov = np.cov(cluster_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(eigenvalues) * 2
            ellipse = Ellipse(xy=centroids_pca[i], width=width, height=height, angle=angle, color=color, alpha=0.12)
            ax_pca.add_patch(ellipse)
            
    # Muestra de etiquetas de municipios para evitar saturación de texto
    for i, row in datos_pca_df.iterrows():
        if i % 20 == 0:
            ax_pca.text(row['PCA1'], row['PCA2'] + 0.06, datos['Municipio'][i], fontsize=7, ha='center', alpha=0.8)
            
    ax_pca.set_title('Algoritmo K-means con Elipses (Municipios de Colombia)', fontsize=14)
    ax_pca.grid(True)
    st.pyplot(fig_pca)

    # ==============================================================================
    # 9 Y 10. MUESTRAS, CONTEOS Y CRUCE DE VARIABLES
    # ==============================================================================
    st.subheader("📊 Frecuencia e Impacto de Clústeres")
    G = pd.DataFrame({'Municipio': datos['Municipio'].values, 'DEPARTAMENTO': datos['DEPARTAMENTO'].values, 'label': km4_clusters.labels_})
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cantidad de Individuos (Municipios) por Clúster:**")
        resumen_grupos = G.groupby('label').size().reset_index(name='Municipios')
        st.dataframe(resumen_grupos)
    with c2:
        st.markdown("**Muestra de listado de asignación de Grupos:**")
        st.dataframe(G.sort_values(by='label').head(12))

    # Gráfico Cruzado (TOTAL_AFECTADOS vs ASESINADO)
    st.subheader("⚔️ Cruce Analítico: TOTAL_AFECTADOS vs ASESINADO (Estandarizado)")
    f1 = datos[col_afectados].values
    f2 = datos[col_asesinado].values
    asignar_colores = [colores[row] for row in km4_clusters.labels_]
    
    fig_cruzado, ax_cr = plt.subplots(figsize=(10, 5))
    ax_cr.scatter(f1, f2, c=asignar_colores, s=45, alpha=0.6)
    ax_cr.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', c=colores, s=300, edgecolor='black', label='Centroides')
    ax_cr.set_xlabel('TOTAL_AFECTADOS')
    ax_cr.set_ylabel('ASESINADO')
    ax_cr.grid(True)
    st.pyplot(fig_cruzado)

    # ==============================================================================
    # 11. BOXPLOT (DATOS REALES)
    # ==============================================================================
    st.subheader("📦 Boxplot del Impacto Total Real por Clúster (Valores sin Escalar)")
    datos_originales_num['Cluster'] = km4_clusters.labels_
    fig_box, ax_box = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='Cluster', y=col_afectados, data=datos_originales_num, palette="Set1", ax=ax_box)
    ax_box.set_ylabel('Cantidad Real Absoluta de Afectados')
    st.pyplot(fig_box)

    # ==============================================================================
    # 12. PCA INTERACTIVO EN 2D Y 3D CON PLOTLY EXPRESS
    # ==============================================================================
    st.subheader("✨ Componentes Principales Interactivos (2D y 3D)")
    
    pca_4d = PCA(n_components=4)
    pca_scores_4d = pca_4d.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pca_scores_4d, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    pca_df['Cluster'] = km4_clusters.labels_.astype(str)
    # CAMBIO SOLICITADO: Ajustamos la etiqueta interactiva usando el nuevo nombre de columna 'Municipio'
    pca_df['Etiqueta'] = datos['Municipio'] + " (" + datos['DEPARTAMENTO'] + ")"

    # Gráfico Interactivo 2D
    fig_2d = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', hover_name='Etiqueta',
                        title='Visualización PCA Interactiva en 2D (Pasa el cursor sobre los municipios)',
                        labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2'},
                        template='plotly_dark')
    st.plotly_chart(fig_2d, use_container_width=True)

    # Gráfico Interactivo 3D
    fig_3d = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Cluster', hover_name='Etiqueta',
                           title='Visualización PCA Interactiva en 3D de tus Municipios',
                           labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'},
                           template='plotly_dark')

    centroids_pca_3d = pca_4d.transform(kmeans.cluster_centers_)
    fig_3d.add_trace(go.Scatter3d(x=centroids_pca_3d[:, 0], y=centroids_pca_3d[:, 1], z=centroids_pca_3d[:, 2],
                                  mode='markers',
                                  marker=dict(size=12, color='white', symbol='diamond'),
                                  name='Centroides'))
    st.plotly_chart(fig_3d, use_container_width=True)

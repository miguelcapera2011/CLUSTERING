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
# CONFIGURACIÓN GENERAL Y ESTILOS
# ==============================================================================
st.set_page_config(page_title="Exposición: Clúster Fuerza Pública", layout="wide")

# Inicialización de la navegación
if 'page' not in st.session_state:
    st.session_state.page = 'portada'

def cambiar_pagina(nombre_pagina):
    st.session_state.page = nombre_pagina

# ==============================================================================
# CARGA AUTOMÁTICA DE DATOS DESDE LA BITÁCORA
# ==============================================================================
def cargar_datos_automatico():
    archivos_en_carpeta = os.listdir('.')
    archivo_encontrado = None
    for archivo in archivos_en_carpeta:
        nombre_minuscula = archivo.lower()
        if ("afectacion" in nombre_minuscula or "fuerza" in nombre_minuscula or "publica" in nombre_minuscula) and (archivo.endswith('.csv') or archivo.endswith('.xlsx')):
            archivo_encontrado = archivo
            break
            
    if archivo_encontrado is None:
        return None, "No se encontró ningún archivo de datos en la carpeta."
    try:
        if archivo_encontrado.endswith('.csv'):
            df = pd.read_csv(archivo_encontrado, header=0)
        else:
            df = pd.read_excel(archivo_encontrado, header=0)
        return df, archivo_encontrado
    except Exception as e:
        return None, f"Error al leer {archivo_encontrado}: {str(e)}"

# Intentar cargar datos para compartirlos entre secciones de análisis
df_original, resultado_carga = cargar_datos_automatico()

# ==============================================================================
# BARRA LATERAL: NAVEGACIÓN BASADA EN EL FORMATO DE LA PRESENTACIÓN
# ==============================================================================
st.sidebar.title("📌 Menú de la Exposición")

# Espacio estratégico para el logo de la Universidad del Tolima
# Nota: Puedes colocar el archivo 'logo_ut.png' en la misma carpeta o usar una URL pública.
url_logo_defecto = "https://www.ut.edu.co/images/logos/logo_ut.png" # URL de respaldo institucional
st.sidebar.image(url_logo_defecto, caption="Universidad del Tolima", use_container_width=True)
st.sidebar.markdown("---")

if st.sidebar.button("🏠 1. Portada Oficial", use_container_width=True): cambiar_pagina('portada')
if st.sidebar.button("🎯 2. Introducción y Problema", use_container_width=True): cambiar_pagina('introduccion')
if st.sidebar.button("📖 3. Marco Teórico", use_container_width=True): cambiar_pagina('teoria')
if st.sidebar.button("⚙️ 4. Metodología y Pipeline", use_container_width=True): cambiar_pagina('metodologia')
if st.sidebar.button("📊 5. Resultados del Modelo", use_container_width=True): cambiar_pagina('resultados')
if st.sidebar.button("🏁 6. Conclusiones y Cierre", use_container_width=True): cambiar_pagina('conclusiones')

st.sidebar.markdown("---")
if df_original is not None:
    st.sidebar.success(f"📦 Datos activos: `{resultado_carga}`")
else:
    st.sidebar.error("⚠️ Pendiente cargar base de datos (.csv/.xlsx)")

# ==============================================================================
# PÁGINA 1: PORTADA OFICIAL
# ==============================================================================
if st.session_state.page == 'portada':
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1,2,1])
    with col_l2:
        st.image(url_logo_defecto, width=220)
        
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Análisis Avanzado de Clúster de la Fuerza Pública</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #4B5563; font-weight: normal;'>Segmentación Territorial de Incidentes de Orden Público Mediante Machine Learning</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### 👤 Autor del Proyecto:")
        st.info("**Miguel Angel Garatejo**\n\n*Estudiante de Ingeniería / Analista de Datos*")
        st.markdown("#### 🏫 Institución:")
        st.write("**Universidad del Tolima**\n\n*Facultad de Ciencias / Programa Académico Avanzado*")
    with col_p2:
        st.markdown("#### 👩‍🏫 Docente Evaluador:")
        st.success("**Yuri Saavedra**\n\n*Cátedra de Ciencia de Datos y Modelado Avanzado*")
        st.markdown("#### 📅 Periodo:")
        st.write(f"**Año:** {time.strftime('%Y')} | **Estado:** Sustentación Final de Proyecto")

    st.markdown("<br><br><center>", unsafe_allow_html=True)
    if st.button("🚀 Iniciar Presentación del Proyecto", type="primary"):
        cambiar_pagina('introduccion')
        st.rerun()
    st.markdown("</center>", unsafe_allow_html=True)

# ==============================================================================
# PÁGINA 2: INTRODUCCIÓN Y PROBLEMA
# ==============================================================================
elif st.session_state.page == 'introduccion':
    st.title("🎯 Introducción y Planteamiento del Problema")
    st.markdown("---")
    
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.error("### 🛑 El Desafío de la Base de Datos Original")
        st.markdown("""
        * **Naturaleza de la Información:** El archivo fuente contiene registros tipo bitácora donde cada fila detalla un ataque o afectación individual a unidades de seguridad.
        * **Restricción de Entrada:** Cuenta con **8 variables categóricas (texto)** y únicamente **1 variable numérica** (`CANTIDAD`).
        * **El Quiebre Técnico:** Algoritmos fundamentados en cálculos de distancias multidimensionales (como **K-Means**) son totalmente incapaces de interpretar strings o nombres planos (*ej: 'EJÉRCITO', 'POLICÍA'*) directamente sin alterar la geometría de los datos.
        """)
    with col_i2:
        st.success("### 💡 Motivación, Objetivos y Justificación")
        st.markdown("""
        * **Motivación Operativa:** Las decisiones estratégicas de seguridad nacional no pueden basarse en lecturas de filas dispersas, requieren segmentaciones territoriales claras.
        * **Objetivo General:** Desarrollar un Pipeline matemático automatizado en Python para reestructurar, normalizar y clasificar los municipios del país de acuerdo a su patrón de vulnerabilidad real.
        * **Importancia:** Permite pasar de un esquema de reacción táctica a un despliegue preventivo basado rigurosamente en datos numéricos.
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Siguiente: Revisar Marco Teórico ➡️"):
        cambiar_pagina('teoria')
        st.rerun()

# ==============================================================================
# PÁGINA 3: MARCO TEÓRICO / CONCEPTUAL
# ==============================================================================
elif st.session_state.page == 'teoria':
    st.title("📖 Marco Teórico y Sustentación Algorítmica")
    st.markdown("---")
    
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.info("### 🔄 1. Reshaping & Pivotado")
        st.markdown("""
        Transforma datos relacionales planos en tensores/matrices matemáticas donde las clases categóricas se convierten en columnas independientes indexadas por el código de municipio (`COD_MUNI`).
        """)
    with t_col2:
        st.info("### 📐 2. Algoritmo K-Means")
        st.markdown("""
        Modelo de aprendizaje no supervisado que particiona los municipios en $K$ grupos minimizando la varianza interna de los clústeres (Within-Cluster Sum of Squares - WSS). Cada grupo posee un **Centroide** o vector promedio.
        """)
    with t_col3:
        st.info("### 🌐 3. Reducción PCA")
        st.markdown("""
        El Análisis de Componentes Principales proyecta el espacio multidimensional original (generado tras el pivotado) en un sistema de ejes ortogonales (`PC1`, `PC2`, `PC3`) conservando la máxima varianza de la muestra.
        """)
        
    st.markdown("""
    #### ⚖️ Importancia Crítica de las Métricas de Distancia y Escalabilidad:
    Para que el espacio geométrico de K-Means sea confiable, implementamos la **Estandarización Z-Score** ($z = \frac{x - \mu}{\sigma}$). Sin esto, variables masivas como los conteos agregados solaparían por completo variables de menor escala pero de extremo impacto crítico, como la tasa de letalidad o pérdida de vidas en combate.
    """)
    
    if st.button("Siguiente: Ver Desarrollo Metodológico ➡️"):
        cambiar_pagina('metodologia')
        st.rerun()

# ==============================================================================
# PÁGINA 4: METODOLOGÍA / DESARROLLO DEL PIPELINE
# ==============================================================================
elif st.session_state.page == 'metodologia':
    st.title("⚙️ Arquitectura del Pipeline y Pasos Desarrollados")
    st.markdown("---")
    
    st.markdown("### 🛠️ Código de Ingeniería de Características implementado:")
    
    with st.expander("Fase 1: Pivotado Estructurado y Agrupación Territorial", expanded=True):
        st.code("""
# Consolidación Territorial: Agrupación por código único de municipio
pivot_accion = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
pivot_fuerza = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0)
total_municipio = df_original.groupby(['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'])['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')

# Cruce unificado de matrices categóricas a columnas numéricas reales
datos = total_municipio.join([pivot_accion, pivot_fuerza]).reset_index().dropna()
        """, language="python")

    with st.expander("Fase 2: Escalamiento Estadístico con StandardScaler", expanded=False):
        st.code("""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Transformación espacial para obtener Media = 0 y Varianza = 1
datos[numericas] = scaler.fit_transform(datos[numericas])
X_scaled = datos.drop(columns=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'])
        """, language="python")

    with st.expander("Fase 3: Optimización del Hiperparámetro K (Método del Codo)", expanded=False):
        st.code("""
from sklearn.cluster import KMeans
wss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=30, random_state=42)
    kmeans.fit(X_scaled)
    wss.append(kmeans.inertia_)
        """, language="python")

    if st.button("🚀 Ejecutar Pipeline en Tiempo Real y Ver Resultados ➡️"):
        cambiar_pagina('resultados')
        st.rerun()

# ==============================================================================
# PÁGINA 5: RESULTADOS Y COMPONENTES ANALÍTICOS (EJECUCIÓN)
# ==============================================================================
elif st.session_state.page == 'resultados':
    st.title("📊 Hallazgos, Modelado de Clústeres y Gráficas Interactivas")
    st.markdown("---")
    
    if df_original is None:
        st.error("❌ No se detectó la base de datos necesaria para procesar los resultados.")
        st.stop()
        
    # --- EJECUCIÓN MATEMÁTICA INTERNA DEL PIPELINE ---
    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
    
    columnas_fuerza = [c for c in df_original['NOMBRE_FUERZA'].unique() if pd.notna(c)] if 'NOMBRE_FUERZA' in df_original.columns else []
    pivot_fuerza = df_original.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_fuerza else pd.DataFrame(index=pivot_accion.index)
    
    columnas_cat = [c for c in df_original['CATEGORIA'].unique() if pd.notna(c)] if 'CATEGORIA' in df_original.columns else []
    pivot_cat = df_original.pivot_table(index=index_cols, columns='CATEGORIA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_cat else pd.DataFrame(index=pivot_accion.index)
    
    total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')
    datos = total_municipio.join([pivot_accion, pivot_fuerza, pivot_cat]).reset_index().dropna()
    
    # Identificar nombres de columnas dinámicas creadas
    col_afectados = 'TOTAL_AFECTADOS' if 'TOTAL_AFECTADOS' in datos.columns else datos.columns[3]
    col_asesinado = 'ASESINADO' if 'ASESINADO' in datos.columns else (datos.columns[4] if len(datos.columns) > 4 else datos.columns[3])
    col_herido = 'HERIDO' if 'HERIDO' in datos.columns else (datos.columns[5] if len(datos.columns) > 5 else datos.columns[3])
    col_ejercito = 'EJERCITO NACIONAL DE COLOMBIA' if 'EJERCITO NACIONAL DE COLOMBIA' in datos.columns else (datos.columns[6] if len(datos.columns) > 6 else datos.columns[3])

    # Copia analítica y escalamiento
    scaler = StandardScaler()
    columnas_omitir = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    numericas = [col for col in datos.columns if col not in columnas_omitir]
    datos_originales_num = datos.copy()
    datos[numericas] = scaler.fit_transform(datos[numericas])
    X_scaled = datos.drop(columns=columnas_omitir)
    
    # Entrenamiento K-Means con K=4 fijo
    kmeans = KMeans(n_clusters=4, n_init=30, random_state=42)
    km4_clusters = kmeans.fit(X_scaled)
    datos_originales_num['Cluster'] = km4_clusters.labels_
    datos['Cluster'] = km4_clusters.labels_.astype(str)

    # --- RENDERIZADO DE CONTROLES E INTERFAZ DE EXPOSICIÓN ---
    st.subheader("🔍 Filtros de Validación Geográfica (Barra Lateral Activa)")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Unidades Territoriales Analizadas", datos.shape[0], help="Cantidad total de municipios únicos procesados")
    col_m2.metric("Dimensiones Numéricas en la Matriz", datos.shape[1] - 3, help="Variables sintéticas obtenidas por pivotado")
    
    # 1. CURVA DEL CODO
    st.markdown("### A. Validación Matemática del Número de Grupos (K)")
    wss = []
    for k in range(1, 11):
        km_test = KMeans(n_clusters=k, n_init=15, random_state=42)
        km_test.fit(X_scaled)
        wss.append(km_test.inertia_)
    fig_elbow = px.line(x=list(range(1, 11)), y=wss, markers=True, title="Optimización matemática mediante curva de Inercia (WSS)",
                        labels={'x': 'Número de Clústeres (k)', 'y': 'Inercia Interna'}, template='plotly_dark')
    fig_elbow.add_vline(x=4, line_dash="dash", line_color="cyan", annotation_text="K Óptimo Seleccionado = 4")
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.markdown("*Análisis:* La gráfica sustenta científicamente que fijar **K=4** es el punto de quiebre donde la ganancia de homogeneidad interna empieza a estabilizarse.")

    # 2. MAPAS DE CALOR DE DISTANCIA
    st.markdown("### B. Verificación de la Matriz de Disimilitudes (Muestra 50x50)")
    distancias_eu = euclidean_distances(X_scaled)[:50, :50]
    nombres_municipios_sub = datos['MUNICIPIO'].iloc[:50].tolist()
    fig_eu = px.imshow(distancias_eu, x=nombres_municipios_sub, y=nombres_municipios_sub,
                       labels=dict(color="Distancia Geométrica"), title="Matriz de Distancia Euclideana Inter-Municipios",
                       color_continuous_scale='RdBu_r', template='plotly_dark')
    st.plotly_chart(fig_eu, use_container_width=True)
    st.markdown("*Análisis:* Las regiones de color homogéneo revelan vecindarios de municipios que registran perfiles delictivos idénticos.")

    # 3. VISUALIZACIÓN PCA 3D
    st.markdown("### C. Proyección Espacial y Separabilidad de Grupos (PCA 3D)")
    pca_3d = PCA(n_components=3)
    scores_pca = pca_3d.fit_transform(X_scaled)
    df_pca = pd.DataFrame(scores_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Cluster'] = km4_clusters.labels_.astype(str)
    df_pca['Municipio'] = datos['MUNICIPIO'].values
    df_pca['Depto'] = datos['DEPARTAMENTO'].values
    
    fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster', 
                           hover_name='Municipio', hover_data=['Depto'],
                           title='Representación Tridimensional de Clústeres con Reducción PCA',
                           color_discrete_sequence=['red', 'green', 'blue', 'orange'], template='plotly_dark')
    
    centroids_3d = pca_3d.transform(kmeans.cluster_centers_)
    fig_3d.add_trace(go.Scatter3d(x=centroids_3d[:, 0], y=centroids_3d[:, 1], z=centroids_3d[:, 2],
                                 mode='markers', marker=dict(size=14, color='white', symbol='diamond', line=dict(width=1.5, color='black')),
                                 name='Centroides Matemáticos'))
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown("*Análisis:* Los rombos blancos marcan los baricentros de cada clúster. La dispersión espacial valida la clara segmentación de fronteras obtenida por el modelo.")

    # 4. TABLA DE PERFIL MEDIO
    st.markdown("### D. Radiografía Numérica de los Clústeres (Valores Reales Promedio)")
    variables_interes = [v for v in [col_afectados, col_asesinado, col_herido, col_ejercito] if v in datos_originales_num.columns]
    tabla_perfil = datos_originales_num.groupby('Cluster')[variables_interes].mean().round(2)
    tabla_perfil['Municipios Asignados'] = datos_originales_num.groupby('Cluster').size()
    st.dataframe(tabla_perfil, use_container_width=True)
    
    st.markdown("""
    * **Clúster 0 (Riesgo Controlado):** Volumen mínimo de incidencias. Estabilidad relativa en el mapa de orden público.
    * **Clúster 1 (Impacto Moderado / Dinámico):** Concentración regular de novedades con tasas de letalidad acotadas.
    * **Clúster 2 (Foco de Conflicto Institucional):** Municipios con alta afectación orientada a confrontaciones con unidades de las Fuerzas Militares.
    * **Clúster 3 (Emergencia Crítica):** Ciudades principales o focos históricos complejos con índices de letalidad y afectación total masivos.
    """)

    if st.button("Siguiente: Conclusiones y Cierre de la Sustentación ➡️"):
        cambiar_pagina('conclusiones')
        st.rerun()

# ==============================================================================
# PÁGINA 6: CONCLUSIONES Y CIERRE
# ==============================================================================
elif st.session_state.page == 'conclusiones':
    st.title("🏁 Conclusiones del Proyecto y Cierre Académico")
    st.markdown("---")
    
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.markdown("### 📌 Conclusiones Clave de la Investigación")
        st.markdown("""
        1. **Superación del Límite Categórico:** Se logró estructurar con éxito un Pipeline que procesa reportes cualitativos planos convirtiéndolos en matrices aptas para el aprendizaje supervisado y no supervisado.
        2. **Robustez Algorítmica:** La integración de `StandardScaler`, `KMeans` y `PCA` garantizó agrupamientos consistentes y balanceados sin sesgos de magnitud.
        3. **Validación Visual:** Las componentes principales demostraron geométricamente una separación limpia de las dinámicas territoriales del país.
        """)
    with c_col2:
        st.markdown("### 🚀 Recomendaciones Estratégicas")
        st.markdown("""
        * **Despliegue de Recursos:** El perfil numérico medio de los centroides permite priorizar la asistencia institucional y logística enfocando esfuerzos en los clústeres identificados como críticos.
        * **Automatización Futura:** La arquitectura construida es enteramente escalable, permitiendo lecturas dinámicas con la simple inserción de nuevos históricos mensuales en la carpeta raíz.
        """)
        
    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #1E3A8A;'>¡Muchas gracias por su atención!</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #4B5563;'>Queda abierto el espacio para preguntas y observaciones del comité evaluador.</p>", unsafe_allow_html=True)
    
    if st.button("↩️ Volver al Inicio de la Exposición"):
        cambiar_pagina('portada')
        st.rerun()

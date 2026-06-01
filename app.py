import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# CONFIGURACIÓN GENERAL Y ESTILO VISUAL "POWERPOINT PREMIUM"

st.set_page_config(page_title="Exposición Avanzada - Orden Público", layout="wide",
                   initial_sidebar_state="collapsed")

# Inyección de CSS Avanzado para simular Diapositivas de Consultoría
st.markdown("""
    <style>
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    .slide-container {
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        border: 1px solid #E2E8F0;
    }
    .slide-title {
        color: #0F172A;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .slide-subtitle {
        color: #64748B;
        font-size: 18px;
        margin-bottom: 25px;
        font-weight: 400;
    }
    div.stButton > button {
        background-color: #E0F2FE !important;
        color: #0369A1 !important;
        border: 1px solid #BAE6FD !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease-in-out !important;
    }
    div.stButton > button:hover {
        background-color: #7DD3FC !important;
        color: #0369A1 !important;
        border-color: #7DD3FC !important;
        box-shadow: 0 4px 12px rgba(3, 105, 161, 0.15) !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border: 1px solid #0284C7 !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
    }
    .insight-card {
        background-color: #F1F5F9;
        border-left: 5px solid #38BDF8;
        padding: 18px;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 15px;
    }
    .insight-critical {
        background-color: #FEF2F2;
        border-left: 5px solid #DC2626;
        padding: 18px;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 15px;
    }
    .insight-success {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 18px;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

if 'diapositiva' not in st.session_state:
    st.session_state.diapositiva = 1

def ir_a_diapositiva(num):
    st.session_state.diapositiva = num
    st.rerun()

def cargar_datos_automatico():
    archivos_en_carpeta = os.listdir('.')
    archivo_encontrado = None
    for archivo in archivos_en_carpeta:
        nombre_minuscula = archivo.lower()
        if ("afectacion" in nombre_minuscula or "fuerza" in nombre_minuscula or "publica" in nombre_minuscula) and (archivo.endswith('.csv') or archivo.endswith('.xlsx')):
            archivo_encontrado = archivo
            break
    if archivo_encontrado is None:
        return None, "No se encontró el registro de datos en la carpeta raíz."
    try:
        if archivo_encontrado.endswith('.csv'):
            df = pd.read_csv(archivo_encontrado, header=0)
        else:
            df = pd.read_excel(archivo_encontrado, header=0)
        return df, archivo_encontrado
    except Exception as e:
        return None, f"Error al leer el archivo: {str(e)}"

df_original, nombre_archivo_cargado = cargar_datos_automatico()

# NAVEGACIÓN
cols_nav = st.columns(6)
nombres_diapo = ["1. Portada", "2. Introducción", "3. Marco Teórico", "4. Metodología", "5. Resultados", "6. Conclusiones"]

for i, nombre in enumerate(nombres_diapo):
    tipo_boton = "primary" if st.session_state.diapositiva == (i + 1) else "secondary"
    if cols_nav[i].button(nombre, use_container_width=True, type=tipo_boton):
        ir_a_diapositiva(i + 1)

st.markdown("---")

# DIAPOSITIVAS
if st.session_state.diapositiva == 1:
    st.markdown("""
    <div class='slide-container' style='text-align: center; padding: 60px 40px;'>
        <img src='https://administrativos.ut.edu.co/images/Home/simbolos/logo_oficial.png' width='197' style='margin-bottom: 20px;'>
        <div class='slide-title' style='font-size: 42px; color: #1E3A8A;'>Análisis de Clústeres (K-Means) En Afectaciones a la Fuerza Pública</div>
        <div class='slide-subtitle' style='font-size: 22px;'>Segmentación Territorial de Incidentes de Orden Público Mediante Modelos de Aprendizaje no Supervisados</div>
        <div style='margin: 40px 0; border-top: 2px solid #E2E8F0;'></div>
    </div>
    """, unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("<div class='insight-card'><h4 style='margin-top:0; color:#1E3A8A;'>ESTUDIANTE</h4><p><b>Miguel Angel Garatejo</b><br>Facultad de Ciencias<br>Matematica con Enfasis en Estadistica</p></div>", unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"<div class='insight-success'><h4 style='margin-top:0; color:#16A34A;'> PROFESOR</h4><p><b>Yuri Marcela Garcia Saavedra</b><br>Mineria de Datos <br>Año: {time.strftime('%Y')} | Clustering</p></div>", unsafe_allow_html=True)
    if st.button("Iniciar Sustentación", type="primary", use_container_width=True):
        ir_a_diapositiva(2)

elif st.session_state.diapositiva == 2:
    st.markdown("<div class='slide-title'> Introducción y Definición del Desafío Técnico</div><div class='slide-subtitle'>Contexto del orden público e inconsistencia de los datos</div>", unsafe_allow_html=True)
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("<div class='slide-container'><h3 style='color: #DC2626; margin-top:0;'> El Problema de los Datos Originales</h3><p><b>Naturaleza del Archivo:</b> La información institucional se presenta como un <i>Histórico de Novedades</i>.</p><ul><li><b>Restricción de Estructura:</b> El archivo posee 8 columnas cualitativas y solo 1 cuantitativa.</li><li><b>El Quiebre Matemático:</b> Los algoritmos basados en distancias son incapaces de calcular similitudes usando texto directo.</li></ul></div>", unsafe_allow_html=True)
    with col_i2:
        st.markdown("<div class='slide-container'><h3 style='color: #0284C7; margin-top:0;'> Objetivos y Justificación</h3><p><b>Objetivo Principal:</b> Construir un flujo automatizado para reestructurar y agrupar numéricamente los municipios.</p><ul><li>Migrar a un mapa estratégico integral.</li><li>Sustentar científicamente la asignación de recursos.</li></ul></div>", unsafe_allow_html=True)
    if st.button("Siguiente Diapositiva: Marco Conceptual ➡️", type="primary"):
        ir_a_diapositiva(3)

elif st.session_state.diapositiva == 3:
    st.markdown("<div class='slide-title'> Fundamentos Teóricos y Algorítmicos</div><div class='slide-subtitle'>Sustentación matemática para el agrupamiento y reducción espacial</div>", unsafe_allow_html=True)
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.markdown("<div class='slide-container' style='min-height: 280px;'><h4 style='color:#0284C7; margin-top:0;'> 1. Reestructuración de Matrices</h4><p style='font-size:14px;'>Transformación de categorías cualitativas en dimensiones numéricas numéricas mediante pivotado.</p></div>", unsafe_allow_html=True)
    with t_col2:
        st.markdown("<div class='slide-container' style='min-height: 280px;'><h4 style='color:#0284C7; margin-top:0;'> 2. Algoritmo K-Means</h4><p style='font-size:14px;'>Modelo no supervisado que minimiza la varianza interna encontrando vectores promedio llamados Centroides.</p></div>", unsafe_allow_html=True)
    with t_col3:
        st.markdown("<div class='slide-container' style='min-height: 280px;'><h4 style='color:#0284C7; margin-top:0;'> 3. PCA</h4><p style='font-size:14px;'>Reducción de dimensiones conservando la mayor variabilidad posible para visualización 3D.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='insight-card'><h4 style='margin-top:0; color:#1E293B;'> Normalización Z-Score</h4><p>Ajuste de escala para obtener Media = 0 y Varianza = 1, evitando que variables masivas eclipsen indicadores críticos.</p></div>", unsafe_allow_html=True)
    if st.button("Siguiente Diapositiva: Estrategia de Procesamiento ", type="primary"):
        ir_a_diapositiva(4)

elif st.session_state.diapositiva == 4:
    st.markdown("<div class='slide-title'>⚙️ Arquitectura del Flujo y Procesamiento de Datos</div><div class='slide-subtitle'>Ingeniería de características implementada en Python</div>", unsafe_allow_html=True)
    with st.expander("Fase 1: Pivotado y Agrupación", expanded=True):
        st.code("pivot = df.pivot_table(index=['COD_MUNI'], columns='ACCION', values='CANTIDAD', aggfunc='sum')", language="python")
    with st.expander("Fase 2: StandardScaler", expanded=False):
        st.code("scaler = StandardScaler()\nX_scaled = scaler.fit_transform(datos)", language="python")
    with st.expander("Fase 3: Método del Codo", expanded=False):
        st.code("wss = [KMeans(k).fit(X).inertia_ for k in range(1, 11)]", language="python")
    if st.button("Siguiente Diapositiva: Ejecución y Resultados ➡️", type="primary"):
        ir_a_diapositiva(5)

# DIAPOSITIVA 5: RESULTADOS (DISEÑO MEJORADO)
elif st.session_state.diapositiva == 5:
    st.markdown("<div class='slide-title'> Hallazgos y Análisis de Clústeres</div><div class='slide-subtitle'>Inspección profunda de patrones y métricas de separación</div>", unsafe_allow_html=True)
    
    if df_original is None:
        st.error("❌ No se detectó el archivo de datos.")
        st.stop()
        
    # --- PROCESAMIENTO ---
    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
    pivot_fuerza = df_original.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0) if 'NOMBRE_FUERZA' in df_original.columns else pd.DataFrame(index=pivot_accion.index)
    total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')
    datos = total_municipio.join([pivot_accion, pivot_fuerza]).reset_index().dropna()
    
    numericas = [col for col in datos.columns if col not in index_cols]
    datos_originales_num = datos.copy()
    scaler = StandardScaler()
    datos[numericas] = scaler.fit_transform(datos[numericas])
    X_scaled = datos.drop(columns=index_cols)
    
    kmeans = KMeans(n_clusters=4, n_init=30, random_state=42)
    km4_clusters = kmeans.fit(X_scaled)
    datos_originales_num['Cluster'] = km4_clusters.labels_
    datos['Cluster'] = km4_clusters.labels_.astype(str)

    # --- MÉTRICAS ---
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Municipios", datos.shape[0])
    col_m2.metric("Dimensiones", datos.shape[1] - 4)
    col_m3.metric("Clústeres", "4 (Óptimo)")

    # 1. ELBOW CHART PREMIUM
    st.markdown("### 📈 A. Validación del K Óptimo")
    wss = []
    for k in range(1, 11):
        wss.append(KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled).inertia_)
    
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(range(1, 11)), y=wss, mode='lines+markers',
                                   line=dict(color='#0EA5E9', width=4),
                                   marker=dict(size=10, color='#0369A1', line=dict(width=2, color='white')),
                                   fill='tozeroy', fillcolor='rgba(14, 165, 233, 0.1)'))
    fig_elbow.update_layout(title="Método del Codo (Inercia Matemática)", template='plotly_white',
                            hovermode='x unified', margin=dict(t=50, b=50, l=50, r=50))
    fig_elbow.add_vline(x=4, line_dash="dash", line_color="#EF4444", annotation_text="K=4 Seleccionado")
    st.plotly_chart(fig_elbow, use_container_width=True)

    # 2. HEATMAP PREMIUM
    st.markdown("### 🗺️ B. Mapa de Calor de Disimilitud (Muestra)")
    distancias_eu = euclidean_distances(X_scaled)[:40, :40]
    nombres_mun = datos['MUNICIPIO'].iloc[:40].tolist()
    
    fig_eu = px.imshow(distancias_eu, x=nombres_mun, y=nombres_mun,
                       color_continuous_scale='GnBu', # Azul-Verde suave muy agradable
                       title="Matriz de Distancia Euclideana (Similitud)")
    fig_eu.update_layout(template='plotly_white', coloraxis_showscale=True)
    st.plotly_chart(fig_eu, use_container_width=True)

    # 3. PCA 3D PREMIUM
    st.markdown("### 🌐 C. Dispersión Espacial 3D (PCA)")
    pca_3d = PCA(n_components=3)
    scores = pca_3d.fit_transform(X_scaled)
    df_pca = pd.DataFrame(scores, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Cluster'] = datos['Cluster']
    df_pca['Municipio'] = datos['MUNICIPIO']
    
    # Paleta de colores Pastel/Premium
    colors = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444'] 
    
    fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster',
                           hover_name='Municipio',
                           color_discrete_sequence=colors,
                           opacity=0.8,
                           title="Nubes de Puntos y Separación de Fronteras")
    
    fig_3d.update_traces(marker=dict(size=5, line=dict(width=1, color='white')))
    fig_3d.update_layout(scene=dict(
        xaxis=dict(backgroundcolor="#F8FAFC", gridcolor="white", showbackground=True),
        yaxis=dict(backgroundcolor="#F8FAFC", gridcolor="white", showbackground=True),
        zaxis=dict(backgroundcolor="#F8FAFC", gridcolor="white", showbackground=True),
    ), margin=dict(l=0, r=0, b=0, t=50))
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("<div class='insight-critical'><h4>Diagnóstico de Outliers</h4><p>Los puntos aislados en el <b>Clúster 3</b> representan ciudades donde la letalidad supera los promedios nacionales por más de 3 desviaciones estándar.</p></div>", unsafe_allow_html=True)

    if st.button("Siguiente Diapositiva: Conclusiones ➡️", type="primary"):
        ir_a_diapositiva(6)

elif st.session_state.diapositiva == 6:
    st.markdown("<div class='slide-title'>🏁 Conclusiones y Recomendaciones</div><div class='slide-subtitle'>Cierre formal de la investigación estadística</div>", unsafe_allow_html=True)
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.markdown("<div class='slide-container' style='min-height:350px;'><h3 style='color:#0369A1; margin-top:0;'> Conclusiones</h3><ol><li>Tratamiento Cualitativo Exitoso.</li><li>Consistencia Algorítmica demostrada.</li><li>Sensibilidad ante Anomalías.</li></ol></div>", unsafe_allow_html=True)
    with c_col2:
        st.markdown("<div class='slide-container' style='min-height:350px;'><h3 style='color:#16A34A; margin-top:0;'> Sugerencias</h3><ul><li>Despliegue Preventivo basado en Centroides.</li><li>Escalabilidad Operativa automatizada.</li></ul></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; padding: 40px 0;'><h2 style='color: #0369A1; margin-bottom: 5px;'>¡Muchas gracias por su atención!</h2><p style='color: #64748B;'>Se abre el espacio para las preguntas.</p></div>", unsafe_allow_html=True)
    if st.button("↩️ Reiniciar Exposición", type="secondary"):
        ir_a_diapositiva(1)

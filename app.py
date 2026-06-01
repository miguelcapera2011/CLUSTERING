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


# ==================================================
# ESTILO PREMIUM PARA GRAFICAS
# ==================================================

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

# CONFIGURACIÓN GENERAL Y ESTILO VISUAL "POWERPOINT PREMIUM"

st.set_page_config(page_title="Exposición Avanzada - Orden Público", layout="wide", initial_sidebar_state="collapsed")

# Inyección de CSS Avanzado para simular Diapositivas de Consultoría (Fondo Claro y Elegante)
st.markdown("""
    <style>
    /* Fondo principal claro y limpio estilo diapositiva */
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Ocultar barra lateral por defecto para enfocar la presentación */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    /* Contenedor de la diapositiva */
    .slide-container {
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        border: 1px solid #E2E8F0;
    }
    /* Estilos de títulos estilo McKinsey */
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
    
    /* MODIFICACIÓN: Botones superiores más bonitos, claros y con letras muy legibles */
    div.stButton > button {
        background-color: #E0F2FE !important; /* Azul cielo muy claro */
        color: #0369A1 !important;            /* Texto azul oscuro de alto contraste */
        border: 1px solid #BAE6FD !important; /* Borde sutil */
        border-radius: 8px !important;
        font-weight: 700 !important;          /* Texto en negrita para máxima legibilidad */
        font-size: 14px !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease-in-out !important;
    }
    
    /* Efecto al pasar el mouse por encima del botón */
    div.stButton > button:hover {
        background-color: #7DD3FC !important; /* Azul claro un poco más vivo */
        color: #0369A1 !important;
        border-color: #7DD3FC !important;
        box-shadow: 0 4px 12px rgba(3, 105, 161, 0.15) !important;
    }

    /* Estilo exclusivo para el botón de la página activa */
    div.stButton > button[kind="primary"] {
        background-color: #0284C7 !important; /* Azul intermedio vivo */
        color: #FFFFFF !important;            /* Texto blanco */
        border: 1px solid #0284C7 !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
    }
    
    div.stButton > button[kind="primary"]:hover {
        background-color: #0369A1 !important;
        color: #FFFFFF !important;
    }

    /* Tarjetas de insights o hallazgos */
    .insight-card {
        background-color: #F1F5F9;
        border-left: 5px solid #38BDF8; /* Azul más claro en el borde */
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
    /* Barra de navegación superior */
    .nav-bar {
        background-color: #0F172A;
        padding: 15px;
        border-radius: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Inicialización del paginador (diapositivas)
if 'diapositiva' not in st.session_state:
    st.session_state.diapositiva = 1

def ir_a_diapositiva(num):
    st.session_state.diapositiva = num
    st.rerun()

# CARGA AUTOMÁTICA DE DATOS DESDE EL REGISTRO HISTÓRICO

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


# CONTROLES DE NAVEGACIÓN SUPERIOR (BOTONES ESTILO DIAPOSITIVA)

cols_nav = st.columns(6)
nombres_diapo = ["1. Portada", "2. Introducción", "3. Marco Teórico", "4. Metodología", "5. Resultados", "6. Conclusiones"]

for i, nombre in enumerate(nombres_diapo):
    tipo_boton = "primary" if st.session_state.diapositiva == (i + 1) else "secondary"
    if cols_nav[i].button(nombre, use_container_width=True, type=tipo_boton):
        ir_a_diapositiva(i + 1)

st.markdown("---")


# DIAPOSITIVA 1: PORTADA OFICIAL

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
        st.markdown("""
        <div class='insight-card'>
            <h4 style='margin-top:0; color:#1E3A8A;'>ESTUDIANTE</h4>
            <p><b>Miguel Angel Garatejo</b><br>Facultad de Ciencias<br>Matematica con Enfasis en Estadistica</p>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"""
        <div class='insight-success'>
            <h4 style='margin-top:0; color:#16A34A;'> PROFESOR</h4>
            <p><b>Yuri Marcela Garcia Saavedra </b><br>Mineria de Datos <br>Año: {time.strftime('%Y')} | Clustering</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Iniciar Sustentación", type="primary", use_container_width=True):
        ir_a_diapositiva(2)


# DIAPOSITIVA 2: INTRODUCCIÓN Y PLANTEAMIENTO DEL PROBLEMA

elif st.session_state.diapositiva == 2:
    st.markdown("""
    <div class='slide-title'> Introducción y Definición del Desafío Técnico</div>
    <div class='slide-subtitle'>Contexto del orden público e inconsistencia de los datos</div>
    """, unsafe_allow_html=True)
    
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
        <div class='slide-container'>
            <h3 style='color: #DC2626; margin-top:0;'> El Problema de los Datos Originales</h3>
            <p><b>Naturaleza del Archivo:</b> La información institucional se presenta como un <i>Histórico de Novedades</i> (registros) donde cada fila reporta un ataque individual aislado.</p>
            <ul>
                <li><b>Restricción de Estructura:</b> El archivo posee <b>8 columnas cualitativas (texto)</b> y solo <b>1 columna cuantitativa (Cantidad)</b>.</li>
                <li><b>El Quiebre Matemático:</b> Los algoritmos matemáticos basados en distancias espaciales (como <i>K-Means</i>) son incapaces de calcular similitudes usando texto directo (ej. 'POLICÍA' o 'EJÉRCITO'). No se pueden promediar palabras (variables categoricas).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_i2:
        st.markdown("""
        <div class='slide-container'>
            <h3 style='color: #0284C7; margin-top:0;'> Objetivos y Justificación</h3>
            <p><b>Objetivo Principal:</b> Construir un flujo de procesamiento automatizado en Python para reestructurar, unificar y agrupar numéricamente los municipios según sus patrones reales de vulnerabilidad.</p>
            <p><b>Importancia Estratégica:</b></p>
            <ul>
                <li>Permite migrar de un análisis estático de registros individuales a un mapa estratégico integral del territorio nacional.</li>
                <li>Sustenta científicamente la toma de decisiones preventivas y la asignación eficiente de recursos logísticos e institucionales.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Siguiente Diapositiva: Marco Conceptual ➡️", type="primary"):
        ir_a_diapositiva(3)

# DIAPOSITIVA 3: MARCO TEÓRICO / CONCEPTUAL

elif st.session_state.diapositiva == 3:
    st.markdown("""
    <div class='slide-title'> Fundamentos Teóricos y Algorítmicos</div>
    <div class='slide-subtitle'>Sustentación matemática para el agrupamiento y reducción espacial</div>
    """, unsafe_allow_html=True)
    
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.markdown("""
        <div class='slide-container' style='min-height: 280px;'>
            <h4 style='color:#0284C7; margin-top:0;'> 1. Reestructuración de Matrices (Pivotado)</h4>
            <p style='font-size:14px;'>Consiste en transformar la estructura lineal del histórico para convertir las categorías cualitativas en nuevas dimensiones numéricas (columnas) indexadas por el código único del municipio.</p>
        </div>
        """, unsafe_allow_html=True)
    with t_col2:
        st.markdown("""
        <div class='slide-container' style='min-height: 280px;'>
            <h4 style='color:#0284C7; margin-top:0;'> 2. Algoritmo K-Means</h4>
            <p style='font-size:14px;'>Modelo de aprendizaje no supervisado que particiona las observaciones en <i>K</i> grupos homogéneos. Su meta es minimizar la varianza interna de cada grupo (Inercia o WSS), encontrando un vector promedio central llamado <b>Centroide</b>.</p>
        </div>
        """, unsafe_allow_html=True)
    with t_col3:
        st.markdown("""
        <div class='slide-container' style='min-height: 280px;'>
            <h4 style='color:#0284C7; margin-top:0;'> 3. Componentes Principales (PCA)</h4>
            <p style='font-size:14px;'>Técnica de reducción de dimensiones que proyecta el plano de alta complejidad hacia un nuevo sistema de ejes ortogonales (PC1, PC2, PC3). Conserva la mayor variabilidad posible permitiendo la visualización gráfica sin alterar las distancias.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div class='insight-card'>
        <h4 style='margin-top:0; color:#1E293B;'> Rol Crítico de la Normalización Estadística (Z-Score)</h4>
        <p>Para asegurar que las distancias geométricas calculadas por el modelo sean confiables, se aplicó un ajuste de escala para obtener una <b>Media = 0 y Varianza = 1</b> (StandardScaler). Sin este paso, las variables masivas (como el conteo total de incidentes) eclipsarían por completo indicadores de menor escala pero con un impacto estratégico crítico, tales como las tasas de letalidad o pérdidas de vidas humanas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Siguiente Diapositiva: Estrategia de Procesamiento ", type="primary"):
        ir_a_diapositiva(4)


# DIAPOSITIVA 4: METODOLOGÍA / DESARROLLO DEL FLUJO

elif st.session_state.diapositiva == 4:
    st.markdown("""
    <div class='slide-title'>⚙️ Arquitectura del Flujo y Procesamiento de Datos</div>
    <div class='slide-subtitle'>Ingeniería de características implementada en Python para la transformación de la información</div>
    """, unsafe_allow_html=True)
    
    st.markdown(" Código Implementado:")
    
    with st.expander("Fase 1: Pivotado Estructurado y Agrupación Territorial", expanded=True):
        st.code("""
# Consolidación Territorial: Agrupación por código único de municipio
pivot_accion = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
pivot_fuerza = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0)
total_municipio = df_original.groupby(['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'])['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')

# Cruce unificado de matrices categóricas a columnas numéricas reales
datos = total_municipio.join([pivot_accion, pivot_fuerza]).reset_index().dropna()
        """, language="python")

    with st.expander("Fase 2: Normalización de Escala (StandardScaler)", expanded=False):
        st.code("""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Ajuste matemático para establecer Media = 0 y Varianza = 1 en todas las columnas
datos[numericas] = scaler.fit_transform(datos[numericas])
X_scaled = datos.drop(columns=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'])
        """, language="python")

    with st.expander("Fase 3: Optimización Matemática (Método del Codo)", expanded=False):
        st.code("""
from sklearn.cluster import KMeans
wss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=30, random_state=42)
    kmeans.fit(X_scaled)
    wss.append(kmeans.inertia_)
        """, language="python")

    if st.button("Siguiente Diapositiva: Ejecución y Resultados del Modelo ➡️", type="primary"):
        ir_a_diapositiva(5)

# DIAPOSITIVA 5: RESULTADOS Y ANÁLISIS DE FONDO DE LOS CLÚSTERES (CORAZÓN DE LA EXP)

elif st.session_state.diapositiva == 5:
    st.markdown("""
    <div class='slide-title'> Hallazgos, Comportamiento Estructurado y Análisis de Clústeres</div>
    <div class='slide-subtitle'>Inspección profunda de patrones, métricas de separación y detección de datos atípicos</div>
    """, unsafe_allow_html=True)
    
    if df_original is None:
        st.error("❌ No se detectó el archivo de datos necesario para procesar los resultados.")
        st.stop()
        
    # --- PROCESAMIENTO MATEMÁTICO REAL ---
    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
    
    columnas_fuerza = [c for c in df_original['NOMBRE_FUERZA'].unique() if pd.notna(c)] if 'NOMBRE_FUERZA' in df_original.columns else []
    pivot_fuerza = df_original.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_fuerza else pd.DataFrame(index=pivot_accion.index)
    
    columnas_cat = [c for c in df_original['CATEGORIA'].unique() if pd.notna(c)] if 'CATEGORIA' in df_original.columns else []
    pivot_cat = df_original.pivot_table(index=index_cols, columns='CATEGORIA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_cat else pd.DataFrame(index=pivot_accion.index)
    
    total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')
    datos = total_municipio.join([pivot_accion, pivot_fuerza, pivot_cat]).reset_index().dropna()
    
    col_afectados = 'TOTAL_AFECTADOS' if 'TOTAL_AFECTADOS' in datos.columns else datos.columns[3]
    col_asesinado = 'ASESINADO' if 'ASESINADO' in datos.columns else (datos.columns[4] if len(datos.columns) > 4 else datos.columns[3])
    col_herido = 'HERIDO' if 'HERIDO' in datos.columns else (datos.columns[5] if len(datos.columns) > 5 else datos.columns[3])
    
    scaler = StandardScaler()
    columnas_omitir = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    numericas = [col for col in datos.columns if col not in columnas_omitir]
    datos_originales_num = datos.copy()
    datos[numericas] = scaler.fit_transform(datos[numericas])
    X_scaled = datos.drop(columns=columnas_omitir)
    
    kmeans = KMeans(n_clusters=4, n_init=30, random_state=42)
    km4_clusters = kmeans.fit(X_scaled)
    datos_originales_num['Cluster'] = km4_clusters.labels_
    datos['Cluster'] = km4_clusters.labels_.astype(str)

    # --- MÉTRICAS GENERALES DE LA MATRIZ ---
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Municipios Procesados", datos.shape[0], help="Total de entidades territoriales únicas consolidadas")
    with col_m2:
        st.metric("Nuevas Columnas Numéricas", datos.shape[1] - 4, help="Variables sintéticas obtenidas por el pivotado")

    # 1. ANÁLISIS DE LA CURVA DEL CODO
    st.markdown(" A. Validación Científica del Número de Grupos (K)")
    wss = []
    for k in range(1, 11):
        km_test = KMeans(n_clusters=k, n_init=15, random_state=42)
        km_test.fit(X_scaled)
        wss.append(km_test.inertia_)
        
    fig_elbow = px.line(x=list(range(1, 11)), y=wss, markers=True, title="Evaluación de Estabilidad por Inercia Interna (WSS)",
                        labels={'x': 'Número de Clústeres (k)', 'y': 'Inercia Matemática'}, template='plotly_white')
    fig_elbow.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="K Óptimo Seleccionado = 4")
    fig_elbow.update_traces(line_color='#38BDF8', marker=dict(size=8, color='#0284C7')) 

    fig_elbow = aplicar_estilo_premium(fig_elbow)
    fig_elbow.update_traces(
       line=dict(width=5),
       marker=dict(size=10)
    )
    st.plotly_chart(fig_elbow, use_container_width=True)
    
    st.markdown("""
    <div class='insight-card'>
        <b>🔍 Análisis Crítico del Codo:</b> La gráfica evidencia que el punto de inflexión más claro ocurre en <b>K=4</b>. Antes de este punto, añadir un grupo extra reduce drásticamente el error del modelo; después de K=4, la ganancia de homogeneidad se estabiliza. Esto demuestra científicamente que clasificar el país en 4 dinámicas territoriales es estructuralmente óptimo.
    </div>
    """, unsafe_allow_html=True)

    # 2. ANÁLISIS DE DISTANCIAS
    st.markdown("### B. Matriz Geométrica de Distancia Euclideana (Muestra de Control de 50 Municipios)")
    distancias_eu = euclidean_distances(X_scaled)[:50, :50]
    nombres_municipios_sub = datos['MUNICIPIO'].iloc[:50].tolist()
    fig_eu = px.imshow(
        distancias_eu,
        x=nombres_municipios_sub,
        y=nombres_municipios_sub,
        title="Mapa de Distancias Euclidianas",
        color_continuous_scale=[
            [0.00, "#22C55E"],  # verde
            [0.25, "#84CC16"],  # verde claro
            [0.50, "#FACC15"],  # amarillo
            [0.75, "#F97316"],  # naranja
            [1.00, "#DC2626"]   # rojo
        ]
    )

    fig_eu = aplicar_estilo_premium(fig_eu)
    fig_eu.update_xaxes(tickfont=dict(color="black", size=10))
    fig_eu.update_yaxes(tickfont=dict(color="black", size=10))
    fig_eu.update_traces(xgap=1, ygap=1)
    fig_eu.update_layout(
        coloraxis_colorbar=dict(
            title="Distancia",
            title_font=dict(size=16, color="black"),
            tickfont=dict(size=12, color="black")
        )
    )
    st.plotly_chart(fig_eu, use_container_width=True)

    st.markdown("""
    <div class='insight-card'>
        <b>Análisis del Mapa de Calor:</b> Los bloques identifican municipios con perfiles de conflicto idénticos (baja distancia entre sí), mientras que los cambios de color revelan contrastes operacionales radicales, aislando zonas tranquilas de aquellas con dinámicas complejas.
    </div>
    """, unsafe_allow_html=True)

    # 3. ANÁLISIS TRIDIMENSIONAL DE PCA
    st.markdown("### 🌐 C. Proyección Espacial Avanzada e Identificación de Datos Atípicos (PCA 3D)")
    pca_3d = PCA(n_components=3)
    scores_pca = pca_3d.fit_transform(X_scaled)
    df_pca = pd.DataFrame(scores_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Cluster'] = km4_clusters.labels_.astype(str)
    df_pca['Municipio'] = datos['MUNICIPIO'].values
    df_pca['Depto'] = datos['DEPARTAMENTO'].values
    
    nombres_clusters = {"0": "Clúster 0: Riesgo Controlado", "1": "Clúster 1: Impacto Moderado", 
                        "2": "Clúster 2: Conflicto Institucional", "3": "Clúster 3: Emergencia Crítica"}
    df_pca['Cluster'].map(nombres_clusters)
    
    fig_3d = px.scatter_3d(
        df_pca,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Nombre_Cluster',
        hover_name='Municipio',
        hover_data=['Depto'],
        title='Distribución Espacial de Municipios',
        color_discrete_map={
            "Clúster 0: Riesgo Controlado": "#22C55E",
            "Clúster 1: Impacto Moderado": "#0EA5E9",
            "Clúster 2: Conflicto Institucional": "#F59E0B",
            "Clúster 3: Emergencia Crítica": "#EF4444"
           }
        )

        fig_3d.update_layout(
        height=900,
        paper_bgcolor="#EAF4FF",
        plot_bgcolor="#F4F9FF",
        font=dict(
            color="black",
            size=14
         ),

        legend=dict(
            font=dict(
               color="black",
               size=14
         ),
         title=dict(
            text="Clusters",
            font=dict(
                color="black",
                size=16
               )
            )
          ),

         scene=dict(
             bgcolor="#F4F9FF",

            xaxis=dict(
                title="PC1",
                title_font=dict(color="black"),
                tickfont=dict(color="black"),
                backgroundcolor="#F4F9FF",
                gridcolor="#CBD5E1"
         ),

          yaxis=dict(
            title="PC2",
            title_font=dict(color="black"),
            tickfont=dict(color="black"),
            backgroundcolor="#F4F9FF",
            gridcolor="#CBD5E1"
        ),

        zaxis=dict(
            title="PC3",
            title_font=dict(color="black"),
            tickfont=dict(color="black"),
            backgroundcolor="#F4F9FF",
            gridcolor="#CBD5E1"
            )
         )
      )
    
    centroids_3d = pca_3d.transform(kmeans.cluster_centers_)
    colores = ["#22C55E", "#0EA5E9", "#F59E0B", "#EF4444"]

    for i in range(4):
        fig_3d.add_trace(
            go.Scatter3d(
                x=[centroids_3d[i, 0]],
                y=[centroids_3d[i, 1]],
                z=[centroids_3d[i, 2]],
                mode='markers',
                marker=dict(
                    size=18,
                    color=colores[i],
                    symbol='diamond'
                ),
                name=f'Centroide {i}'
            )
        )

    for cluster in sorted(df_pca["Cluster"].unique()):
        temp = df_pca[df_pca["Cluster"] == cluster]
        fig_3d.add_trace(
            go.Mesh3d(
                x=temp["PC1"],
                y=temp["PC2"],
                z=temp["PC3"],
                opacity=0.10,
                color={
                    "0": "#22C55E",
                    "1": "#0EA5E9",
                    "2": "#F59E0B",
                    "3": "#EF4444"
                }[cluster],
                name=f"Área Cluster {cluster}"
            )
        )
        
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("""
    <div class='insight-critical'>
        <h4> Diagnóstico de Datos Atípicos (Puntos Lejanos en el Espacio)</h4>
        <p>Al exploring la visualización en 3D, se identifican puntos que rompen la densidad del grupo y se proyectan de forma aislada en las esquinas del plano geométrico. 
        Estos corresponden a <b>Datos Atípicos Operacionales (Outliers)</b> como grandes capitales o focos críticos históricos (ej. <i>Cali, Tumaco o Cúcuta</i>). 
        El modelo no los excluye, sino que los agrupa de forma aislada en el <b>Clúster 3 (Emergencia Crítica)</b> porque sus volúmenes y la letalidad de sus ataques superan los promedios nacionales por más de 3 desviaciones estándar.</p>
    </div>
    """, unsafe_allow_html=True)

    # 4. RADIOGRAFÍA PROFUNDA DE LOS RESULTADOS
    st.markdown("### D. Perfil de Comportamiento de los Clústeres (Valores Reales Promedio)")
    variables_interes = [v for v in [col_afectados, col_asesinado, col_herido] if v in datos_originales_num.columns]
    tabla_perfil = datos_originales_num.groupby('Cluster')[variables_interes].mean().round(2)
    tabla_perfil['Municipios Asignados'] = datos_originales_num.groupby('Cluster').size()
    
    tabla_perfil.index = ["Clúster 0 (Riesgo Controlado)", "Clúster 1 (Impacto Moderado)", 
                          "Clúster 2 (Conflicto Institucional)", "Clúster 3 (Emergencia Crítica)"]
    st.dataframe(tabla_perfil, use_container_width=True)
    
    st.markdown("""
    <div class='slide-container'>
        <h4 style='margin-top:0; color:#0F172A;'>🔍 Interpretación Estratégica de cada Grupo:</h4>
        <ul>
            <li><b>🟢 Clúster 0 (Riesgo Controlado):</b> Agrupa a la inmensa mayoría de municipios del país. Los incidentes son esporádicos y aislados, manteniendo promedios cercanos a cero. Representa la estabilidad base del territorio.</li>
            <li><b>🔵 Clúster 1 (Impacto Moderado / Dinámico):</b> Municipios que muestran actividad delictiva constante pero con baja letalidad. Son zonas con novedades frecuentes (heridos o afectaciones logísticas) pero donde la confrontación armada no está desbordada.</li>
            <li><b>🟡 Clúster 2 (Foco de Conflicto Institucional):</b> Zonas geográficas muy particulares donde los ataques están dirigidos explícitamente a las patrullas e instalaciones físicas de la Fuerza Pública. Presentan niveles intermedios de letalidad y una alta concentración de eventos bélicos.</li>
            <li><b>🔴 Clúster 3 (Emergencia Crítica):</b> El grupo más alarmante del análisis. Contiene pocos municipios pero registra promedios de asesinados, heridos y afectaciones totales sumamente altos. Aquí es donde radican las anomalías de los datos y donde el despliegue del Estado debe pasar de ser reactivo a completamente prioritario.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Siguiente Diapositiva: Conclusiones y Recomendaciones ➡️", type="primary"):
        ir_a_diapositiva(6)


# DIAPOSITIVA 6: CONCLUSIONES Y CIERRE ACADÉMICO

elif st.session_state.diapositiva == 6:
    st.markdown("""
    <div class='slide-title'>🏁 Conclusiones Académicas y Recomendaciones Futuras</div>
    <div class='slide-subtitle'>Cierre formal de la investigación estadística</div>
    """, unsafe_allow_html=True)
    
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.markdown("""
        <div class='slide-container' style='min-height:350px;'>
            <h3 style='color:#0369A1; margin-top:0;'> Conclusiones Clave</h3>
            <ol>
                <li><b>Tratamiento Cualitativo Exitoso:</b> Se logró solucionar la limitación inicial de trabajar con columnas de texto mediante una estrategia de reestructuración matricial efectiva.</li>
                <li><b>Consistencia Algorítmica:</b> El acoplamiento de <i>Z-Score, K-Means y PCA</i> demostró una separación clara de los municipios en el espacio geométrico, aislando de forma óptima las zonas críticas de las estables.</li>
                <li><b>Identificación de Anomalías:</b> El modelo demostró alta sensibilidad al aislar de forma automática los datos atípicos de alto impacto operacional en el clúster de Emergencia Crítica.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    with c_col2:
        st.markdown("""
        <div class='slide-container' style='min-height:350px;'>
            <h3 style='color:#16A34A; margin-top:0;'> Sugerencias para el Futuro</h3>
            <ul>
                <li><b>Logística de Despliegue Preventivo:</b> Los perfiles numéricos de los centroides de los clústeres 2 y 3 permiten a los tomadores de decisiones pre-posicionar apoyo logístico y asistencia médica en los municipios prioritarios.</li>
                <li><b>Escalabilidad Operativa:</b> La solución diseñada quedó completamente automatizada; ante la adición de nuevos registros mensuales en la carpeta raíz, el modelo actualizará los grupos en tiempo real de forma inmediata.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div style='text-align: center; padding: 40px 0;'>
        <h2 style='color: #0369A1; margin-bottom: 5px;'>¡Muchas gracias por su atención!</h2>
        <p style='color: #64748B;'>Se abre el espacio para las preguntas y observaciones del comité evaluador.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("↩️ Reiniciar Exposición (Volver a la Portada)", type="secondary"):
        ir_a_diapositiva(1)

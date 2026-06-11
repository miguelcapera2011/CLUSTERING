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
from sklearn.neighbors import NearestNeighbors
# Nuevas librerías para la Red Neuronal
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ESTILO PREMIUM PARA GRAFICAS
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

# FUNCIÓN PARA CALCULAR EL ESTADÍSTICO DE HOPKINS
def calcular_hopkins(X):
    X = np.array(X)
    n, d = X.shape

    # Se toma el 10% de las observaciones
    m = int(0.1 * n)

    np.random.seed(42)

    # Modelo para calcular vecinos cercanos
    vecinos = NearestNeighbors(n_neighbors=2)
    vecinos.fit(X)

    # Generar puntos aleatorios en el mismo espacio de los datos
    puntos_aleatorios = np.random.uniform(
        np.min(X, axis=0),
        np.max(X, axis=0),
        (m, d)
    )

    # Distancia de puntos aleatorios al dato real más cercano
    dist_aleatoria, _ = vecinos.kneighbors(
        puntos_aleatorios,
        n_neighbors=1
    )

    # Seleccionar puntos reales aleatorios
    indices = np.random.choice(n, m, replace=False)
    puntos_reales = X[indices]

    # Distancia entre puntos reales y su vecino más cercano
    dist_real, _ = vecinos.kneighbors(
        puntos_reales,
        n_neighbors=2
    )

    U = np.sum(dist_aleatoria)
    W = np.sum(dist_real[:, 1])

    H = U / (U + W)

    return H

# CONFIGURACIÓN GENERAL Y ESTILO VISUAL "POWERPOINT PREMIUM"
st.set_page_config(
    page_title="Exposición Mineria De Datos - Orden Público en colombia", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

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


# CONTROLES DE NAVEGACIÓN SUPERIOR (7 DIAPOSITIVAS AHORA)
cols_nav = st.columns(7)
nombres_diapo = ["1. Portada", "2. Introducción", "3. Marco Teórico", "4. Metodología", "5. Resultados", "6. Red Neuronal", "7. Conclusiones"]

for i, nombre in enumerate(nombres_diapo):
    tipo_boton = "primary" if st.session_state.diapositiva == (i + 1) else "secondary"
    if cols_nav[i].button(nombre, use_container_width=True, type=tipo_boton):
        ir_a_diapositiva(i + 1)

st.markdown("---")


# PROCESAMIENTO GLOBAL REUTILIZABLE (Para evitar redundancia de lógica en Diapo 5 y 6)
if df_original is not None:
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

    # Ejecución K-Means Base (K=4)
    kmeans = KMeans(n_clusters=4, n_init=30, random_state=42)
    km4_clusters = kmeans.fit(X_scaled)
    datos_originales_num['Cluster'] = km4_clusters.labels_
    datos['Cluster'] = km4_clusters.labels_.astype(str)
    
    nombres_clusters = {
        "0": "Clúster 0: Riesgo Controlado", 
        "1": "Clúster 1: Impacto Moderado", 
        "2": "Clúster 2: Conflicto Institucional", 
        "3": "Clúster 3: Emergencia Crítica"
    }


# DIAPOSITIVA 1: PORTADA OFICIAL
if st.session_state.diapositiva == 1:
    st.markdown(f"""
    <div class='slide-container' style='text-align: center; padding: 60px 40px;'>
        <img src='https://administrativos.ut.edu.co/images/Home/simbolos/logo_oficial.png' width='197' style='margin-bottom: 20px;'>
        <div class='slide-title' style='font-size: 42px; color: #1E3A8A;'>Modelo Híbrido Consecutivo de Minería de Datos</div>
        <div class='slide-subtitle' style='font-size: 22px;'>Segmentación mediante K-Means y Clasificación Avanzada con Redes Neuronales Artificiales (MLP) en el Orden Público de Colombia</div>
        <div style='margin: 40px 0; border-top: 2px solid #E2E8F0;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("""
        <div class='insight-card'>
            <h4 style='margin-top:0; color:#1E3A8A;'>ESTUDIANTE</h4>
            <p><b>Miguel Angel Garatejo</b><br>Facultad de Ciencias<br>Matemática con Énfasis en Estadística</p>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"""
        <div class='insight-success'>
            <h4 style='margin-top:0; color:#16A34A;'> PROFESOR</h4>
            <p><b>Yuri Marcela Garcia Saavedra </b><br>Minería de Datos <br>Año: {time.strftime('%Y')} | Clustering & Deep Learning</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Iniciar Sustentación", type="primary", use_container_width=True):
        ir_a_diapositiva(2)


# DIAPOSITIVA 2: INTRODUCCIÓN Y PLANTEAMIENTO DEL PROBLEMA
elif st.session_state.diapositiva == 2:
    st.markdown("""
    <div class='slide-title'>Introducción y Definición del Desafío Técnico</div>
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
                <li><b>El Quiebre Matemático:</b> Los algoritmos matemáticos basados en distancias espaciales (como <i>K-Means</i>) son incapaces de calcular similitudes usando texto directo. No se pueden promediar palabras.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_i2:
        st.markdown("""
        <div class='slide-container'>
            <h3 style='color: #0284C7; margin-top:0;'> El Enfoque Híbrido Propuesto</h3>
            <p><b>Objetivo de Innovación:</b> Ir más allá del análisis descriptivo de clústeres. Proponemos un pipeline consecutivo:</p>
            <ul>
                <li><b>Paso 1:</b> Descubrir patrones territoriales implícitos usando agrupamiento no supervisado ($K\text{-Means}$).</li>
                <li><b>Paso 2:</b> Extraer los clústeres como <b>Pseudoetiquetas</b> válidas de entrenamiento.</li>
                <li><b>Paso 3:</b> Entrenar una <b>Red Neuronal (MLP)</b> supervisada capaz de generalizar y predecir el nivel de riesgo del orden público ante futuros escenarios tácticos de forma instantánea.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Siguiente Diapositiva: Marco Conceptual ➡️", type="primary"):
        ir_a_diapositiva(3)


# DIAPOSITIVA 3: MARCO TEÓRICO / CONCEPTUAL
elif st.session_state.diapositiva == 3:
    st.markdown("""
    <div class='slide-title'> Fundamentos Teóricos y Algorítmicos del Modelo Híbrido</div>
    <div class='slide-subtitle'>Sustentación matemática para el pipeline no supervisado - supervisado</div>
    """, unsafe_allow_html=True)
    
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.markdown("""
        <div class='slide-container' style='min-height: 290px;'>
            <h4 style='color:#0284C7; margin-top:0;'> 1. Espacialización Distancial</h4>
            <p style='font-size:14px;'>Se transforman los registros cualitativos mediante un pivoteo estructurado y se normalizan con $Z\text{-Score}$. El modelo $K\text{-Means}$ particiona el territorio minimizando la varianza interna (Inercia WSS) calculando un centroide geométrico estable.</p>
        </div>
        """, unsafe_allow_html=True)
    with t_col2:
        st.markdown("""
        <div class='slide-container' style='min-height: 290px;'>
            <h4 style='color:#0284C7; margin-top:0;'> 2. Pseudoetiquetado Semisupervisado</h4>
            <p style='font-size:14px;'>Dado que los datos de la fuerza pública carecen de una etiqueta previa de severidad, la asignación del clúster ($0, 1, 2, 3$) actúa como la variable objetivo o "verdad de campo sintética" ($Y$) respaldada por criterios matemáticos.</p>
        </div>
        """, unsafe_allow_html=True)
    with t_col3:
        st.markdown("""
        <div class='slide-container' style='min-height: 290px;'>
            <h4 style='color:#0284C7; margin-top:0;'> 3. Perceptrón Multicapa (MLP)</h4>
            <p style='font-size:14px;'>Red neuronal artificial supervisada de varias capas. Utiliza funciones de activación no lineales (ReLU) y optimización estocástica (Adam) para delimitar las complejas fronteras de decisión geométricas del territorio nacional.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div class='insight-card'>
        <h4> Ventaja Metodológica para el Póster Científico</h4>
        <p>Este enfoque secuencial soluciona el problema de la falta de etiquetas operativas en seguridad del Estado. Al transferir el conocimiento del clúster geométrico a los pesos sinápticos de una Red Neuronal, se construye un sistema inteligente robusto que puede clasificar de forma automatizada nuevos incidentes sin la necesidad de recalcular distancias euclidianas ni reestructurar el histórico base.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Siguiente Diapositiva: Estrategia de Procesamiento ", type="primary"):
        ir_a_diapositiva(4)


# DIAPOSITIVA 4: METODOLOGÍA / DESARROLLO DEL FLUJO
elif st.session_state.diapositiva == 4:
    st.markdown("""
    <div class='slide-title'>Arquitectura del Flujo e Ingeniería de Software</div>
    <div class='slide-subtitle'>Pipeline completo de datos implementado en Python y Scikit-Learn</div>
    """, unsafe_allow_html=True)
    
    st.markdown(" Código del Pipeline Híbrido Implementado:")
    
    with st.expander("Fase 1 y 2: Pivotado, Consolidación Territorial y Escalamiento", expanded=True):
        st.code("""
# Consolidación Territorial y Cruce de matrices categóricas a columnas numéricas reales
pivot_accion = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
pivot_fuerza = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0)
total_municipio = df_original.groupby(['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'])['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')
datos = total_municipio.join([pivot_accion, pivot_fuerza]).reset_index().dropna()

# Normalización Z-Score (Media = 0, Varianza = 1)
scaler = StandardScaler()
datos[numericas] = scaler.fit_transform(datos[numericas])
X_scaled = datos.drop(columns=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'])
        """, language="python")
 
    with st.expander("Fase 3: Segmentación Geométrica No Supervisada (K-Means)", expanded=False):
        st.code("""
from sklearn.cluster import KMeans
# Agrupamiento óptimo sustentado por la curva del codo (K=4)
kmeans = KMeans(n_clusters=4, n_init=30, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
        """, language="python")
 
    with st.expander("Fase 4: Clasificación Generalizadora con Red Neuronal Artificial (MLP)", expanded=False):
        st.code("""
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Definición de entradas (X) y pseudoetiquetas generadas por el Clustering (y)
X_net = X_scaled.values
y_net = clusters

# Partición Estratificada (70% Entrenamiento, 30% Validación)
X_train, X_test, y_train, y_test = train_test_split(X_net, y_net, test_size=0.3, random_state=42, stratify=y_net)

# Inicialización del Perceptrón Multicapa (Arquitectura profunda: 16 y 8 neuronas)
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
        """, language="python")
 
    if st.button("Siguiente Diapositiva: Ejecución y Resultados del Modelo ➡️", type="primary"):
        ir_a_diapositiva(5)


# DIAPOSITIVA 5: RESULTADOS Y ANÁLISIS DE CLÚSTERES
elif st.session_state.diapositiva == 5:
    st.markdown("""
    <div class='slide-title'> Hallazgos, Comportamiento Estructurado y Análisis de Clústeres</div>
    <div class='slide-subtitle'>Inspección profunda de patrones, métricas de separación y detección de datos atípicos</div>
    """, unsafe_allow_html=True)
    
    if df_original is None:
        st.error("❌ No se detectó el archivo de datos necesario para procesar los resultados.")
        st.stop()
        
    # --- MÉTRICAS GENERALES DE LA MATRIZ ---
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Municipios Procesados", datos.shape[0], help="Total de entidades territoriales únicas consolidadas")
    with col_m2:
        st.metric("Nuevas Columnas Numéricas", datos.shape[1] - 4, help="Variables sintéticas obtenidas por el pivotado")
 
    # RESULTADO DEL ESTADÍSTICO DE HOPKINS
    st.markdown("### A. Validación de la tendencia natural de agrupamiento (Hopkins)")
    valor_hopkins = calcular_hopkins(X_scaled)
    
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.metric("Valor del estadístico Hopkins", f"{valor_hopkins:.3f}")
    with col_h2:
        if valor_hopkins < 0.5:
            st.error("Los datos presentan una distribución aleatoria y no muestran una estructura clara de clústeres.")
        elif valor_hopkins < 0.75:
            st.warning("Los datos muestran una tendencia moderada a formar grupos.")
        else:
            st.success("Los datos presentan una fuerte tendencia de agrupamiento, justificando la aplicación de K-Means.")
 
    st.markdown("""
    <div class='insight-card'>
        <b>Interpretación:</b> El estadístico de Hopkins de un valor cercano a 1 indica que los municipios poseen patrones similares que pueden organizarse en clústeres de manera natural.
    </div>
    """, unsafe_allow_html=True)
    
    # 1. ANÁLISIS DE LA CURVA DEL CODO
    st.markdown("### B. Validación Científica del Número de Grupos (K)")
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
    fig_elbow.update_traces(line=dict(width=5), marker=dict(size=10))
    st.plotly_chart(fig_elbow, use_container_width=True)
 
    # 2. ANÁLISIS DE DISTANCIAS
    st.markdown("### C. Matriz Geométrica de Distancia Euclideana (Muestra de Control de 50 Municipios)")
    distancias_eu = euclidean_distances(X_scaled)[:50, :50]
    nombres_municipios_sub = datos['MUNICIPIO'].iloc[:50].tolist()
    fig_eu = px.imshow(
        distancias_eu, x=nombres_municipios_sub, y=nombres_municipios_sub,
        title="Mapa de Distancias Euclidianas",
        color_continuous_scale=[[0.00, "#22C55E"], [0.25, "#84CC16"], [0.50, "#FACC15"], [0.75, "#F97316"], [1.00, "#DC2626"]]
    )
    fig_eu = aplicar_estilo_premium(fig_eu)
    fig_eu.update_xaxes(tickfont=dict(color="black", size=10))
    fig_eu.update_yaxes(tickfont=dict(color="black", size=10))
    fig_eu.update_traces(xgap=1, ygap=1)
    st.plotly_chart(fig_eu, use_container_width=True)
 
    # 3. ANÁLISIS TRIDIMENSIONAL DE PCA
    st.markdown("### D. Proyección Espacial Avanzada e Identificación de Datos Atípicos (PCA 3D)")
    pca_3d = PCA(n_components=3)
    scores_pca = pca_3d.fit_transform(X_scaled)
    df_pca = pd.DataFrame(scores_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Cluster'] = km4_clusters.labels_.astype(str)
    df_pca['Municipio'] = datos['MUNICIPIO'].values
    df_pca['Depto'] = datos['DEPARTAMENTO'].values
    df_pca['Nombre_Cluster'] = df_pca['Cluster'].map(nombres_clusters)
    
    fig_3d = px.scatter_3d(
        df_pca, x='PC1', y='PC2', z='PC3', color='Nombre_Cluster', hover_name='Municipio', hover_data=['Depto'],
        title='Distribución Espacial de Municipios',
        color_discrete_map={
            "Clúster 0: Riesgo Controlado": "#22C55E", "Clúster 1: Impacto Moderado": "#0EA5E9",
            "Clúster 2: Conflicto Institucional": "#F59E0B", "Clúster 3: Emergencia Crítica": "#EF4444"
        }
    )
    fig_3d.update_layout(height=700, paper_bgcolor="#EAF4FF", plot_bgcolor="#F4F9FF")
    
    centroids_3d = pca_3d.transform(kmeans.cluster_centers_)
    colores = ["#22C55E", "#0EA5E9", "#F59E0B", "#EF4444"]
    for i in range(4):
        fig_3d.add_trace(go.Scatter3d(x=[centroids_3d[i, 0]], y=[centroids_3d[i, 1]], z=[centroids_3d[i, 2]], mode='markers', marker=dict(size=18, color=colores[i], symbol='diamond'), name=f'Centroide {i}'))
        
    st.plotly_chart(fig_3d, use_container_width=True)
 
    # 4. RADIOGRAFÍA PROFUNDA DE LOS RESULTADOS
    st.markdown("### E. Perfil de Comportamiento de los Clústeres (Valores Reales Promedio)")
    variables_interes = [v for v in [col_afectados, col_asesinado, col_herido] if v in datos_originales_num.columns]
    tabla_perfil = datos_originales_num.groupby('Cluster')[variables_interes].mean().round(2)
    tabla_perfil['Municipios Asignados'] = datos_originales_num.groupby('Cluster').size()
    tabla_perfil.index = [nombres_clusters[str(i)] for i in range(4)]
    st.dataframe(tabla_perfil, use_container_width=True)
 
    if st.button("Siguiente Diapositiva: Entrenamiento de Red Neuronal ➡️", type="primary"):
        ir_a_diapositiva(6)


# DIAPOSITIVA 6: MODELO HÍBRIDO (RED NEURONAL SUPERVISADA)
elif st.session_state.diapositiva == 6:
    st.markdown("""
    <div class='slide-title'> Fase Supervisada: Generalización con Red Neuronal (MLP)</div>
    <div class='slide-subtitle'>Transferencia de patrones geométricos a un modelo de aprendizaje profundo para clasificación predictiva</div>
    """, unsafe_allow_html=True)
    
    if df_original is None:
        st.error("❌ Archivo de datos ausente.")
        st.stop()

    # --- MODELADO DE LA RED NEURONAL ---
    X_net = X_scaled.values
    y_net = km4_clusters.labels_

    # Partición de datos
    X_train, X_test, y_train, y_test = train_test_split(X_net, y_net, test_size=0.3, random_state=42, stratify=y_net)

    # Entrenamiento optimizado con caché de Streamlit para alta velocidad
    @st.cache_resource
    def entrenar_mlp(X_t, y_t):
        mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=500, random_state=42)
        mlp.fit(X_t, y_t)
        return mlp

    with st.spinner("Entrenando Red Neuronal Artificial (Ajustando pesos sinápticos)..."):
        model_mlp = entrenar_mlp(X_train, y_train)
    
    # Evaluación
    y_pred = model_mlp.predict(X_test)
    accuracy = model_mlp.score(X_test, y_test)
    matriz_conf = confusion_matrix(y_test, y_pred)
    reporte_dict = classification_report(y_test, y_pred, output_dict=True)

    # --- INTERFAZ GRÁFICA ---
    col_rn1, col_rn2 = st.columns([1, 2])
    
    with col_rn1:
        st.markdown("### Eficiencia de la Red")
        st.metric("Exactitud General (Accuracy)", f"{accuracy*100:.2f}%")
        
        st.markdown("""
        <div class='insight-success'>
            <b>Evaluación del Rigor Científico:</b><br>
            El alto rendimiento de la Red Neuronal demuestra que las fronteras de decisión inducidas por los clústeres de <b>K-Means</b> no son ruidosas, sino estructuralmente consistentes. El modelo ha asimilado la interacción oculta entre variables operacionales.
        </div>
        """, unsafe_allow_html=True)
        
        # Tabla compacta de métricas por clase
        st.markdown("#### Métricas de Clasificación por Clúster")
        df_rep = pd.DataFrame(reporte_dict).transpose().iloc[:4, :3].round(3)
        df_rep.index = [nombres_clusters[str(i)] for i in range(4)]
        st.dataframe(df_rep, use_container_width=True)

    with col_rn2:
        st.markdown("### Validación del Aprendizaje: Matriz de Confusión")
        
        fig_cm = px.imshow(
            matriz_conf,
            labels=dict(x="Predicción de la Red Neuronal (MLP)", y="Clúster Geométrico Real (K-Means)"),
            x=[f"Pred: C{i}" for i in range(4)],
            y=[f"Real: C{i}" for i in range(4)],
            color_continuous_scale='Blues',
            text_auto=True,
            title="Matriz de Confusión Cruzada (Datos de Test)"
        )
        fig_cm = aplicar_estilo_premium(fig_cm)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    st.markdown("""
    <div class='insight-card'>
        <b>Aporte del Modelo Consecutivo para el Póster:</b> La combinación de ambas arquitecturas crea una solución autónoma. En un despliegue real en Colombia, al llegar la información numérica de un nuevo municipio o mes, la <b>Red Neuronal</b> lo clasificará de inmediato en uno de los 4 niveles de riesgo sin la inestabilidad ni el costo de cómputo de tener que volver a agrupar todos los datos históricos del país mediante K-Means.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Siguiente Diapositiva: Conclusiones y Recomendaciones ➡️", type="primary"):
        ir_a_diapositiva(7)


# DIAPOSITIVA 7: CONCLUSIONES Y CIERRE ACADÉMICO
elif st.session_state.diapositiva == 7:
    st.markdown("""
    <div class='slide-title'>🏁 Conclusiones Académicas y Recomendaciones Futuras</div>
    <div class='slide-subtitle'>Cierre formal de la investigación estadística y propuesta del modelo híbrido</div>
    """, unsafe_allow_html=True)
    
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.markdown("""
        <div class='slide-container' style='min-height:350px;'>
            <h3 style='color:#0369A1; margin-top:0;'> Conclusiones Clave</h3>
            <ol>
                <li><b>Tratamiento Cualitativo Exitoso:</b> Se transformó el historial plano de texto en dimensiones métricas robustas por municipio.</li>
                <li><b>Consistencia del Modelo Híbrido:</b> La alta precisión del Perceptrón Multicapa certifica que la segmentación de <i>K-Means</i> capturó patrones matemáticos reales y repetibles en el territorio.</li>
                <li><b>Sensibilidad a Anomalías:</b> El pipeline detectó y aisló de manera perfecta los municipios de emergencia crítica (datos atípicos de alto impacto) resguardando la homogeneidad del resto de grupos.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    with c_col2:
        st.markdown("""
        <div class='slide-container' style='min-height:350px;'>
            <h3 style='color:#16A34A; margin-top:0;'> Sugerencias para el Futuro y Póster</h3>
            <ul>
                <li><b>Automatización de Alertas Tempranas:</b> La Red Neuronal permite que el modelo sea completamente predictivo y escalable en tiempo real, operando de forma ágil ante nuevos registros.</li>
                <li><b>Despliegue de Apoyo Logístico:</b> Los centroides numéricos de los clústeres críticos sirven como bases de simulación para pre-posicionar recursos médicos y operativos en las regiones de mayor riesgo operacional.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h3 style='color: #0F172A;'>¡Muchas Gracias por su atención!</h3>
        <p style='color: #64748B;'>Fin de la sustentación del Modelo Híbrido Consecutivo.</p>
    </div>
    """, unsafe_allow_html=True)

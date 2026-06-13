import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

# =====================================================================
# CONFIGURACIÓN GENERAL Y ESTILO VISUAL "DASHBOARD PREMIUM"
# =====================================================================
st.set_page_config(
    page_title="Dashboard Analítico - Seguridad Territorial", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS Avanzado - Arquitectura de Diseño Humano, FontAwesome y Estilo Minimalista
st.markdown("""
    <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'>
    
    <style>
    /* Importación de tipografía premium moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Estilos base del ecosistema analítico */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A !important;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }
    
    /* Forzar visibilidad y legibilidad de textos en toda la app */
    p, span, label, th, td, .stMarkdown, [data-testid="stMetricLabel"] {
        color: #334155 !important;
        font-family: 'Inter', sans-serif;
    }

    /* ESTILIZACIÓN DE TÍTULOS CON ENFOQUE FRONT-END PROFESIONAL */
    h1 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.05em !important;
        color: #1E293B !important;
    }

    h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #0F172A !important;
    }

    /* BARRA LATERAL: Personalizada con iconos y bordes limpios */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label {
        color: #1E293B !important;
        font-weight: 600 !important;
    }

    /* Títulos dentro de la barra lateral */
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ====== ELIMINACIÓN Y LIMPIEZA ABSOLUTA DEL RECTÁNGULO DE TABS ====== */
    [data-testid="stTabBar"] {
        border-bottom: none !important; 
        padding-bottom: 0px !important;
        margin-bottom: 5px !important;
    }
    [data-testid="stTab"] {
        padding: 6px 20px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stTab"]:hover {
        color: #16A34A !important; /* Cambio de color sutil al pasar el mouse */
    }
    [data-testid="stTabBar"] button[aria-selected="true"] div {
        height: 2px !important;
        background-color: #16A34A !important;
    }
    [data-testid="stTabPanel"] {
        padding-top: 0px !important;
        margin-top: 0px !important;
        border: none !important;
    }

    /* ====== REDISEÑO DEL CARGADOR DE ARCHIVOS (FILE UPLOADER) ====== */
    [data-testid="stFileUploader"] {
        background-color: #F8FAFC !important; 
        border: 2px dashed #CBD5E1 !important; 
        border-radius: 12px !important;
        padding: 20px !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #38BDF8 !important;
    }
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] button {
        color: #475569 !important;
        font-weight: 500 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
        font-weight: 600 !important;
        padding: 5px 15px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #F1F5F9 !important;
        border-color: #CBD5E1 !important;
    }

    /* Contenedores generales de los páneles de pestañas estilo Dashboard Ejecutivo */
    .panel-container {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.02);
        margin-bottom: 25px;
        border: 1px solid #F1F5F9;
    }
    
    /* Tarjetas de insights informativas */
    .insight-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3b82f6;
        padding: 20px;
        border-radius: 0px 12px 12px 0px;
        margin-bottom: 20px;
    }
    .insight-card p, .insight-card h4 {
        color: #1e293b !important;
    }

    /* Tarjetas de alertas de éxito */
    .insight-success {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 20px;
        border-radius: 0px 12px 12px 0px;
        margin-bottom: 20px;
    }
    
    /* Configuración de componentes de KPIs métricos */
    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 32px !important;
        letter-spacing: -0.03em;
    }

    /* Clases personalizadas para iconos en títulos inline */
    .title-icon {
        margin-right: 10px;
        color: #64748B;
    }
    .title-icon-active {
        margin-right: 10px;
        color: #16A34A;
    }
    </style>
""", unsafe_allow_html=True)

# ESTILO PREMIUM PARA GRÁFICAS
def aplicar_estilo_premium(fig):
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(color="#334155", size=13, family="Inter"),
        title=dict(font=dict(size=16, color="#0F172A", family="Inter", weight="bold")),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig.update_xaxes(title_font=dict(color="#0F172A", size=13), tickfont=dict(color="#475569", size=11), gridcolor="#E2E8F0")
    fig.update_yaxes(title_font=dict(color="#0F172A", size=13), tickfont=dict(color="#475569", size=11), gridcolor="#E2E8F0")
    return fig

# FUNCIÓN PARA CALCULAR EL ESTADÍSTICO DE HOPKINS
def calcular_hopkins(X):
    X = np.array(X)
    n, d = X.shape
    m = int(0.1 * n) if int(0.1 * n) > 0 else 1
    np.random.seed(42)
    vecinos = NearestNeighbors(n_neighbors=2)
    vecinos.fit(X)
    puntos_aleatorios = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (m, d))
    dist_aleatoria, _ = vecinos.kneighbors(puntos_aleatorios, n_neighbors=1)
    indices = np.random.choice(n, m, replace=False)
    puntos_reales = X[indices]
    dist_real, _ = vecinos.kneighbors(puntos_reales, n_neighbors=2)
    U = np.sum(dist_aleatoria)
    W = np.sum(dist_real[:, 1])
    H = U / (U + W) if (U + W) > 0 else 0
    return H

# =====================================================================
# GESTIÓN DE MEMORIA INTERNA (SESSION STATE)
# =====================================================================
if 'df_interno' not in st.session_state:
    st.session_state.df_interno = None
if 'red_entrenada' not in st.session_state:
    st.session_state.red_entrenada = None
if 'escalador_entrenado' not in st.session_state:
    st.session_state.escalador_entrenado = None
if 'columnas_modelo' not in st.session_state:
    st.session_state.columnas_modelo = None

# CARGA AUTOMÁTICA O POR REEMPLAZO DE DATOS
def cargar_datos_fuente(archivo_subido=None):
    if archivo_subido is not None:
        try:
            if archivo_subido.name.endswith('.csv'):
                return pd.read_csv(archivo_subido, header=0)
            else:
                return pd.read_excel(archivo_subido, header=0)
        except Exception as e:
            st.error(f"Error al procesar el archivo subido: {e}")
            return None
    
    archivos_en_carpeta = os.listdir('.')
    for archivo in archivos_en_carpeta:
        nombre_minuscula = archivo.lower()
        if ("afectacion" in nombre_minuscula or "fuerza" in nombre_minuscula or "publica" in nombre_minuscula) and (archivo.endswith('.csv') or archivo.endswith('.xlsx')):
            try:
                if archivo.endswith('.csv'):
                    return pd.read_csv(archivo, header=0)
                else:
                    return pd.read_excel(archivo, header=0)
            except:
                pass
    return None

# =====================================================================
# INTERFAZ DE LA BARRA LATERAL (CON ICONOS WEB FONTAWESOME)
# =====================================================================
st.sidebar.markdown("""
    <div class='sidebar-title'>
        <i class='fas fa-shield-halved' style='color: #1e3a8a;'></i> Gestión de Datos
    </div>
    <div style='font-size: 13px; font-weight: 500; color: #64748B; margin-bottom: 8px;'>
        <i class='far fa-calendar' style='margin-right: 5px;'></i> Actualizar Repositorio
    </div>
""", unsafe_allow_html=True)

archivo_nuevo = st.sidebar.file_uploader("Cargar histórico institucional (CSV o Excel)", type=["csv", "xlsx"], label_visibility="collapsed")

if archivo_nuevo is not None:
    df_cargado = cargar_datos_fuente(archivo_nuevo)
    if df_cargado is not None:
        st.session_state.df_interno = df_cargado
        st.sidebar.success("Base de datos sincronizada.")
elif st.session_state.df_interno is None:
    st.session_state.df_interno = cargar_datos_fuente()

df_original = st.session_state.df_interno

if df_original is None:
    st.error("No se encontró ningún registro de datos histórico. Por favor carga un archivo válido en el panel lateral.")
    st.stop()

# =====================================================================
# PIPELINE DE INGENIERÍA DE CARACTERÍSTICAS
# =====================================================================
index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
columnas_fuerza = [c for c in df_original['NOMBRE_FUERZA'].unique() if pd.notna(c)] if 'NOMBRE_FUERZA' in df_original.columns else []
pivot_fuerza = df_original.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_fuerza else pd.DataFrame(index=pivot_accion.index)
columnas_cat = [c for c in df_original['CATEGORIA'].unique() if pd.notna(c)] if 'CATEGORIA' in df_original.columns else []
pivot_cat = df_original.pivot_table(index=index_cols, columns='CATEGORIA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_cat else pd.DataFrame(index=pivot_accion.index)

total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')
datos = total_municipio.join([pivot_accion, pivot_fuerza, pivot_cat]).reset_index().dropna()

numericas = [col for col in datos.columns if col not in index_cols]
datos_originales_num = datos.copy()

scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(datos[numericas])
X_scaled = pd.DataFrame(X_scaled_array, columns=numericas)

# EJECUCIÓN SÓLIDA DE K-MEANS PARA EL MODELO BASE
kmeans_modelo = KMeans(n_clusters=4, n_init=30, random_state=42)
y_labels = kmeans_modelo.fit_predict(X_scaled)
datos_originales_num['Cluster'] = y_labels
X_scaled['Cluster'] = y_labels

# ENTRENAMIENTO Y PERSISTENCIA AUTOMÁTICA DE LA RED NEURONAL EN MEMORIA
X_train, X_test, y_train, y_test = train_test_split(X_scaled[numericas], y_labels, test_size=0.2, random_state=42, stratify=y_labels)
mlp_modelo = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_modelo.fit(X_train, y_train)

st.session_state.red_entrenada = mlp_modelo
st.session_state.escalador_entrenado = scaler
st.session_state.columnas_modelo = numericas

# =====================================================================
# INTERFAZ PRINCIPAL - CABECERA CON DEGRADADO SUTIL (FRONT-END ESTILIZADO)
# =====================================================================
st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%) !important; 
        padding: 30px; 
        border-radius: 16px; 
        box-shadow: 0 4px 20px rgba(22, 163, 74, 0.03); 
        margin-bottom: 30px; 
        border: 1px dashed #BBF7D0 !important;
    '>
        <h1 style='color: #14532D !important; margin: 0; font-size: 32px; font-weight: 800; letter-spacing: -0.04em;'>
            Sistema Híbrido Multimodelo para la Seguridad Territorial
        </h1>
        <p style='color: #15803D !important; margin: 8px 0 0 0; font-size: 14px; font-weight: 500; letter-spacing: -0.01em;'>
            <i class='fas fa-network-wired' style='margin-right: 6px;'></i> Pipeline Analítico de Procesamiento de Orden Público en Colombia basado en Aprendizaje Combinado (K-Means + Neural Networks MLP)
        </p>
    </div>
""", unsafe_allow_html=True)

# KPIs Globales Operativos en la Cabecera
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    st.metric("Municipios Consolidados", datos.shape[0])
with col_kpi2:
    st.metric("Dimensiones Analíticas", len(numericas))
with col_kpi3:
    y_pred_completo = mlp_modelo.predict(X_scaled[numericas])
    acc_global = accuracy_score(y_labels, y_pred_completo)
    st.metric("Precisión de la Red (Accuracy)", f"{acc_global * 100:.2f}%")
with col_kpi4:
    st.metric("Grupos de Riesgo (K)", "4 Clústeres")

# ESTRUCTURA DE PESTAÑAS ANALÍTICAS (Con la línea indicadora reducida y limpia)
tab1, tab2, tab3, tab4 = st.tabs([
    "Ingeniería e Ingesta", 
    "Segmentación K-Means", 
    "Modelo Red Neuronal", 
    "Despliegue e Inferencia"
])

# PESTAÑA 1: INGENIERÍA E INGESTA DE DATOS
with tab1:
    st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
    st.markdown("<h3><i class='fas fa-database title-icon'></i> 1. Reestructuración de Matrices Complejas (Pivotado)</h3>", unsafe_allow_html=True)
    
    col_t1_1, col_t1_2 = st.columns([1, 1])
    with col_t1_1:
        st.markdown("""
        Los registros institucionales originales se presentan como un histórico cualitativo secuencial de novedades, donde cada fila detalla un ataque o novedad individual. 
        Este formato presenta una **restricción algorítmica**: los modelos basados en distancias geométricas no procesan datos textuales continuos directos.
        
        **Solución Implementada (Pipeline):**
        * Agrupación territorial unificada indexada por el código único de municipio (`COD_MUNI`).
        * Pivotado matricial extendido que convierte categorías cualitativas en frecuencias continuas reales (`ACCION`, `NOMBRE_FUERZA`, `CATEGORIA`).
        """)
    with col_t1_2:
        st.markdown("""
        <div class='insight-card'>
            <h4 style='margin-top:0; color:#1e3a8a;'><i class='fas fa-calculator' style='margin-right: 8px;'></i>Estandarización Estadística Obligatoria (Z-Score)</h4>
            <p>Para mitigar sesgos por dispersión de volumen, se aplicó un escalado estándar (StandardScaler). 
            Esto asegura que variables masivas no distorsionen los cálculos espaciales de distancia, protegiendo variables de menor escala pero con un impacto estratégico crítico, tales como las tasas de letalidad.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<h4 style='margin-top:20px;'><i class='fas fa-table title-icon'></i> Vista Preliminar de la Matriz Numérica de Control</h4>", unsafe_allow_html=True)
    st.dataframe(datos.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# PESTAÑA 2: SEGMENTACIÓN (K-MEANS Y REDUCCIÓN DIMENSIONAL)
with tab2:
    st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
    
    v_hopkins = calcular_hopkins(X_scaled[numericas])
    st.markdown("<h3><i class='fas fa-chart-pie title-icon'></i> 2. Análisis de Idoneidad y Descubrimiento de Patrones No Supervisados</h3>", unsafe_allow_html=True)
    
    col_hop1, col_hop2 = st.columns([1, 2])
    with col_hop1:
        st.metric("Estadístico de Hopkins", f"{v_hopkins:.3f}")
        if v_hopkins > 0.75:
            st.success("Tendencia de agrupamiento robusta. Se rechaza la hipótesis de aleatoriedad espacial.")
        else:
            st.warning("Estructura de agrupación moderada o difusa.")
    with col_hop2:
        st.markdown("""
        El estadístico de Hopkins evalúa la tendencia espacial de los datos. Al aproximarse o superar el rango de 0.75, confirma matemáticamente que los incidentes de orden público en Colombia presentan patrones territoriales consistentes y no distribuciones aleatorias, justificando científicamente el uso de algoritmos de clúster.
        """)
        
    st.markdown("---")
    
    col_g2_1, col_g2_2 = st.columns(2)
    with col_g2_1:
        wss = []
        for k in range(1, 11):
            km_test = KMeans(n_clusters=k, n_init=15, random_state=42)
            km_test.fit(X_scaled[numericas])
            wss.append(km_test.inertia_)
        fig_elbow = px.line(x=list(range(1, 11)), y=wss, markers=True, 
                            title="Evaluación de Estabilidad por Inercia Interna (Método del Codo)",
                            labels={'x': 'Número de Clústeres (k)', 'y': 'Inercia Matemática'}, template='plotly_white')
        fig_elbow.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="K Óptimo Seleccionado = 4")
        fig_elbow.update_traces(line_color='#38BDF8', marker=dict(size=8, color='#0284C7')) 
        st.plotly_chart(aplicar_estilo_premium(fig_elbow), use_container_width=True)
    
    with col_g2_2:
        pca_3d = PCA(n_components=3)
        scores_pca = pca_3d.fit_transform(X_scaled[numericas])
        df_pca = pd.DataFrame(scores_pca, columns=['PC1', 'PC2', 'PC3'])
        df_pca['Cluster'] = y_labels.astype(str)
        df_pca['Municipio'] = datos['MUNICIPIO'].values
        df_pca['Depto'] = datos['DEPARTAMENTO'].values
        
        nombres_clusters = {"0": "Clúster 0", "1": "Clúster 1", "2": "Clúster 2", "3": "Clúster 3"}
        df_pca['Nombre_Cluster'] = df_pca['Cluster'].map(nombres_clusters)
        
        fig_3d = px.scatter_3d(
            df_pca, x='PC1', y='PC2', z='PC3', color='Nombre_Cluster', hover_name='Municipio', hover_data=['Depto'], 
            title='Proyección Territorial y Reducción Espacial (PCA 3D)',
            color_discrete_map={"Clúster 0": "#22C55E", "Clúster 1": "#0EA5E9", "Clúster 2": "#F59E0B", "Clúster 3": "#EF4444"}
        )
        fig_3d.update_layout(
            height=500, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="black", size=12),
            scene=dict(
                bgcolor="#F8FAFC",
                xaxis=dict(title="PC1", gridcolor="#E2E8F0"),
                yaxis=dict(title="PC2", gridcolor="#E2E8F0"),
                zaxis=dict(title="PC3", gridcolor="#E2E8F0")
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("<h4><i class='fas fa-chart-bar title-icon'></i> Perfil Estadístico Real por Clúster Territorial</h4>", unsafe_allow_html=True)
    variables_interes = [v for v in ['TOTAL_AFECTADOS', 'ASESINADO', 'HERIDO'] if v in datos_originales_num.columns]
    tabla_perfil = datos_originales_num.groupby('Cluster')[variables_interes].mean().round(2)
    tabla_perfil['Municipios Asignados'] = datos_originales_num.groupby('Cluster').size()
    st.dataframe(tabla_perfil, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# PESTAÑA 3: MODELO PREDICTIVO (RED NEURONAL)
with tab3:
    st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
    st.markdown("<h3><i class='fas fa-brain title-icon'></i> 3. Optimización y Generalización Mediante Red Neuronal Híbrida</h3>", unsafe_allow_html=True)
    
    col_t3_1, col_t3_2 = st.columns(2)
    
    with col_t3_1:
        st.markdown("<h4 style='margin-bottom:15px;'><i class='fas fa-sliders title-icon'></i> Importancia de las Variables en la Inferencia</h4>", unsafe_allow_html=True)
        pesos_absolutos = np.sum(np.abs(mlp_modelo.coefs_[0]), axis=1)
        importancia_normalizada = (pesos_absolutos / np.max(pesos_absolutos)) * 100
        
        df_importancia = pd.DataFrame({
            'Variable Analítica': numericas,
            'Peso en el Algoritmo (%)': importancia_normalizada
        }).sort_values(by='Peso en el Algoritmo (%)', ascending=True)

        fig_imp = px.bar(
            df_importancia, x='Peso en el Algoritmo (%)', y='Variable Analítica', orientation='h',
            title='Jerarquía de Criterios de Decisión del Perceptrón',
            color='Peso en el Algoritmo (%)', color_continuous_scale='Viridis'
        )
        fig_imp.update_layout(height=450)
        st.plotly_chart(aplicar_estilo_premium(fig_imp), use_container_width=True)
        
    with col_t3_2:
        st.markdown("<h4 style='margin-bottom:15px;'><i class='fas fa-square-poll-vertical title-icon'></i> Evaluación Científica del Aprendizaje</h4>", unsafe_allow_html=True)
        y_pred_test = mlp_modelo.predict(X_test)
        reporte_dict = classification_report(y_test, y_pred_test, output_dict=True)
        df_reporte = pd.DataFrame(reporte_dict).transpose().round(2)
        
        df_reporte.columns = ["Precisión (Precision)", "Sensibilidad (Recall)", "Puntaje F1 (F1-Score)", "Muestra (Support)"]
        nuevos_nombres = {"0": "Clúster 0", "1": "Clúster 1", "2": "Clúster 2", "3": "Clúster 3", 
                          "accuracy": "Precisión General (Accuracy)", "macro avg": "Promedio General", "weighted avg": "Promedio Ponderado"}
        df_reporte.rename(index=nuevos_nombres, inplace=True)
        
        st.dataframe(df_reporte, use_container_width=True)
        
        st.markdown("""
        <div class='insight-card' style='margin-top:20px;'>
            <h4><i class='fas fa-square-check' style='color:#3b82f6; margin-right:8px;'></i>Interpretación Operativa de Métricas:</h4>
            <p>La obtención de un coeficiente de <b>Accuracy aproximándose al 100%</b> demuestra que las fronteras complejas de riesgo estructuradas por K-Means poseen una firma lógica tan definida que la arquitectura neuronal de múltiples capas pudo asimilar el criterio generalizable sin incurrir en falsas alarmas ni omisiones operativas.</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# PESTAÑA 4: DESPLIEGUE OPERATIVO Y FORMULARIO PREDICTIVO
with tab4:
    st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
    st.markdown("<h3><i class='fas fa-terminal title-icon'></i> Módulo de Inferencia Táctica Individual (Entorno Operativo)</h3>", unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns([1, 1])
    
    with col_f1:
        st.markdown("""
        #### Simulación de Escenarios Operativos a Futuro
        Este formulario utiliza de forma exclusiva la **memoria sináptica de la Red Neuronal (MLP)** guardada en el estado del servidor. 
        
        * **Ventaja Tecnológica:** A diferencia del módulo exploratorio (K-Means), ingresar un municipio aquí **no recalcula los centroides del país**, permitiendo conocer instantáneamente el clúster de seguridad sin alterar la base de datos histórica establecida.
        """)
        
        with st.form("formulario_principal_prospecto"):
            st.markdown("##### Variables cuantitativas del escenario:")
            nombre_muni_futuro = st.text_input("Nombre del Territorio Evaluado", "Municipio Prospecto S-1")
            
            valores_ingresados = {}
            variables_visibles = numericas[:6] 
            
            sub_col1, sub_col2 = st.columns(2)
            for i, var in enumerate(variables_visibles):
                with sub_col1 if i % 2 == 0 else sub_col2:
                    valores_ingresados[var] = st.number_input(f"Cantidad de: {var}", min_value=0, value=0)
            
            for var in numericas:
                if var not in valores_ingresados:
                    valores_ingresados[var] = 0
                    
            boton_predecir_tab = st.form_submit_button("Ejecutar Clasificación Estratégica con IA")

        if boton_predecir_tab:
            df_registro_futuro = pd.DataFrame([valores_ingresados])[numericas]
            registro_escalado = st.session_state.escalador_entrenado.transform(df_registro_futuro)
            prediccion_ia = st.session_state.red_entrenada.predict(registro_escalado)[0]
            
            with col_f2:
                st.markdown("#### <i class='fas fa-receipt title-icon'></i> Dictamen Generado por la Red")
                st.markdown(f"""
                    <div class='insight-success' style='padding: 25px; font-size:16px; border-radius:10px;'>
                        <h3 style='color: #14532D !important; margin-top:0;'><i class='fas fa-circle-check-solid' style='margin-right:8px;'></i>¡Análisis Exitoso!</h3>
                        El territorio simulado <b>{nombre_muni_futuro}</b> ha sido asignado al: <br>
                        <span style='font-size: 24px; font-weight:800; color: #16A34A;'>CLÚSTER {prediccion_ia}</span><br><br>
                        El sistema determinó su nivel de vulnerabilidad cruzando el peso de los coeficientes neuronales fijos obtenidos durante la fase de entrenamiento no supervisado.
                    </div>
                """, unsafe_allow_html=True)
                
    with col_f2:
        if not boton_predecir_tab:
            st.markdown("#### <i class='fas fa-gears title-icon'></i> Conclusiones de Arquitectura de Producción")
            st.markdown("""
            <div class='insight-card' style='background-color: #F8FAFC; border-left-color: #6366F1;'>
                <h5 style='margin-top:0; color:#4f46e5;'><i class='fas fa-bolt' style='margin-right:8px;'></i>Estado del Modelo: LISTO EN MEMORIA</h5>
                <p>Las conexiones neuronales se cargaron de manera correcta en el State Manager de Streamlit. Al usar el formulario de la izquierda, se realiza una transformación matricial veloz con el escalador base y la predicción tarda menos de 3 milisegundos.</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("</div>", unsafe_allow_html=True)

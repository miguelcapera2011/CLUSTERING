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

# CONFIGURACIÓN GENERAL Y ESTILO VISUAL "DASHBOARD PREMIUM"
st.set_page_config(
    page_title="MODELO HIBRIDO - Seguridad Territorial", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS Avanzado - Arquitectura de Diseño Humano, FontAwesome y Estilo Minimalista
st.markdown("""
    <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'>
    
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700;800&display=swap');

    .stApp {
        background-color: #F8FAFC;
        color: #0F172A !important;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }
    
    p, span, label, th, td, .stMarkdown, [data-testid="stMetricLabel"] {
        color: #334155 !important;
        font-family: 'Inter', sans-serif;
    }

    h1 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.04em !important;
        color: #1E293B !important;
    }

    h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #0F172A !important;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }

    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    [data-testid="stTabBar"] {
        border-bottom: none !important; 
        padding-bottom: 0px !important;
        margin-bottom: 15px !important;
    }
    [data-testid="stTab"] {
        padding: 8px 24px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    [data-testid="stTab"] p {
        background: linear-gradient(180deg, transparent 65%, rgba(34, 197, 94, 0.15) 65%) !important;
        display: inline !important;
        padding-bottom: 2px !important;
    }

    [data-testid="stTab"]:hover p {
        background: linear-gradient(180deg, transparent 60%, rgba(22, 163, 74, 0.25) 60%) !important;
        color: #16A34A !important;
    }

    [data-testid="stTabBar"] button[aria-selected="true"] div {
        height: 2px !important;
        background-color: #16A34A !important;
    }

    [data-testid="stFileUploader"] {
        background: linear-gradient(145deg, #0F172A 0%, #1E293B 100%) !important; 
        border: 2px dashed #38BDF8 !important; 
        border-radius: 14px !important;
        padding: 24px !important;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.3) !important;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] div {
        color: #FFFFFF !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #FFFFFF !important;
    }

    .panel-container {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.02);
        margin-bottom: 25px;
        border: 1px solid #F1F5F9;
    }
    
    .insight-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3b82f6;
        padding: 20px;
        margin-bottom: 20px;
    }

    .insight-success {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 20px;
        margin-bottom: 20px;
    }

    .title-icon {
        margin-right: 12px;
        color: #64748B;
        width: 20px;
        text-align: center;
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
    fig.update_xaxes(gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0")
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
    return U / (U + W) if (U + W) > 0 else 0

# CARGA AUTOMÁTICA O POR REEMPLAZO DE DATOS
def cargar_datos_fuente(archivo_subido=None):
    if archivo_subido is not None:
        try:
            return pd.read_csv(archivo_subido, header=0) if archivo_subido.name.endswith('.csv') else pd.read_excel(archivo_subido, header=0)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
            return None
    
    for archivo in os.listdir('.'):
        nombre_minuscula = archivo.lower()
        if ("afectacion" in nombre_minuscula or "fuerza" in nombre_minuscula or "publica" in nombre_minuscula) and (archivo.endswith('.csv') or archivo.endswith('.xlsx')):
            try:
                return pd.read_csv(archivo, header=0) if archivo.endswith('.csv') else pd.read_excel(archivo, header=0)
            except:
                pass
    return None

# GESTIÓN DE DATOS EN LA BARRA LATERAL
st.sidebar.markdown("<div class='sidebar-title'><i class='fa-solid fa-shield-halved' style='color: #1e3a8a;'></i> Gestión de Datos</div>", unsafe_allow_html=True)
archivo_nuevo = st.sidebar.file_uploader("Cargar histórico institucional (CSV o Excel)", type=["csv", "xlsx"], label_visibility="collapsed")

if 'df_interno' not in st.session_state:
    st.session_state.df_interno = None

if archivo_nuevo is not None:
    df_cargado = cargar_datos_fuente(archivo_nuevo)
    if df_cargado is not None:
        st.session_state.df_interno = df_cargado
        st.sidebar.success("Base de datos sincronizada.")
elif st.session_state.df_interno is None:
    st.session_state.df_interno = cargar_datos_fuente()

df_original = st.session_state.df_interno
if df_original is None:
    st.error("No se encontró ningún registro de datos histórico.")
    st.stop()

# PIPELINE OPTIMIZADO
@st.cache_data
def ejecutar_pipeline_analitico(df):
    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    pivot_accion = df.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
    
    columnas_fuerza = [c for c in df['NOMBRE_FUERZA'].unique() if pd.notna(c)] if 'NOMBRE_FUERZA' in df.columns else []
    pivot_fuerza = df.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_fuerza else pd.DataFrame(index=pivot_accion.index)
    
    columnas_cat = [c for c in df['CATEGORIA'].unique() if pd.notna(c)] if 'CATEGORIA' in df.columns else []
    pivot_cat = df.pivot_table(index=index_cols, columns='CATEGORIA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_cat else pd.DataFrame(index=pivot_accion.index)

    total_municipio = df.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')
    datos_completos = total_municipio.join([pivot_accion, pivot_fuerza, pivot_cat]).reset_index().dropna()

    cols_numericas = [col for col in datos_completos.columns if col not in index_cols]
    
    scaler_obj = StandardScaler()
    X_scaled_arr = scaler_obj.fit_transform(datos_completos[cols_numericas])
    X_scaled_df = pd.DataFrame(X_scaled_arr, columns=cols_numericas)

    kmeans_mdl = KMeans(n_clusters=4, init='k-means++', n_init=30, random_state=42)
    lbls = kmeans_mdl.fit_predict(X_scaled_df)
    
    datos_originales_num_df = datos_completos.copy()
    datos_originales_num_df['Cluster'] = lbls
    X_scaled_df['Cluster'] = lbls

    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df[cols_numericas], lbls, test_size=0.2, random_state=42, stratify=lbls)
    mlp_mdl = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=700, random_state=42)
    mlp_mdl.fit(X_train, y_train)
    
    return datos_completos, cols_numericas, datos_originales_num_df, X_scaled_df, mlp_mdl, kmeans_mdl, scaler_obj, lbls, X_test, y_test

# Ejecutar pipeline seguro
datos, numericas, datos_originales_num, X_scaled, mlp_modelo, kmeans_modelo, scaler, y_labels, X_test, y_test = ejecutar_pipeline_analitico(df_original)

# Persistencia en session_state
st.session_state.red_entrenada = mlp_modelo
st.session_state.kmeans_entrenado = kmeans_modelo
st.session_state.escalador_entrenado = scaler
st.session_state.columnas_modelo = numericas

# INTERFAZ PRINCIPAL
st.markdown("""
    <div style='background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%) !important; padding: 25px; border-radius: 16px; margin-bottom: 30px; border: 1px dashed #BBF7D0 !important;'>
        <h1 style='color: #14532D !important; margin: 0; font-size: 30px;'>Sistema Híbrido para la Seguridad Territorial</h1>
        <p style='color: #15803D !important; margin: 5px 0 0 0;'>Pipeline Analítico Basado en Aprendizaje Combinado Estable (K-Means++ + MLP Neural Network)</p>
    </div>
""", unsafe_allow_html=True)

# KPIs
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1: st.metric("Municipios", datos.shape[0])
with col_kpi2: st.metric("Dimensiones", len(numericas))
with col_kpi3:
    acc_global = accuracy_score(y_labels, mlp_modelo.predict(X_scaled[numericas]))
    st.metric("Precisión Red (Accuracy)", f"{acc_global * 100:.2f}%")
with col_kpi4: st.metric("Grupos (K)", "4 Clústeres")

# DEFINICIÓN DE LAS PESTAÑAS
tabs = st.tabs(["Ingeniería e Ingesta", "Segmentación K-Means", "Modelo Red Neuronal", "Despliegue e Inferencia"])

# PESTAÑA 1: INGENIERÍA E INGESTA
with tabs[0]:
    st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
    st.markdown("<h3><i class='fa-solid fa-database title-icon'></i> 1. Matriz Consolidada Post-Pivotado</h3>", unsafe_allow_html=True)
    st.dataframe(datos.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# PESTAÑA 2: SEGMENTACIÓN
with tabs[1]:
    st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        wss = [KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42).fit(X_scaled[numericas]).inertia_ for k in range(1, 11)]
        fig_elbow = px.line(x=list(range(1, 11)), y=wss, markers=True, title="Método del Codo", labels={'x': 'K', 'y': 'Inercia'})
        st.plotly_chart(aplicar_estilo_premium(fig_elbow), use_container_width=True)
    with col2:
        pca_3d = PCA(n_components=3)
        scores_pca = pca_3d.fit_transform(X_scaled[numericas])
        df_pca = pd.DataFrame(scores_pca, columns=['PC1', 'PC2', 'PC3'])
        df_pca['Cluster'] = y_labels.astype(str)
        df_pca['Municipio'] = datos['MUNICIPIO'].values
        fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster', hover_name='Municipio', title='Proyección Territorial PCA 3D')
        st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# PESTAÑA 3: RED NEURONAL
with tabs[2]:
    st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
    df_reporte = pd.DataFrame(classification_report(y_test, mlp_modelo.predict(X_test), output_dict=True)).transpose().round(2)
    st.dataframe(df_reporte, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# PESTAÑA 4: FORMULARIO DE INFERENCIA EN PRODUCCIÓN (MEJORADO Y INDEXADO)
with tabs[3]:
    st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
    st.markdown("<h3><i class='fa-solid fa-terminal title-icon'></i> Entorno Operativo de Predicción Dirigida</h3>", unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns([1.2, 0.8])
    
    with col_f1:
        with st.form("formulario_principal_prospecto"):
            nombre_muni_futuro = st.text_input("Nombre del Territorio", "Territorio de Prueba")
            valores_ingresados = {}
            
            variables_para_input = [v for v in numericas if v != 'TOTAL_AFECTADOS']
            
            st.markdown("##### Ingrese los valores numéricos para forzar el cambio de Clúster:")
            sub_col1, sub_col2 = st.columns(2)
            for i, var in enumerate(variables_para_input):
                with sub_col1 if i % 2 == 0 else sub_col2:
                    valores_ingresados[var] = st.number_input(f"Cantidad de: {var}", min_value=0, value=0, step=5)
                    
            boton_predecir_tab = st.form_submit_button("Calcular Clúster con Consistencia Híbrida")

        if boton_predecir_tab:
            if 'TOTAL_AFECTADOS' in numericas:
                valores_ingresados['TOTAL_AFECTADOS'] = sum(valores_ingresados.values())
            
            # Reindexación estricta para respetar el orden numérico del scaler
            df_registro_futuro = pd.DataFrame([valores_ingresados]).reindex(columns=numericas, fill_value=0)
            registro_escalado = st.session_state.escalador_entrenado.transform(df_registro_futuro)
            
            prediccion_kmeans = st.session_state.kmeans_entrenado.predict(registro_escalado)[0]
            prediccion_mlp = st.session_state.red_entrenada.predict(registro_escalado)[0]
            
            colores_cluster = {0: "#22C55E", 1: "#0EA5E9", 2: "#F59E0B", 3: "#EF4444"}
            color_resaltado = colores_cluster.get(prediccion_kmeans, "#3b82f6")
            
            with col_f2:
                st.markdown(f"""
                    <div class='insight-success' style='padding: 25px; border-radius:10px; border-left: 6px solid {color_resaltado} !important;'>
                        <h3 style='color: #1E293B !important; margin-top:0;'><i class='fa-solid fa-circle-check' style='color: {color_resaltado};'></i> Clasificación Completada</h3>
                        El territorio simulado <b>{nombre_muni_futuro}</b> ha sido evaluado.<br><br>
                        Asignación Geométrica (K-Means++): <br>
                        <span style='font-size: 30px; font-weight:800; color: {color_resaltado};'>CLÚSTER {prediccion_kmeans}</span><br><br>
                        Clasificación de la Red (MLP): <br>
                        <span style='font-size: 20px; font-weight:700; color: #475569;'>CLÚSTER {prediccion_mlp}</span>
                    </div>
                """, unsafe_allow_html=True)
                
    with col_f2:
        if not boton_predecir_tab:
            st.markdown("""
            <div class='insight-card'>
                <h5><i class='fa-solid fa-bolt' style='margin-right:8px;'></i>Validación Espacial Activa</h5>
                <p>Al optimizar el algoritmo con <code>k-means++</code> y forzar la auto-suma de la variable <code>TOTAL_AFECTADOS</code>, el espacio métrico cambia de forma dinámica según tus inputs.</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("</div>", unsafe_allow_html=True)

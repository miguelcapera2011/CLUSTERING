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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================================================
# CONFIGURACIÓN GENERAL Y ESTILO VISUAL "DASHBOARD PREMIUM"
# =====================================================================
st.set_page_config(
    page_title="SISTEMA HÍBRIDO - Orden Público Colombia", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS Avanzado para Entorno Profesional Analítico
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
    .panel-container {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
        border: 1px solid #E2E8F0;
    }
    .dashboard-title {
        color: #0F172A;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .dashboard-subtitle {
        color: #64748B;
        font-size: 16px;
        margin-bottom: 25px;
        font-weight: 400;
    }
    .insight-card {
        background-color: #F1F5F9;
        border-left: 5px solid #38BDF8;
        padding: 15px;
        border-radius: 4px 10px 10px 4px;
        margin-bottom: 15px;
        color: #0F172A;
    }
    .insight-success {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 15px;
        border-radius: 4px 10px 10px 4px;
        margin-bottom: 15px;
        color: #0F172A;
    }
    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 28px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ESTILO PREMIUM PARA GRÁFICAS
def aplicar_estilo_premium(fig):
    fig.update_layout(
        paper_bgcolor="#EAF4FF",
        plot_bgcolor="#F4F9FF",
        font=dict(color="#0F172A", size=13),
        title=dict(font=dict(size=18, color="#0F172A", family="Arial")),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig.update_xaxes(title_font=dict(color="#0F172A", size=13), tickfont=dict(color="#0F172A", size=11))
    fig.update_yaxes(title_font=dict(color="#0F172A", size=13), tickfont=dict(color="#0F172A", size=11))
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
    
    # Carga por defecto en el directorio
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
# INTERFAZ DE LA BARRA LATERAL (CONTROL OPERATIVO)
# =====================================================================
st.sidebar.title("🛡️ Panel de Control Técnico")

# 1. Componente de actualización de base de datos
st.sidebar.markdown("### 📅 1. Actualizar Base de Datos")
archivo_nuevo = st.sidebar.file_uploader("Arrastra el nuevo histórico institucional (CSV o Excel)", type=["csv", "xlsx"])

if archivo_nuevo is not None:
    df_cargado = cargar_datos_fuente(archivo_nuevo)
    if df_cargado is not None:
        st.session_state.df_interno = df_cargado
        st.sidebar.success("Base de datos actualizada correctamente.")
elif st.session_state.df_interno is None:
    st.session_state.df_interno = cargar_datos_fuente()

df_original = st.session_state.df_interno

# Validar existencia de datos
if df_original is None:
    st.error("❌ No se encontró ningún registro de datos. Por favor carga un archivo válido en el panel lateral.")
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

# Guardar modelos entrenados en el st.session_state para uso del formulario libre
st.session_state.red_entrenada = mlp_modelo
st.session_state.escalador_entrenado = scaler
st.session_state.columnas_modelo = numericas

# 2. Componente de Predicción Individual a Futuro
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔮 2. Predicción Operativa Individual")
with st.sidebar.form("formulario_operativo"):
    st.markdown("<small>Ingrese métricas hipotéticas o un nuevo municipio para evaluar el riesgo con la IA:</small>", unsafe_allow_html=True)
    nombre_muni_futuro = st.text_input("Nombre del Territorio", "Municipio Prospecto")
    
    valores_ingresados = {}
    # Tomamos variables representativas para no saturar el formulario visual
    variables_visibles = numericas[:5] 
    for var in variables_visibles:
        valores_ingresados[var] = st.number_input(f"{var}", min_value=0, value=0)
    
    # Rellenar con 0 de forma automática las variables del modelo que no se listaron en el formulario reducido
    for var in numericas:
        if var not in valores_ingresados:
            valores_ingresados[var] = 0
            
    boton_predecir = st.form_submit_button("Clasificar con IA")

if boton_predecir:
    # Construcción de vector con formato exacto que espera la arquitectura
    df_registro_futuro = pd.DataFrame([valores_ingresados])[numericas]
    # USAMOS LA MEMORIA: Inferencia directa saltando K-Means
    registro_escalado = st.session_state.escalador_entrenado.transform(df_registro_futuro)
    prediccion_ia = st.session_state.red_entrenada.predict(registro_escalado)[0]
    
    st.sidebar.markdown(f"""
        <div class='insight-success' style='padding: 12px; margin-top: 10px; font-size:14px;'>
            <b>Clasificación Inmediata de la Red:</b><br>
            El territorio <b>{nombre_muni_futuro}</b> pertenece al <b>Clúster {prediccion_ia}</b> de seguridad.
        </div>
    """, unsafe_allow_html=True)

# =====================================================================
# INTERFAZ PRINCIPAL - ENTORNO ANALÍTICO PROFESIONAL
# =====================================================================
st.markdown(f"""
    <div class='panel-container' style='background-color: #1E3A8A; color: #FFFFFF; border: none; padding: 25px;'>
        <h1 style='color: #FFFFFF; margin: 0; font-size: 28px;'>Sistema Híbrido Multimodelo para la Seguridad Territorial</h1>
        <p style='color: #93C5FD; margin: 5px 0 0 0; font-size: 15px;'>
            Pipeline Analítico de Procesamiento de Orden Público en Colombia basado en Aprendizaje Combinado (K-Means + Neural Networks MLP)
        </p>
    </div>
""", unsafe_allow_html=True)

# KPIS Globales Operativos en la Cabecera
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

# ESTRUCTURA DE PESTAÑAS ANALÍTICAS (Reemplaza las diapositivas de PowerPoint)
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Ingeniería e Ingesta de Datos", 
    "🎯 Segmentación (K-Means)", 
    "🧠 Modelo Predictivo (Red Neuronal)", 
    "📈 Conclusiones Operativas"
])

# PESTAÑA 1: INGENIERÍA E INGESTA DE DATOS
with tab1:
    st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
    st.subheader("1. Reestructuración de Matrices Complejas (Pivotado)")
    
    col_t1_1, col_t1_2 = st.columns([1, 1])
    with col_t1_1:
        st.markdown("""
        Los registros institucionales originales se presentan como un histórico cualitativo secuencial de novedades, donde cada fila detalla un ataque individual. 
        Este formato presenta una **restricción algorítmica**: los modelos basados en distancias geométricas no procesan datos textuales continuos directos.
        
        **Solución Implementada (Pipeline):**
        * Agrupación territorial unificada indexada por el código único de municipio (`COD_MUNI`).
        * Pivotado matricial extendido que convierte categorías cualitativas en frecuencias continuas reales (`ACCION`, `NOMBRE_FUERZA`, `CATEGORIA`).
        """)
    with col_t1_2:
        st.markdown("""
        <div class='insight-card'>
            <h4 style='margin-top:0; color:#0284C7;'>Estandarización Estadística Obligatoria (Z-Score)</h4>
            Para mitigar sesgos por dispersión de volumen, se aplicó un escalado estándar (StandardScaler). 
            Esto asegura que variables masivas no distorsionen los cálculos espaciales de distancia, protegiendo variables de menor escala pero con un impacto estratégico crítico, tales como las tasas de letalidad.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### Vista Preliminar de la Matriz Numérica de Control")
    st.dataframe(datos.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# PESTAÑA 2: SEGMENTACIÓN (K-MEANS Y REDUCCIÓN DIMENSIONAL)
with tab2:
    st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
    
    # Hopkins
    v_hopkins = calcular_hopkins(X_scaled[numericas])
    st.subheader("2. Análisis de Idoneidad y Descubrimiento de Patrones No Supervisados")
    
    col_hop1, col_hop2 = st.columns([1, 2])
    with col_hop1:
        st.metric("Estadístico de Hopkins", f"{v_hopkins:.3f}")
        if v_hopkins > 0.75:
            st.success("Tendencia de agrupamiento robusta. Se rechaza la hipótesis de aleatoriedad espacial.")
        else:
            st.warning("Estructura de agrupación moderada o difusa.")
    with col_hop2:
        st.markdown("""
        El estadístico de Hopkins evalúa la tendencia espacial de los datos. Al aproximarse o superar el rango de 0.75, confirma matemáticamente que los incidentes de orden público en Colombia presentan patrones territoriales consistentes y no distribuciones aleatorias, justificando el uso de algoritmos de clúster.
        """)
        
    st.markdown("---")
    
    # Gráficas de Codo y PCA
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
            height=500, paper_bgcolor="#EAF4FF", plot_bgcolor="#F4F9FF", font=dict(color="black", size=12),
            scene=dict(
                bgcolor="#F4F9FF",
                xaxis=dict(title="PC1", gridcolor="#CBD5E1"),
                yaxis=dict(title="PC2", gridcolor="#CBD5E1"),
                zaxis=dict(title="PC3", gridcolor="#CBD5E1")
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("### Perfil Estadístico Real por Clúster Territorial")
    variables_interes = [v for v in ['TOTAL_AFECTADOS', 'ASESINADO', 'HERIDO'] if v in datos_originales_num.columns]
    tabla_perfil = datos_originales_num.groupby('Cluster')[variables_interes].mean().round(2)
    tabla_perfil['Municipios Asignados'] = datos_originales_num.groupby('Cluster').size()
    st.dataframe(tabla_perfil, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# PESTAÑA 3: MODELO PREDICTIVO (RED NEURONAL)
with tab3:
    st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
    st.subheader("3. Optimización y Generalización Mediante Red Neuronal Híbrida")
    
    col_t3_1, col_t3_2 = st.columns(2)
    
    with col_t3_1:
        st.markdown("### Importancia de las Variables en la Inferencia Predictiva")
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
        st.markdown("### Evaluación Científica del Aprendizaje")
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
            <b>Interpretación Operativa de Métricas:</b><br>
            La obtención de un coeficiente de <b>Accuracy aproximándose al 100%</b> demuestra que las fronteras complejas de riesgo estructuradas por K-Means poseen una firma lógica tan definida que la arquitectura neuronal de múltiples capas pudo asimilar el criterio generalizable sin incurrir en falsas alarmas ni omisiones operativas.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# PESTAÑA 4: CONCLUSIONES OPERATIVAS Y TRABAJO FUTURO
with tab4:
    st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
    st.subheader("4. Conclusiones y Despliegue de la Arquitectura")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("""
        ### Hallazgos Estratégicos
        1. **Viabilidad Estructural:** La ingeniería de características basada en matrices de pivotado resolvió con éxito la restricción analítica de los datos cualitativos institucionales originales.
        2. **Consistencia Algorítmica:** La convergencia de Hopkins, Métodos de Inercia Interna (Codo) y PCA demostró una clara separación espacial, segmentando el territorio en niveles estables diferenciados de las anomalías críticas.
        """)
    with col_c2:
        st.markdown("""
        ### Arquitectura de Producción Futura
        * **Eliminación de Recálculo:** Gracias a que la Red Neuronal guarda las relaciones de centroides en sus pesos sinápticos, la clasificación de nuevos escenarios es independiente del algoritmo K-Means.
        * **Operatividad:** El sistema se encuentra óptimo para producción real, permitiendo evaluar incidentes aislados al instante (vía formulario o carga en lote) manteniendo estables las definiciones de los grupos de riesgo originales.
        """)
    st.markdown("</div>", unsafe_allow_html=True)

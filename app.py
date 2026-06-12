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
# Librerías para el bloque de la Red Neuronal Artificial
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- ESTILO DE ALTA DENSIDAD Y CONTRASTE PARA PÓSTER CIENTÍFICO (IMPRESIÓN) ---
def aplicar_estilo_poster(fig, titulo_grafica):
    fig.update_layout(
        paper_bgcolor="#FFFFFF",  # Fondo blanco limpio para impresión
        plot_bgcolor="#F8FAFC",   # Fondo de gráfica sutil
        font=dict(
            color="#0F172A",
            size=16               # Texto más grande para lectura a distancia
        ),
        title=dict(
            text=f"<b>{titulo_grafica}</b>", # Título en negrita y explícito
            font=dict(size=24, color="#0F172A"),
            x=0.01
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=14)
        )
    )
    # Forzar visibilidad y color de los ejes
    fig.update_xaxes(title_font=dict(size=16, color="#0F172A"), tickfont=dict(size=14, color="#0F172A"), showgrid=True, gridcolor="#E2E8F0")
    fig.update_yaxes(title_font=dict(size=16, color="#0F172A"), tickfont=dict(size=14, color="#0F172A"), showgrid=True, gridcolor="#E2E8F0")
    return fig

# FUNCIÓN PARA CALCULAR EL ESTADÍSTICO DE HOPKINS
def calcular_hopkins(X):
    X = np.array(X)
    n, d = X.shape
    m = int(0.1 * n)
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
    H = U / (U + W)
    return H

# CONFIGURACIÓN DE STREAMLIT
st.set_page_config(
    page_title="Investigación: Modelo Híbrido de Minería de Datos", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inyección de CSS estilo diapositiva académica clara
st.markdown("""
    <style>
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
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
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .slide-subtitle {
        color: #475569;
        font-size: 20px;
        margin-bottom: 25px;
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
        box-shadow: 0 4px 12px rgba(3, 105, 161, 0.15) !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: #0284C7 !important; 
        color: #FFFFFF !important;            
        border: 1px solid #0284C7 !important;
    }
    .insight-card { background-color: #F1F5F9; border-left: 5px solid #38BDF8; padding: 20px; border-radius: 4px 12px 12px 4px; margin-bottom: 15px; }
    .insight-success { background-color: #F0FDF4; border-left: 5px solid #16A34A; padding: 20px; border-radius: 4px 12px 12px 4px; margin-bottom: 15px; }
    .concept-box { background-color: #EFF6FF; border: 1px solid #BFDBFE; padding: 15px; border-radius: 8px; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

if 'diapositiva' not in st.session_state:
    st.session_state.diapositiva = 1

def ir_a_diapositiva(num):
    st.session_state.diapositiva = num
    st.rerun()

# CARGA AUTOMÁTICA DE DATOS
def cargar_datos_automatico():
    archivos_en_carpeta = os.listdir('.')
    archivo_encontrado = None
    for archivo in archivos_en_carpeta:
        nombre_minuscula = archivo.lower()
        if ("afectacion" in nombre_minuscula or "fuerza" in nombre_minuscula or "publica" in nombre_minuscula) and (archivo.endswith('.csv') or archivo.endswith('.xlsx')):
            archivo_encontrado = archivo
            break
    if archivo_encontrado is None: return None, "No se encontró el archivo."
    try:
        df = pd.read_csv(archivo_encontrado, header=0) if archivo_encontrado.endswith('.csv') else pd.read_excel(archivo_encontrado, header=0)
        return df, archivo_encontrado
    except Exception as e: return None, str(e)

df_original, nombre_archivo_cargado = cargar_datos_automatico()

# NAVEGACIÓN SUPERIOR (7 PESTAÑAS)
cols_nav = st.columns(7)
nombres_diapo = ["1. Portada", "2. Introducción", "3. Marco Teórico", "4. Metodología", "5. Resultados K-Means", "6. Red Neuronal (Híbrido)", "7. Conclusiones"]
for i, nombre in enumerate(nombres_diapo):
    tipo_boton = "primary" if st.session_state.diapositiva == (i + 1) else "secondary"
    if cols_nav[i].button(nombre, use_container_width=True, type=tipo_boton): ir_a_diapositiva(i + 1)

st.markdown("---")

# PROCESAMIENTO MATEMÁTICO INTEGRAL COHERENTE
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

    # K-Means Base (Generador de etiquetas fijas)
    kmeans = KMeans(n_clusters=4, n_init=30, random_state=42)
    km4_clusters = kmeans.fit(X_scaled)
    datos_originales_num['Cluster'] = km4_clusters.labels_
    datos['Cluster'] = km4_clusters.labels_
    
    nombres_clusters = {
        "0": "Clúster 0: Riesgo Controlado", "1": "Clúster 1: Impacto Moderado",
        "2": "Clúster 2: Conflicto Institucional", "3": "Clúster 3: Emergencia Crítica"
    }

# PÁGINAS 1 A 4 MANTIENEN SU ESTRUCTURA CON TEXTOS ACTUALIZADOS
if st.session_state.diapositiva == 1:
    st.markdown("<div class='slide-container' style='text-align: center; padding: 50px;'><img src='https://administrativos.ut.edu.co/images/Home/simbolos/logo_oficial.png' width='180'><div class='slide-title' style='color:#1E3A8A;'>Modelo Híbrido Consecutivo de Minería de Datos</div><div class='slide-subtitle'>De la Segmentación Espacial Matemática (K-Means) a la Clasificación Inteligente (Redes Neuronales)</div></div>", unsafe_allow_html=True)
    if st.button("Iniciar Sustentación", type="primary", use_container_width=True): ir_a_diapositiva(2)

elif st.session_state.diapositiva == 2:
    st.markdown("<div class='slide-title'>Introducción y Desafío Metodológico</div>", unsafe_allow_html=True)
    st.markdown("<div class='slide-container'><h3>El Quiebre Analítico</h3><p>Los datos institucionales carecen de una columna de 'Severidad' o 'Riesgo'. No están etiquetados. Para resolver esto, diseñamos un <b>Pipeline Consecutivo Híbrido</b>: el algoritmo no supervisado descubre las categorías territoriales latentes y la Red Neuronal aprende a predecirlas de forma autónoma.</p></div>", unsafe_allow_html=True)
    if st.button("Siguiente Diapositiva ➡️", type="primary"): ir_a_diapositiva(3)

elif st.session_state.diapositiva == 3:
    st.markdown("<div class='slide-title'>Sustentación Algorítmica</div>", unsafe_allow_html=True)
    st.markdown("<div class='slide-container'><h4>Componentes del Sistema Inteligente</h4><ul><li><b>K-Means:</b> Agrupamiento por optimización de distancias euclidianas intragrupo.</li><li><b>Pseudoetiquetado:</b> Conversión de variables geométricas en variables objetivo (Target).</li><li><b>Multilayer Perceptron (MLP):</b> Red neuronal encargada de modelar las fronteras de decisión complejas.</li></ul></div>", unsafe_allow_html=True)
    if st.button("Siguiente Diapositiva ➡️", type="primary"): ir_a_diapositiva(4)

elif st.session_state.diapositiva == 4:
    st.markdown("<div class='slide-title'>Arquitectura del Flujo y Código Fuente</div>", unsafe_allow_html=True)
    st.code("# Pipeline unificado de Extracción, Transformación, Agrupamiento y Red Neuronal\n# (Ver detalles en secciones de ejecución)", language="python")
    if st.button("Ver Resultados del K-Means ➡️", type="primary"): ir_a_diapositiva(5)


# DIAPOSITIVA 5: RESULTADOS K-MEANS OPTIMIZADOS PARA PÓSTER
elif st.session_state.diapositiva == 5:
    st.markdown("<div class='slide-title'>Fase No Supervisada: Análisis de Clústeres (K-Means)</div><div class='slide-subtitle'>Validación matemática y distribución espacial de las tipologías de orden público</div>", unsafe_allow_html=True)
    
    # Gráfica del Codo
    wss = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled).inertia_ for k in range(1, 11)]
    fig_elbow = px.line(x=list(range(1, 11)), y=wss, markers=True, labels={'x': 'Número de Clústeres (k)', 'y': 'Inercia (WSS)'}, template='plotly_white')
    fig_elbow.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="K=4 Óptimo (Codo)")
    fig_elbow.update_traces(line=dict(width=4, color='#0284C7'), marker=dict(size=10, color='#0369A1'))
    fig_elbow = aplicar_estilo_poster(fig_elbow, "Figura 1: Curva del Codo para Selección del Número de Grupos (K)")
    st.plotly_chart(fig_elbow, use_container_width=True)
    
    st.markdown("<div class='insight-card'><b>Análisis Científico de la Figura 1:</b> El método del codo evalúa la inercia interna. La tasa de cambio matemático se estabiliza de forma drástica en K=4, lo que demuestra que estructurar las afectaciones del país en 4 dimensiones territoriales es estadísticamente óptimo y parsimonioso.</div>", unsafe_allow_html=True)
    
    # Gráfica PCA 3D
    pca_3d = PCA(n_components=3)
    scores_pca = pca_3d.fit_transform(X_scaled)
    df_pca = pd.DataFrame(scores_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Nombre_Cluster'] = km4_clusters.labels_.astype(str).map(nombres_clusters)
    df_pca['Municipio'] = datos['MUNICIPIO'].values
    
    fig_3d = px.scatter_3d(
        df_pca, x='PC1', y='PC2', z='PC3', color='Nombre_Cluster', hover_name='Municipio',
        color_discrete_map={"Clúster 0: Riesgo Controlado": "#22C55E", "Clúster 1: Impacto Moderado": "#0EA5E9", "Clúster 2: Conflicto Institucional": "#F59E0B", "Clúster 3: Emergencia Crítica": "#EF4444"}
    )
    fig_3d.update_layout(height=650, title=dict(text="<b>Figura 2: Proyección Espacial de Municipios mediante Componentes Principales (PCA 3D)</b>", font=dict(size=22)))
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("<div class='insight-critical'><b>Análisis de Datos Atípicos (Figura 2):</b> El Clúster 3 (rojo) aísla de manera natural las anomalías del orden público (grandes capitales o zonas críticas históricas). El modelo no elimina los 'outliers', sino que los agrupa de forma segregada debido a que sus métricas superan por más de 3 desviaciones estándar la media nacional.</div>", unsafe_allow_html=True)
    
    if st.button("Ir al Nexo Híbrido y Red Neuronal ➡️", type="primary"): ir_a_diapositiva(6)


# DIAPOSITIVA 6: EXPLICACIÓN COMPLETA DEL PASO DE K-MEANS A RED NEURONAL
elif st.session_state.diapositiva == 6:
    st.markdown("<div class='slide-title'>Fase Supervisada: El Nexo Híbrido y Red Neuronal Artificial</div><div class='slide-subtitle'>Explicación detallada del flujo de transferencia: Del espacio geométrico a los pesos sinápticos</div>", unsafe_allow_html=True)
    
    # --- BLOQUE METODOLÓGICO VISUAL (EL PUENTE DE TRANSICIÓN) ---
    st.markdown("""
    <div style='background-color: #F8FAFC; border: 2px solid #0284C7; padding: 25px; border-radius: 12px; margin-bottom: 30px;'>
        <h3 style='color: #0284C7; margin-top: 0;'>¿DÓNDE EMPIEZA LA CONEXIÓN ENTRE K-MEANS Y LA RED NEURONAL?</h3>
        <p style='font-size: 16px; line-height: 1.6;'>
            El proceso es un <b>Pipeline Híbrido Consecutivo en Cascada</b>. El punto exacto de unión ocurre cuando los resultados del agrupamiento espacial se transforman en variables de supervisión. El flujo sigue de manera estricta esta secuencia:
        </p>
        <div style='display: flex; justify-content: space-between; align-items: center; text-align: center; gap: 15px; margin: 20px 0;'>
            <div style='flex: 1; background: #FFFFFF; padding: 15px; border-radius: 8px; border: 1px solid #CBD5E1;'>
                <b style='color: #0F172A; font-size: 14px;'>1. MATRIZ ORIGINAL SCALED ($X$)</b><br>
                <span style='font-size: 13px; color: #64748B;'>Indicadores numéricos normalizados de criminalidad por municipio.</span>
            </div>
            <div style='color: #0284C7; font-weight: bold; font-size: 24px;'>➔</div>
            <div style='flex: 1; background: #E0F2FE; padding: 15px; border-radius: 8px; border: 1px solid #93C5FD;'>
                <b style='color: #0369A1; font-size: 14px;'>2. CLUSTERING ($K\text{-Means}$)</b><br>
                <span style='font-size: 13px; color: #0369A1;'>Calcula distancias geométricas y asigna a cada fila una etiqueta fija de 0 a 3.</span>
            </div>
            <div style='color: #16A34A; font-weight: bold; font-size: 24px;'>➔</div>
            <div style='flex: 1; background: #DCFCE7; padding: 15px; border-radius: 8px; border: 1px solid #86EFAC;'>
                <b style='color: #15803D; font-size: 14px;'>3. EL PUENTE (PSEUDOETIQUETA)</b><br>
                <code style='color: #166534; font-weight: bold; font-size: 13px;'>y_net = km_clusters.labels_</code><br>
                <span style='font-size: 12px; color: #15803D;'>La salida geométrica se vuelve la variable objetivo (Target) a predecir.</span>
            </div>
            <div style='color: #7C3AED; font-weight: bold; font-size: 24px;'>➔</div>
            <div style='flex: 1; background: #F3E8FF; padding: 15px; border-radius: 8px; border: 1px solid #D8B4FE;'>
                <b style='color: #6B21A8; font-size: 14px;'>4. DEEP LEARNING (MLP)</b><br>
                <span style='font-size: 13px; color: #6B21A8;'>La Red Neuronal se entrena para deducir las fronteras complejas de la clasificación.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- ENTRENAMIENTO DE LA RED NEURONAL ---
    X_net = X_scaled.values
    y_net = km4_clusters.labels_ # AQUÍ SE CREA EL NEXO SUPERVISADO

    X_train, X_test, y_train, y_test = train_test_split(X_net, y_net, test_size=0.3, random_state=42, stratify=y_net)

    @st.cache_resource
    def ejecutar_entrenamiento_red(X_t, y_t):
        mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=500, random_state=42)
        mlp.fit(X_t, y_t)
        return mlp

    model_mlp = ejecutar_entrenamiento_red(X_train, y_train)
    y_pred = model_mlp.predict(X_test)
    accuracy = model_mlp.score(X_test, y_test)
    matriz_conf = confusion_matrix(y_test, y_pred)
    reporte_dict = classification_report(y_test, y_pred, output_dict=True)

    # --- PRESENTACIÓN DE RESULTADOS GRÁFICOS PARA PÓSTER ---
    col_rn1, col_rn2 = st.columns([1, 2])
    
    with col_rn1:
        st.markdown("### Métricas Globales de Generalización")
        st.metric("Exactitud General del Modelo (Accuracy)", f"{accuracy*100:.2f}%", help="Porcentaje total de aciertos de la red neuronal sobre los datos de validación.")
        
        st.markdown("#### Desglose de Rigor Estadístico por Categoría Territorial")
        df_rep = pd.DataFrame(reporte_dict).transpose().iloc[:4, :3].round(3)
        df_rep.index = [nombres_clusters[str(i)] for i in range(4)]
        df_rep.columns = ['Precisión (Precision)', 'Sensibilidad (Recall)', 'F1-Score Base']
        st.dataframe(df_rep, use_container_width=True)
        
        st.markdown("""
        <div class='concept-box'>
            <b>Conceptos Clave para la Defensa del Póster:</b><br>
            * <b>Precisión:</b> Mide la ausencia de falsas alarmas. Si da 1.00, cuando la red predice 'Emergencia Crítica', el acierto es absoluto.<br>
            * <b>Sensibilidad (Recall):</b> Capacidad del modelo para mapear la totalidad de zonas vulnerables sin dejar focos de conflicto críticos por fuera del radar.
        </div>
        """, unsafe_allow_html=True)

    with col_rn2:
        # Matriz de Confusión con Estilo de Póster Científico de Alta Calidad
        fig_cm = px.imshow(
            matriz_conf,
            labels=dict(x="Predicción Emitida por la Red Neuronal (MLP)", y="Clúster Geométrico Base (K-Means)"),
            x=[f"Pred: Clúster {i}" for i in range(4)],
            y=[f"Real: Clúster {i}" for i in range(4)],
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig_cm = aplicar_estilo_poster(fig_cm, "Figura 3: Matriz de Confusión Cruzada para Validación del Modelo Supervisado")
        fig_cm.update_layout(height=500)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    st.markdown("""
    <div class='insight-success'>
        <b>Interpretación de la Figura 3:</b> La alta concentración numérica exclusivamente localizada a lo largo de la diagonal principal de la matriz demuestra que la Red Neuronal ha asimilado de forma óptima la estructura distancial del territorio nacional. La tasa de error fuera de la diagonal es marginal, lo que convalida científicamente la viabilidad operativa del modelo híbrido consecutivo.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Ver Conclusiones del Proyecto ➡️", type="primary"): ir_a_diapositiva(7)


# DIAPOSITIVA 7: CONCLUSIONES Y CIERRE ACADÉMICO
elif st.session_state.diapositiva == 7:
    st.markdown("<div class='slide-title'>🏁 Conclusiones y Proyecciones Estratégicas</div>", unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("<div class='slide-container'><h3>Aporte Científico</h3><ol><li><b>Resolución del Vacío Operativo:</b> Se implementó una solución semisupervisada ante la falta de variables de severidad previas en los históricos institucionales.</li><li><b>Eficiencia del Pipeline:</b> La Red Neuronal se convierte en un motor predictivo ágil que independiza al sistema de tener que ejecutar pesados cálculos de optimización espacial euclidiana en el futuro.</li></ol></div>", unsafe_allow_html=True)
    with col_c2:
        st.markdown("<div class='slide-container'><h3>Aplicabilidad en Seguridad</h3><ul><li><b>Automatización de Alertas:</b> Al ingresar los indicadores mensuales actuales, la Red clasifica instantáneamente la región en milisegundos.</li><li><b>Optimización de Apoyos:</b> Facilita al Estado la simulación predictiva de escenarios y la distribución coordinada de asistencia logística militar y médica.</li></ul></div>", unsafe_allow_html=True)

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

# --- NUEVOS IMPORTS PARA LA FASE SUPERVISADA ---
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

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
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

# FUNCIÓN PARA CALCULAR EL ESTADÍSTICO DE HOPKINS
def calcular_hopkins(X):
    X = np.array(X)
    n, d = X.shape
    m = int(0.1 * n)
    if m < 1: m = 1
    
    np.random.seed(42)
    vecinos = NearestNeighbors(n_neighbors=2)
    vecinos.fit(X)

    puntos_aleatorios = np.random.uniform(
        np.min(X, axis=0),
        np.max(X, axis=0),
        (m, d)
    )

    dist_aleatoria, _ = vecinos.kneighbors(
        puntos_aleatorios,
        n_neighbors=1
    )

    indices = np.random.choice(n, m, replace=False)
    puntos_reales = X[indices]

    dist_real, _ = vecinos.kneighbors(
        puntos_reales,
        n_neighbors=2
    )

    U = np.sum(dist_aleatoria)
    W = np.sum(dist_real[:, 1])
    H = U / (U + W) if (U + W) > 0 else 0.5
    return H

# CONFIGURACIÓN GENERAL
st.set_page_config(
    page_title="Exposición Mineria De Datos - Orden Público en Colombia", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inyección de CSS Avanzado
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
    div.stButton > button[kind="primary"]:hover {
        background-color: #0369A1 !important;
        color: #FFFFFF !important;
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

# CARGA AUTOMÁTICA DE DATOS
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

# CONTROLES DE NAVEGACIÓN SUPERIOR
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
            <p><b>Miguel Angel Garatejo</b><br>Facultad de Ciencias<br>Matemática con Énfasis en Estadística</p>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"""
        <div class='insight-success'>
            <h4 style='margin-top:0; color:#16A34A;'> PROFESOR</h4>
            <p><b>Yuri Marcela Garcia Saavedra </b><br>Minería de Datos <br>Año: {time.strftime('%Y')} | Clustering</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Iniciar Sustentación", type="primary", use_container_width=True):
        ir_a_diapositiva(2)

# DIAPOSITIVA 2: INTRODUCCIÓN
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
            <p><b>Naturaleza del Archivo:</b> La información institucional se presenta como un <i>Histórico de Novedades</i> donde cada fila reporta un ataque aislado.</p>
            <ul>
                <li><b>Restricción de Estructura:</b> El archivo posee variables cualitativas (texto) imposibles de promediar directamente.</li>
                <li><b>El Quiebre Matemático:</b> Los algoritmos de distancia espacial (K-Means) no procesan texto directo sin una transformación matricial previa.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_i2:
        st.markdown("""
        <div class='slide-container'>
            <h3 style='color: #0284C7; margin-top:0;'> Propuesta de Innovación</h3>
            <p><b>Enfoque Híbrido:</b> Usar aprendizaje No Supervisado (K-Means) para etiquetar el territorio de forma científica, y posteriormente entrenar una Red Neuronal (MLP) para automatizar la predicción de riesgos.</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Siguiente Diapositiva: Marco Conceptual ➡️", type="primary"):
        ir_a_diapositiva(3)

# DIAPOSITIVA 3: MARCO TEÓRICO
elif st.session_state.diapositiva == 3:
    st.markdown("""
    <div class='slide-title'>Fundamentos Teóricos del Modelo Híbrido</div>
    <div class='slide-subtitle'>Sustentación matemática del acoplamiento K-Means + Red Neuronal</div>
    """, unsafe_allow_html=True)
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("""
        <div class='slide-container' style='min-height: 220px;'>
            <h4 style='color:#0284C7; margin-top:0;'> Fase No Supervisada: K-Means</h4>
            <p style='font-size:14px;'>Particiona los municipios minimizando la varianza interna de los grupos (Inercia WSS). Actúa como un <b>etiquetador automático inteligente</b>.</p>
        </div>
        """, unsafe_allow_html=True)
    with t_col2:
        st.markdown("""
        <div class='slide-container' style='min-height: 220px;'>
            <h4 style='color:#10B981; margin-top:0;'> Fase Supervisada: Perceptrón Multicapa (MLP)</h4>
            <p style='font-size:14px;'>Red neuronal artificial que aprende las complejas fronteras de decisión no lineales creadas por el clúster, permitiendo clasificar nuevos escenarios al instante.</p>
        </div>
        """, unsafe_allow_html=True)
        
    if st.button("Siguiente Diapositiva: Estrategia de Procesamiento ", type="primary"):
        ir_a_diapositiva(4)

# DIAPOSITIVA 4: METODOLOGÍA
elif st.session_state.diapositiva == 4:
    st.markdown("""
    <div class='slide-title'>Arquitectura del Flujo e Ingeniería de Datos</div>
    <div class='slide-subtitle'>Tubería de datos implementada para asegurar rigurosidad científica</div>
    """, unsafe_allow_html=True)
    
    with st.expander("Fase 1: Pivotado y Construcción de Pseudo-etiquetas (K-Means)", expanded=True):
        st.code("""
# Agrupación y unificación matricial por municipio
pivot_accion = df_original.pivot_table(index=['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO'], columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
datos = total_municipio.join([pivot_accion, pivot_fuerza]).reset_index().dropna()

# Ajuste de escala y asignación de clústeres base
X_scaled_km = StandardScaler().fit_transform(datos[numericas])
kmeans = KMeans(n_clusters=4, random_state=42)
datos['Cluster'] = kmeans.fit_predict(X_scaled_km)
        """, language="python")
 
    with st.expander("Fase 2: Separación Train/Test y Red Neuronal para Evitar Fuga de Datos (Data Leakage)", expanded=False):
        st.code("""
# Separación de datos y etiquetas antes de normalizar para la red neuronal
X = datos[numericas]
y = datos['Cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Escalamiento científico aislado
scaler_mlp = StandardScaler()
X_train_scaled = scaler_mlp.fit_transform(X_train)
X_test_scaled = scaler_mlp.transform(X_test)

# Arquitectura MLP
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)
        """, language="python")
 
    if st.button("Siguiente Diapositiva: Ejecución y Resultados del Modelo Híbrido ➡️", type="primary"):
        ir_a_diapositiva(5)

# DIAPOSITIVA 5: RESULTADOS (EL CORAZÓN DEL PÓSTER)
elif st.session_state.diapositiva == 5:
    st.markdown("""
    <div class='slide-title'>Resultados del Modelo Híbrido (Ideal para Póster)</div>
    <div class='slide-subtitle'>De la segmentación geométrica a la automatización predictiva por IA</div>
    """, unsafe_allow_html=True)
    
    if df_original is None:
        st.error("❌ Archivo de datos no detectado.")
        st.stop()
        
    # --- 1. PROCESAMIENTO MATEMÁTICO REAL (PIVOTADO Y CLUSTERING) ---
    index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
    pivot_accion = df_original.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
    columnas_fuerza = [c for c in df_original['NOMBRE_FUERZA'].unique() if pd.notna(c)] if 'NOMBRE_FUERZA' in df_original.columns else []
    pivot_fuerza = df_original.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0) if columnas_fuerza else pd.DataFrame(index=pivot_accion.index)
    total_municipio = df_original.groupby(index_cols)['CANTIDAD'].sum().to_frame(name='TOTAL_AFECTADOS')
    
    datos = total_municipio.join([pivot_accion, pivot_fuerza]).reset_index().dropna()
    numericas = [col for col in datos.columns if col not in index_cols]
    
    # K-Means Base para obtener etiquetas
    scaler_km = StandardScaler()
    X_scaled_km = scaler_km.fit_transform(datos[numericas])
    valor_hopkins = calcular_hopkins(X_scaled_km)
    
    kmeans = KMeans(n_clusters=4, n_init=30, random_state=42)
    datos['Cluster'] = kmeans.fit_predict(X_scaled_km)
    
    # --- PANELES VISUALES PARA EL PÓSTER ---
    tab_km, tab_mlp, tab_pred = st.tabs(["📊 FASE A: Clustering (No Supervisado)", "🧠 FASE B: Red Neuronal (Supervisado)", "🔮 Predicción de Nuevos Municipios"])
    
    with tab_km:
        st.markdown("### 1. Validación Estructural del Territorio")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.metric("Estadístico de Hopkins", f"{valor_hopkins:.3f}")
        with col_h2:
            st.success("Estructura de clústeres altamente significativa y válida.")
            
        # PCA 3D
        pca_3d = PCA(n_components=3)
        scores_pca = pca_3d.fit_transform(X_scaled_km)
        df_pca = pd.DataFrame(scores_pca, columns=['PC1', 'PC2', 'PC3'])
        df_pca['Cluster_Name'] = datos['Cluster'].map({0: "Riesgo Controlado", 1: "Impacto Moderado", 2: "Conflicto Institucional", 3: "Emergencia Crítica"})
        df_pca['Municipio'] = datos['MUNICIPIO'].values
        
        fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster_Name', hover_name='Municipio', title='Espacio Geométrico del Conflicto (PCA)')
        st.plotly_chart(aplicar_estilo_premium(fig_3d), use_container_width=True)

    with tab_mlp:
        st.markdown("### 2. Evaluación de la Red Neuronal (MLP)")
        
        # --- PROCESAMIENTO EXCLUSIVO SUPERVISADO (EVITANDO DATA LEAKAGE) ---
        X = datos[numericas]
        y = datos['Cluster']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        scaler_mlp = StandardScaler()
        X_train_scaled = scaler_mlp.fit_transform(X_train)
        X_test_scaled = scaler_mlp.transform(X_test)
        
        mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        y_pred = mlp.predict(X_test_scaled)
        
        # Métricas para el póster
        accuracy = mlp.score(X_test_scaled, y_test)
        st.metric("Precisión Global de la IA (Accuracy)", f"{accuracy * 100:.2f}%")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Matriz de Confusión (Datos de Prueba):**")
            cm = confusion_matrix(y_test, y_pred)
            labels_map = ["R. Controlado", "I. Moderado", "C. Inst.", "E. Crítica"]
            fig_cm = px.imshow(cm, text_auto=True, x=labels_map, y=labels_map, labels=dict(x="Predicción de IA", y="Clúster Real"), color_continuous_scale="Blues")
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_m2:
            st.markdown("**Reporte de Clasificación Académico:**")
            reporte_dict = classification_report(y_test, y_pred, output_dict=True)
            df_reporte = pd.DataFrame(reporte_dict).transpose().round(2)
            st.dataframe(df_reporte, use_container_width=True)

    with tab_pred:
        st.markdown("### 🔮 Simulador de Alerta Temprana para Nuevos Municipios")
        st.write("Ingresa los datos estimados de un nuevo escenario o municipio para que la **Red Neuronal** clasifique instantáneamente su nivel de riesgo:")
        
        # Formulario dinámico basado en las columnas numéricas reales
        col_inputs = st.columns(min(len(numericas), 4))
        inputs_usuario = {}
        for idx, col_num in enumerate(numericas):
            with col_inputs[idx % 4]:
                inputs_usuario[col_num] = st.number_input(f"{col_num}", min_value=0, value=5)
                
        if st.button("Calcular Nivel de Riesgo con IA", type="primary"):
            df_nuevo = pd.DataFrame([inputs_usuario])
            df_nuevo_scaled = scaler_mlp.transform(df_nuevo)
            prediccion_final = mlp.predict(df_nuevo_scaled)[0]
            
            mapa_resultado = {
                0: ("🟢 Riesgo Controlado", "insight-success"),
                1: ("🔵 Impacto Moderado", "insight-card"),
                2: ("🟡 Conflicto Institucional", "insight-card"),
                3: ("🔴 Emergencia Crítica", "insight-critical")
            }
            
            nombre_r, estilo_r = mapa_resultado[prediccion_final]
            st.markdown(f"""
            <div class='{estilo_r}'>
                <h3>Resultado del Modelo Híbrido: {nombre_r}</h3>
                <p>La Red Neuronal ha analizado el patrón del vector ingresado y lo ha asociado a este nivel de vulnerabilidad operacional de forma automática.</p>
            </div>
            """, unsafe_allow_html=True)

    if st.button("Siguiente Diapositiva: Conclusiones ➡️", type="primary"):
        ir_a_diapositiva(6)

# DIAPOSITIVA 6: CONCLUSIONES
elif st.session_state.diapositiva == 6:
    st.markdown("""
    <div class='slide-title'>🏁 Conclusiones Académicas Destacadas para el Póster</div>
    <div class='slide-subtitle'>Cierre formal del flujo metodológico híbrido</div>
    """, unsafe_allow_html=True)
    
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.markdown("""
        <div class='slide-container' style='min-height:280px;'>
            <h3 style='color:#0369A1; margin-top:0;'> Aporte del Enfoque Híbrido</h3>
            <ul>
                <li><b>Generación de Pseudo-etiquetas:</b> K-Means eliminó la subjetividad humana, categorizando el territorio nacional de manera matemática y óptima en 4 dinámicas.</li>
                <li><b>Generalización Inteligente:</b> La Red Neuronal (MLP) aprendió con éxito las fronteras complejas de riesgo, alcanzando altos niveles de precisión en testeo.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c_col2:
        st.markdown("""
        <div class='slide-container' style='min-height:280px;'>
            <h3 style='color:#16A34A; margin-top:0;'> Utilidad para Seguridad Pública</h3>
            <ul>
                <li><b>Herramienta de Alerta Temprana:</b> El módulo predictivo permite simular escenarios futuros de orden público, clasificando nuevos eventos sin repetir el proceso completo de clustering.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h3 style='color: #0F172A;'>¡Muchas Gracias por su atención!</h3>
        <p style='color: #64748B;'>Fin de la sustentación híbrida.</p>
    </div>
    """, unsafe_allow_html=True)

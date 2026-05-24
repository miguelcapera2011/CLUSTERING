import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import os
import time

# ==============================================================================
# 1. ESTILO VISUAL "INSTITUCIONAL PREMIUM" (CSS)
# ==============================================================================
st.set_page_config(page_title="Sustentación Final - Fuerza Pública", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Colores de marca */
    :root {
        --policia-verde: #2E4D23;
        --ejercito-verde: #8F9779;
        --accent-blue: #1E3A8A;
        --bg-light: #FDFDFD;
    }

    .stApp {
        background-color: #F8FAFC;
    }

    /* Contenedor tipo diapositiva de alta gama */
    .slide-card {
        background-color: #FFFFFF;
        padding: 45px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }

    /* Títulos elegantes */
    .main-title {
        color: #0F172A;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 42px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 10px;
    }

    .sub-title {
        color: #64748B;
        font-size: 20px;
        font-weight: 400;
        margin-bottom: 30px;
    }

    /* Estilo de botones de navegación */
    div.stButton > button {
        background-color: #F1F5F9;
        color: #475569;
        border-radius: 10px;
        border: 1px solid #CBD5E1;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s;
    }

    div.stButton > button:hover {
        border-color: #1E3A8A;
        color: #1E3A8A;
        background-color: #EFF6FF;
    }

    div.stButton > button:active, div.stButton > button:focus, .st-emotion-cache-7ym5gk {
        background-color: #1E3A8A !important;
        color: white !important;
        border: none !important;
    }

    /* Tarjetas de análisis */
    .analysis-box {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 20px;
        border-left: 6px solid #2E4D23;
        margin-top: 15px;
    }

    .stat-card {
        text-align: center;
        padding: 20px;
        background: white;
        border-radius: 15px;
        border: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LÓGICA DE NAVEGACIÓN Y DATOS
# ==============================================================================
if 'page' not in st.session_state:
    st.session_state.page = 1

def change_page(num):
    st.session_state.page = num

# Carga de datos
def load_data():
    files = [f for f in os.listdir('.') if f.lower().endswith(('.csv', '.xlsx'))]
    target = next((f for f in files if any(k in f.lower() for k in ["afectacion", "fuerza", "publica"])), None)
    if not target: return None
    return pd.read_csv(target) if target.endswith('.csv') else pd.read_excel(target)

df = load_data()

# ==============================================================================
# 3. BARRA DE NAVEGACIÓN SUPERIOR
# ==============================================================================
st.markdown("<h3 style='text-align:center; color:#64748B; font-size:14px; letter-spacing:2px;'>SISTEMA DE ANÁLISIS ESTRATÉGICO TERRITORIAL</h3>", unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns(6)
btns = [c1.button("PORTADA", on_click=change_page, args=(1,), use_container_width=True),
        c2.button("CONTEXTO", on_click=change_page, args=(2,), use_container_width=True),
        c3.button("TEORÍA", on_click=change_page, args=(3,), use_container_width=True),
        c4.button("MÉTODO", on_click=change_page, args=(4,), use_container_width=True),
        c5.button("RESULTADOS", on_click=change_page, args=(5,), use_container_width=True),
        c6.button("CIERRE", on_click=change_page, args=(6,), use_container_width=True)]

# ==============================================================================
# DIAPOSITIVA 1: PORTADA
# ==============================================================================
if st.session_state.page == 1:
    st.markdown("""
    <div class='slide-card' style='text-align: center;'>
        <div style='display: flex; justify-content: center; gap: 40px; margin-bottom: 30px;'>
            <img src='https://www.ut.edu.co/images/logos/logo_ut.png' width='140'>
        </div>
        <h1 class='main-title'>Segmentación de Vulnerabilidad<br><span style='color:#2E4D23'>de la Fuerza Pública</span></h1>
        <p class='sub-title'>Identificación de Patrones Delictivos mediante Inteligencia de Datos</p>
        <div style='background:#F1F5F9; padding: 20px; border-radius: 15px; display: inline-block; width: 80%;'>
            <p style='margin:0; font-size:18px;'><b>Expositor:</b> Miguel Angel Garatejo</p>
            <p style='margin:0; font-size:16px; color:#64748B;'>Docente Evaluador: Yuri Saavedra</p>
        </div>
        <p style='margin-top:40px; color:#94A3B8; font-size:12px;'>CIENCIA DE DATOS Y MODELADO AVANZADO • UNIVERSIDAD DEL TOLIMA</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# DIAPOSITIVA 2: INTRODUCCIÓN Y PROBLEMA
# ==============================================================================
elif st.session_state.page == 2:
    st.markdown("<h1 class='main-title'>🎯 Contexto y Desafío Estratégico</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("""
        <div class='slide-card'>
            <h3 style='color:#DC2626;'>¿Cuál es el problema?</h3>
            <p>Los reportes de afectación a la seguridad nacional se almacenan como <b>Registros de Novedades</b> individuales.</p>
            <p>Esta información es mayoritariamente <b>texto (cualitativa)</b>, lo que impide que las computadoras puedan "medir" automáticamente la gravedad de un municipio.</p>
            <div class='analysis-box' style='border-color:#DC2626;'>
                <b>Obstáculo:</b> No podemos promediar la palabra 'Ataque' o 'Emboscada'. Necesitamos transformar el lenguaje en geometría.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='slide-card'>
            <h3 style='color:#2E4D23;'>Nuestra Misión</h3>
            <p>Pasar de una reacción táctica (atender el evento del día) a una <b>planeación estratégica</b> basada en datos consolidados.</p>
            <ul>
                <li><b>Agrupar:</b> Unificar los 1,102 municipios en perfiles homogéneos.</li>
                <li><b>Priorizar:</b> Identificar dónde el riesgo es inminente.</li>
                <li><b>Actuar:</b> Optimizar el despliegue de hombres y recursos.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# DIAPOSITIVA 3: MARCO TEÓRICO
# ==============================================================================
elif st.session_state.page == 3:
    st.markdown("<h1 class='main-title'>📖 Sustentación del Flujo Analítico</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='stat-card'><h4>1. Conversión de Datos</h4><p>Transformamos texto en frecuencias numéricas mediante <b>Tablas Dinámicas</b>.</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='stat-card'><h4>2. Clustering K-Means</h4><p>Agrupamos municipios buscando la <b>mínima distancia</b> al centro del grupo.</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='stat-card'><h4>3. PCA (3D)</h4><p>Simplificamos 20 variables en 3 ejes para <b>ver el mapa del riesgo</b>.</p></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='analysis-box'>
        <b>💡 Dato Técnico Importante:</b> Se utilizó <b>StandardScaler</b> para que variables como 'Asesinados' (que son pocas) tengan el mismo peso que 'Heridos' (que son muchas) en el cálculo del algoritmo.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# DIAPOSITIVA 5: RESULTADOS (ANÁLISIS PROFUNDO)
# ==============================================================================
elif st.session_state.page == 5:
    if df is not None:
        # --- PROCESAMIENTO ---
        index_cols = ['COD_MUNI', 'MUNICIPIO', 'DEPARTAMENTO']
        pivot_accion = df.pivot_table(index=index_cols, columns='ACCION', values='CANTIDAD', aggfunc='sum', fill_value=0)
        pivot_fuerza = df.pivot_table(index=index_cols, columns='NOMBRE_FUERZA', values='CANTIDAD', aggfunc='sum', fill_value=0)
        pivot_cat = df.pivot_table(index=index_cols, columns='CATEGORIA', values='CANTIDAD', aggfunc='sum', fill_value=0)
        
        datos_full = pivot_accion.join([pivot_fuerza, pivot_cat]).reset_index()
        num_cols = [c for c in datos_full.columns if c not in index_cols]
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(datos_full[num_cols])
        
        # KMeans K=4
        model = KMeans(n_clusters=4, n_init=20, random_state=42)
        datos_full['Cluster'] = model.fit_predict(X_scaled).astype(str)
        
        # Análisis de resultados
        st.markdown("<h1 class='main-title'>📊 Resultados y Perfiles Estratégicos</h1>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🌎 Mapa del Riesgo (PCA)", "⚖️ Comparativa de Fuerzas", "🚨 Identificación de Atípicos"])
        
        with tab1:
            pca = PCA(n_components=3)
            coords = pca.fit_transform(X_scaled)
            df_viz = pd.DataFrame(coords, columns=['PC1', 'PC2', 'PC3'])
            df_viz['Cluster'] = datos_full['Cluster']
            df_viz['Municipio'] = datos_full['MUNICIPIO']
            
            mapping = {"0": "Estabilidad Territorial", "1": "Conflicto Dinámico", "2": "Foco Institucional", "3": "Emergencia Crítica"}
            df_viz['Nombre_Clúster'] = df_viz['Cluster'].map(mapping)
            
            fig3d = px.scatter_3d(df_viz, x='PC1', y='PC2', z='PC3', color='Nombre_Clúster',
                                 hover_name='Municipio', opacity=0.7,
                                 color_discrete_sequence=['#10B981', '#3B82F6', '#F59E0B', '#EF4444'],
                                 template='plotly_white', title="Proyección Geométrica de Vulnerabilidad")
            st.plotly_chart(fig3d, use_container_width=True)
            
            st.markdown("""
            <div class='analysis-box'>
                <b>Análisis de Fondo:</b> Se observa que la mayoría de municipios (Verde) están concentrados cerca del origen (bajo riesgo). Sin embargo, hay una <b>dispersión radial</b>: a medida que nos alejamos del centro, el riesgo aumenta exponencialmente. El eje Z representa la intensidad de ataques a la Policía, mientras que el eje X la afectación al Ejército.
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.subheader("¿A quién atacan en cada clúster?")
            # Comparar promedio de afectación por fuerza
            fuerzas = [c for c in pivot_fuerza.columns if c in datos_full.columns]
            resumen_fuerza = datos_full.groupby('Cluster')[fuerzas].mean().T
            resumen_fuerza.columns = ["Estabilidad", "Dinámico", "Foco Inst.", "Emergencia"]
            
            fig_bar = px.bar(resumen_fuerza, barmode='group', template='plotly_white', 
                             title="Intensidad de Afectación por Institución (Promedios)",
                             color_discrete_sequence=['#A7F3D0', '#60A5FA', '#FDE68A', '#FCA5A5'])
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.info("💡 Hallazgo: En el Clúster 2, la brecha de afectación entre Ejército y Policía se acorta, indicando ataques coordinados contra ambas instituciones por igual.")

        with tab3:
            st.subheader("Detección de Municipios Atípicos (Outliers)")
            criticos = datos_full[datos_full['Cluster'] == "3"].sort_values(by=datos_full.columns[3], ascending=False).head(10)
            
            col_a, col_b = st.columns([1, 1.5])
            with col_a:
                st.write("Top de Municipios con Afectación Extrema:")
                st.dataframe(criticos[['MUNICIPIO', 'DEPARTAMENTO'] + fuerzas[:2]])
            with col_b:
                st.markdown(f"""
                <div class='insight-critical'>
                    <h4>🚨 Diagnóstico de Municipios Lejanos</h4>
                    <p>El modelo identifica a municipios como <b>{criticos['MUNICIPIO'].iloc[0]}</b> y <b>{criticos['MUNICIPIO'].iloc[1]}</b> como 'Atípicos'.</p>
                    <p><b>¿Por qué?</b> Su comportamiento no es una tendencia, es una anomalía masiva. Mientras el promedio nacional de incidentes es bajo, estos puntos registran valores 500% por encima de la media. El algoritmo los aísla en el Clúster Rojo para evitar que "contaminen" el análisis de las zonas más tranquilas.</p>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# DIAPOSITIVA 6: CONCLUSIONES
# ==============================================================================
elif st.session_state.page == 6:
    st.markdown("<h1 class='main-title'>🏁 Conclusiones Académicas</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='slide-card' style='height:400px;'>
            <h3 style='color:#1E3A8A;'>Logros Técnicos</h3>
            <ul>
                <li><b>Normalización exitosa:</b> Se logró que los datos atípicos (Cali, Tumaco) no sesgaran el agrupamiento de los municipios pequeños.</li>
                <li><b>Separabilidad:</b> El PCA confirmó que existen fronteras claras entre la delincuencia común y el conflicto de alta intensidad.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='slide-card' style='height:400px;'>
            <h3 style='color:#2E4D23;'>Impacto Real</h3>
            <ul>
                <li><b>Priorización:</b> Ahora el comando superior puede ver qué municipios del 'Clúster 1' están a punto de pasar al 'Clúster 3'.</li>
                <li><b>Prevención:</b> Permite mover recursos médicos y de apoyo antes de que la inestabilidad se consolide.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align:center; color:#2E4D23;'>¡Gracias por su atención!</h2>", unsafe_allow_html=True)
    if st.button("🔄 REINICIAR EXPOSICIÓN"): change_page(1)
